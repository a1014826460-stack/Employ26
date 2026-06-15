# Occupation Detail Matches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `public.occupation_detail_matches` as the canonical full-dataset occupation detail recognition result table, keeping retrieval Top10 while default downstream output remains Top1.

**Architecture:** The new table is a narrow PostgreSQL result layer keyed by `recruitment_record_id`, fed by batched reads from `public.recruitment_jobs_normalized`. Matching reuses the existing BGE occupation matcher and the current best finetuned `v1 + bge-large` artifact, then stores Top1 compatibility columns plus a `top10_candidates jsonb` audit/review payload. Existing analysis code is updated to prefer the new table without overwriting `public.skill_extraction_requirement_matches`.

**Tech Stack:** Python, pandas, SQLAlchemy, PostgreSQL JSONB, sentence-transformers, pytest.

---

## File Structure

- Create `src/db/occupation_detail_matches.py` for table DDL, candidate JSON serialization, and batch upsert.
- Create `src/data_pipeline/occupation_detail_match_full.py` for resumable full-dataset batch matching.
- Modify `src/skill_extraction/config.py` to expose `occupation_detail_match_table` and `occupation_detail_model_path`.
- Modify `src/analysis/structured_pg_source.py` to prefer `occupation_detail_match_table` for structured analysis.
- Modify `config/database.yaml` to register `public.occupation_detail_matches` and the best finetuned model path.
- Modify `Employ26-database.md` and `src/data_pipeline/README.md` to document the new table and run commands.
- Add tests in `src/tests/test_occupation_detail_matches.py`, `src/tests/test_occupation_detail_match_full.py`, `src/tests/test_skill_extraction_config.py`, and `src/tests/test_structured_pg_source.py`.

---

### Task 1: Add PostgreSQL Helper For The New Result Table

**Files:**
- Create: `src/db/occupation_detail_matches.py`
- Test: `src/tests/test_occupation_detail_matches.py`

- [ ] **Step 1: Write failing tests for candidate JSON serialization and upsert row shape**

Add this to `src/tests/test_occupation_detail_matches.py`:

```python
import json

import pandas as pd

from src.db.occupation_detail_matches import (
    DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
    build_occupation_detail_match_records,
    build_top_candidates,
)


def test_build_top_candidates_keeps_top10_order_and_fields():
    row = {
        "top1_code": "1",
        "top1_title": "人工智能工程技术人员",
        "top1_score": 0.91,
        "top1_detail_path": "工程技术 > 人工智能工程技术人员",
        "top1_detail_name": "人工智能工程技术人员",
        "top10_code": "10",
        "top10_title": "其他工程技术人员",
        "top10_score": 0.51,
        "top10_detail_path": "工程技术 > 其他工程技术人员",
        "top10_detail_name": "其他工程技术人员",
    }

    candidates = build_top_candidates(row, top_k=10)

    assert len(candidates) == 10
    assert candidates[0] == {
        "rank": 1,
        "code": "1",
        "title": "人工智能工程技术人员",
        "score": 0.91,
        "detail_path": "工程技术 > 人工智能工程技术人员",
        "detail_name": "人工智能工程技术人员",
    }
    assert candidates[-1]["rank"] == 10
    assert candidates[-1]["code"] == "10"


def test_build_occupation_detail_match_records_uses_top1_as_final_output():
    source_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "rid-1",
                "source_platform": "Liepin",
                "source_table": '"Liepin".raw_data',
                "source_row_number": 7,
                "job_title": "算法工程师",
            }
        ]
    )
    matched_df = pd.DataFrame(
        [
            {
                "query_text": "算法工程师。负责机器学习模型研发",
                "query_source": "job_title+job_description_raw",
                "selected_candidate_rank": 1,
                "top1_code": "2-02-10-09",
                "top1_title": "人工智能工程技术人员",
                "top1_score": 0.91,
                "大类": "专业技术人员",
                "中类": "工程技术人员",
                "小类": "信息和通信工程技术人员",
                "细类": "人工智能工程技术人员",
                "top1_detail_path": "专业技术人员 > 工程技术人员 > 信息和通信工程技术人员 > 人工智能工程技术人员",
                "top1_detail_name": "人工智能工程技术人员",
            }
        ]
    )

    records = build_occupation_detail_match_records(
        source_df=source_df,
        matched_df=matched_df,
        run_id="test-run",
        model_recipe="v1",
        base_model="bge-large-zh-v1.5",
        model_path="output/penghui/rag_round2_training/bge-large-round2-finetuned",
        top_k=10,
    )

    assert len(records) == 1
    record = records[0]
    assert record["target_table"] == DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE
    assert record["recruitment_record_id"] == "rid-1"
    assert record["occupation_code"] == "2-02-10-09"
    assert record["occupation_title"] == "人工智能工程技术人员"
    assert record["selected_candidate_rank"] == 1
    assert record["top_k"] == 10
    assert json.loads(record["top10_candidates"])[0]["rank"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_matches.py -v
```

Expected: FAIL because `src.db.occupation_detail_matches` does not exist.

- [ ] **Step 3: Implement the table helper**

Create `src/db/occupation_detail_matches.py`:

```python
"""PostgreSQL helpers for full-dataset occupation detail match results."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.db.postgres import create_pg_engine, ensure_schema
from src.db.recruitment_jobs_normalized import quote_table_name, split_table_name


DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE = "public.occupation_detail_matches"


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensure_occupation_detail_matches_table(
    connection,
    table_name: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
) -> None:
    """Ensure the canonical occupation detail result table exists."""
    schema_name, raw_table_name = split_table_name(table_name)
    ensure_schema(connection, schema_name)
    qualified_table = quote_table_name(table_name)
    object_prefix = raw_table_name.replace('"', "").replace(".", "_").strip("_")

    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                recruitment_record_id text PRIMARY KEY,
                source_platform text NOT NULL DEFAULT '',
                source_table text NOT NULL DEFAULT '',
                source_row_number bigint,
                job_title text NOT NULL DEFAULT '',
                query_text text NOT NULL DEFAULT '',
                query_source text NOT NULL DEFAULT '',
                occupation_code text NOT NULL DEFAULT '',
                occupation_title text NOT NULL DEFAULT '',
                "大类" text NOT NULL DEFAULT '',
                "中类" text NOT NULL DEFAULT '',
                "小类" text NOT NULL DEFAULT '',
                "细类" text NOT NULL DEFAULT '',
                top1_score double precision NOT NULL DEFAULT 0,
                is_matched boolean NOT NULL DEFAULT false,
                selected_candidate_rank integer NOT NULL DEFAULT 1,
                top_k integer NOT NULL DEFAULT 10,
                top10_candidates jsonb NOT NULL DEFAULT '[]'::jsonb,
                model_recipe text NOT NULL DEFAULT '',
                base_model text NOT NULL DEFAULT '',
                model_path text NOT NULL DEFAULT '',
                run_id text NOT NULL DEFAULT '',
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_occupation_code
            ON {qualified_table} (occupation_code)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_is_matched
            ON {qualified_table} (is_matched)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_updated_at
            ON {qualified_table} (updated_at)
            """
        )
    )


def build_top_candidates(matched_row: dict[str, Any] | pd.Series, top_k: int = 10) -> list[dict[str, Any]]:
    """Build ordered Top-K candidate JSON payload from matcher wide columns."""
    row = dict(matched_row)
    candidates: list[dict[str, Any]] = []
    for rank in range(1, int(top_k) + 1):
        prefix = f"top{rank}"
        candidates.append(
            {
                "rank": rank,
                "code": _safe_text(row.get(f"{prefix}_code", "")),
                "title": _safe_text(row.get(f"{prefix}_title", "")),
                "score": _safe_float(row.get(f"{prefix}_score", 0)),
                "detail_path": _safe_text(row.get(f"{prefix}_detail_path", "")),
                "detail_name": _safe_text(row.get(f"{prefix}_detail_name", "")),
            }
        )
    return candidates


def build_occupation_detail_match_records(
    *,
    source_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    run_id: str,
    model_recipe: str,
    base_model: str,
    model_path: str,
    top_k: int = 10,
    target_table: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
) -> list[dict[str, Any]]:
    """Convert source and matcher outputs into upsert-ready records."""
    if len(source_df) != len(matched_df):
        raise ValueError("source_df and matched_df must have the same row count")

    records: list[dict[str, Any]] = []
    for source_row, matched_row in zip(
        source_df.to_dict(orient="records"),
        matched_df.to_dict(orient="records"),
    ):
        top1_code = _safe_text(matched_row.get("top1_code", ""))
        top1_title = _safe_text(matched_row.get("top1_title", ""))
        records.append(
            {
                "target_table": target_table,
                "recruitment_record_id": _safe_text(source_row.get("recruitment_record_id", "")),
                "source_platform": _safe_text(source_row.get("source_platform", "")),
                "source_table": _safe_text(source_row.get("source_table", "")),
                "source_row_number": source_row.get("source_row_number", None),
                "job_title": _safe_text(source_row.get("job_title", "")),
                "query_text": _safe_text(matched_row.get("query_text", "")),
                "query_source": _safe_text(matched_row.get("query_source", "job_title+job_description_raw")),
                "occupation_code": top1_code,
                "occupation_title": top1_title,
                "大类": _safe_text(matched_row.get("大类", "")),
                "中类": _safe_text(matched_row.get("中类", "")),
                "小类": _safe_text(matched_row.get("小类", "")),
                "细类": _safe_text(matched_row.get("细类", "")),
                "top1_score": _safe_float(matched_row.get("top1_score", 0)),
                "is_matched": bool(top1_code and top1_title),
                "selected_candidate_rank": 1,
                "top_k": int(top_k),
                "top10_candidates": json.dumps(
                    build_top_candidates(matched_row, top_k=top_k),
                    ensure_ascii=False,
                ),
                "model_recipe": model_recipe,
                "base_model": base_model,
                "model_path": model_path,
                "run_id": run_id,
            }
        )
    return records


def upsert_occupation_detail_match_records(
    records: list[dict[str, Any]],
    table_name: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
) -> int:
    """Batch upsert occupation detail match records."""
    if not records:
        return 0

    qualified_table = quote_table_name(table_name)
    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_occupation_detail_matches_table(connection, table_name=table_name)
            connection.execute(
                text(
                    f"""
                    INSERT INTO {qualified_table} (
                        recruitment_record_id,
                        source_platform,
                        source_table,
                        source_row_number,
                        job_title,
                        query_text,
                        query_source,
                        occupation_code,
                        occupation_title,
                        "大类",
                        "中类",
                        "小类",
                        "细类",
                        top1_score,
                        is_matched,
                        selected_candidate_rank,
                        top_k,
                        top10_candidates,
                        model_recipe,
                        base_model,
                        model_path,
                        run_id
                    )
                    VALUES (
                        :recruitment_record_id,
                        :source_platform,
                        :source_table,
                        :source_row_number,
                        :job_title,
                        :query_text,
                        :query_source,
                        :occupation_code,
                        :occupation_title,
                        :大类,
                        :中类,
                        :小类,
                        :细类,
                        :top1_score,
                        :is_matched,
                        :selected_candidate_rank,
                        :top_k,
                        CAST(:top10_candidates AS jsonb),
                        :model_recipe,
                        :base_model,
                        :model_path,
                        :run_id
                    )
                    ON CONFLICT (recruitment_record_id)
                    DO UPDATE SET
                        source_platform = EXCLUDED.source_platform,
                        source_table = EXCLUDED.source_table,
                        source_row_number = EXCLUDED.source_row_number,
                        job_title = EXCLUDED.job_title,
                        query_text = EXCLUDED.query_text,
                        query_source = EXCLUDED.query_source,
                        occupation_code = EXCLUDED.occupation_code,
                        occupation_title = EXCLUDED.occupation_title,
                        "大类" = EXCLUDED."大类",
                        "中类" = EXCLUDED."中类",
                        "小类" = EXCLUDED."小类",
                        "细类" = EXCLUDED."细类",
                        top1_score = EXCLUDED.top1_score,
                        is_matched = EXCLUDED.is_matched,
                        selected_candidate_rank = EXCLUDED.selected_candidate_rank,
                        top_k = EXCLUDED.top_k,
                        top10_candidates = EXCLUDED.top10_candidates,
                        model_recipe = EXCLUDED.model_recipe,
                        base_model = EXCLUDED.base_model,
                        model_path = EXCLUDED.model_path,
                        run_id = EXCLUDED.run_id,
                        updated_at = now()
                    """
                ),
                records,
            )
    finally:
        engine.dispose()
    return len(records)
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_matches.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/db/occupation_detail_matches.py src/tests/test_occupation_detail_matches.py
git commit -m "feat: add occupation detail match result table helper"
```

---

### Task 2: Add Full-Dataset Batch Matching CLI

**Files:**
- Create: `src/data_pipeline/occupation_detail_match_full.py`
- Test: `src/tests/test_occupation_detail_match_full.py`

- [ ] **Step 1: Write failing tests for query construction and batch SQL**

Add this to `src/tests/test_occupation_detail_match_full.py`:

```python
import pandas as pd

from src.data_pipeline.occupation_detail_match_full import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MODEL_RECIPE,
    DEFAULT_TOP_K,
    build_match_input_dataframe,
    build_unmatched_batch_query,
)


def test_build_match_input_dataframe_uses_normalized_columns():
    jobs_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "rid-1",
                "source_platform": "Liepin",
                "source_table": '"Liepin".raw_data',
                "source_row_number": 1,
                "job_title": "算法工程师",
                "job_description_raw": "负责推荐系统模型研发。",
            }
        ]
    )

    result = build_match_input_dataframe(jobs_df)

    assert result.loc[0, "岗位名称"] == "算法工程师"
    assert result.loc[0, "岗位描述"] == "负责推荐系统模型研发。"
    assert result.loc[0, "职业匹配来源"] == "job_title+job_description_raw"
    assert DEFAULT_TOP_K == 10
    assert DEFAULT_MODEL_RECIPE == "v1"
    assert DEFAULT_BASE_MODEL == "bge-large-zh-v1.5"


def test_build_unmatched_batch_query_skips_existing_rows_when_resume_enabled():
    query = build_unmatched_batch_query(
        normalized_table="public.recruitment_jobs_normalized",
        target_table="public.occupation_detail_matches",
        resume=True,
    )

    assert '"public"."recruitment_jobs_normalized"' in query
    assert '"public"."occupation_detail_matches"' in query
    assert "not exists" in query.lower()
    assert "limit :batch_size" in query.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_match_full.py -v
```

Expected: FAIL because `src.data_pipeline.occupation_detail_match_full` does not exist.

- [ ] **Step 3: Implement the CLI**

Create `src/data_pipeline/occupation_detail_match_full.py`:

```python
"""Full-dataset occupation detail matching into PostgreSQL."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db.occupation_detail_matches import (
    DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
    build_occupation_detail_match_records,
    ensure_occupation_detail_matches_table,
    upsert_occupation_detail_match_records,
)
from src.db.postgres import create_pg_engine
from src.db.recruitment_jobs_normalized import (
    DEFAULT_NORMALIZED_TABLE,
    quote_table_name,
)
from src.skill_extraction.config import load_skill_extraction_config


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_TOP_K = 10
DEFAULT_MODEL_RECIPE = "v1"
DEFAULT_BASE_MODEL = "bge-large-zh-v1.5"
DEFAULT_MODEL_PATH = "output/penghui/rag_round2_training/bge-large-round2-finetuned"


def build_match_input_dataframe(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized jobs into the matcher input contract."""
    work_df = jobs_df.copy()
    work_df["岗位名称"] = work_df["job_title"].fillna("").astype(str)
    work_df["岗位描述"] = work_df["job_description_raw"].fillna("").astype(str)
    work_df["职业匹配来源"] = "job_title+job_description_raw"
    return work_df


def build_unmatched_batch_query(
    *,
    normalized_table: str,
    target_table: str,
    resume: bool,
) -> str:
    """Build stable batch query for normalized jobs."""
    normalized = quote_table_name(normalized_table)
    target = quote_table_name(target_table)
    resume_filter = (
        f"""
        AND NOT EXISTS (
            SELECT 1
            FROM {target} m
            WHERE m.recruitment_record_id = n.recruitment_record_id
        )
        """
        if resume
        else ""
    )
    return f"""
        SELECT
            n.recruitment_record_id,
            n.source_platform,
            n.source_table,
            n.source_row_number,
            n.job_title,
            n.job_description_raw
        FROM {normalized} n
        WHERE COALESCE(n.recruitment_record_id, '') <> ''
          AND COALESCE(n.job_title, '') <> ''
          {resume_filter}
        ORDER BY n.recruitment_record_id
        LIMIT :batch_size
    """


def load_job_batch(
    *,
    connection,
    normalized_table: str,
    target_table: str,
    batch_size: int,
    resume: bool,
) -> pd.DataFrame:
    """Load one unmatched batch from PostgreSQL."""
    query = build_unmatched_batch_query(
        normalized_table=normalized_table,
        target_table=target_table,
        resume=resume,
    )
    return pd.read_sql_query(text(query), connection, params={"batch_size": int(batch_size)})


def run_full_occupation_detail_matching(
    *,
    database_config_path: str | Path | None = None,
    normalized_table: str = DEFAULT_NORMALIZED_TABLE,
    target_table: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
    model_path: str = DEFAULT_MODEL_PATH,
    model_recipe: str = DEFAULT_MODEL_RECIPE,
    base_model: str = DEFAULT_BASE_MODEL,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 20000,
    limit_rows: int | None = None,
    resume: bool = True,
    run_id: str | None = None,
) -> int:
    """Run resumable full-dataset occupation detail matching."""
    config = load_skill_extraction_config(database_config_path=database_config_path)
    matcher_config = config.__class__(
        **{
            **config.__dict__,
            "embedding_model_path": Path(model_path),
            "catalog_embedding_cache_path": config.cache_dir
            / f"occupation_catalog_embeddings_{Path(model_path).name}.npy",
        }
    )
    from src.skill_extraction.bge_matcher import OccupationBGEMatcher

    run_id = run_id or datetime.now().strftime("occupation-detail-%Y%m%d-%H%M%S")
    matcher = OccupationBGEMatcher(matcher_config)
    matcher.build_index()

    total_written = 0
    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_occupation_detail_matches_table(connection, table_name=target_table)

        while True:
            if limit_rows is not None and total_written >= int(limit_rows):
                break

            effective_batch_size = int(batch_size)
            if limit_rows is not None:
                effective_batch_size = min(effective_batch_size, int(limit_rows) - total_written)

            with engine.connect() as connection:
                jobs_df = load_job_batch(
                    connection=connection,
                    normalized_table=normalized_table,
                    target_table=target_table,
                    batch_size=effective_batch_size,
                    resume=resume,
                )

            if jobs_df.empty:
                logger.info("没有更多待匹配招聘记录，结束。")
                break

            match_input_df = build_match_input_dataframe(jobs_df)
            matched_df = matcher.match_jobs(match_input_df, top_k=top_k)
            records = build_occupation_detail_match_records(
                source_df=jobs_df,
                matched_df=matched_df,
                run_id=run_id,
                model_recipe=model_recipe,
                base_model=base_model,
                model_path=model_path,
                top_k=top_k,
                target_table=target_table,
            )
            written = upsert_occupation_detail_match_records(records, table_name=target_table)
            total_written += written
            logger.info("本批写入 %s 行，累计写入 %s 行", written, total_written)
    finally:
        engine.dispose()
    return total_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="全量职业细类识别: Top10 检索 + Top1 默认输出 + PostgreSQL 入库")
    parser.add_argument("--database-config", default=None, help="数据库配置文件路径，默认 config/database.yaml")
    parser.add_argument("--normalized-table", default=DEFAULT_NORMALIZED_TABLE, help="招聘统一规范层表")
    parser.add_argument("--target-table", default=DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE, help="职业细类识别结果表")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="finetuned embedding 模型路径")
    parser.add_argument("--model-recipe", default=DEFAULT_MODEL_RECIPE, help="模型配方名")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="底座模型名")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="底层检索候选数，默认 10")
    parser.add_argument("--batch-size", type=int, default=20000, help="每批招聘记录数")
    parser.add_argument("--limit-rows", type=int, default=None, help="调试/试跑时限制总处理行数")
    parser.add_argument("--run-id", default=None, help="本次运行 ID，默认自动生成")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已有 recruitment_record_id，强制重算覆盖")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_full_occupation_detail_matching(
        database_config_path=args.database_config,
        normalized_table=args.normalized_table,
        target_table=args.target_table,
        model_path=args.model_path,
        model_recipe=args.model_recipe,
        base_model=args.base_model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows,
        resume=not args.no_resume,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_match_full.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/data_pipeline/occupation_detail_match_full.py src/tests/test_occupation_detail_match_full.py
git commit -m "feat: add full occupation detail matching pipeline"
```

---

### Task 3: Add Config Entries For The New Canonical Table And Model

**Files:**
- Modify: `config/database.yaml`
- Modify: `src/skill_extraction/config.py`
- Test: `src/tests/test_skill_extraction_config.py`

- [ ] **Step 1: Write failing config test**

Append this to `src/tests/test_skill_extraction_config.py`:

```python
from src.skill_extraction.config import load_skill_extraction_config


def test_skill_extraction_config_exposes_occupation_detail_match_defaults():
    config = load_skill_extraction_config()

    assert config.occupation_detail_match_table == "public.occupation_detail_matches"
    assert str(config.occupation_detail_model_path).replace("\\", "/").endswith(
        "output/penghui/rag_round2_training/bge-large-round2-finetuned"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_skill_extraction_config.py -v
```

Expected: FAIL because the config object has no `occupation_detail_match_table` field.

- [ ] **Step 3: Update `config/database.yaml`**

Add these entries:

```yaml
tables:
  processing_results:
    occupation_detail_matches: public.occupation_detail_matches

skill_extraction:
  occupation_detail_match_table: public.occupation_detail_matches
  occupation_detail_model_path: output/penghui/rag_round2_training/bge-large-round2-finetuned
  occupation_detail_top_k: 10
```

Keep the existing `skill_extraction.requirement_match_table: public.skill_extraction_requirement_matches` unchanged so the old requirement-prep table is not overwritten.

- [ ] **Step 4: Update `SkillExtractionConfig`**

In `src/skill_extraction/config.py`, add fields to the dataclass:

```python
    occupation_detail_match_table: str
    occupation_detail_model_path: Path
    occupation_detail_top_k: int
```

Then add these constructor arguments in `load_skill_extraction_config`:

```python
        occupation_detail_match_table=qualify_table_name(
            skill_settings.get(
                "occupation_detail_match_table",
                raw_config.get("tables", {}).get("processing_results", {}).get(
                    "occupation_detail_matches",
                    "public.occupation_detail_matches",
                ),
            )
        ),
        occupation_detail_model_path=PROJECT_ROOT
        / skill_settings.get(
            "occupation_detail_model_path",
            "output/penghui/rag_round2_training/bge-large-round2-finetuned",
        ),
        occupation_detail_top_k=max(1, int(skill_settings.get("occupation_detail_top_k", 10))),
```

- [ ] **Step 5: Run config tests**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_skill_extraction_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add config/database.yaml src/skill_extraction/config.py src/tests/test_skill_extraction_config.py
git commit -m "feat: configure canonical occupation detail matches"
```

---

### Task 4: Point Structured Analysis At The New Result Table

**Files:**
- Modify: `src/analysis/structured_pg_source.py`
- Test: `src/tests/test_structured_pg_source.py`

- [ ] **Step 1: Add failing test for config preference**

Append this to `src/tests/test_structured_pg_source.py`:

```python
from pathlib import Path

from src.analysis.structured_pg_source import load_default_structured_source_config


class _FakeSkillConfig:
    recruitment_normalized_table = "public.recruitment_jobs_normalized"
    requirement_match_table = "public.skill_extraction_requirement_matches"
    occupation_detail_match_table = "public.occupation_detail_matches"


def test_structured_source_prefers_occupation_detail_match_table(monkeypatch):
    monkeypatch.setattr(
        "src.analysis.structured_pg_source.load_skill_extraction_config",
        lambda: _FakeSkillConfig(),
    )

    config = load_default_structured_source_config()

    assert config.normalized_table == "public.recruitment_jobs_normalized"
    assert config.occupation_match_table == "public.occupation_detail_matches"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_structured_pg_source.py::test_structured_source_prefers_occupation_detail_match_table -v
```

Expected: FAIL because `load_default_structured_source_config` still reads `requirement_match_table`.

- [ ] **Step 3: Update structured source config loading**

In `src/analysis/structured_pg_source.py`, replace `load_default_structured_source_config` with:

```python
def load_default_structured_source_config() -> StructuredSourceConfig:
    """从项目配置读取结构化统计主输入表。"""
    skill_config = load_skill_extraction_config()
    return StructuredSourceConfig(
        normalized_table=skill_config.recruitment_normalized_table,
        occupation_match_table=getattr(
            skill_config,
            "occupation_detail_match_table",
            skill_config.requirement_match_table,
        ),
    )
```

- [ ] **Step 4: Run structured source tests**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_structured_pg_source.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add src/analysis/structured_pg_source.py src/tests/test_structured_pg_source.py
git commit -m "feat: use canonical occupation detail matches in analysis"
```

---

### Task 5: Wire CLI Defaults To Config Values

**Files:**
- Modify: `src/data_pipeline/occupation_detail_match_full.py`
- Test: `src/tests/test_occupation_detail_match_full.py`

- [ ] **Step 1: Add failing test for config-driven defaults**

Append this to `src/tests/test_occupation_detail_match_full.py`:

```python
from pathlib import Path

from src.data_pipeline.occupation_detail_match_full import resolve_runtime_defaults


class _FakeConfig:
    recruitment_normalized_table = "public.recruitment_jobs_normalized"
    occupation_detail_match_table = "public.occupation_detail_matches"
    occupation_detail_model_path = Path("output/penghui/rag_round2_training/bge-large-round2-finetuned")
    occupation_detail_top_k = 10


def test_resolve_runtime_defaults_uses_config_when_args_are_empty():
    defaults = resolve_runtime_defaults(_FakeConfig())

    assert defaults["normalized_table"] == "public.recruitment_jobs_normalized"
    assert defaults["target_table"] == "public.occupation_detail_matches"
    assert defaults["top_k"] == 10
    assert defaults["model_path"].endswith("bge-large-round2-finetuned")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_match_full.py::test_resolve_runtime_defaults_uses_config_when_args_are_empty -v
```

Expected: FAIL because `resolve_runtime_defaults` does not exist.

- [ ] **Step 3: Implement config default resolver**

Add this to `src/data_pipeline/occupation_detail_match_full.py`:

```python
def resolve_runtime_defaults(config) -> dict[str, object]:
    """Resolve runtime defaults from project config."""
    return {
        "normalized_table": config.recruitment_normalized_table,
        "target_table": config.occupation_detail_match_table,
        "model_path": str(config.occupation_detail_model_path),
        "top_k": int(config.occupation_detail_top_k),
    }
```

Then update `run_full_occupation_detail_matching` so `normalized_table`, `target_table`, `model_path`, and `top_k` accept `None` and are filled after config load:

```python
    config = load_skill_extraction_config(database_config_path=database_config_path)
    defaults = resolve_runtime_defaults(config)
    normalized_table = normalized_table or str(defaults["normalized_table"])
    target_table = target_table or str(defaults["target_table"])
    model_path = model_path or str(defaults["model_path"])
    top_k = int(top_k or defaults["top_k"])
```

Update CLI parser defaults for these arguments to `None`:

```python
    parser.add_argument("--normalized-table", default=None, help="招聘统一规范层表")
    parser.add_argument("--target-table", default=None, help="职业细类识别结果表")
    parser.add_argument("--model-path", default=None, help="finetuned embedding 模型路径")
    parser.add_argument("--top-k", type=int, default=None, help="底层检索候选数，默认读取配置")
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_match_full.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add src/data_pipeline/occupation_detail_match_full.py src/tests/test_occupation_detail_match_full.py
git commit -m "feat: load occupation detail matching defaults from config"
```

---

### Task 6: Document The New Table And Operating Commands

**Files:**
- Modify: `Employ26-database.md`
- Modify: `src/data_pipeline/README.md`

- [ ] **Step 1: Update database documentation**

In `Employ26-database.md`, add a section near `public.skill_extraction_requirement_matches`:

```markdown
##### `public.occupation_detail_matches`

- 用途：全量招聘记录的正式职业细类识别结果层
- 主键：`recruitment_record_id`
- 输入来源：`public.recruitment_jobs_normalized`
- 默认模型：`output/penghui/rag_round2_training/bge-large-round2-finetuned`
- 检索策略：底层保留 Top10，默认职业输出取 Top1

关键字段：

- `recruitment_record_id`
- `source_platform`
- `source_table`
- `source_row_number`
- `job_title`
- `query_text`
- `query_source`
- `occupation_code`
- `occupation_title`
- `大类`
- `中类`
- `小类`
- `细类`
- `top1_score`
- `is_matched`
- `selected_candidate_rank`
- `top_k`
- `top10_candidates` (`jsonb`)
- `model_recipe`
- `base_model`
- `model_path`
- `run_id`
- `created_at`
- `updated_at`

说明：

- 这张表是后续结构化统计默认读取的职业识别结果层。
- `occupation_code` / `occupation_title` 固定表示 Top1 结果。
- `top10_candidates` 用于人工复核、误差分析和后续 rerank，不改变默认输出口径。
- `public.skill_extraction_requirement_matches` 保留为旧的任职要求切分与技能抽取预处理结果表，不再承担 1052 万行全量职业识别结果层职责。
```

Also add indexes:

```sql
create index if not exists idx_occupation_detail_matches_occupation_code
on public.occupation_detail_matches (occupation_code);

create index if not exists idx_occupation_detail_matches_is_matched
on public.occupation_detail_matches (is_matched);

create index if not exists idx_occupation_detail_matches_updated_at
on public.occupation_detail_matches (updated_at);
```

- [ ] **Step 2: Update pipeline README**

In `src/data_pipeline/README.md`, add:

```markdown
### `occupation_detail_match_full.py`

全量职业细类识别脚本。它从 `public.recruitment_jobs_normalized` 分批读取招聘记录，使用当前最佳 `v1 + bge-large` finetuned 模型检索职业细类，底层保留 Top10 候选，默认结果字段仍取 Top1，并写入 `public.occupation_detail_matches`。

试跑 1000 行：

```bash
python -m src.data_pipeline.occupation_detail_match_full --limit-rows 1000 --batch-size 1000
```

正式断点续跑：

```bash
python -m src.data_pipeline.occupation_detail_match_full --batch-size 20000
```

强制重算覆盖已有结果：

```bash
python -m src.data_pipeline.occupation_detail_match_full --batch-size 20000 --no-resume
```

默认配置来自 `config/database.yaml`：

- `skill_extraction.occupation_detail_match_table`
- `skill_extraction.occupation_detail_model_path`
- `skill_extraction.occupation_detail_top_k`
```

- [ ] **Step 3: Commit Task 6**

Run:

```bash
git add Employ26-database.md src/data_pipeline/README.md
git commit -m "docs: document canonical occupation detail matching table"
```

---

### Task 7: Run Verification And A Safe Dry Run

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
.\.conda\python.exe -m pytest src/tests/test_occupation_detail_matches.py src/tests/test_occupation_detail_match_full.py src/tests/test_skill_extraction_config.py src/tests/test_structured_pg_source.py -v
```

Expected: PASS.

- [ ] **Step 2: Run a 100-row database dry run**

Run:

```bash
.\.conda\python.exe -m src.data_pipeline.occupation_detail_match_full --limit-rows 100 --batch-size 100
```

Expected:

- The script creates `public.occupation_detail_matches` if needed.
- The script writes 100 rows.
- Logs show `top_k=10` behavior through the matcher.

- [ ] **Step 3: Validate PostgreSQL output shape**

Run:

```bash
.\.conda\python.exe - <<'PY'
from sqlalchemy import text
from src.db.postgres import create_pg_engine

engine = create_pg_engine()
with engine.connect() as conn:
    row = conn.execute(text("""
        select
            count(*) as rows,
            count(*) filter (where occupation_code <> '') as matched_rows,
            min(jsonb_array_length(top10_candidates)) as min_candidates,
            max(jsonb_array_length(top10_candidates)) as max_candidates
        from public.occupation_detail_matches
    """)).mappings().one()
    print(dict(row))
engine.dispose()
PY
```

Expected:

- `rows >= 100`
- `matched_rows >= 1`
- `min_candidates = 10`
- `max_candidates = 10`

- [ ] **Step 4: Validate structured analysis can read the new table**

Run:

```bash
.\.conda\python.exe - <<'PY'
from src.analysis.structured_pg_source import build_structured_source_coverage

print(build_structured_source_coverage())
PY
```

Expected:

- `occupation_match_table` is `public.occupation_detail_matches`.
- `match_join_key` is `recruitment_record_id`.
- No missing-column error is raised.

- [ ] **Step 5: Commit verification-only doc updates if any**

If dry-run findings require documentation changes, commit them:

```bash
git add Employ26-database.md src/data_pipeline/README.md
git commit -m "docs: record occupation detail matching dry run notes"
```

If there are no changes, skip this commit.

---

### Task 8: Start Full Run Only After Dry Run Looks Healthy

**Files:**
- No code changes expected.

- [ ] **Step 1: Confirm dry-run row count and candidate shape**

Run:

```bash
.\.conda\python.exe - <<'PY'
from sqlalchemy import text
from src.db.postgres import create_pg_engine

engine = create_pg_engine()
with engine.connect() as conn:
    print(conn.execute(text("""
        select
            count(*) as result_rows,
            count(*) filter (where is_matched) as matched_rows,
            round(avg(top1_score)::numeric, 6) as avg_top1_score
        from public.occupation_detail_matches
    """)).mappings().one())
engine.dispose()
PY
```

Expected: The result rows from the dry run are present and average score is not null.

- [ ] **Step 2: Run the full resumable job**

Run:

```bash
.\.conda\python.exe -m src.data_pipeline.occupation_detail_match_full --batch-size 20000
```

Expected:

- The script skips existing `recruitment_record_id` rows.
- It writes batches until all rows from `public.recruitment_jobs_normalized` with non-empty `job_title` are covered.
- It can be stopped and restarted with the same command.

- [ ] **Step 3: Monitor coverage during the run**

Run this in a separate terminal:

```bash
.\.conda\python.exe - <<'PY'
from sqlalchemy import text
from src.db.postgres import create_pg_engine

engine = create_pg_engine()
with engine.connect() as conn:
    row = conn.execute(text("""
        select
            (select count(*) from public.recruitment_jobs_normalized where coalesce(job_title, '') <> '') as eligible_rows,
            (select count(*) from public.occupation_detail_matches) as matched_rows
    """)).mappings().one()
    print(dict(row))
engine.dispose()
PY
```

Expected: `matched_rows` increases toward `eligible_rows`.

---

## Self-Review

- Spec coverage: The plan creates the new canonical result table, stores Top10 as JSONB, keeps Top1 as default output, avoids overwriting `public.skill_extraction_requirement_matches`, and updates structured analysis to read the new table.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: Table names, config fields, function names, and CLI flags are consistent across tasks.
- Risk: The full 1052 万行 run is intentionally gated behind a dry run because encoding volume and PostgreSQL write throughput may require batch-size tuning.
