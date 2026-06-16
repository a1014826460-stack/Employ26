"""
职业细类 BGE 匹配模块。

职责：
1. 使用本地微调模型 `D:\\model\\bge-base-zh-finetuned` 对岗位文本和职业细类做向量编码；
2. 返回 Top-K 候选；
3. 按“Top1 -> Top5 依次回退”的规则选择首个可用细类。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from src.db.postgres import create_pg_engine
from src.db.recruitment_jobs_normalized import quote_table_name
from .config import SkillExtractionConfig


logger = logging.getLogger(__name__)


def _safe_text(value: object) -> str:
    """安全转文本，统一去除空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _format_occupation_code(value: object) -> str:
    """Format 7-digit occupation codes as x-xx-xx-xx when possible."""
    text = _safe_text(value)
    digits = re.sub(r"\D", "", text)
    if len(digits) == 7:
        return f"{digits[0]}-{digits[1:3]}-{digits[3:5]}-{digits[5:7]}"
    return text


def _build_catalog_search_text(row: pd.Series) -> str:
    """构造职业细类向量检索文本。"""
    parts = [
        _safe_text(row.get("大类", "")),
        _safe_text(row.get("中类", "")),
        _safe_text(row.get("小类", "")),
        _safe_text(row.get("title", "")) or _safe_text(row.get("title_clean", "")),
        _safe_text(row.get("desc_clean", "")) or _safe_text(row.get("desc", "")),
        _safe_text(row.get("task_text_joined", "")) or _safe_text(row.get("tasks", "")),
    ]
    return " | ".join([part for part in parts if part])


def _build_job_query_text(row: pd.Series, desc_limit: int = 220) -> str:
    """构造岗位查询文本，优先保留岗位名称，再补充描述片段。"""
    title = _safe_text(row.get("岗位名称", ""))
    desc = _safe_text(row.get("岗位描述", ""))
    if desc and len(desc) > desc_limit:
        desc = desc[:desc_limit]

    parts = [part for part in [title, desc] if part]
    return "。".join(parts)


@dataclass(frozen=True)
class OccupationCandidate:
    """职业细类候选。"""

    rank: int
    score: float
    code: str
    title: str
    大类: str
    中类: str
    小类: str
    细类: str
    detail_path: str


class OccupationBGEMatcher:
    """基于 BGE 的职业细类匹配器。"""

    def __init__(self, config: SkillExtractionConfig):
        self.config = config
        self.model = SentenceTransformer(
            str(config.embedding_model_path),
            device=config.embedding_device,
        )
        self.catalog_df = pd.DataFrame()
        self.catalog_embeddings: np.ndarray | None = None

    def load_catalog(self) -> pd.DataFrame:
        """加载职业目录，并仅保留可映射到细类的记录。"""
        qualified_table = quote_table_name(self.config.catalog_preprocessed_table)
        query = f"""
            SELECT
                code,
                title,
                "desc" AS desc,
                tasks,
                "大类" AS 大类,
                "中类" AS 中类,
                "小类" AS 小类,
                "细类" AS 细类,
                task_text_joined,
                title_clean,
                desc_clean,
                hierarchy_text
            FROM {qualified_table}
            WHERE node_type = 'occupation_leaf'
        """
        engine = create_pg_engine(application_name="occupation_bge_catalog")
        try:
            with engine.connect() as connection:
                catalog_df = pd.read_sql_query(text(query), connection)
        except Exception as exc:
            logger.warning("从 PostgreSQL 加载职业目录失败，回退 DuckDB: %s", exc)
            duckdb_query = query.replace(qualified_table, self.config.catalog_preprocessed_table)
            with duckdb.connect(str(self.config.db_path)) as conn:
                conn.execute(f"PRAGMA threads={self.config.duckdb_threads}")
                catalog_df = conn.execute(duckdb_query).df()
        finally:
            engine.dispose()

        if catalog_df.empty:
            raise ValueError("职业目录表为空，无法进行职业细类匹配")

        work_df = catalog_df.copy()
        work_df["code"] = work_df["code"].map(_format_occupation_code)
        work_df["title"] = work_df["title"].fillna("").astype(str).str.strip()
        work_df = work_df[(work_df["code"] != "") & (work_df["title"] != "")].copy()
        if work_df.empty:
            raise ValueError("职业目录中没有可用的职业细类记录")

        # The unified catalog can contain inconsistent hierarchy detail fields.
        # For occupation leaves, code/title are the canonical fine-grained label.
        work_df["细类"] = work_df["title"]
        work_df["detail_path"] = (
            work_df[["大类", "中类", "小类", "细类"]]
            .fillna("")
            .astype(str)
            .apply(lambda row: " > ".join([part.strip() for part in row.tolist() if part.strip()]), axis=1)
        )
        work_df["detail_name"] = work_df["细类"].astype(str)
        work_df["search_text"] = work_df.apply(_build_catalog_search_text, axis=1)
        work_df = work_df.reset_index(drop=True)

        self.catalog_df = work_df
        logger.info("已加载职业细类目录: %s 条", len(work_df))
        return self.catalog_df

    def build_index(self, force_rebuild: bool = False) -> None:
        """为职业细类目录构建或加载向量缓存。"""
        if self.catalog_df.empty:
            self.load_catalog()

        cache_path = self.config.catalog_embedding_cache_path
        if cache_path.exists() and not force_rebuild:
            self.catalog_embeddings = np.load(cache_path)
            logger.info("已加载职业细类向量缓存: %s", cache_path)
            return

        texts = self.catalog_df["search_text"].fillna("").astype(str).tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.catalog_embeddings = np.asarray(embeddings, dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, self.catalog_embeddings)
        logger.info("已生成职业细类向量缓存: %s", cache_path)

    def _match_query_texts(
        self,
        query_df: pd.DataFrame,
        query_texts: List[str],
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """执行底层向量检索，并统一整理输出字段。"""
        if query_df.empty:
            return pd.DataFrame()
        if self.catalog_embeddings is None:
            self.build_index()

        top_k = min(int(top_k or self.config.match_top_k), len(self.catalog_df))
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

        score_matrix = np.matmul(query_embeddings, self.catalog_embeddings.T)
        top_indices = np.argpartition(score_matrix, -top_k, axis=1)[:, -top_k:]
        top_scores = np.take_along_axis(score_matrix, top_indices, axis=1)

        order = np.argsort(-top_scores, axis=1)
        sorted_indices = np.take_along_axis(top_indices, order, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)

        matched_rows: List[Dict] = []
        for row_index, job_row in enumerate(query_df.to_dict(orient="records")):
            candidates: List[OccupationCandidate] = []
            for rank_index in range(top_k):
                catalog_index = int(sorted_indices[row_index, rank_index])
                catalog_row = self.catalog_df.iloc[catalog_index]
                candidates.append(
                    OccupationCandidate(
                        rank=rank_index + 1,
                        score=float(sorted_scores[row_index, rank_index]),
                        code=_safe_text(catalog_row.get("code", "")),
                        title=_safe_text(catalog_row.get("title", "")),
                        大类=_safe_text(catalog_row.get("大类", "")),
                        中类=_safe_text(catalog_row.get("中类", "")),
                        小类=_safe_text(catalog_row.get("小类", "")),
                        细类=_safe_text(catalog_row.get("细类", "")),
                        detail_path=_safe_text(catalog_row.get("detail_path", "")),
                    )
                )

            selected = self._select_candidate(candidates)
            result_row = {
                "job_id": job_row.get("job_id", None),
                "job_title": _safe_text(job_row.get("岗位名称", "")),
                "query_text": query_texts[row_index],
                "query_source": _safe_text(job_row.get("职业匹配来源", "")),
                "selected_candidate_rank": selected.rank if selected else None,
                "top1_code": selected.code if selected else "",
                "top1_title": selected.title if selected else "",
                "top1_score": selected.score if selected else 0.0,
                "detail_path": selected.detail_path if selected else "",
                "detail_name": selected.细类 if selected else "",
                "大类": selected.大类 if selected else "",
                "中类": selected.中类 if selected else "",
                "小类": selected.小类 if selected else "",
                "细类": selected.细类 if selected else "",
                "is_matched": bool(selected),
            }

            for candidate in candidates:
                prefix = f"top{candidate.rank}"
                result_row[f"{prefix}_code"] = candidate.code
                result_row[f"{prefix}_title"] = candidate.title
                result_row[f"{prefix}_score"] = candidate.score
                result_row[f"{prefix}_detail_path"] = candidate.detail_path
                result_row[f"{prefix}_detail_name"] = candidate.细类

            matched_rows.append(result_row)

        return pd.DataFrame(matched_rows)

    def match_jobs(self, jobs_df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        """批量匹配岗位到职业细类，并保留 Top-K 候选。"""
        if jobs_df.empty:
            return pd.DataFrame()
        query_texts = jobs_df.apply(_build_job_query_text, axis=1).tolist()
        return self._match_query_texts(query_df=jobs_df, query_texts=query_texts, top_k=top_k)

    def match_requirement_texts(
        self,
        requirement_df: pd.DataFrame,
        query_col: str = "职业匹配文本",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """基于任职要求文本批量匹配职业细类。

        技能词典流程会优先使用 `任职要求_items_text`，缺失时回退到 `RAG匹配文本`，
        然后通过这个入口调用 BGE 模型完成职业细类语义匹配。
        """
        if requirement_df.empty:
            return pd.DataFrame()
        if query_col not in requirement_df.columns:
            raise KeyError(f"缺少匹配文本列: {query_col}")

        query_texts = requirement_df[query_col].fillna("").astype(str).map(_safe_text).tolist()
        return self._match_query_texts(query_df=requirement_df, query_texts=query_texts, top_k=top_k)

    @staticmethod
    def _select_candidate(candidates: List[OccupationCandidate]) -> OccupationCandidate | None:
        """按 Top1 到 Top5 依次回退，选择首个可用细类。"""
        for candidate in candidates[:5]:
            if candidate.detail_path and candidate.细类:
                return candidate
        return None
