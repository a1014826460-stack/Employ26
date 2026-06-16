"""Common utilities for automatic LLM-based dataset labeling."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from ..skill_extraction.config import SkillExtractionConfig, load_skill_extraction_config
from ..db.postgres import create_pg_engine
from .llm_router import LLMRouter, extract_json_from_response


logger = logging.getLogger(__name__)

SOURCE_COLUMNS: Sequence[str] = (
    "recruitment_record_id",
    "source_table",
    "source_row_number",
    "岗位名称",
    "岗位描述_清洗",
    "任职要求_items_text",
    "岗位职责_items_text",
    "sections_brief",
    "occupation_title",
    "occupation_code",
)

TEXT_FIELD_PRIORITY: Sequence[str] = (
    "任职要求_items_text",
    "岗位职责_items_text",
    "岗位描述_清洗",
    "text",
)


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_skill_key(value: object) -> str:
    return safe_text(value).casefold()


def extract_match_text(row: Dict[str, object]) -> str:
    for field_name in TEXT_FIELD_PRIORITY:
        text = safe_text(row.get(field_name, ""))
        if text:
            return text
    return ""


def build_sample_id(row: Dict[str, object], fallback_index: int) -> str:
    for field_name in ("sample_id", "recruitment_record_id", "source_row_number"):
        value = safe_text(row.get(field_name, ""))
        if value:
            return value
    return f"sample_{fallback_index:07d}"


def load_requirement_match_rows(
    config: SkillExtractionConfig | None = None,
    source_table: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    config = config or load_skill_extraction_config()
    source_table = source_table or config.requirement_match_table
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            column_query = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = '{source_table.split('.', 1)[0].strip('"')}'
                  AND table_name = '{source_table.split('.', 1)[1].strip('"')}'
                ORDER BY ordinal_position
            """
            available_columns = {
                str(row[0])
                for row in connection.exec_driver_sql(column_query).fetchall()
            }
            select_expressions: List[str] = []
            for column_name in SOURCE_COLUMNS:
                if column_name in available_columns:
                    select_expressions.append(_quote_identifier(column_name))
                else:
                    select_expressions.append(f"NULL AS {_quote_identifier(column_name)}")
            query = f"""
                SELECT
                    {", ".join(select_expressions)}
                FROM {source_table}
                {limit_clause}
            """
            return pd.read_sql_query(query, connection)
    finally:
        engine.dispose()


def prepare_labeling_frame(
    source_df: pd.DataFrame,
    max_text_chars: int = 900,
    min_text_chars: int = 20,
) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    seen_texts: set[str] = set()

    for row_index, row in enumerate(source_df.to_dict(orient="records")):
        text = extract_match_text(row)
        if len(text) < int(min_text_chars):
            continue

        normalized_key = text[: max_text_chars * 2].casefold()
        if normalized_key in seen_texts:
            continue
        seen_texts.add(normalized_key)

        records.append(
            {
                "sample_id": build_sample_id(row, row_index),
                "job_title": safe_text(row.get("岗位名称", "")),
                "occupation_title": safe_text(row.get("occupation_title", "")),
                "occupation_code": safe_text(row.get("occupation_code", "")),
                "source_table": safe_text(row.get("source_table", "")),
                "source_row_number": safe_text(row.get("source_row_number", "")),
                "text": text[:max_text_chars],
            }
        )

    return pd.DataFrame(records)


def stratified_sample_frame(
    frame: pd.DataFrame,
    sample_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    if frame.empty or sample_size <= 0 or len(frame) <= sample_size:
        return frame.copy()

    rng = random.Random(seed)
    group_rows: List[List[Dict[str, str]]] = []
    group_key_series = frame["occupation_code"].fillna("").astype(str)
    fallback_series = frame["occupation_title"].fillna("").astype(str)

    frame = frame.copy()
    frame["_group_key"] = group_key_series.where(group_key_series != "", fallback_series)
    frame["_group_key"] = frame["_group_key"].replace("", "__ungrouped__")

    for _, group_df in frame.groupby("_group_key"):
        rows = group_df.drop(columns=["_group_key"]).to_dict(orient="records")
        rng.shuffle(rows)
        group_rows.append(rows)

    rng.shuffle(group_rows)
    sampled_records: List[Dict[str, str]] = []
    while len(sampled_records) < sample_size and any(group_rows):
        next_group_rows: List[List[Dict[str, str]]] = []
        for rows in group_rows:
            if not rows:
                continue
            sampled_records.append(rows.pop())
            if rows:
                next_group_rows.append(rows)
            if len(sampled_records) >= sample_size:
                break
        group_rows = next_group_rows

    return pd.DataFrame(sampled_records)


def build_chat_prompts(
    llm,
    prompt_pairs: Sequence[Tuple[str, str]],
) -> List[str]:
    tokenizer = llm.get_tokenizer()
    rendered_prompts: List[str] = []
    for system_prompt, user_prompt in prompt_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"system: {system_prompt}\nuser: {user_prompt}\nassistant:"
        rendered_prompts.append(prompt)
    return rendered_prompts


def init_labeling_llm(
    model_path: str | Path,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
):
    """Initialize local vLLM when the runtime is available."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping local LLM init because torch import failed: %s", exc)
        return None

    if not torch.cuda.is_available():
        logger.warning("Skipping local LLM init because CUDA is unavailable")
        return None

    try:
        from src.skill_extraction.dictionary.merge import init_vllm_engine
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping local LLM init because vLLM import failed: %s", exc)
        return None

    try:
        return init_vllm_engine(
            model_path=str(model_path),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local LLM init failed, falling back to API routing: %s", exc)
        return None


def run_prompt_pairs(
    model_path: str | Path,
    prompt_pairs: Sequence[Tuple[str, str]],
    batch_size: int = 16,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
    temperature: float = 0.1,
    max_tokens: int = 1536,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    llm=None,
) -> List[str]:
    """Run prompt pairs with local vLLM when available, else use API routing."""
    if llm is not None:
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("vLLM runtime unavailable at inference time, falling back to API: %s", exc)
        else:
            formatted_prompts = build_chat_prompts(llm, prompt_pairs)
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            outputs: List[str] = []
            effective_batch_size = max(1, int(batch_size))
            total_batches = (len(formatted_prompts) + effective_batch_size - 1) // effective_batch_size
            for batch_index in range(total_batches):
                start = batch_index * effective_batch_size
                end = min(start + effective_batch_size, len(formatted_prompts))
                logger.info(
                    "Running local vLLM batch %d/%d for prompt pairs %d-%d",
                    batch_index + 1,
                    total_batches,
                    start + 1,
                    end,
                )
                batch_start = time.time()
                batch_outputs = llm.generate(formatted_prompts[start:end], sampling_params)
                logger.info(
                    "Finished local vLLM batch %d/%d in %.1fs",
                    batch_index + 1,
                    total_batches,
                    time.time() - batch_start,
                )
                for output in batch_outputs:
                    outputs.append(output.outputs[0].text)
            return outputs

    config = load_skill_extraction_config()
    router = LLMRouter.from_env(config.llm_env_file)
    return router.batch_complete_text(
        prompt_pairs,
        strength="cheap",
        max_output_tokens=max_tokens,
        reasoning_effort="low",
    )


def _extract_openai_response_text(response) -> str:
    from .llm_router import extract_response_text

    return extract_response_text(response)


def run_openai_prompt_pairs(
    prompt_pairs: Sequence[Tuple[str, str]],
    model: str = "openai/gpt-5.4-mini",
    max_output_tokens: int = 2048,
    reasoning_effort: str | None = "low",
    api_key: str | None = None,
) -> List[str]:
    config = load_skill_extraction_config()
    router = LLMRouter.from_env(config.llm_env_file)
    outputs: List[str] = []
    total = len(prompt_pairs)

    for index, (system_prompt, user_prompt) in enumerate(prompt_pairs, start=1):
        logger.info(
            "Running final-review prompt %d/%d with model %s",
            index,
            total,
            model,
        )
        strength = "strong" if model == config.llm_strong_model else "cheap"
        outputs.append(
            router.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strength=strength,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
        )

    return outputs


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
