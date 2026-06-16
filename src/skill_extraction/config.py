"""`skill_extraction` 模块的统一配置入口。

本模块负责三件事：
1. 从 `config/database.yaml` 读取项目配置；
2. 将配置解析为结构化的 `SkillExtractionConfig`；
3. 统一约定技能抽取相关的输入表、输出目录和模型路径。

设计目标是让上层脚本不再分别拼接路径、猜测表名或硬编码模型目录，
而是共享同一份可复用、可追踪的配置对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.utils.llm_router import (
    DEFAULT_API_MODE,
    DEFAULT_BASE_URL,
    DEFAULT_CHEAP_MODEL,
    DEFAULT_STRONG_MODEL,
)

from src.utils.utils import load_database_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_table_names(value: object) -> List[str]:
    """解析 `jobs_table` 配置项，兼容列表和逗号分隔字符串。

    参数:
        value: 原始配置值，可能为 `None`、YAML 列表或字符串。

    返回:
        List[str]: 过滤空值并去除首尾空白后的表名列表。
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def qualify_table_name(table_name: str) -> str:
    """校验并返回配置中的 PostgreSQL 表名。

    参数:
        table_name: 原始表名，推荐使用 schema.table。

    返回:
        str: 去除首尾空白后的表名。

    说明:
        这里不再自动补齐 `recruit.main` 前缀，避免把封存的 DuckDB 表名
        误当成当前 PostgreSQL 表引用。
    """
    normalized = str(table_name).strip()
    if not normalized:
        raise ValueError("table_name 不能为空")
    return normalized


@dataclass(frozen=True)
class SkillExtractionConfig:
    """技能抽取主链路使用的结构化配置对象。

    该对象集中保存：
    - DuckDB 连接信息
    - Embedding / LLM / BERT 模型路径
    - 输入表名
    - 词典、缓存、报告和中间产物路径
    """

    project_root: Path
    db_path: Path
    duckdb_threads: int
    embedding_model_path: Path
    llm_model_path: Path
    bert_model_path: Path
    preferred_local_llm_path: Path
    fallback_local_llm_paths: List[Path]
    model_download_dir: Path
    embedding_device: str
    llm_base_url: str
    llm_api_mode: str
    llm_cheap_model: str
    llm_strong_model: str
    llm_env_file: Path
    embedding_batch_size: int
    match_top_k: int
    catalog_table: str
    catalog_preprocessed_table: str
    jobs_tables: List[str]
    requirement_match_table: str
    occupation_detail_match_table: str
    occupation_detail_model_path: Path
    occupation_detail_top_k: int
    recruitment_normalized_table: str
    output_dir: Path
    prompt_train_dir: Path
    prompt_supplement_dir: Path
    report_dir: Path
    cache_dir: Path
    dict_dir: Path
    dictionary_path: Path
    catalog_embedding_cache_path: Path
    training_manifest_path: Path
    training_requirements_path: Path
    validation_pool_path: Path
    category_summary_path: Path
    state_path: Path

    def ensure_dirs(self) -> None:
        """确保技能抽取流程需要的目录全部存在。

        该方法会在配置加载完成后执行，用于提前创建：
        - prompt 目录
        - report 目录
        - cache 目录
        - dictionary 目录
        从而避免后续脚本在写文件时因为目录缺失而失败。
        """
        for target in [
            self.output_dir,
            self.prompt_train_dir,
            self.prompt_supplement_dir,
            self.report_dir,
            self.cache_dir,
            self.dict_dir,
        ]:
            target.mkdir(parents=True, exist_ok=True)


def _resolve_local_model_path(
    primary: object,
    fallbacks: List[Path],
) -> Path:
    primary_path = Path(str(primary))
    if primary_path.exists():
        return primary_path
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return primary_path


def load_skill_extraction_config(
    database_config_path: str | Path | None = None,
) -> SkillExtractionConfig:
    """读取 YAML 配置并构建 `SkillExtractionConfig`。

    参数:
        database_config_path: 可选的配置文件路径；为空时读取默认配置。

    返回:
        SkillExtractionConfig: 可直接被技能抽取各脚本复用的配置对象。

    异常:
        ValueError: 当关键配置缺失，例如未配置 `jobs_table` 时抛出。
    """
    raw_config = load_database_config(database_config_path)
    try:
        import torch  # type: ignore

        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        embedding_device = "cpu"
    db_settings = raw_config.get("database", {})
    parsing_settings = raw_config.get("job_title_parsing", {})
    skill_settings = raw_config.get("skill_extraction", {})

    db_path = PROJECT_ROOT / db_settings.get("duckdb_path", "output/recruit.duckdb")
    jobs_tables = [
        qualify_table_name(table_name)
        for table_name in _split_table_names(parsing_settings.get("jobs_table"))
    ]

    if not jobs_tables:
        raise ValueError("config/database.yaml 未配置 job_title_parsing.jobs_table")

    output_dir = PROJECT_ROOT / "output" / "skill_extraction"
    prompt_dir = output_dir / "prompts"
    report_dir = output_dir / "reports"
    cache_dir = output_dir / "cache"
    dict_dir = PROJECT_ROOT / "dicts"
    model_download_dir = PROJECT_ROOT / "models" / "hf"
    default_llm_path = raw_config.get("model_paths", {}).get("llm", "models/Qwen3-8B")
    preferred_local_llm_path = Path(
        skill_settings.get("preferred_local_llm_path", default_llm_path)
    )
    fallback_local_llm_paths = [
        PROJECT_ROOT / "models" / "hf" / "Qwen2.5-14B-Instruct",
        PROJECT_ROOT / "models" / "hf" / "DeepSeek-R1-Distill-Qwen-14B",
        PROJECT_ROOT / "models" / "hf" / "Qwen2.5-7B-Instruct",
        PROJECT_ROOT / str(default_llm_path),
    ]
    resolved_local_llm_path = _resolve_local_model_path(preferred_local_llm_path, fallback_local_llm_paths)

    config = SkillExtractionConfig(
        project_root=PROJECT_ROOT,
        db_path=db_path,
        duckdb_threads=max(1, int(db_settings.get("duckdb_threads", 8))),
        embedding_model_path=Path(
            raw_config.get(
                "model_paths", {}
            ).get(
                "bge", "models/bge-base-zh-v1.5"
            )
        ),
        llm_model_path=resolved_local_llm_path,
        bert_model_path=Path(
            raw_config.get("model_paths", {}).get(
                "bert",
                raw_config.get(
                    "BERT_path",
                    skill_settings.get("bert_model_path", "models/chinese-roberta-wwm-ext"),
                ),
            )
        ),
        preferred_local_llm_path=preferred_local_llm_path,
        fallback_local_llm_paths=fallback_local_llm_paths,
        model_download_dir=model_download_dir,
        embedding_device=embedding_device,
        llm_base_url=skill_settings.get("llm_base_url", DEFAULT_BASE_URL),
        llm_api_mode=skill_settings.get("llm_api_mode", DEFAULT_API_MODE),
        llm_cheap_model=skill_settings.get("llm_cheap_model", DEFAULT_CHEAP_MODEL),
        llm_strong_model=skill_settings.get("llm_strong_model", DEFAULT_STRONG_MODEL),
        llm_env_file=PROJECT_ROOT / skill_settings.get("llm_env_file", ".env.local"),
        embedding_batch_size=128,
        match_top_k=5,
        catalog_table=qualify_table_name(
            parsing_settings.get(
                "catalog_table",
                "public.occ_dict_unified",
            )
        ),
        catalog_preprocessed_table=qualify_table_name(
            parsing_settings.get(
                "catalog_preprocessed_table",
                "public.occ_dict_unified",
            )
        ),
        jobs_tables=jobs_tables,
        requirement_match_table=qualify_table_name(
            skill_settings.get(
                "requirement_match_table",
                "public.skill_extraction_requirement_matches",
            )
        ),
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
            "output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned",
        ),
        occupation_detail_top_k=max(1, int(skill_settings.get("occupation_detail_top_k", 10))),
        recruitment_normalized_table=qualify_table_name(
            raw_config.get("tables", {}).get("processing_results", {}).get(
                "recruitment_jobs_normalized",
                "public.recruitment_jobs_normalized",
            )
        ),
        output_dir=output_dir,
        prompt_train_dir=prompt_dir / "train",
        prompt_supplement_dir=prompt_dir / "supplement",
        report_dir=report_dir,
        cache_dir=cache_dir,
        dict_dir=dict_dir,
        dictionary_path=dict_dir / "occupation_skill_dictionary.json",
        catalog_embedding_cache_path=cache_dir / "occupation_catalog_embeddings.npy",
        training_manifest_path=output_dir / "occupation_skill_training_manifest.csv",
        training_requirements_path=output_dir / "occupation_skill_training_requirements.csv",
        validation_pool_path=output_dir / "occupation_skill_validation_pool.csv",
        category_summary_path=output_dir / "occupation_skill_category_summary.csv",
        state_path=output_dir / "occupation_skill_iteration_state.json",
    )
    config.ensure_dirs()
    return config

