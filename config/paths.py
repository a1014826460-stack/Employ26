"""集中化的项目路径与 PostgreSQL 配置管理。

本模块提供 `ProjectPaths` dataclass，将所有模型路径、PostgreSQL 连接参数
和输出路径集中到单一配置对象中。

配置优先级:
    显式参数 > 环境变量 > config/database.yaml > 模块内默认值

用法:
    from config.paths import get_project_paths
    paths = get_project_paths()
    pg = paths.pg_connection_params  # dict: host, port, dbname, user, password
    url = paths.pg_sqlalchemy_url()
    model = paths.bge_model_path     # 读取环境变量或默认值

环境变量覆盖:
    EMPLOYDATA_PG_HOST     — PostgreSQL 主机地址
    EMPLOYDATA_PG_PORT     — PostgreSQL 端口
    EMPLOYDATA_PG_DBNAME   — 数据库名
    EMPLOYDATA_PG_USER     — 用户名
    EMPLOYDATA_PG_PASSWORD — 密码
    EMPLOYDATA_PG_SCHEMA   — 默认 schema
    EMPLOYDATA_BGE_MODEL_PATH — BGE embedding 模型路径
    EMPLOYDATA_QWEN_MODEL_PATH — Qwen LLM 模型路径
    EMPLOYDATA_BERT_MODEL_PATH — BERT 模型路径
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


def _project_root() -> Path:
    """返回项目根目录（config/ 的父目录）。"""
    return Path(__file__).resolve().parent.parent


def _parse_scalar(value: str) -> Any:
    """解析本项目配置文件中使用的简单 YAML 标量。"""
    text = value.strip()
    if text in {"", "null", "None"}:
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def _simple_yaml_load(path: Path) -> dict[str, Any]:
    """读取不依赖第三方库的简单 YAML 配置。

    这里只支持项目当前 `config/database.yaml` 使用的映射、列表和标量。
    """
    parsed_lines: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        parsed_lines.append((indent, raw.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, (indent, line) in enumerate(parsed_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML 列表缩进不合法: {line}")
            parent.append(_parse_scalar(line[2:]))
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_is_list = False
        if index + 1 < len(parsed_lines):
            next_indent, next_line = parsed_lines[index + 1]
            next_is_list = next_indent > indent and next_line.startswith("- ")

        container: Any = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def load_database_yaml(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载 `config/database.yaml`。

    Args:
        config_path: 可选配置路径；为空时使用项目根目录下的默认配置。

    Returns:
        dict[str, Any]: 数据库与模型配置。文件不存在时返回空字典。
    """
    root = _project_root()
    target = Path(config_path) if config_path else root / "config" / "database.yaml"
    if not target.exists() or not target.read_text(encoding="utf-8").strip():
        return {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return _simple_yaml_load(target)


@dataclass(frozen=True)
class ProjectPaths:
    """项目集中配置对象。

    所有模型路径、PostgreSQL 连接参数和历史 DuckDB 兼容路径的统一来源。
    每个字段都可通过同名环境变量覆盖，默认值基于项目根目录的相对路径。
    """

    project_root: Path
    pg_connection_params: dict = field(default_factory=dict)
    pg_default_schema: str = "public"
    pg_available_schemas: tuple[str, ...] = ()
    table_mappings: dict[str, Any] = field(default_factory=dict)
    model_paths: dict[str, Any] = field(default_factory=dict)
    duckdb_path: Path = field(default_factory=Path)
    duckdb_threads: int = 8
    bge_model_path: Path = field(default_factory=Path)
    qwen_model_path: Path = field(default_factory=Path)
    bert_model_path: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    dict_dir: Path = field(default_factory=Path)
    config_dir: Path = field(default_factory=Path)

    @property
    def pg_host(self) -> str:
        """PostgreSQL 主机地址。"""
        return self.pg_connection_params.get("host", "localhost")

    @property
    def pg_port(self) -> int:
        """PostgreSQL 端口。"""
        return int(self.pg_connection_params.get("port", 5432))

    @property
    def pg_dbname(self) -> str:
        """PostgreSQL 数据库名。"""
        return self.pg_connection_params.get("dbname", "Employ26")

    @property
    def pg_user(self) -> str:
        """PostgreSQL 用户名。"""
        return self.pg_connection_params.get("user", "postgres")

    @property
    def pg_password(self) -> str:
        """PostgreSQL 密码。"""
        return self.pg_connection_params.get("password", "")

    @property
    def pg_schema(self) -> str:
        """默认 PostgreSQL schema。

        注意:
            这只是默认 schema，不代表所有表都在同一个 schema 下。
            当前 Employ26 业务 schema 详见 `Employ26-database.md`。
        """
        return self.pg_default_schema

    def pg_sqlalchemy_url(self, dbname: str | None = None) -> str:
        """返回 SQLAlchemy 使用的 PostgreSQL 连接 URL。"""
        target_db = dbname or self.pg_dbname
        user = quote_plus(str(self.pg_user))
        password = quote_plus(str(self.pg_password))
        auth = f"{user}:{password}" if password else user
        return f"postgresql+psycopg://{auth}@{self.pg_host}:{self.pg_port}/{target_db}"

    def get_table_name(
        self,
        section: str,
        key: str,
        default: str | None = None,
    ) -> str:
        """从 `database.yaml` 的 `tables` 区块读取推荐 PostgreSQL 表名。

        Args:
            section: 表分组名，例如 `annotations`。
            key: 表别名，例如 `tasks_v2`。
            default: 找不到配置时返回的兜底值。

        Returns:
            str: 推荐使用的 PostgreSQL 全限定表名。
        """
        section_map = self.table_mappings.get(section, {})
        if isinstance(section_map, dict) and key in section_map:
            return str(section_map[key])
        if default is not None:
            return default
        raise KeyError(f"未配置 tables.{section}.{key}")

    @property
    def skill_extraction_output_dir(self) -> Path:
        """技能抽取产物的输出目录。"""
        return self.output_dir / "skill_extraction"

    @property
    def cache_dir(self) -> Path:
        """中间产物缓存目录。"""
        return self.skill_extraction_output_dir / "cache"

    @property
    def report_dir(self) -> Path:
        """报告输出目录。"""
        return self.skill_extraction_output_dir / "reports"

    @property
    def prompt_dir(self) -> Path:
        """LLM prompt 模板目录。"""
        return self.skill_extraction_output_dir / "prompts"

    @property
    def rag_index_dir(self) -> Path:
        """RAG 向量索引产物目录。"""
        return self.project_root / "src" / "rag" / "artifacts"


def get_project_paths(
    *,
    pg_params: dict | None = None,
    bge_model_path: str | Path | None = None,
    qwen_model_path: str | Path | None = None,
    bert_model_path: str | Path | None = None,
) -> ProjectPaths:
    """构建 `ProjectPaths` 实例，优先使用显式参数，其次环境变量，最后默认值。

    Args:
        pg_params: 覆盖 PostgreSQL 连接参数（dict，键: host/port/dbname/user/password/schema）。
        bge_model_path: 覆盖 BGE embedding 模型路径。
        qwen_model_path: 覆盖 Qwen LLM 模型路径。
        bert_model_path: 覆盖 BERT 模型路径。

    Returns:
        ProjectPaths: 冻结的路径与连接配置对象。
    """
    root = _project_root()

    def _resolve(key: str, explicit: str | Path | None, default: str | Path) -> Path:
        """按 显式参数 > 环境变量 > 默认值 的优先级解析路径。"""
        if explicit is not None:
            return Path(explicit)
        env_val = os.getenv(key)
        if env_val:
            return Path(env_val)
        return Path(default)

    yaml_config = load_database_yaml()
    yaml_database = yaml_config.get("database", {}) if isinstance(yaml_config, dict) else {}
    yaml_models = yaml_config.get("model_paths", {}) if isinstance(yaml_config, dict) else {}
    yaml_tables = yaml_config.get("tables", {}) if isinstance(yaml_config, dict) else {}
    if pg_params is None:
        pg_params = {}

    def _resolve_setting(
        key: str,
        env_key: str,
        default: Any,
    ) -> Any:
        """按 显式参数 > 环境变量 > YAML > 默认值 解析配置值。"""
        if key in pg_params and pg_params[key] is not None:
            return pg_params[key]
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val
        if key in yaml_database and yaml_database[key] is not None:
            return yaml_database[key]
        return default

    # Postgres 连接参数：显式 dict > 环境变量 > database.yaml > 默认值
    pg_connection_params = {
        "host": _resolve_setting("host", "EMPLOYDATA_PG_HOST", "localhost"),
        "port": int(_resolve_setting("port", "EMPLOYDATA_PG_PORT", 5432)),
        "dbname": _resolve_setting("dbname", "EMPLOYDATA_PG_DBNAME", "Employ26"),
        "user": _resolve_setting("user", "EMPLOYDATA_PG_USER", "postgres"),
        "password": _resolve_setting("password", "EMPLOYDATA_PG_PASSWORD", ""),
    }
    pg_default_schema = _resolve_setting("schema", "EMPLOYDATA_PG_SCHEMA", "public")
    yaml_schemas = yaml_database.get("schemas", []) if isinstance(yaml_database, dict) else []
    pg_available_schemas = tuple(str(schema) for schema in yaml_schemas)
    duckdb_path = root / str(yaml_database.get("duckdb_path", "output/recruit.duckdb"))
    duckdb_threads = int(yaml_database.get("duckdb_threads", 8))

    return ProjectPaths(
        project_root=root,
        pg_connection_params=pg_connection_params,
        pg_default_schema=pg_default_schema,
        pg_available_schemas=pg_available_schemas,
        table_mappings=yaml_tables if isinstance(yaml_tables, dict) else {},
        model_paths=yaml_models if isinstance(yaml_models, dict) else {},
        duckdb_path=duckdb_path,
        duckdb_threads=duckdb_threads,
        bge_model_path=_resolve(
            "EMPLOYDATA_BGE_MODEL_PATH",
            bge_model_path,
            root / str(yaml_models.get("bge", "models/bge-base-zh-v1.5")),
        ),
        qwen_model_path=_resolve(
            "EMPLOYDATA_QWEN_MODEL_PATH",
            qwen_model_path,
            root / str(yaml_models.get("llm", "models/Qwen3-8B")),
        ),
        bert_model_path=_resolve(
            "EMPLOYDATA_BERT_MODEL_PATH",
            bert_model_path,
            root / str(yaml_models.get("bert", "models/chinese-roberta-wwm-ext")),
        ),
        output_dir=root / "output",
        dict_dir=root / "dicts",
        config_dir=root / "config",
    )
