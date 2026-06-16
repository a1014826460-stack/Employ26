"""统一职业词典表的建表、迁移与兼容视图管理。"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from src.db.postgres import create_pg_engine, ensure_schema


DEFAULT_UNIFIED_TABLE = "public.occ_dict_unified"
ARCHIVE_SCHEMA = "archive_occ"

LEGACY_TABLES = (
    "occ_dict",
    "occ_dict_detailed",
    "occ_dict_pro",
    "occ_dict_class",
)


def _object_exists(connection, *, schema_name: str, object_name: str, object_type: str) -> bool:
    """判断表或视图是否存在。"""
    if object_type == "table":
        sql = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema_name
              AND table_name = :object_name
              AND table_type = 'BASE TABLE'
        """
    elif object_type == "view":
        sql = """
            SELECT 1
            FROM information_schema.views
            WHERE table_schema = :schema_name
              AND table_name = :object_name
        """
    else:
        raise ValueError(f"不支持的 object_type: {object_type}")

    return connection.execute(
        text(sql),
        {"schema_name": schema_name, "object_name": object_name},
    ).scalar_one_or_none() is not None


def _qualify(schema_name: str, table_name: str) -> str:
    """构造安全的 schema.table。"""
    return f'"{schema_name}"."{table_name}"'


def _trim_sql(expression: str) -> str:
    """生成兼容全角空格的 trim SQL。"""
    return f"nullif(trim(replace(CAST({expression} AS text), '　', ' ')), '')"


def _resolve_source_table(connection, table_name: str) -> str:
    """优先读取 archive 表，其次读取 public 物理表。"""
    if _object_exists(connection, schema_name=ARCHIVE_SCHEMA, object_name=table_name, object_type="table"):
        return _qualify(ARCHIVE_SCHEMA, table_name)
    if _object_exists(connection, schema_name="public", object_name=table_name, object_type="table"):
        return _qualify("public", table_name)
    raise RuntimeError(f"未找到迁移源表: {table_name}")


def ensure_occ_dict_unified_table(connection, table_name: str = DEFAULT_UNIFIED_TABLE) -> None:
    """确保统一职业词典表存在。"""
    ensure_schema(connection, "public")
    qualified_table = _qualify("public", table_name.split(".", 1)[1] if "." in table_name else table_name)

    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                node_key text PRIMARY KEY,
                node_type text NOT NULL CHECK (node_type IN ('occupation_leaf', 'class_backbone')),
                is_terminal boolean NOT NULL,

                code text,
                title text,
                "desc" text,
                tasks text,

                "级别" text,
                "分类代码" text,
                "职业代码" text,
                "大类" text,
                "中类" text,
                "小类" text,
                "细类" text,

                task_list text[],
                task_text_joined text,
                title_clean text,
                desc_clean text,
                hierarchy_text text,
                aliases text[],
                retrieval_title_text text,
                retrieval_desc_text text,
                retrieval_task_text text,

                has_occ_dict boolean NOT NULL DEFAULT false,
                has_occ_dict_detailed boolean NOT NULL DEFAULT false,
                has_occ_dict_pro boolean NOT NULL DEFAULT false,
                has_occ_dict_class boolean NOT NULL DEFAULT false,

                canonical_source text,
                source_versions jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_occ_dict_unified_leaf_code
            ON {qualified_table} (code)
            WHERE node_type = 'occupation_leaf' AND code IS NOT NULL
            """
        )
    )
    connection.execute(text(f'CREATE INDEX IF NOT EXISTS idx_occ_dict_unified_node_type ON {qualified_table} (node_type)'))
    connection.execute(text(f'CREATE INDEX IF NOT EXISTS idx_occ_dict_unified_occ_code ON {qualified_table} ("职业代码")'))
    connection.execute(text(f'CREATE INDEX IF NOT EXISTS idx_occ_dict_unified_class_code ON {qualified_table} ("分类代码")'))


def load_legacy_tables_into_unified(connection, table_name: str = DEFAULT_UNIFIED_TABLE) -> None:
    """把四张旧表全量灌入统一职业词典表。"""
    qualified_table = _qualify("public", table_name.split(".", 1)[1] if "." in table_name else table_name)
    occ_dict_src = _resolve_source_table(connection, "occ_dict")
    occ_dict_detailed_src = _resolve_source_table(connection, "occ_dict_detailed")
    occ_dict_pro_src = _resolve_source_table(connection, "occ_dict_pro")
    occ_dict_class_src = _resolve_source_table(connection, "occ_dict_class")

    connection.execute(text(f"TRUNCATE TABLE {qualified_table}"))

    connection.execute(
        text(
            f"""
            INSERT INTO {qualified_table} (
                node_key, node_type, is_terminal,
                code, title, "desc", tasks,
                has_occ_dict, canonical_source, source_versions
            )
            SELECT
                'leaf:' || trim(replace(CAST(code AS text), '　', ' ')) AS node_key,
                'occupation_leaf' AS node_type,
                true AS is_terminal,
                {_trim_sql("code")} AS code,
                {_trim_sql("title")} AS title,
                {_trim_sql('"desc"')} AS "desc",
                {_trim_sql("tasks")} AS tasks,
                true AS has_occ_dict,
                'occ_dict' AS canonical_source,
                jsonb_build_object('occ_dict', true)
            FROM {occ_dict_src}
            WHERE {_trim_sql("code")} IS NOT NULL
            """
        )
    )

    connection.execute(
        text(
            f"""
            INSERT INTO {qualified_table} (
                node_key, node_type, is_terminal,
                code, title, "desc", tasks,
                "级别", "分类代码", "职业代码", "大类", "中类", "小类", "细类",
                has_occ_dict_detailed, canonical_source, source_versions
            )
            SELECT
                'leaf:' || trim(replace(CAST(code AS text), '　', ' ')) AS node_key,
                'occupation_leaf' AS node_type,
                true AS is_terminal,
                {_trim_sql("code")} AS code,
                {_trim_sql("title")} AS title,
                {_trim_sql('"desc"')} AS "desc",
                {_trim_sql("tasks")} AS tasks,
                {_trim_sql('"级别"')} AS "级别",
                {_trim_sql('"分类代码"')} AS "分类代码",
                {_trim_sql('"职业代码"')} AS "职业代码",
                {_trim_sql('"大类"')} AS "大类",
                {_trim_sql('"中类"')} AS "中类",
                {_trim_sql('"小类"')} AS "小类",
                {_trim_sql('"细类"')} AS "细类",
                true AS has_occ_dict_detailed,
                'occ_dict_detailed' AS canonical_source,
                jsonb_build_object('occ_dict_detailed', true)
            FROM {occ_dict_detailed_src}
            WHERE {_trim_sql("code")} IS NOT NULL
            ON CONFLICT (node_key)
            DO UPDATE SET
                title = COALESCE(EXCLUDED.title, {qualified_table}.title),
                "desc" = COALESCE(EXCLUDED."desc", {qualified_table}."desc"),
                tasks = COALESCE(EXCLUDED.tasks, {qualified_table}.tasks),
                "级别" = COALESCE(EXCLUDED."级别", {qualified_table}."级别"),
                "分类代码" = COALESCE(EXCLUDED."分类代码", {qualified_table}."分类代码"),
                "职业代码" = COALESCE(EXCLUDED."职业代码", {qualified_table}."职业代码"),
                "大类" = COALESCE(EXCLUDED."大类", {qualified_table}."大类"),
                "中类" = COALESCE(EXCLUDED."中类", {qualified_table}."中类"),
                "小类" = COALESCE(EXCLUDED."小类", {qualified_table}."小类"),
                "细类" = COALESCE(EXCLUDED."细类", {qualified_table}."细类"),
                has_occ_dict_detailed = true,
                canonical_source = 'occ_dict_detailed',
                source_versions = {qualified_table}.source_versions || jsonb_build_object('occ_dict_detailed', true),
                updated_at = now()
            """
        )
    )

    connection.execute(
        text(
            f"""
            INSERT INTO {qualified_table} (
                node_key, node_type, is_terminal,
                code, title, "desc", tasks,
                "级别", "分类代码", "职业代码", "大类", "中类", "小类", "细类",
                task_list, task_text_joined, title_clean, desc_clean,
                hierarchy_text, aliases, retrieval_title_text, retrieval_desc_text, retrieval_task_text,
                has_occ_dict_pro, canonical_source, source_versions
            )
            SELECT
                'leaf:' || trim(replace(CAST(code AS text), '　', ' ')) AS node_key,
                'occupation_leaf' AS node_type,
                true AS is_terminal,
                {_trim_sql("code")} AS code,
                {_trim_sql("title")} AS title,
                {_trim_sql('"desc"')} AS "desc",
                {_trim_sql("tasks")} AS tasks,
                {_trim_sql('"级别"')} AS "级别",
                {_trim_sql('"分类代码"')} AS "分类代码",
                {_trim_sql('"职业代码"')} AS "职业代码",
                {_trim_sql('"大类"')} AS "大类",
                {_trim_sql('"中类"')} AS "中类",
                {_trim_sql('"小类"')} AS "小类",
                {_trim_sql('"细类"')} AS "细类",
                task_list,
                {_trim_sql("task_text_joined")} AS task_text_joined,
                {_trim_sql("title_clean")} AS title_clean,
                {_trim_sql("desc_clean")} AS desc_clean,
                {_trim_sql("hierarchy_text")} AS hierarchy_text,
                aliases,
                {_trim_sql("retrieval_title_text")} AS retrieval_title_text,
                {_trim_sql("retrieval_desc_text")} AS retrieval_desc_text,
                {_trim_sql("retrieval_task_text")} AS retrieval_task_text,
                true AS has_occ_dict_pro,
                'occ_dict_pro' AS canonical_source,
                jsonb_build_object('occ_dict_pro', true)
            FROM {occ_dict_pro_src}
            WHERE {_trim_sql("code")} IS NOT NULL
            ON CONFLICT (node_key)
            DO UPDATE SET
                title = COALESCE(EXCLUDED.title, {qualified_table}.title),
                "desc" = COALESCE(EXCLUDED."desc", {qualified_table}."desc"),
                tasks = COALESCE(EXCLUDED.tasks, {qualified_table}.tasks),
                "级别" = EXCLUDED."级别",
                "分类代码" = EXCLUDED."分类代码",
                "职业代码" = EXCLUDED."职业代码",
                "大类" = EXCLUDED."大类",
                "中类" = EXCLUDED."中类",
                "小类" = EXCLUDED."小类",
                "细类" = EXCLUDED."细类",
                task_list = COALESCE(EXCLUDED.task_list, {qualified_table}.task_list),
                task_text_joined = COALESCE(EXCLUDED.task_text_joined, {qualified_table}.task_text_joined),
                title_clean = COALESCE(EXCLUDED.title_clean, {qualified_table}.title_clean),
                desc_clean = COALESCE(EXCLUDED.desc_clean, {qualified_table}.desc_clean),
                hierarchy_text = EXCLUDED.hierarchy_text,
                aliases = COALESCE(EXCLUDED.aliases, {qualified_table}.aliases),
                retrieval_title_text = COALESCE(EXCLUDED.retrieval_title_text, {qualified_table}.retrieval_title_text),
                retrieval_desc_text = COALESCE(EXCLUDED.retrieval_desc_text, {qualified_table}.retrieval_desc_text),
                retrieval_task_text = COALESCE(EXCLUDED.retrieval_task_text, {qualified_table}.retrieval_task_text),
                has_occ_dict_pro = true,
                canonical_source = 'occ_dict_pro',
                source_versions = {qualified_table}.source_versions || jsonb_build_object('occ_dict_pro', true),
                updated_at = now()
            """
        )
    )

    connection.execute(
        text(
            f"""
            INSERT INTO {qualified_table} (
                node_key, node_type, is_terminal,
                "级别", "分类代码", "职业代码", "大类", "中类", "小类", "细类",
                has_occ_dict_class, canonical_source, source_versions
            )
            SELECT
                'class:' ||
                COALESCE({_trim_sql('"级别"')}, '') || ':' ||
                COALESCE({_trim_sql('"分类代码"')}, '') || ':' ||
                COALESCE({_trim_sql('"职业代码"')}, '') || ':' ||
                COALESCE({_trim_sql('"细类"')}, '') AS node_key,
                'class_backbone' AS node_type,
                true AS is_terminal,
                {_trim_sql('"级别"')} AS "级别",
                {_trim_sql('"分类代码"')} AS "分类代码",
                {_trim_sql('"职业代码"')} AS "职业代码",
                {_trim_sql('"大类"')} AS "大类",
                {_trim_sql('"中类"')} AS "中类",
                {_trim_sql('"小类"')} AS "小类",
                {_trim_sql('"细类"')} AS "细类",
                true AS has_occ_dict_class,
                'occ_dict_class' AS canonical_source,
                jsonb_build_object('occ_dict_class', true)
            FROM {occ_dict_class_src}
            ON CONFLICT (node_key)
            DO UPDATE SET
                "级别" = COALESCE(EXCLUDED."级别", {qualified_table}."级别"),
                "分类代码" = COALESCE(EXCLUDED."分类代码", {qualified_table}."分类代码"),
                "职业代码" = COALESCE(EXCLUDED."职业代码", {qualified_table}."职业代码"),
                "大类" = COALESCE(EXCLUDED."大类", {qualified_table}."大类"),
                "中类" = COALESCE(EXCLUDED."中类", {qualified_table}."中类"),
                "小类" = COALESCE(EXCLUDED."小类", {qualified_table}."小类"),
                "细类" = COALESCE(EXCLUDED."细类", {qualified_table}."细类"),
                has_occ_dict_class = true,
                canonical_source = 'occ_dict_class',
                source_versions = {qualified_table}.source_versions || jsonb_build_object('occ_dict_class', true),
                updated_at = now()
            """
        )
    )
    

def retire_legacy_physical_tables(connection, table_name: str = DEFAULT_UNIFIED_TABLE) -> None:
    """退役 public 下旧物理表，并重建兼容视图。"""
    ensure_schema(connection, ARCHIVE_SCHEMA)
    qualified_unified = _qualify("public", table_name.split(".", 1)[1] if "." in table_name else table_name)

    for legacy_table in LEGACY_TABLES:
        if _object_exists(connection, schema_name="public", object_name=legacy_table, object_type="view"):
            connection.execute(text(f'DROP VIEW IF EXISTS {_qualify("public", legacy_table)}'))
        if _object_exists(connection, schema_name="public", object_name=legacy_table, object_type="table"):
            if _object_exists(connection, schema_name=ARCHIVE_SCHEMA, object_name=legacy_table, object_type="table"):
                connection.execute(text(f'DROP TABLE {_qualify(ARCHIVE_SCHEMA, legacy_table)}'))
            connection.execute(text(f'ALTER TABLE {_qualify("public", legacy_table)} SET SCHEMA "{ARCHIVE_SCHEMA}"'))

    connection.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW public.occ_dict AS
            SELECT code, title, "desc", tasks
            FROM {qualified_unified}
            WHERE node_type = 'occupation_leaf'
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW public.occ_dict_detailed AS
            SELECT
                code, title, "desc", tasks,
                "级别", "分类代码", "职业代码", "大类", "中类", "小类", "细类"
            FROM {qualified_unified}
            WHERE node_type = 'occupation_leaf'
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW public.occ_dict_pro AS
            SELECT
                code, title, "desc", tasks,
                "级别", "分类代码", "职业代码", "大类", "中类", "小类", "细类",
                task_list, task_text_joined, title_clean, desc_clean,
                hierarchy_text, aliases, retrieval_title_text, retrieval_desc_text, retrieval_task_text
            FROM {qualified_unified}
            WHERE node_type = 'occupation_leaf'
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW public.occ_dict_class AS
            SELECT
                "级别", "分类代码", "职业代码", "大类", "中类", "小类", "细类"
            FROM {qualified_unified}
            WHERE node_type = 'class_backbone'
            """
        )
    )


def migrate_occ_dicts_to_unified(table_name: str = DEFAULT_UNIFIED_TABLE) -> dict[str, int]:
    """执行统一职业词典迁移。"""
    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_occ_dict_unified_table(connection, table_name=table_name)
            load_legacy_tables_into_unified(connection, table_name=table_name)
            retire_legacy_physical_tables(connection, table_name=table_name)

            qualified_unified = _qualify("public", table_name.split(".", 1)[1] if "." in table_name else table_name)
            leaf_count = int(connection.execute(text(f"SELECT count(*) FROM {qualified_unified} WHERE node_type = 'occupation_leaf'")).scalar_one())
            class_count = int(connection.execute(text(f"SELECT count(*) FROM {qualified_unified} WHERE node_type = 'class_backbone'")).scalar_one())
            return {
                "occupation_leaf_rows": leaf_count,
                "class_backbone_rows": class_count,
                "total_rows": leaf_count + class_count,
            }
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="统一职业词典迁移工具")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="创建 public.occ_dict_unified，灌入四张旧表，并退役旧物理表为兼容视图",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    if args.migrate:
        print(migrate_occ_dicts_to_unified())


if __name__ == "__main__":
    main()
