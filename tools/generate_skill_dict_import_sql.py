"""Generate idempotent PostgreSQL import SQL for skill dictionaries."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "skill_extraction" / "sql" / "import_skill_dictionaries_to_dict.sql"
CUTOFF_SQL = "TIMESTAMPTZ '2023-01-01 00:00:00+00'"


def parse_ts(value: Any, fallback: str) -> str:
    """Parse source timestamp to UTC ISO string, falling back when absent."""
    if not value:
        return fallback
    text = str(value).strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def file_mtime(path: Path) -> str:
    """Return file modification timestamp in UTC ISO format."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()


def make_code(prefix: str, key: str) -> str:
    """Generate stable alphanumeric code <= 20 chars."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest().upper()[:16]
    return f"{prefix}{digest}"


def load_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], str]:
    """Load dictionaries and convert them to normalized import rows."""
    hard_path = ROOT / "dicts" / "flat_skill_dictionary.json"
    soft_version = (ROOT / "dicts" / "soft_skill" / "current.txt").read_text(
        encoding="utf-8"
    ).strip()
    soft_path = ROOT / "dicts" / "soft_skill" / f"{soft_version}.json"
    blacklist_path = ROOT / "dicts" / "blacklist_soft_skills.txt"

    hard_data = json.loads(hard_path.read_text(encoding="utf-8"))
    soft_data = json.loads(soft_path.read_text(encoding="utf-8"))
    blacklisted = set()
    if blacklist_path.exists():
        blacklisted = {
            line.strip()
            for line in blacklist_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    hard_updated = parse_ts(
        hard_data.get("metadata", {}).get("updated_at"), file_mtime(hard_path)
    )
    soft_updated = parse_ts(
        soft_data.get("metadata", {}).get("updated_at"), file_mtime(soft_path)
    )

    hard_rows: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for item in hard_data.get("skills", []):
        name = str(item.get("name") or "").strip()[:100]
        skill_code = make_code("H", "hard:" + name.casefold())
        if skill_code in seen_codes:
            skill_code = make_code(
                "H",
                "hard:" + name.casefold() + ":" + json.dumps(item, ensure_ascii=False),
            )
        seen_codes.add(skill_code)
        hard_rows.append(
            {
                "skill_code": skill_code,
                "skill_name": name,
                "category": str(item.get("category") or "uncategorized").strip()[:64],
                "last_updated": parse_ts(
                    item.get("last_updated") or item.get("version_date"), hard_updated
                ),
                "version_date": item.get("version_date"),
                "aliases": item.get("aliases") or [],
                "skill_type": item.get("skill_type"),
                "notes": item.get("notes"),
                "proficiency_level": item.get("proficiency_level"),
                "parent_skill_code": item.get("parent_skill_code"),
                "source_version": str(hard_data.get("metadata", {}).get("schema_version", "")),
                "source_file": "dicts/flat_skill_dictionary.json",
                "source_payload": item,
            }
        )

    soft_rows: list[dict[str, Any]] = []
    seen_codes.clear()
    for dim_key, dim_info in soft_data.get("dimensions", {}).items():
        dim_name = str(dim_info.get("name") or dim_key)
        for item in dim_info.get("skills", []):
            name = str(item.get("name") or "").strip()[:100]
            category = str(item.get("dimension") or dim_key).strip()[:64]
            skill_code = make_code("S", f"soft:{category.casefold()}:{name.casefold()}")
            if skill_code in seen_codes:
                skill_code = make_code(
                    "S",
                    f"soft:{category.casefold()}:{name.casefold()}:"
                    + json.dumps(item, ensure_ascii=False),
                )
            seen_codes.add(skill_code)
            is_blacklisted = name in blacklisted
            soft_rows.append(
                {
                    "skill_code": skill_code,
                    "skill_name": name,
                    "category": category,
                    "dimension_name": dim_name,
                    "last_updated": parse_ts(
                        item.get("last_updated") or item.get("version_date"), soft_updated
                    ),
                    "version_date": item.get("version_date"),
                    "aliases": item.get("aliases") or [],
                    "proficiency_level": item.get("proficiency_level"),
                    "parent_skill_code": item.get("parent_skill_code"),
                    "is_blacklisted": is_blacklisted,
                    "is_active": not is_blacklisted,
                    "source_version": soft_version,
                    "source_file": f"dicts/soft_skill/{soft_version}.json",
                    "source_payload": item,
                }
            )

    hard_category_names = {
        "programming_language": "编程语言",
        "framework": "框架",
        "database": "数据库",
        "tool": "工具软件",
        "office": "办公软件",
        "equipment": "设备/仪器",
        "process": "工艺方法",
        "certification": "证书/资质",
        "uncategorized": "未分类",
    }
    category_rows: list[dict[str, str]] = []
    for category in sorted({row["category"] for row in hard_rows}):
        category_rows.append(
            {
                "skill_kind": "hard",
                "category_code": category,
                "category_name": hard_category_names.get(category, category),
                "source_file": "dicts/flat_skill_dictionary.json",
            }
        )
    for dim_key, dim_info in sorted(soft_data.get("dimensions", {}).items()):
        category_rows.append(
            {
                "skill_kind": "soft",
                "category_code": dim_key,
                "category_name": str(dim_info.get("name") or dim_key),
                "source_file": f"dicts/soft_skill/{soft_version}.json",
            }
        )

    code_re = re.compile(r"^[A-Za-z0-9]{1,20}$")
    for rows in (hard_rows, soft_rows):
        for row in rows:
            if not code_re.fullmatch(row["skill_code"]):
                raise ValueError(f"invalid generated skill_code: {row['skill_code']}")
            if not row["skill_name"]:
                raise ValueError(f"empty skill_name for {row['skill_code']}")
    return hard_rows, soft_rows, category_rows, soft_version


def json_literal(rows: list[dict[str, Any]]) -> str:
    """Return compact JSON for SQL dollar-quoted literal."""
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def build_sql() -> str:
    """Build PostgreSQL SQL script."""
    hard_rows, soft_rows, category_rows, soft_version = load_sources()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ddl_summary = (
        "schema=dict; tables=skill_categories, hard_skills, soft_skills, "
        "skill_dictionary_import_log; indexes=category btree, last_updated desc, aliases gin"
    )

    return f"""-- Auto-generated by tools/generate_skill_dict_import_sql.py at {generated_at}
-- Run with: psql -d Employ26 -v ON_ERROR_STOP=1 -f output/skill_extraction/sql/import_skill_dictionaries_to_dict.sql
-- Source: dicts/flat_skill_dictionary.json ({len(hard_rows)} hard skill candidates)
-- Source: dicts/soft_skill/{soft_version}.json ({len(soft_rows)} soft skill candidates)
-- DDL summary: {ddl_summary}

\\set ON_ERROR_STOP on

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'Employ26' THEN
        RAISE EXCEPTION 'Refusing to import into database %. Connect to Employ26 first.', current_database();
    END IF;
END $$;

CREATE SCHEMA IF NOT EXISTS dict;
COMMENT ON SCHEMA dict IS 'Skill dictionary schema for Employ26.';

CREATE TABLE IF NOT EXISTS dict.skill_categories (
    skill_kind TEXT NOT NULL CHECK (skill_kind IN ('hard', 'soft')),
    category_code VARCHAR(64) NOT NULL,
    category_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (skill_kind, category_code)
);
COMMENT ON TABLE dict.skill_categories IS 'Category metadata for hard skill categories and soft skill dimensions.';
COMMENT ON COLUMN dict.skill_categories.skill_kind IS 'Dictionary kind: hard or soft.';
COMMENT ON COLUMN dict.skill_categories.category_code IS 'Category or dimension code.';
COMMENT ON COLUMN dict.skill_categories.category_name IS 'Human-readable category or dimension name.';
COMMENT ON COLUMN dict.skill_categories.source_file IS 'Repository source file.';
COMMENT ON COLUMN dict.skill_categories.imported_at IS 'Import timestamp.';

CREATE TABLE IF NOT EXISTS dict.hard_skills (
    skill_code VARCHAR(20) PRIMARY KEY CHECK (skill_code ~ '^[A-Za-z0-9]{{1,20}}$'),
    skill_name VARCHAR(100) NOT NULL CHECK (length(btrim(skill_name)) > 0),
    skill_kind TEXT NOT NULL DEFAULT 'hard' CHECK (skill_kind = 'hard'),
    category VARCHAR(64) NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(aliases) = 'array'),
    skill_type TEXT,
    notes TEXT,
    proficiency_level TEXT,
    parent_skill_code VARCHAR(20),
    last_updated TIMESTAMPTZ NOT NULL,
    source_version TEXT,
    source_file TEXT NOT NULL,
    source_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT hard_skills_category_fk FOREIGN KEY (skill_kind, category)
        REFERENCES dict.skill_categories(skill_kind, category_code),
    CONSTRAINT hard_skills_parent_fk FOREIGN KEY (parent_skill_code)
        REFERENCES dict.hard_skills(skill_code)
);
COMMENT ON TABLE dict.hard_skills IS 'Validated hard skill dictionary imported from dicts/flat_skill_dictionary.json.';
COMMENT ON COLUMN dict.hard_skills.skill_code IS 'Stable alphanumeric hard skill identifier; primary key.';
COMMENT ON COLUMN dict.hard_skills.skill_name IS 'Canonical hard skill name.';
COMMENT ON COLUMN dict.hard_skills.category IS 'Hard skill category code.';
COMMENT ON COLUMN dict.hard_skills.aliases IS 'JSON array of aliases used for matching.';
COMMENT ON COLUMN dict.hard_skills.skill_type IS 'Original source skill_type.';
COMMENT ON COLUMN dict.hard_skills.notes IS 'Source notes or curation comments.';
COMMENT ON COLUMN dict.hard_skills.proficiency_level IS 'Optional proficiency metadata if present.';
COMMENT ON COLUMN dict.hard_skills.parent_skill_code IS 'Optional parent hard skill reference.';
COMMENT ON COLUMN dict.hard_skills.last_updated IS 'Version timestamp; records earlier than 2023-01-01 are excluded.';
COMMENT ON COLUMN dict.hard_skills.source_file IS 'Repository source file path.';
COMMENT ON COLUMN dict.hard_skills.source_payload IS 'Original JSON source payload.';

CREATE TABLE IF NOT EXISTS dict.soft_skills (
    skill_code VARCHAR(20) PRIMARY KEY CHECK (skill_code ~ '^[A-Za-z0-9]{{1,20}}$'),
    skill_name VARCHAR(100) NOT NULL CHECK (length(btrim(skill_name)) > 0),
    skill_kind TEXT NOT NULL DEFAULT 'soft' CHECK (skill_kind = 'soft'),
    category VARCHAR(64) NOT NULL,
    dimension_name TEXT,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(aliases) = 'array'),
    proficiency_level TEXT,
    parent_skill_code VARCHAR(20),
    is_blacklisted BOOLEAN NOT NULL DEFAULT false,
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_updated TIMESTAMPTZ NOT NULL,
    source_version TEXT,
    source_file TEXT NOT NULL,
    source_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT soft_skills_category_fk FOREIGN KEY (skill_kind, category)
        REFERENCES dict.skill_categories(skill_kind, category_code),
    CONSTRAINT soft_skills_parent_fk FOREIGN KEY (parent_skill_code)
        REFERENCES dict.soft_skills(skill_code)
);
COMMENT ON TABLE dict.soft_skills IS 'Validated soft skill dictionary imported from the current dicts/soft_skill version.';
COMMENT ON COLUMN dict.soft_skills.skill_code IS 'Stable alphanumeric soft skill identifier; primary key.';
COMMENT ON COLUMN dict.soft_skills.skill_name IS 'Canonical soft skill name.';
COMMENT ON COLUMN dict.soft_skills.category IS 'Big Five dimension code.';
COMMENT ON COLUMN dict.soft_skills.dimension_name IS 'Human-readable dimension name.';
COMMENT ON COLUMN dict.soft_skills.aliases IS 'JSON array of aliases used for matching.';
COMMENT ON COLUMN dict.soft_skills.proficiency_level IS 'Optional proficiency metadata if present.';
COMMENT ON COLUMN dict.soft_skills.parent_skill_code IS 'Optional parent soft skill reference.';
COMMENT ON COLUMN dict.soft_skills.is_blacklisted IS 'True when listed in dicts/blacklist_soft_skills.txt.';
COMMENT ON COLUMN dict.soft_skills.is_active IS 'False for blacklisted high-false-positive skills.';
COMMENT ON COLUMN dict.soft_skills.last_updated IS 'Version timestamp; records earlier than 2023-01-01 are excluded.';
COMMENT ON COLUMN dict.soft_skills.source_file IS 'Repository source file path.';
COMMENT ON COLUMN dict.soft_skills.source_payload IS 'Original JSON source payload.';

CREATE INDEX IF NOT EXISTS idx_hard_skills_category ON dict.hard_skills USING btree(category);
CREATE INDEX IF NOT EXISTS idx_hard_skills_last_updated_desc ON dict.hard_skills(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_hard_skills_aliases_gin ON dict.hard_skills USING gin(aliases);
CREATE INDEX IF NOT EXISTS idx_soft_skills_category ON dict.soft_skills USING btree(category);
CREATE INDEX IF NOT EXISTS idx_soft_skills_last_updated_desc ON dict.soft_skills(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_soft_skills_aliases_gin ON dict.soft_skills USING gin(aliases);
CREATE INDEX IF NOT EXISTS idx_soft_skills_active ON dict.soft_skills(is_active);

CREATE TABLE IF NOT EXISTS dict.skill_dictionary_import_log (
    import_id BIGSERIAL PRIMARY KEY,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    raw_count INTEGER NOT NULL,
    invalid_count INTEGER NOT NULL,
    outdated_count INTEGER NOT NULL,
    duplicate_count INTEGER NOT NULL,
    bad_reference_count INTEGER NOT NULL,
    imported_count INTEGER NOT NULL,
    ddl_summary TEXT NOT NULL
);
COMMENT ON TABLE dict.skill_dictionary_import_log IS 'Audit log for idempotent skill dictionary imports.';

CREATE TEMP TABLE _skill_categories_raw AS
SELECT * FROM jsonb_to_recordset($json${json_literal(category_rows)}$json$::jsonb)
AS x(skill_kind TEXT, category_code TEXT, category_name TEXT, source_file TEXT);

INSERT INTO dict.skill_categories(skill_kind, category_code, category_name, source_file, imported_at)
SELECT skill_kind, category_code, category_name, source_file, now()
FROM _skill_categories_raw
ON CONFLICT (skill_kind, category_code) DO UPDATE SET
    category_name = EXCLUDED.category_name,
    source_file = EXCLUDED.source_file,
    imported_at = now();

CREATE TEMP TABLE _hard_skills_raw AS
SELECT * FROM jsonb_to_recordset($json${json_literal(hard_rows)}$json$::jsonb)
AS x(skill_code TEXT, skill_name TEXT, category TEXT, last_updated TIMESTAMPTZ, version_date TEXT, aliases JSONB, skill_type TEXT, notes TEXT, proficiency_level TEXT, parent_skill_code TEXT, source_version TEXT, source_file TEXT, source_payload JSONB);

CREATE TEMP TABLE _soft_skills_raw AS
SELECT * FROM jsonb_to_recordset($json${json_literal(soft_rows)}$json$::jsonb)
AS x(skill_code TEXT, skill_name TEXT, category TEXT, dimension_name TEXT, last_updated TIMESTAMPTZ, version_date TEXT, aliases JSONB, proficiency_level TEXT, parent_skill_code TEXT, is_blacklisted BOOLEAN, is_active BOOLEAN, source_version TEXT, source_file TEXT, source_payload JSONB);

CREATE TEMP TABLE _hard_skills_filtered AS
WITH annotated AS (
    SELECT r.*,
           (r.skill_code ~ '^[A-Za-z0-9]{{1,20}}$' AND length(btrim(coalesce(r.skill_name, ''))) BETWEEN 1 AND 100) AS is_valid,
           (r.last_updated >= {CUTOFF_SQL}) AS is_current,
           row_number() OVER (PARTITION BY r.skill_code ORDER BY r.last_updated DESC NULLS LAST, r.skill_name) AS rn
    FROM _hard_skills_raw r
), deduped AS (
    SELECT * FROM annotated WHERE is_valid AND is_current AND rn = 1
), referenced AS (
    SELECT d.*, (d.parent_skill_code IS NULL OR EXISTS (SELECT 1 FROM deduped p WHERE p.skill_code = d.parent_skill_code)) AS has_valid_reference
    FROM deduped d
)
SELECT * FROM referenced WHERE has_valid_reference;

CREATE TEMP TABLE _soft_skills_filtered AS
WITH annotated AS (
    SELECT r.*,
           (r.skill_code ~ '^[A-Za-z0-9]{{1,20}}$' AND length(btrim(coalesce(r.skill_name, ''))) BETWEEN 1 AND 100) AS is_valid,
           (r.last_updated >= {CUTOFF_SQL}) AS is_current,
           row_number() OVER (PARTITION BY r.skill_code ORDER BY r.last_updated DESC NULLS LAST, r.skill_name) AS rn
    FROM _soft_skills_raw r
), deduped AS (
    SELECT * FROM annotated WHERE is_valid AND is_current AND rn = 1
), referenced AS (
    SELECT d.*, (d.parent_skill_code IS NULL OR EXISTS (SELECT 1 FROM deduped p WHERE p.skill_code = d.parent_skill_code)) AS has_valid_reference
    FROM deduped d
)
SELECT * FROM referenced WHERE has_valid_reference;

INSERT INTO dict.hard_skills(skill_code, skill_name, skill_kind, category, aliases, skill_type, notes, proficiency_level, parent_skill_code, last_updated, source_version, source_file, source_payload, imported_at)
SELECT skill_code, skill_name, 'hard', category, coalesce(aliases, '[]'::jsonb), skill_type, notes, proficiency_level, parent_skill_code, last_updated, source_version, source_file, coalesce(source_payload, '{{}}'::jsonb), now()
FROM _hard_skills_filtered
ON CONFLICT (skill_code) DO UPDATE SET
    skill_name = EXCLUDED.skill_name, category = EXCLUDED.category, aliases = EXCLUDED.aliases,
    skill_type = EXCLUDED.skill_type, notes = EXCLUDED.notes, proficiency_level = EXCLUDED.proficiency_level,
    parent_skill_code = EXCLUDED.parent_skill_code, last_updated = EXCLUDED.last_updated,
    source_version = EXCLUDED.source_version, source_file = EXCLUDED.source_file,
    source_payload = EXCLUDED.source_payload, imported_at = now()
WHERE dict.hard_skills.last_updated <= EXCLUDED.last_updated;

INSERT INTO dict.soft_skills(skill_code, skill_name, skill_kind, category, dimension_name, aliases, proficiency_level, parent_skill_code, is_blacklisted, is_active, last_updated, source_version, source_file, source_payload, imported_at)
SELECT skill_code, skill_name, 'soft', category, dimension_name, coalesce(aliases, '[]'::jsonb), proficiency_level, parent_skill_code, coalesce(is_blacklisted, false), coalesce(is_active, true), last_updated, source_version, source_file, coalesce(source_payload, '{{}}'::jsonb), now()
FROM _soft_skills_filtered
ON CONFLICT (skill_code) DO UPDATE SET
    skill_name = EXCLUDED.skill_name, category = EXCLUDED.category, dimension_name = EXCLUDED.dimension_name,
    aliases = EXCLUDED.aliases, proficiency_level = EXCLUDED.proficiency_level,
    parent_skill_code = EXCLUDED.parent_skill_code, is_blacklisted = EXCLUDED.is_blacklisted,
    is_active = EXCLUDED.is_active, last_updated = EXCLUDED.last_updated,
    source_version = EXCLUDED.source_version, source_file = EXCLUDED.source_file,
    source_payload = EXCLUDED.source_payload, imported_at = now()
WHERE dict.soft_skills.last_updated <= EXCLUDED.last_updated;

WITH hard_stats AS (
    SELECT
        (SELECT count(*) FROM _hard_skills_raw)::int AS raw_count,
        (SELECT count(*) FROM _hard_skills_raw WHERE NOT (skill_code ~ '^[A-Za-z0-9]{{1,20}}$' AND length(btrim(coalesce(skill_name, ''))) BETWEEN 1 AND 100))::int AS invalid_count,
        (SELECT count(*) FROM _hard_skills_raw WHERE last_updated < {CUTOFF_SQL})::int AS outdated_count,
        (SELECT count(*) FROM (SELECT skill_code FROM _hard_skills_raw GROUP BY skill_code HAVING count(*) > 1) d)::int AS duplicate_count,
        (SELECT count(*) FROM _hard_skills_raw r WHERE r.parent_skill_code IS NOT NULL AND NOT EXISTS (SELECT 1 FROM _hard_skills_raw p WHERE p.skill_code = r.parent_skill_code))::int AS bad_reference_count,
        (SELECT count(*) FROM _hard_skills_filtered)::int AS imported_count
), soft_stats AS (
    SELECT
        (SELECT count(*) FROM _soft_skills_raw)::int AS raw_count,
        (SELECT count(*) FROM _soft_skills_raw WHERE NOT (skill_code ~ '^[A-Za-z0-9]{{1,20}}$' AND length(btrim(coalesce(skill_name, ''))) BETWEEN 1 AND 100))::int AS invalid_count,
        (SELECT count(*) FROM _soft_skills_raw WHERE last_updated < {CUTOFF_SQL})::int AS outdated_count,
        (SELECT count(*) FROM (SELECT skill_code FROM _soft_skills_raw GROUP BY skill_code HAVING count(*) > 1) d)::int AS duplicate_count,
        (SELECT count(*) FROM _soft_skills_raw r WHERE r.parent_skill_code IS NOT NULL AND NOT EXISTS (SELECT 1 FROM _soft_skills_raw p WHERE p.skill_code = r.parent_skill_code))::int AS bad_reference_count,
        (SELECT count(*) FROM _soft_skills_filtered)::int AS imported_count
)
INSERT INTO dict.skill_dictionary_import_log(source_name, source_file, raw_count, invalid_count, outdated_count, duplicate_count, bad_reference_count, imported_count, ddl_summary)
SELECT 'hard_skills', 'dicts/flat_skill_dictionary.json', raw_count, invalid_count, outdated_count, duplicate_count, bad_reference_count, imported_count, '{ddl_summary}' FROM hard_stats
UNION ALL
SELECT 'soft_skills', 'dicts/soft_skill/{soft_version}.json', raw_count, invalid_count, outdated_count, duplicate_count, bad_reference_count, imported_count, '{ddl_summary}' FROM soft_stats;

DO $$
DECLARE
    h RECORD;
    s RECORD;
BEGIN
    SELECT * INTO h FROM dict.skill_dictionary_import_log WHERE source_name = 'hard_skills' ORDER BY import_id DESC LIMIT 1;
    SELECT * INTO s FROM dict.skill_dictionary_import_log WHERE source_name = 'soft_skills' ORDER BY import_id DESC LIMIT 1;
    RAISE NOTICE 'hard_skills import: raw=%, invalid=%, outdated=%, duplicate_skill_code_groups=%, bad_references=%, imported=%', h.raw_count, h.invalid_count, h.outdated_count, h.duplicate_count, h.bad_reference_count, h.imported_count;
    RAISE NOTICE 'soft_skills import: raw=%, invalid=%, outdated=%, duplicate_skill_code_groups=%, bad_references=%, imported=%', s.raw_count, s.invalid_count, s.outdated_count, s.duplicate_count, s.bad_reference_count, s.imported_count;
    RAISE NOTICE 'DDL summary: {ddl_summary}';
END $$;

COMMIT;
"""


def main() -> None:
    """Generate the SQL file."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_sql(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
