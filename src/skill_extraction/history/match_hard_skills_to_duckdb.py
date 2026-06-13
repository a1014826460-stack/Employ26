"""
把硬技能词典匹配到 DuckDB 样本表，并直接产出“一行一个岗位”的完整结果表。

这版脚本和上一版最大的区别是：
1. 不再输出“一个 skill 一行”的明细表；
2. 直接保留完整样本字段；
3. 只额外新增一个 `skill_name` 列；
4. `skill_name` 使用 JSON 数组字符串，保存该岗位命中的全部硬技能主名称。

为什么这里不只用正则：
1. 中文技能词更适合做归一化后的包含匹配，例如“质量管理体系”“财务分析”；
2. 英文缩写、符号技能更需要边界意识，例如 `ERP`、`SQL`、`C++`、`APQP`；
3. 词典里同时有 `name` 和 `aliases`，单纯一套正则很难兼顾可维护性与准确率；
4. 大批量样本下，全部拼成超长正则既难维护，也更容易出现边界误匹配。

因此这里采用“混合匹配”：
1. 中文 / 混合技能词：对文本和词条都做轻量归一化，再做包含匹配；
2. 英文 / 缩写技能词：使用大小写无关、带边界约束的正则匹配；
3. 每个岗位优先使用其 `detail_path` 对应职业细类下的技能词，同时附加通用硬技能池；
4. 最终只保留去重后的 skill 主名称，写入 `skill_name` JSON 数组。

默认数据源：
- `recruit.main.skill_extraction_requirement_matches`

默认输出表：
- `recruit.main.hard_skill_match_results`

用法示例：
python -m src.skill_extraction.history.match_hard_skills_to_duckdb ^
  --dictionary dicts/occupation_skill_dictionary_v2.4.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import duckdb
import pandas as pd

from ..config import load_skill_extraction_config


GLOBAL_POOL_PATH = "通用技能池 > 跨职业通用硬技能"

# 这些模式用于排除明显不是硬技能的条目，避免在“任职要求 / 岗位职责”切分项里
# 把学历、年龄、工作制度等噪音当成技能文本继续匹配。
NON_SKILL_ITEM_PATTERNS = [
    re.compile(r"(本科|大专|硕士|博士|学历|学位)"),
    re.compile(r"(年龄|性别|形象|身高|户籍)"),
    re.compile(r"(\d+\s*年\s*经验|工作经验|经验要求)"),
    re.compile(r"(可接受.*(出差|加班|轮班|夜班)|能接受.*(出差|加班|轮班|夜班))"),
    re.compile(r"(责任心|抗压|沟通能力|团队协作|执行力|学习能力|稳定性)"),
]

# 最终写回 DuckDB 的字段顺序。这里严格按用户要求控制输出列，不保留调试列。
OUTPUT_COLUMNS = [
    "岗位名称",
    "岗位描述",
    "任职要求_items_text",
    "岗位职责_items_text",
    "sections_brief",
    "occupation_title",
    "occupation_code",
    "skill_name",
]

# 这些词通常不是“具体硬技能名”，而更像技能容器、能力集合或泛化主题。
# 当 alias 命中时，如果父 skill 名落入这类模式，我们优先把 alias 本身解析为最终技能名，
# 避免把 `Java` 匹配成“编程语言”或者把 `MySQL` 匹配成“数据库技术”。
GENERIC_SKILL_NAME_PATTERNS = [
    re.compile(r"(编程语言|数据库技术|网络编程|开发技术|开发能力|框架应用|框架开发)"),
    re.compile(r"(技术栈|生态组件|组件与原理|原理与应用|工具使用|软件使用)"),
    re.compile(r"(系统开发|系统设计|项目开发|平台开发|平台技术|开发实践)"),
    re.compile(r"(数据库操作|数据库应用|数据库开发|前端开发技术|后端开发技术)"),
    re.compile(r"(编程语言技能|语言技能|框架技能|开发框架|开发$|编程$)"),
]

# alias 归一化映射。
# 这里只放高确定性的缩写/变体，避免过度“智能化”带来错误归并。
ALIAS_CANONICAL_MAP = {
    "java": "Java",
    "java ee": "Java EE",
    "java web": "Java Web",
    "springmvc": "Spring MVC",
    "spring mvc": "Spring MVC",
    "springboot": "Spring Boot",
    "spring boot": "Spring Boot",
    "springcloud": "Spring Cloud",
    "spring cloud": "Spring Cloud",
    "mybatis": "MyBatis",
    "mysql": "MySQL",
    "redis": "Redis",
    "sql": "SQL",
    "sqlserver": "SQL Server",
    "sql server": "SQL Server",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "dubbo": "Dubbo",
}


def _safe_text(value: object) -> str:
    """安全转字符串并去首尾空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_match_text(text: str) -> str:
    """
    归一化匹配文本。

    设计目标不是做复杂 NLP，而是做“足够稳定的轻量归一化”：
    - 统一小写；
    - 去掉空白；
    - 去掉中英文常见标点；
    - 保留汉字、字母、数字主体，便于做字符串包含匹配。
    """
    lowered = str(text).lower()
    lowered = re.sub(r"[\s\u3000]+", "", lowered)
    lowered = re.sub(r"[，,。；;：:（）()\[\]【】{}<>《》“”\"'‘’、/\\|_.-]", "", lowered)
    return lowered


def safe_lower_text(text: str) -> str:
    """
    为英文缩写匹配准备文本。

    这里不删除所有标点，是为了保留 `C++` / `C#` / `Pro/E` 这类词的边界信息。
    """
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def split_items(text: str) -> List[str]:
    """
    把结构化字段拆成条目。

    当前上游切分脚本普遍使用 ` | ` 作为条目拼接符，因此这里优先按这个符号切。
    如果文本本身没有这个分隔符，则整体作为一个条目处理。
    """
    content = _safe_text(text)
    if not content:
        return []
    if " | " in content:
        items = [item.strip() for item in content.split(" | ")]
        return [item for item in items if item]
    return [content]


def is_skill_like_item(item: str) -> bool:
    """
    判断某条文本是否像“值得进入硬技能匹配”的内容。

    这里只做保守过滤，避免明显无效文本进入匹配流程。
    如果条目很短、或者明显是学历/年龄/软技能要求，就直接跳过。
    """
    text = _safe_text(item)
    if not text or len(text) <= 2:
        return False
    for pattern in NON_SKILL_ITEM_PATTERNS:
        if pattern.search(text):
            return False
    return True


@dataclass(frozen=True)
class TermEntry:
    """单个可匹配词项。"""

    skill_name: str
    term_text: str
    term_role: str
    is_ascii_like: bool
    normalized_term: str
    category: str | None = None


class HardSkillMatcher:
    """
    基于职业细类词典的硬技能匹配器。

    词典结构仍然按职业细类分层，但产出结果已经改成“一行一个岗位”。
    这样做的好处是：
    - 仍能利用 `detail_path` 缩小候选技能范围，减少误匹配；
    - 最终表结构足够轻，便于直接接入下游分析。
    """

    def __init__(self, dictionary: Dict):
        self.dictionary = dictionary
        self.term_index = self._build_term_index(dictionary)

    @staticmethod
    def _is_ascii_like_term(text: str) -> bool:
        """
        判断词项是否更适合走英文边界匹配。

        典型样例：
        - SQL
        - ERP
        - Java
        - C++
        - C#
        - Pro/E
        """
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9.+#/\-]*", text))

    def _build_term_index(self, dictionary: Dict) -> Dict[str, List[TermEntry]]:
        """为每个职业细类构建可匹配词项列表。"""
        categories = dictionary.get("categories", {})
        index: Dict[str, List[TermEntry]] = {}

        for detail_path, category in categories.items():
            entries: List[TermEntry] = []
            seen = set()
            for skill in category.get("skills", []) or []:
                skill_name = _safe_text(skill.get("name", ""))
                if not skill_name:
                    continue

                terms = [(skill_name, "name")]
                terms.extend([(_safe_text(alias), "alias") for alias in (skill.get("aliases", []) or [])])

                for term_text, term_role in terms:
                    if not term_text:
                        continue
                    key = (skill_name.casefold(), term_text.casefold())
                    if key in seen:
                        continue
                    seen.add(key)

                    entries.append(
                        TermEntry(
                            skill_name=skill_name,
                            term_text=term_text,
                            term_role=term_role,
                            is_ascii_like=self._is_ascii_like_term(term_text),
                            normalized_term=normalize_match_text(term_text),
                        )
                    )

            # 长词优先，减少短词抢匹配造成的噪音。
            entries.sort(key=lambda item: (len(item.normalized_term), len(item.term_text)), reverse=True)
            index[detail_path] = entries
        return index

    def _iter_entries(self) -> Iterable[TermEntry]:
        """
        读取全词典可用的技能词项。

        这里不再限制 `detail_path`，因为用户明确要求放开职业细类限制。
        这样会提升召回，但也会带来更多跨职业噪音，因此后续更依赖词典质量。
        """
        seen = set()
        for entries in self.term_index.values():
            for entry in entries:
                key = (entry.skill_name.casefold(), entry.term_text.casefold())
                if key in seen:
                    continue
                seen.add(key)
                yield entry

    @staticmethod
    def _is_generic_skill_name(skill_name: str) -> bool:
        """判断父 skill 名是否过于泛化。"""
        text = _safe_text(skill_name)
        if not text:
            return False
        return any(pattern.search(text) for pattern in GENERIC_SKILL_NAME_PATTERNS)

    @staticmethod
    def _canonicalize_alias(term_text: str) -> str:
        """
        把 alias 规范成更适合落表展示的技能名。

        例如：
        - `java` -> `Java`
        - `springmvc` -> `Spring MVC`
        - `mysql` -> `MySQL`
        """
        text = _safe_text(term_text)
        if not text:
            return ""
        compact_key = re.sub(r"\s+", " ", text).strip().casefold()
        canonical = ALIAS_CANONICAL_MAP.get(compact_key)
        if canonical:
            return canonical

        alias_stripped = re.sub(r"(语言|开发|编程|框架|数据库|技术)$", "", text, flags=re.IGNORECASE).strip()
        stripped_key = alias_stripped.casefold()
        canonical = ALIAS_CANONICAL_MAP.get(stripped_key)
        if canonical:
            return canonical
        return text

    def _resolve_output_skill_name(self, entry: TermEntry) -> str:
        """
        决定最终写入结果表的技能名。

        规则：
        1. 如果命中的是 skill 主名称，直接返回主名称；
        2. 如果命中的是 alias，且父 skill 名比较泛，则优先返回 alias 的规范名；
        3. 否则仍然返回父 skill 名，保证和词典主名称保持一致。
        """
        if entry.term_role == "name":
            return entry.skill_name
        if self._is_generic_skill_name(entry.skill_name):
            alias_name = self._canonicalize_alias(entry.term_text)
            if alias_name:
                return alias_name
        return entry.skill_name

    def match_text(self, text: str) -> List[str]:
        """
        对单段文本做技能匹配，并返回命中的 skill 主名称列表。

        这里返回的是“用于最终落表的技能名”。
        大多数情况下它等于词典中的 `skill_name`；
        但当命中的只是泛化父词条下的 alias 时，会优先回写更具体的 alias 规范名。
        """
        normalized_text = normalize_match_text(text)
        raw_text = safe_lower_text(text)
        if not normalized_text:
            return []

        matched_skill_names: List[str] = []
        seen_skills = set()

        for entry in self._iter_entries():
            if not entry.normalized_term:
                continue

            if entry.is_ascii_like:
                pattern = rf"(?<![a-z0-9]){re.escape(entry.term_text.lower())}(?![a-z0-9])"
                is_match = bool(re.search(pattern, raw_text))
            else:
                is_match = entry.normalized_term in normalized_text

            if not is_match:
                continue

            resolved_skill_name = self._resolve_output_skill_name(entry)
            skill_key = resolved_skill_name.casefold()
            if skill_key in seen_skills:
                continue
            seen_skills.add(skill_key)
            matched_skill_names.append(resolved_skill_name)

        return matched_skill_names


def load_dictionary(path: Path) -> Dict:
    """读取硬技能词典 JSON。"""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def fetch_source_rows(
    conn: duckdb.DuckDBPyConnection,
    source_table: str,
    limit: int | None,
) -> pd.DataFrame:
    """
    读取待匹配样本。

    这里仍然额外取 `detail_path`，因为匹配时要用它定位职业细类词典。
    但写回结果表时，会严格裁剪到用户要求的 8 列。
    """
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    query = f"""
        SELECT
            岗位名称,
            岗位描述,
            任职要求_items_text,
            岗位职责_items_text,
            sections_brief,
            occupation_title,
            occupation_code,
            detail_path
        FROM {source_table}
        {limit_clause}
    """
    return conn.execute(query).df()


def _collect_candidate_texts(row: Dict[str, object]) -> List[str]:
    """
    从单条岗位样本中收集待匹配文本。

    处理原则：
    - 对结构化条目字段进行切分；
    - 过滤明显不是技能条目的内容；
    - 保留全文字段作为召回补充；
    - 最终按原顺序去重，减少重复匹配开销。
    """
    candidates: List[str] = []

    for field_name in ["任职要求_items_text", "岗位职责_items_text"]:
        for item in split_items(_safe_text(row.get(field_name, ""))):
            if is_skill_like_item(item):
                candidates.append(item)

    cleaned_sections_brief = _clean_sections_brief(_safe_text(row.get("sections_brief", "")))
    for item in split_items(cleaned_sections_brief):
        if is_skill_like_item(item):
            candidates.append(item)

    job_desc = _safe_text(row.get("岗位描述", ""))
    if job_desc:
        candidates.append(job_desc)

    seen = set()
    deduped_candidates: List[str] = []
    for text in candidates:
        key = normalize_match_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(text)
    return deduped_candidates


def _clean_sections_brief(text: str) -> str:
    """
    清洗 `sections_brief`，避免把已经拆分到单独字段里的职责/要求再次重复塞进结果表。

    上游 `sections_brief` 常常是一个总摘要，里面会混入：
    - `岗位职责: ...`
    - `任职要求: ...`
    但这两部分已经分别写入 `岗位职责_items_text` 和 `任职要求_items_text`，
    因此这里把它们删掉，只保留其余摘要信息。
    """
    content = _safe_text(text)
    if not content:
        return ""

    kept_parts: List[str] = []
    for part in split_items(content):
        compact = part.strip()
        if re.match(r"^(岗位职责|任职要求)\s*[:：]", compact):
            continue
        kept_parts.append(compact)
    return " | ".join(kept_parts)


def match_dataframe(source_df: pd.DataFrame, matcher: HardSkillMatcher) -> pd.DataFrame:
    """
    对样本 DataFrame 做硬技能匹配，并返回最终产表。

    最终表特点：
    - 一行一个岗位样本；
    - 原始岗位字段完整保留；
    - `skill_name` 是 JSON 数组字符串，保存去重后的技能主名称。
    """
    output_rows: List[Dict[str, str]] = []

    for row in source_df.to_dict(orient="records"):
        matched_skills: List[str] = []
        seen_skills = set()
        cleaned_sections_brief = _clean_sections_brief(_safe_text(row.get("sections_brief", "")))

        for text in _collect_candidate_texts(row):
            for skill_name in matcher.match_text(text=text):
                skill_key = skill_name.casefold()
                if skill_key in seen_skills:
                    continue
                seen_skills.add(skill_key)
                matched_skills.append(skill_name)

        output_rows.append(
            {
                "岗位名称": _safe_text(row.get("岗位名称", "")),
                "岗位描述": _safe_text(row.get("岗位描述", "")),
                "任职要求_items_text": _safe_text(row.get("任职要求_items_text", "")),
                "岗位职责_items_text": _safe_text(row.get("岗位职责_items_text", "")),
                "sections_brief": cleaned_sections_brief,
                "occupation_title": _safe_text(row.get("occupation_title", "")),
                "occupation_code": _safe_text(row.get("occupation_code", "")),
                "skill_name": json.dumps(matched_skills, ensure_ascii=False),
            }
        )

    result_df = pd.DataFrame(output_rows)
    if result_df.empty:
        result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    return result_df[OUTPUT_COLUMNS]


def create_result_table(
    conn: duckdb.DuckDBPyConnection,
    result_df: pd.DataFrame,
    output_table: str,
) -> None:
    """
    把最终结果表写回 DuckDB。

    这里显式把 `skill_name` 转成 DuckDB 的 JSON 类型，而不是仅仅保存成 VARCHAR。
    这样下游如果要做：
    - `json_extract`
    - 数组展开
    - JSON 长度统计
    会更方便。
    """
    conn.register("tmp_hard_skill_match_results", result_df)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT
            岗位名称,
            岗位描述,
            任职要求_items_text,
            岗位职责_items_text,
            sections_brief,
            occupation_title,
            occupation_code,
            CAST(skill_name AS JSON) AS skill_name
        FROM tmp_hard_skill_match_results
        """
    )
    conn.unregister("tmp_hard_skill_match_results")


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="把硬技能词典匹配到 DuckDB 样本，并生成完整岗位结果表")
    parser.add_argument("--dictionary", required=True, help="硬技能词典 JSON 路径")
    parser.add_argument(
        "--source-table",
        default=None,
        help="DuckDB 源表；未指定时读取 config/database.yaml 中的 requirement_match_table",
    )
    parser.add_argument(
        "--output-table",
        default="recruit.main.hard_skill_match_results",
        help="最终结果表名；表结构固定为完整样本字段 + skill_name JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="调试用，只处理前 N 条样本")
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    config = load_skill_extraction_config()
    source_table = args.source_table or config.requirement_match_table
    dictionary = load_dictionary(Path(args.dictionary))
    matcher = HardSkillMatcher(dictionary)

    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute(f"PRAGMA threads={config.duckdb_threads}")
        source_df = fetch_source_rows(conn=conn, source_table=source_table, limit=args.limit)
        result_df = match_dataframe(source_df=source_df, matcher=matcher)
        create_result_table(conn=conn, result_df=result_df, output_table=args.output_table)

    matched_row_count = 0
    if not result_df.empty:
        matched_row_count = sum(
            1
            for value in result_df["skill_name"].tolist()
            if isinstance(value, str) and len(json.loads(value)) > 0
        )

    print(f"词典路径: {args.dictionary}")
    print(f"源表: {source_table}")
    print(f"输出表: {args.output_table}")
    print(f"处理样本数: {len(result_df)}")
    print(f"命中技能的样本数: {matched_row_count}")


if __name__ == "__main__":
    main()
