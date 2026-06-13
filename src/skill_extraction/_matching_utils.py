"""硬技能匹配共享工具（无 DuckDB 依赖）。

从 ``history/match_hard_skills_to_duckdb.py`` 中抽取的纯函数和常量，
供 ``hard_skill_matcher.py`` 等 PostgreSQL 链路安全引用。

本模块不导入 ``duckdb``、``pandas`` 或任何数据库驱动。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


# ============================================================================
#  常量
# ============================================================================

# 这些词通常不是"具体硬技能名"，而更像技能容器、能力集合或泛化主题。
# 当 alias 命中时，如果父 skill 名落入这类模式，我们优先把 alias 本身解析为最终技能名，
# 避免把 `Java` 匹配成"编程语言"或者把 `MySQL` 匹配成"数据库技术"。
GENERIC_SKILL_NAME_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(编程语言|数据库技术|网络编程|开发技术|开发能力|框架应用|框架开发)"),
    re.compile(r"(技术栈|生态组件|组件与原理|原理与应用|工具使用|软件使用)"),
    re.compile(r"(系统开发|系统设计|项目开发|平台开发|平台技术|开发实践)"),
    re.compile(r"(数据库操作|数据库应用|数据库开发|前端开发技术|后端开发技术)"),
    re.compile(r"(编程语言技能|语言技能|框架技能|开发框架|开发$|编程$)"),
]

# alias 归一化映射。
# 这里只放高确定性的缩写/变体，避免过度"智能化"带来错误归并。
ALIAS_CANONICAL_MAP: dict[str, str] = {
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


# ============================================================================
#  工具函数
# ============================================================================


def _safe_text(value: object) -> str:
    """安全转字符串并去首尾空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_match_text(text: str) -> str:
    """归一化匹配文本。

    设计目标不是做复杂 NLP，而是做"足够稳定的轻量归一化"：
    - 统一小写；
    - 去掉空白；
    - 去掉中英文常见标点；
    - 保留汉字、字母、数字主体，便于做字符串包含匹配。
    """
    lowered = str(text).lower()
    lowered = re.sub(r"[\s　]+", "", lowered)
    lowered = re.sub(
        r"[，,。；;：:（）()\[\]【】{}<>《》“”\"'‘’、/\\|_.-]", "", lowered
    )
    return lowered


def safe_lower_text(text: str) -> str:
    """为英文缩写匹配准备文本。

    这里不删除所有标点，是为了保留 ``C++`` / ``C#`` / ``Pro/E`` 这类词的边界信息。
    """
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def is_ascii_like_term(text: str) -> bool:
    """判断词项是否更适合走英文边界匹配。

    典型样例：
    - SQL
    - ERP
    - Java
    - C++
    - C#
    - Pro/E
    """
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9.+#/\-]*", text))


def is_generic_skill_name(skill_name: str) -> bool:
    """判断父 skill 名是否过于泛化。"""
    text = _safe_text(skill_name)
    if not text:
        return False
    return any(pattern.search(text) for pattern in GENERIC_SKILL_NAME_PATTERNS)


def canonicalize_alias(term_text: str) -> str:
    """把 alias 规范成更适合落表展示的技能名。

    例如：
    - ``java`` -> ``Java``
    - ``springmvc`` -> ``Spring MVC``
    - ``mysql`` -> ``MySQL``
    """
    text = _safe_text(term_text)
    if not text:
        return ""
    compact_key = re.sub(r"\s+", " ", text).strip().casefold()
    canonical = ALIAS_CANONICAL_MAP.get(compact_key)
    if canonical:
        return canonical

    alias_stripped = re.sub(
        r"(语言|开发|编程|框架|数据库|技术)$", "", text, flags=re.IGNORECASE
    ).strip()
    stripped_key = alias_stripped.casefold()
    canonical = ALIAS_CANONICAL_MAP.get(stripped_key)
    if canonical:
        return canonical
    return text


# ============================================================================
#  数据结构
# ============================================================================


@dataclass(frozen=True)
class TermEntry:
    """单个可匹配词项。"""

    skill_name: str
    term_text: str
    term_role: str
    is_ascii_like: bool
    normalized_term: str
    category: str | None = None
