"""从标注数据中提取软技能种子词，并按大五人格维度分组。

职责：
1. 查询 `annotations.label_studio_annotations_v2` 表的 `soft_skill` 字段；
2. 统计每个软技能标注词的出现频次，过滤掉频次 < 3 的低频词；
3. 将高频种子词映射到大五人格维度（基于关键词匹配）；
4. 当数据库不可用时，使用维度定义中的预设关键词作为兜底种子。

使用方式:
    python -m src.skill_extraction.soft_skill_seed_extractor

返回结构:
    dict[str, list[str]]: 维度名 -> 种子词列表
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

from config.paths import get_project_paths

logger = logging.getLogger(__name__)

# 大五人格维度关键词映射表
# 每个维度对应一组中文关键词，用于将种子词归类
DIMENSION_KEYWORDS: Dict[str, List[str]] = {
    "openness": [
        "创新", "好奇心", "审美", "想象力", "灵活", "学习能力",
        "开放", "创造力", "求知", "探索", "变通", "适应力",
        "思维活跃", "接受新事物", "发散思维",
    ],
    "conscientiousness": [
        "细心", "责任心", "自律", "计划性", "严谨", "高效",
        "认真", "踏实", "细致", "执行力", "条理", "守时",
        "尽责", "耐心", "专注", "自我管理", "时间管理",
    ],
    "extraversion": [
        "沟通能力", "团队协作", "领导力", "表达能力", "活跃",
        "外向", "社交", "演讲", "谈判", "影响力", "感染力",
        "人际交往", "组织能力", "协调能力", "公众表达",
    ],
    "agreeableness": [
        "合作", "同理心", "友善", "包容", "服务意识",
        "亲和力", "谦虚", "信任", "助人", "体贴", "谦逊",
        "尊重", "善解人意", "换位思考",
    ],
    "neuroticism": [
        "抗压能力", "情绪管理", "冷静", "心理承受力",
        "情绪稳定", "韧性", "自控力", "沉着", "镇定",
        "心理素质", "压力管理", "心态好",
    ],
}

# 最低频次阈值：低于此值的标注词不纳入种子词
MIN_FREQUENCY = 3


def _map_to_dimension(skill_text: str) -> Optional[str]:
    """将一个软技能文本映射到大五人格维度。

    遍历每个维度的关键词列表，如果技能文本中包含某个关键词，
    则归类到该维度。优先返回第一个命中的维度。

    参数:
        skill_text: 软技能标注文本。

    返回:
        Optional[str]: 命中的维度名，未命中时返回 None。
    """
    if not skill_text or not skill_text.strip():
        return None

    text = skill_text.strip()
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return dimension
    return None


def _fetch_soft_skills_from_db() -> List[str]:
    """从 PostgreSQL 查询所有非空的 soft_skill 标注值。

    查询 `annotations.label_studio_annotations_v2` 表，
    获取 `soft_skill` 字段中所有非空的标注文本。

    返回:
        List[str]: 所有非空 soft_skill 值的列表。

    异常:
        当数据库连接失败时抛出异常，由调用方决定是否使用兜底逻辑。
    """
    from sqlalchemy import create_engine, text

    paths = get_project_paths()
    engine = create_engine(paths.pg_sqlalchemy_url(), future=True)

    query = text(
        """
        SELECT soft_skill
        FROM annotations.label_studio_annotations_v2
        WHERE soft_skill IS NOT NULL
          AND TRIM(soft_skill) <> ''
        """
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        return [str(row[0]).strip() for row in rows if row[0]]
    finally:
        engine.dispose()


def _build_fallback_seeds() -> Dict[str, List[str]]:
    """当数据库不可用时，使用维度定义中的关键词作为兜底种子词。

    每个维度取其关键词列表中的前 5 个作为种子词。

    返回:
        dict[str, list[str]]: 维度名 -> 兜底种子词列表。
    """
    logger.warning("数据库不可用，使用维度定义中的预设关键词作为兜底种子词")
    fallback: Dict[str, List[str]] = {}
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        fallback[dimension] = keywords[:5]
    return fallback


def _group_by_dimension(
    skill_counts: Counter,
    min_frequency: int = MIN_FREQUENCY,
) -> Dict[str, List[str]]:
    """将高频技能词按大五维度分组。

    参数:
        skill_counts: 技能词频次统计。
        min_frequency: 最低频次阈值。

    返回:
        dict[str, list[str]]: 维度名 -> 按频次降序排列的种子词列表。
    """
    result: Dict[str, List[str]] = {
        dim: [] for dim in DIMENSION_KEYWORDS
    }

    # 按频次降序排列，优先处理高频词
    for skill, count in skill_counts.most_common():
        if count < min_frequency:
            continue
        dimension = _map_to_dimension(skill)
        if dimension is not None:
            result[dimension].append(skill)

    # 每个维度内部按频次降序排列
    for dimension in result:
        result[dimension].sort(
            key=lambda s: skill_counts.get(s, 0),
            reverse=True,
        )

    return result


def extract_soft_skill_seeds(
    *,
    min_frequency: int = MIN_FREQUENCY,
    fallback_on_db_error: bool = True,
) -> Dict[str, List[str]]:
    """从标注数据中提取软技能种子词并按大五维度分组。

    主入口函数。查询数据库中 `annotations.label_studio_annotations_v2`
    的 `soft_skill` 字段，统计频次，过滤低频词，映射到大五维度。

    参数:
        min_frequency: 最低频次阈值，默认 3。
        fallback_on_db_error: 数据库不可用时是否使用兜底种子词。
            设为 False 时，数据库错误会直接抛出。

    返回:
        dict[str, list[str]]: 大五维度名 -> 种子词列表。
        维度名为: openness, conscientiousness, extraversion,
        agreeableness, neuroticism。

    示例:
        >>> seeds = extract_soft_skill_seeds()
        >>> seeds["openness"]  # 包含"创新"等开放性相关词
        ['创新', '学习能力', ...]
    """
    try:
        raw_skills = _fetch_soft_skills_from_db()
    except Exception as exc:
        if fallback_on_db_error:
            logger.warning("数据库查询失败 (%s)，使用兜底种子词", exc)
            return _build_fallback_seeds()
        raise

    if not raw_skills:
        logger.info("数据库中未找到非空 soft_skill 标注，使用兜底种子词")
        return _build_fallback_seeds()

    # 统计频次
    skill_counts: Counter = Counter()
    for skill in raw_skills:
        # 处理可能的多值字段（如逗号分隔的多个技能）
        parts = [p.strip() for p in skill.replace("，", ",").split(",") if p.strip()]
        for part in parts:
            skill_counts[part] += 1

    logger.info(
        "共提取 %d 个唯一软技能标注，其中 %d 个频次 >= %d",
        len(skill_counts),
        sum(1 for _, c in skill_counts.items() if c >= min_frequency),
        min_frequency,
    )

    return _group_by_dimension(skill_counts, min_frequency=min_frequency)


def main() -> None:
    """CLI 入口：打印提取结果。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    seeds = extract_soft_skill_seeds()

    print("\n=== 软技能种子词（大五维度分组）===\n")
    for dimension, words in seeds.items():
        print(f"【{dimension}】({len(words)} 个)")
        if words:
            for w in words:
                print(f"  - {w}")
        else:
            print("  (无)")
        print()


if __name__ == "__main__":
    main()
