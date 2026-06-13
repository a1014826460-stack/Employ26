"""软技能词典构建器：基于种子词，用 LLM 扩展每个大五人格维度的变体词。

职责：
1. 接收种子词字典（来自 soft_skill_seed_extractor）；
2. 对每个维度，调用 LLM 生成语义相近的扩展词、别名；
3. 组装为统一的词典结构并输出 JSON 文件；
4. 当 LLM 不可用时，基于种子词和维度关键词做静态扩展。

使用方式:
    python -m src.skill_extraction.soft_skill_dictionary_builder

输出格式:
    {
      "schema_version": 1,
      "dimensions": {
        "openness": {
          "name": "开放性",
          "skills": [
            {"name": "创新", "aliases": ["创新能力", "创新思维"], "dimension": "openness"},
            ...
          ]
        },
        ...
      }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._dict_paths import get_current_soft_skill_dict_path

logger = logging.getLogger(__name__)

# 大五维度的中文名称映射
DIMENSION_DISPLAY_NAMES: Dict[str, str] = {
    "openness": "开放性",
    "conscientiousness": "尽责性",
    "extraversion": "外向性",
    "agreeableness": "宜人性",
    "neuroticism": "情绪稳定性",
}

# 每个维度的静态别名映射，用于 LLM 不可用时的兜底扩展
# 键为种子词，值为该词的常见别名/变体
_STATIC_ALIASES: Dict[str, List[str]] = {
    # ── openness ──
    "创新": ["创新能力", "创新思维", "创新意识", "开拓创新"],
    "好奇心": ["求知欲", "好奇心强", "探索欲"],
    "审美": ["审美能力", "审美观", "艺术鉴赏"],
    "想象力": ["想象创造力", "发散思维", "创想能力"],
    "灵活": ["灵活应变", "变通能力", "灵活性", "应变能力"],
    "学习能力": ["快速学习", "自主学习", "学习意愿", "求知好学"],
    "开放": ["思维开放", "开放心态", "心态开放"],
    "创造力": ["创造能力", "创造性思维", "创意能力"],
    "求知": ["求知欲", "好学", "主动学习"],
    "探索": ["探索精神", "勇于探索", "敢于尝试"],
    "变通": ["灵活变通", "随机应变", "通权达变"],
    "适应力": ["适应能力", "环境适应力", "快速适应"],
    "思维活跃": ["思维敏捷", "思路开阔", "头脑灵活"],
    "接受新事物": ["拥抱变化", "接纳新知", "勇于尝新"],
    "发散思维": ["多角度思考", "创造性思维", "逆向思维"],
    # ── conscientiousness ──
    "细心": ["细致", "注重细节", "一丝不苟"],
    "责任心": ["责任感", "有担当", "有责任心"],
    "自律": ["自我约束", "自我管理", "严格要求自己"],
    "计划性": ["有计划", "规划能力", "目标管理"],
    "严谨": ["严谨认真", "一丝不苟", "严格要求"],
    "高效": ["高效执行", "高效率", "效率高"],
    "认真": ["认真负责", "态度认真", "一丝不苟"],
    "踏实": ["脚踏实地", "务实", "稳扎稳打"],
    "细致": ["细致入微", "精细", "注重细节"],
    "执行力": ["执行能力强", "行动力强", "落地能力"],
    "条理": ["有条理", "逻辑清晰", "条理分明"],
    "守时": ["准时", "时间观念强", "按时完成"],
    "尽责": ["尽职尽责", "恪尽职守", "勤勉尽责"],
    "耐心": ["有耐心", "耐性好", "不急不躁"],
    "专注": ["专心致志", "心无旁骛", "注意力集中"],
    "自我管理": ["自律性强", "自我驱动", "自我提升"],
    "时间管理": ["时间观念强", "合理安排时间", "统筹时间"],
    # ── extraversion ──
    "沟通能力": ["沟通技巧", "表达沟通", "有效沟通"],
    "团队协作": ["团队合作", "协作精神", "配合默契"],
    "领导力": ["领导能力", "领导才能", "带队能力"],
    "表达能力": ["语言表达", "口头表达", "表达清晰"],
    "活跃": ["积极主动", "热情", "活泼开朗"],
    "外向": ["性格外向", "开朗", "健谈"],
    "社交": ["社交能力", "社交技巧", "善于交际"],
    "演讲": ["演讲能力", "公众演讲", "口才好"],
    "谈判": ["谈判能力", "谈判技巧", "商务谈判"],
    "影响力": ["个人影响力", "感召力", "号召力"],
    "感染力": ["情绪感染力", "热情感染", "鼓舞人心"],
    "人际交往": ["人际关系", "社交能力", "善于交际"],
    "组织能力": ["统筹能力", "策划能力", "活动组织"],
    "协调能力": ["统筹协调", "多方协调", "资源协调"],
    "公众表达": ["公开演讲", "台上表达", "汇报能力"],
    # ── agreeableness ──
    "合作": ["团队合作", "配合度高", "善于合作"],
    "同理心": ["共情能力", "换位思考", "理解他人"],
    "友善": ["友好", "和善", "待人友善"],
    "包容": ["宽容大度", "包容性强", "海纳百川"],
    "服务意识": ["服务精神", "客户导向", "乐于助人"],
    "亲和力": ["亲和感", "平易近人", "有亲和力"],
    "谦虚": ["谦逊", "低调", "虚心学习"],
    "信任": ["值得信赖", "诚信", "诚实守信"],
    "助人": ["乐于助人", "热心帮助", "有爱心"],
    "体贴": ["善解人意", "关心他人", "细致体贴"],
    "谦逊": ["谦虚谨慎", "不骄不躁", "虚心求教"],
    "尊重": ["尊重他人", "相互尊重", "尊重差异"],
    "善解人意": ["体谅他人", "理解力强", "感受力强"],
    "换位思考": ["同理心强", "站在对方角度", "理解对方立场"],
    # ── neuroticism ──
    "抗压能力": ["抗压性强", "能抗压", "压力承受力"],
    "情绪管理": ["情绪调节", "情绪控制", "管理情绪"],
    "冷静": ["沉着冷静", "临危不乱", "遇事冷静"],
    "心理承受力": ["心理素质好", "心理韧性", "心理强大"],
    "情绪稳定": ["情绪平稳", "心态平和", "不易波动"],
    "韧性": ["韧性强", "坚韧不拔", "百折不挠"],
    "自控力": ["自我控制", "克制力", "自制力"],
    "沉着": ["沉着应对", "镇定自若", "泰然处之"],
    "镇定": ["镇定从容", "沉着镇定", "面不改色"],
    "心理素质": ["心理素质过硬", "心理过关", "心态稳定"],
    "压力管理": ["压力调节", "化解压力", "排解压力"],
    "心态好": ["心态积极", "乐观心态", "正面心态"],
}

# LLM 扩展的系统提示词
_LLM_SYSTEM_PROMPT = """你是一个人力资源和组织行为学专家。
你的任务是为给定的软技能种子词生成语义相近的变体词和别名。

要求：
1. 变体词必须与种子词语义相近，可以出现在招聘描述中
2. 每个种子词生成 3-5 个变体词/别名
3. 变体词应为中文，2-6 个字为宜
4. 避免与种子词完全重复
5. 输出格式为纯 JSON 数组，例如: ["变体1", "变体2", "变体3"]"""

_LLM_USER_PROMPT_TEMPLATE = """维度: {dimension_name}（{dimension_key}）
种子词: {seed_word}

请为以上种子词生成 3-5 个语义相近的变体词/别名，输出为 JSON 数组。"""


def _expand_with_llm(
    seed_word: str,
    dimension_key: str,
    dimension_name: str,
    llm_client: Any,
) -> List[str]:
    """用 LLM 为单个种子词生成变体词。

    参数:
        seed_word: 种子词文本。
        dimension_key: 维度键名（如 openness）。
        dimension_name: 维度中文名（如 开放性）。
        llm_client: LLM 客户端，需支持 complete_text()。

    返回:
        List[str]: LLM 生成的变体词列表，失败时返回空列表。
    """
    try:
        user_prompt = _LLM_USER_PROMPT_TEMPLATE.format(
            dimension_name=dimension_name,
            dimension_key=dimension_key,
            seed_word=seed_word,
        )
        response = llm_client.complete_text(
            system_prompt=_LLM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        # 尝试从响应中解析 JSON 数组
        response = response.strip()
        if response.startswith("["):
            parsed = json.loads(response)
        else:
            # 尝试提取 JSON 部分
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                parsed = json.loads(response[start:end + 1])
            else:
                logger.warning("LLM 返回无法解析为 JSON 数组: %s", response[:100])
                return []

        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
        return []
    except Exception as exc:
        logger.warning("LLM 扩展失败 seed=%s error=%s", seed_word, exc)
        return []


def _build_skill_entry(
    name: str,
    dimension_key: str,
    aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """构建单个技能条目。

    参数:
        name: 技能名称。
        dimension_key: 所属维度键名。
        aliases: 别名列表。

    返回:
        dict: 标准化的技能条目。
    """
    return {
        "name": name,
        "aliases": aliases or [],
        "dimension": dimension_key,
    }


def _get_static_aliases(seed_word: str) -> List[str]:
    """获取种子词的静态别名。

    参数:
        seed_word: 种子词。

    返回:
        List[str]: 静态别名列表，未找到时返回空列表。
    """
    return list(_STATIC_ALIASES.get(seed_word, []))


def build_soft_skill_dictionary(
    seeds: Dict[str, List[str]],
    llm_client: Any = None,
) -> Dict[str, Any]:
    """基于种子词构建完整的软技能词典。

    对每个维度的每个种子词：
    - 若提供 llm_client，调用 LLM 生成变体词；
    - 否则使用静态别名映射；
    - 去重后组装为标准词典结构。

    参数:
        seeds: 大五维度种子词字典，格式为 {维度名: [种子词列表]}。
            来自 soft_skill_seed_extractor.extract_soft_skill_seeds()。
        llm_client: 可选的 LLM 客户端。为 None 时使用静态别名扩展。

    返回:
        dict: 完整的软技能词典，格式参见模块 docstring。

    示例:
        >>> from src.skill_extraction.soft_skill_seed_extractor import (
        ...     extract_soft_skill_seeds,
        ... )
        >>> seeds = extract_soft_skill_seeds()
        >>> dictionary = build_soft_skill_dictionary(seeds)
        >>> dictionary["schema_version"]
        1
    """
    logger.info("开始构建软技能词典，共 %d 个维度", len(seeds))

    dimensions: Dict[str, Any] = {}

    for dimension_key, seed_words in seeds.items():
        dimension_name = DIMENSION_DISPLAY_NAMES.get(dimension_key, dimension_key)
        logger.info("处理维度: %s (%s)，%d 个种子词", dimension_name, dimension_key, len(seed_words))

        skills: List[Dict[str, Any]] = []
        seen_names: set = set()

        for seed_word in seed_words:
            if seed_word in seen_names:
                continue
            seen_names.add(seed_word)

            # 获取别名：优先 LLM，兜底静态映射
            if llm_client is not None:
                llm_aliases = _expand_with_llm(
                    seed_word, dimension_key, dimension_name, llm_client
                )
                static_aliases = _get_static_aliases(seed_word)
                # 合并去重，LLM 结果优先
                all_aliases = llm_aliases + [
                    a for a in static_aliases if a not in set(llm_aliases)
                ]
            else:
                all_aliases = _get_static_aliases(seed_word)

            # 从别名中去掉与 name 重复的
            all_aliases = [a for a in all_aliases if a != seed_word]

            # 将别名中未见过的也加入 seen_names 防止后续重复
            for alias in all_aliases:
                seen_names.add(alias)

            entry = _build_skill_entry(
                name=seed_word,
                dimension_key=dimension_key,
                aliases=all_aliases,
            )
            skills.append(entry)

        dimensions[dimension_key] = {
            "name": dimension_name,
            "skills": skills,
        }

        logger.info(
            "维度 %s: %d 个技能条目",
            dimension_name,
            len(skills),
        )

    dictionary = {
        "schema_version": 1,
        "dimensions": dimensions,
    }

    total_skills = sum(len(d["skills"]) for d in dimensions.values())
    logger.info("词典构建完成，共 %d 个维度，%d 个技能条目", len(dimensions), total_skills)

    return dictionary


def save_dictionary(
    dictionary: Dict[str, Any],
    output_path: Path | str,
) -> Path:
    """将词典保存为 JSON 文件。

    参数:
        dictionary: 词典字典。
        output_path: 输出文件路径。

    返回:
        Path: 实际写入的文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    logger.info("词典已保存至 %s", output_path)
    return output_path


def load_dictionary(path: Path | str) -> Dict[str, Any]:
    """从 JSON 文件加载词典。

    参数:
        path: 词典文件路径。

    返回:
        dict: 词典字典。

    异常:
        FileNotFoundError: 文件不存在。
        ValueError: 文件格式不合法。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"词典文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "schema_version" not in data:
        raise ValueError(f"词典文件格式不合法: {path}")

    return data


def main() -> None:
    """CLI 入口：基于种子词构建软技能词典并保存。"""
    import argparse

    from config.paths import get_project_paths

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="构建软技能词典")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="使用 LLM 扩展变体词（默认使用静态别名）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认 dicts/soft_skill_dictionary.json）",
    )
    args = parser.parse_args()

    paths = get_project_paths()
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认输出到版本化词典路径；若 current.txt 不存在，兜底使用旧路径
        try:
            output_path = get_current_soft_skill_dict_path()
        except (FileNotFoundError, ImportError):
            output_path = paths.project_root / "dicts" / "soft_skill_dictionary.json"

    # 获取种子词
    from src.skill_extraction.soft_skill_seed_extractor import extract_soft_skill_seeds

    seeds = extract_soft_skill_seeds()
    logger.info("种子词维度: %s", list(seeds.keys()))

    # 可选：创建 LLM 客户端
    llm_client = None
    if args.use_llm:
        from src.model_platform.llm import create_llm_client

        llm_client = create_llm_client()
        logger.info("已创建 LLM 客户端")

    # 构建词典
    dictionary = build_soft_skill_dictionary(seeds, llm_client=llm_client)

    # 保存
    save_dictionary(dictionary, output_path)

    # 打印摘要
    print("\n=== 软技能词典构建完成 ===\n")
    print(f"Schema 版本: {dictionary['schema_version']}")
    print(f"维度数量: {len(dictionary['dimensions'])}")
    total = sum(len(d['skills']) for d in dictionary['dimensions'].values())
    print(f"技能条目总数: {total}")
    print()
    for key, dim in dictionary["dimensions"].items():
        print(f"  {dim['name']} ({key}): {len(dim['skills'])} 个技能")
    print(f"\n输出文件: {output_path}")


if __name__ == "__main__":
    main()
