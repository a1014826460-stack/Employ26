"""技能类别映射模块。

本模块提供从 `skill_type` 到标准 8 类的映射功能，包括：
1. 基于规则的确定性映射（覆盖约 100 种已知 skill_type 值）
2. 基于关键词的启发式分类（覆盖规则未命中的技能）
3. 对仍未命中的技能通过 LLM 进行分类
4. 批量为技能词典增加 `category` 字段

8 个标准类别：
- programming_language: 编程语言
- framework: 框架
- database: 数据库
- tool: 工具软件
- office: 办公软件
- equipment: 设备/仪器
- process: 工艺方法
- certification: 证书/资质
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 8 个合法 category 值
VALID_CATEGORIES: frozenset[str] = frozenset({
    "programming_language",
    "framework",
    "database",
    "tool",
    "office",
    "equipment",
    "process",
    "certification",
})

# ─── 启发式分类关键词集 ───────────────────────────────────

_PL_KEYWORDS: frozenset[str] = frozenset({
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Golang",
    "Rust", "Swift", "Kotlin", "Scala", "Ruby", "PHP", "Perl", "R语言", "Lua",
    "MATLAB", "Haskell", "Erlang", "Elixir", "Clojure", "F#", "Objective-C",
    "Shell", "Bash", "PowerShell", "SQL", "PL/SQL", "T-SQL", "HTML", "CSS",
    "XML", "YAML", "Markdown", "LaTeX", "Verilog", "VHDL", "Assembly",
    "汇编", "ABAP", "Groovy", "Dart", "Julia", "VBA", "Fortran", "COBOL",
    "AJAX", "JSON",
})

_FW_KEYWORDS: frozenset[str] = frozenset({
    "Spring", "Spring Boot", "Spring Cloud", "Django", "Flask", "FastAPI",
    "React", "Vue", "Vue.js", "Angular", "Next.js", "Nuxt.js", "Express",
    "Node.js", "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "Hadoop",
    "Spark", "Flink", "Kafka", "Qt", ".NET", ".NET Core", "Hibernate",
    "MyBatis", "MyBatis-Plus", "Dubbo", "gRPC", "Thrift",
    "Transformer模型", "RNN", "DPDK", "VPP", "OVS",
})

_DB_KEYWORDS: frozenset[str] = frozenset({
    "MySQL", "PostgreSQL", "Redis", "MongoDB", "Oracle", "SQL Server",
    "Elasticsearch", "ClickHouse", "HBase", "Cassandra", "Neo4j",
    "InfluxDB", "TiDB", "达梦", "人大金仓", "OpenGauss", "数据仓库",
})

_OFFICE_KEYWORDS: frozenset[str] = frozenset({
    "Microsoft Office", "Microsoft Excel", "Microsoft Word",
    "Microsoft PowerPoint", "Microsoft Outlook", "WPS Office", "WPS",
    "金山文档", "腾讯文档", "飞书", "钉钉", "企业微信",
})

_EQUIP_KEYWORDS: frozenset[str] = frozenset({
    "示波器", "万用表", "频谱分析仪", "信号发生器", "逻辑分析仪",
    "PLC", "CNC", "光谱仪", "显微镜", "3D打印机", "焊接设备",
    "贴片机", "回流焊", "波峰焊", "AOI", "X-Ray",
    "CentOS", "Windows", "Windows Server",
    "中央空调", "空气源热泵", "水热泵",
})

_CERT_KEYWORDS: frozenset[str] = frozenset({
    "PMP", "CPA", "CFA", "FRM", "ACCA", "CMA", "CIA",
    "一级建造师", "二级建造师", "注册会计师", "注册电气工程师",
    "CET-4", "CET-6", "雅思", "托福", "GRE", "GMAT",
    "AWS认证", "Azure认证", "HCIE", "HCIP", "CCIE", "CCNP",
    "A-Level", "A-Level课程体系", "AP课程", "AP", "IB课程体系", "IBDP",
    "STEAM创客编程教育", "雅思考试",
    "ISO", "ISO9001", "ISO14001", "ISO16949", "ISO22000", "ISO标准",
    "ISO 9001质量管理体系", "ISO9001体系", "ISO9001/14001体系",
    "ISO9001质量管理体系", "ISO12647",
    "IATF 16949管理体系", "IATF16949", "IATF16949体系标准", "16949体系",
    "TS16949", "QS9000", "AS9100",
    "GMP", "GMP规范", "HACCP", "HACCP体系",
    "FDA", "MDR法规", "ICHQ10研发质量体系",
    "GB14880", "GB2760", "IPC-A-610",
    "VDA", "VDA6.3审核", "RoHS标准",
    "CCAR-91/135/136/141部", "民航运行标准", "航空发动机适航规范",
    "船级社规范", "船舶设计规范",
    "消防国标", "消防标准", "消防行业标准", "国家消防安全法律法规",
    "《电业安全工作规程》", "《调度规程》",
    "环保法规", "食品法规", "食品添加剂法规", "食品营养强化剂法规",
    "饲料法律法规", "行业标准与法规",
    "质量体系", "质量管理", "知识产权法规", "知识产权实务经验",
    "物业管理法律法规", "物业管理法规", "物业管理相关法律法规",
    "刑事诉讼法", "刑法", "法律法规", "法律文书撰写", "法律文书起草",
    "贸易法规", "海关政策", "海关业务流程",
    "SOP制作与更新", "PFMEA", "PPAP", "APQP", "APQP流程",
    "SPC", "CPK分析", "IE", "KPI", "OKR", "BLM模型", "SCI论文写作",
    "DAMA", "DCMM", "安规要求", "内控管理", "内部控制", "FQC", "BOM",
    "项目管理",
})

# skill_type 语义兜底映射
_TYPE_FALLBACK: dict[str, str] = {
    "system": "equipment",
    "专业技能": "process",
    "专业领域": "process",
    "产品开发": "process",
    "产品研发": "process",
    "产品设计": "process",
    "仓储管理": "process",
    "农业技术": "process",
    "基因组学": "process",
    "多语言内容制作": "process",
    "实验室分析": "process",
    "实验技术": "process",
    "工业互联网": "process",
    "市场开发": "process",
    "影像制作": "process",
    "影视制作": "process",
    "技术": "process",
    "技能": "process",
    "操作系统": "tool",
    "故障诊断": "process",
    "教学技能": "process",
    "新能源汽车技术": "process",
    "服务技能": "process",
    "服装设计": "process",
    "机械设备": "equipment",
    "机械设计": "process",
    "材料科学": "process",
    "汽车工程": "process",
    "活动管理": "process",
    "物流管理": "process",
    "环保技术": "process",
    "生产管理": "process",
    "生物制药": "process",
    "电子技术基础": "process",
    "电气工程": "process",
    "细胞培养": "process",
    "网络技术": "process",
    "设计工程": "process",
    "软件开发": "process",
    "软件开发技能": "process",
    "项目管理": "process",
}


def _is_programming_language(name: str) -> bool:
    """判断是否为编程语言。"""
    if name in _PL_KEYWORDS:
        return True
    return bool(re.search(r"编程语言|脚本语言", name))


def _is_framework(name: str) -> bool:
    """判断是否为框架/库/运行时。"""
    if name in _FW_KEYWORDS:
        return True
    return bool(re.search(r"框架|引擎|runtime|运行时", name, re.IGNORECASE))


def _is_database(name: str) -> bool:
    """判断是否为数据库。"""
    if name in _DB_KEYWORDS:
        return True
    return bool(re.search(r"数据库|数据仓库|数据存储|NoSQL|NewSQL", name, re.IGNORECASE))


def _is_office(name: str) -> bool:
    """判断是否为办公软件。"""
    if name in _OFFICE_KEYWORDS:
        return True
    return bool(re.search(r"办公软件|文档处理|表格处理|演示文稿", name))


def _is_equipment(name: str) -> bool:
    """判断是否为设备/仪器。"""
    if name in _EQUIP_KEYWORDS:
        return True
    if re.search(r"设备|仪器|仪表|机床|机器人|控制器|传感器|探头|夹具|治具|工装", name):
        return True
    return bool(re.search(r"PLC|CNC|DCS|SCADA|HMI|变频器|伺服", name, re.IGNORECASE))


def _is_certification(name: str) -> bool:
    """判断是否为证书/资质/标准体系。"""
    if name in _CERT_KEYWORDS:
        return True
    if re.search(r"证书|资质|认证|资格|执照|牌照", name):
        return True
    if re.search(r"体系|标准|规范|法规|法律|规程|制度|规定", name):
        return True
    return bool(re.search(r"ISO|IATF|GMP|HACCP|FDA|VDA|GB\d|TS\d", name, re.IGNORECASE))


def _is_process(name: str) -> bool:
    """判断是否为工艺方法/技术方法。"""
    if re.search(
        r"工艺|方法|技术|流程|操作|实验|分析|检测|检验|测试|调试|优化|开发|设计|制造|生产|加工",
        name,
    ):
        return True
    return False


def classify_by_heuristic(name: str, skill_type: str) -> str | None:
    """使用启发式规则对单个技能进行分类。

    按优先级依次检查：编程语言 > 数据库 > 办公 > 证书 > 框架 > 设备 > 工艺。
    若均未命中，按 skill_type 语义兜底；"专业知识" 默认归为 process。

    参数:
        name: 技能名称。
        skill_type: 原始 skill_type 值。

    返回:
        str | None: 标准类别名称，无法判断时返回 None。
    """
    if _is_programming_language(name):
        return "programming_language"
    if _is_database(name):
        return "database"
    if _is_office(name):
        return "office"
    if _is_certification(name):
        return "certification"
    if _is_framework(name):
        return "framework"
    if _is_equipment(name):
        return "equipment"
    if _is_process(name):
        return "process"

    # skill_type 语义兜底
    result = _TYPE_FALLBACK.get(skill_type)
    if result:
        return result

    # "专业知识" 是最大的未映射类别，归为 process
    if skill_type == "专业知识":
        return "process"

    return None


# 默认规则文件路径（相对于项目根目录）
_DEFAULT_RULES_RELATIVE = Path("dicts/skill_category_rules.json")


def _project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parents[2]


def _default_rules_path() -> Path:
    """返回默认映射规则文件路径。"""
    return _project_root() / _DEFAULT_RULES_RELATIVE


def load_category_rules(rules_path: str | Path | None = None) -> dict[str, Any]:
    """加载技能类别映射规则文件。

    参数:
        rules_path: 规则文件路径，为空时使用默认路径 dicts/skill_category_rules.json。

    返回:
        dict: 包含 'mapping_rules'、'categories' 和 'llm_classification_prompt' 的字典。

    异常:
        FileNotFoundError: 规则文件不存在。
        ValueError: JSON 解析失败或缺少必要字段。
    """
    path = Path(rules_path) if rules_path else _default_rules_path()
    if not path.exists():
        raise FileNotFoundError(f"映射规则文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "mapping_rules" not in data:
        raise ValueError(f"规则文件缺少 'mapping_rules' 字段: {path}")
    if "categories" not in data:
        raise ValueError(f"规则文件缺少 'categories' 字段: {path}")

    return data


def map_skill_type(
    skill_type: str,
    rules: dict[str, str] | None = None,
    rules_data: dict[str, Any] | None = None,
) -> str | None:
    """将单个 skill_type 值映射到标准类别。

    参数:
        skill_type: 原始 skill_type 值。
        rules: 映射规则字典（skill_type -> category），与 rules_data 二选一。
        rules_data: 完整规则数据（含 mapping_rules 字段），与 rules 二选一。

    返回:
        str | None: 标准类别名称，规则未命中时返回 None。
        "needs_llm" 标记的 skill_type 也会返回 None，与未知 skill_type 行为一致，
        由调用方决定是否将 None 结果交给 LLM 分类。
    """
    if rules is None:
        if rules_data is None:
            rules_data = load_category_rules()
        rules = rules_data.get("mapping_rules", {})

    category = rules.get(skill_type)
    if category == "needs_llm":
        return None
    return category


def _build_llm_classification_prompt(
    skill_names: list[str],
    categories: dict[str, Any],
    prompt_config: dict[str, Any],
) -> tuple[str, str]:
    """构建 LLM 分类的 system_prompt 和 user_prompt。

    参数:
        skill_names: 待分类的技能名称列表。
        categories: 类别定义字典。
        prompt_config: prompt 模板配置。

    返回:
        tuple[str, str]: (system_prompt, user_prompt)
    """
    system_prompt = prompt_config.get("system_prompt", "")

    # 构建技能列表文本
    skills_text = "\n".join(f"- {name}" for name in skill_names)

    user_prompt = (
        f"请对以下 {len(skill_names)} 个技能进行分类。\n\n"
        f"技能列表：\n{skills_text}\n\n"
        f"请返回一个 JSON 数组，每个元素格式为 {{\"skill\": \"技能名\", \"category\": \"类别英文标识\"}}。\n"
        f"类别必须是以下之一：{', '.join(sorted(VALID_CATEGORIES))}\n"
        f"只返回 JSON 数组，不要返回其他内容。"
    )

    return system_prompt, user_prompt


def classify_batch_by_llm(
    skill_names: list[str],
    rules_data: dict[str, Any] | None = None,
    llm_client: Any = None,
    batch_size: int = 50,
) -> dict[str, str]:
    """通过 LLM 对未映射的技能进行批量分类。

    参数:
        skill_names: 待分类的技能名称列表。
        rules_data: 完整规则数据，用于获取类别定义和 prompt 模板。
        llm_client: LLM client 实例，为空时通过 create_llm_client() 创建。
        batch_size: 每批处理的技能数量，默认 50。

    返回:
        dict[str, str]: 技能名称 -> 类别英文标识的映射。
        LLM 返回无效类别时该技能会被跳过。
    """
    if not skill_names:
        return {}

    if rules_data is None:
        rules_data = load_category_rules()

    categories = rules_data.get("categories", {})
    prompt_config = rules_data.get("llm_classification_prompt", {})

    if llm_client is None:
        from src.model_platform.llm import create_llm_client

        llm_client = create_llm_client()

    results: dict[str, str] = {}

    # 分批处理
    for i in range(0, len(skill_names), batch_size):
        batch = skill_names[i : i + batch_size]
        system_prompt, user_prompt = _build_llm_classification_prompt(
            batch, categories, prompt_config
        )

        try:
            parsed = llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "skill" in item and "category" in item:
                        skill = item["skill"]
                        category = item["category"]
                        if category in VALID_CATEGORIES:
                            results[skill] = category
                        else:
                            logger.warning(
                                "LLM 返回无效类别 '%s'（技能: %s），已跳过",
                                category,
                                skill,
                            )
            elif isinstance(parsed, dict):
                # 兼容单个技能时返回对象的情况
                if "skill" in parsed and "category" in parsed:
                    skill = parsed["skill"]
                    category = parsed["category"]
                    if category in VALID_CATEGORIES:
                        results[skill] = category

        except Exception as exc:
            logger.error("LLM 分类批次失败 (batch %d): %s", i // batch_size, exc)
            # 尝试逐个重试
            for skill_name in batch:
                try:
                    single_prompt = prompt_config.get("user_prompt_template", "").format(
                        skill_name=skill_name, context=""
                    )
                    single_result = llm_client.complete_json(
                        system_prompt=system_prompt,
                        user_prompt=single_prompt,
                    )
                    if isinstance(single_result, dict) and "category" in single_result:
                        category = single_result["category"]
                        if category in VALID_CATEGORIES:
                            results[skill_name] = category
                except Exception as single_exc:
                    logger.error("单个技能分类失败 (%s): %s", skill_name, single_exc)

    return results


def apply_categories_to_dictionary(
    dict_path: str | Path | None = None,
    output_path: str | Path | None = None,
    rules_data: dict[str, Any] | None = None,
    llm_client: Any = None,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """为技能词典中每个技能增加 category 字段。

    处理流程：
    1. 加载映射规则
    2. 对每个技能的 skill_type 尝试规则映射
    3. 对规则未命中的技能（或 skill_type 为 "专业知识"），通过 LLM 分类
    4. 在每个技能对象中增加 "category" 字段
    5. 返回更新后的词典数据

    参数:
        dict_path: 词典文件路径，为空时使用默认路径。
        output_path: 输出文件路径，为空时不写文件。
        rules_data: 完整规则数据，为空时自动加载。
        llm_client: LLM client 实例，为空时自动创建。
        skip_llm: 是否跳过 LLM 分类（未映射技能的 category 设为 None）。

    返回:
        dict: 更新后的词典数据，包含 category 字段。
    """
    from config.paths import get_project_paths

    if dict_path is None:
        paths = get_project_paths()
        dict_path = paths.dict_dir / "flat_skill_dictionary.json"
    dict_path = Path(dict_path)

    if rules_data is None:
        rules_data = load_category_rules()

    mapping_rules = rules_data.get("mapping_rules", {})

    with open(dict_path, "r", encoding="utf-8") as f:
        dict_data = json.load(f)

    skills = dict_data.get("skills", [])
    unmapped_skills: list[str] = []

    # 第一轮：规则映射
    for skill in skills:
        skill_type = skill.get("skill_type", "")
        category = map_skill_type(skill_type, rules=mapping_rules)
        if category is not None:
            skill["category"] = category
        else:
            unmapped_skills.append(skill["name"])

    # 第二轮：启发式分类（规则未命中的技能）
    for skill in skills:
        if skill.get("category") is not None:
            continue
        name = skill["name"]
        skill_type = skill.get("skill_type", "")
        heuristic_cat = classify_by_heuristic(name, skill_type)
        if heuristic_cat is not None:
            skill["category"] = heuristic_cat
            # 从未映射列表中移除
            if name in unmapped_skills:
                unmapped_skills.remove(name)

    # 第三轮：LLM 分类（仅对启发式仍未命中的技能，且 skip_llm=False）
    remaining_for_llm = [
        s["name"] for s in skills
        if s.get("category") is None and not skip_llm
    ]
    if remaining_for_llm:
        llm_results = classify_batch_by_llm(
            remaining_for_llm,
            rules_data=rules_data,
            llm_client=llm_client,
        )
        for skill in skills:
            if skill.get("category") is None and skill["name"] in llm_results:
                skill["category"] = llm_results[skill["name"]]

    # 兜底：仍然未映射的技能设为 None
    for skill in skills:
        if skill.get("category") is None:
            skill["category"] = None

    # 统计结果
    category_counts: dict[str | None, int] = {}
    for skill in skills:
        cat = skill.get("category")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("类别分布: %s", category_counts)

    # 写入输出文件
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, ensure_ascii=False, indent=2)
        logger.info("已写入分类结果到: %s", output_path)

    return dict_data


def get_category_definitions(
    rules_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """获取 8 个标准类别的定义信息。

    参数:
        rules_data: 完整规则数据，为空时自动加载。

    返回:
        dict: 类别英文标识 -> {name_zh, description, examples} 的映射。
    """
    if rules_data is None:
        rules_data = load_category_rules()
    return rules_data.get("categories", {})
