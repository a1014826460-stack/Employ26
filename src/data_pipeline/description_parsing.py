"""岗位描述结构化切分工具。"""

import argparse
import re
import json
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sqlalchemy import bindparam, text

from src.data_pipeline.description_schema import (
    ALIAS_TO_STD,
    DEFAULT_PARSED_TABLE,
    DESCRIPTION_SECTIONS_JSON_COL,
    DUTIES_TEXT_COL,
    JOB_DESCRIPTION_CLEAN_COL,
    OPTIONAL_NOTE_RE,
    PARSER_VERSION,
    RAG_QUERY_SOURCE_COL,
    RAG_QUERY_TEXT_COL,
    REQUIREMENTS_TEXT_COL,
    SECTIONS_BRIEF_COL,
    UNCLASSIFIED_TEXT_COL,
)
from src.data_pipeline.text_cleaning import normalize_text, remove_noise, sanitize_item
from src.db.job_description_parsed import (
    build_parsed_pg_rows,
    quote_table_name,
    split_table_name,
    write_parsed_rows_to_postgres,
)
from src.db.postgres import create_pg_engine, table_exists


ALIAS_PATTERNS = [
    (re.compile(r'[\s\?？·•●▪◆★※~_]*'.join(map(re.escape, alias)), re.IGNORECASE), alias, std)
    for alias, std in ALIAS_TO_STD
]

ITEM_TOKEN = r'(?:\d+\s*[、．,，)）]|\d+\.(?!\d)|[（(]\d+[)）]|[一二三四五六七八九十]+\s*[、.．]|[（(][一二三四五六七八九十]+[)）]|[-•●▪◆★])'
ITEM_START_RE = re.compile(rf'(?:(?<=^)|(?<=[\n;；]))\s*(?=(?:"|“)?{ITEM_TOKEN})')
ITEM_LEAD_RE = re.compile(rf'^\s*(?:"|“)?{ITEM_TOKEN}\s*')
PREFIX_ENUM_RE = re.compile(r'^\s*(?:[（(]?[一二三四五六七八九十0-9]+[)）]?[、.．]?\s*)')
NOISE_ONLY_RE = re.compile(r'^\s*(?:[（(]?[一二三四五六七八九十0-9]+[)）]?[、.．。]?)\s*$')
BENEFIT_ITEM_RE = re.compile(
    r"薪资|薪酬|工资|底薪|提成|奖金|绩效|补贴|津贴|福利|五险|社保|公积金|包吃|包住|住宿|餐补|房补|"
    r"年假|双休|大小周|节假日|月休|做六休一|法定|体检|团建|下午茶|日结|预支|报销车费"
)
OTHER_ITEM_RE = re.compile(
    r"工作地点|上班地点|工作地址|办公地点|地址|职能类别|关键字|关键词|联系方式|联系人|公司简介|"
    r"公司介绍|备注|友情提醒|重要提示|招聘人数|公司服务|岗位优势|原标题|温馨提示|投递简历|详情欢迎咨询"
)
DUTY_ITEM_RE = re.compile(
    r"负责|参与|开展|执行|制定|完成|跟进|处理|管理|维护|开发|设计|研究|协助|推进|撰写|拓展|"
    r"走访|下达|调试|改善|培训|接待|销售|测试|支持|收集|分析|组织|协调|配送|分拣|打包|"
    r"扫描|贴标签|取餐|送餐|跑腿|煮饭|辅导|编程|调机|生产|加工|组装|操作|主要跑|日常负责"
    r"|短途配送|配送"
)
REQUIREMENT_ITEM_RE = re.compile(
    r"学历|专业|经验|优先|熟悉|熟练|掌握|能力|责任心|沟通|任职|要求|资格|抗压|本科|大专|硕士|"
    r"博士|证书|职称|年龄|CET|GPA|身体健康|可接受|无犯罪|纹身|吃苦耐劳|会使用|服从|不限"
    r"|无需经验|无经验|接受兼职|接受短期|短期|长期"
)
EMBEDDED_HEADING_RE = re.compile(
    r"(岗位职责|工作职责|职责描述|主要职责|任职要求|岗位要求|任职资格|职位要求|招聘要求|"
    r"应聘要求|员工要求|司机要求|基本要求|必须条件|优先条件|职业要求|相关要求|招聘需求|年龄要求|採用要求|采用要求|要求|我们要找这样的人才|"
    r"工作内容|职位描述|职位简介|工作薪资|薪资待遇|"
    r"薪资福利|福利待遇|其他福利|工作岗位|工作要求|技能要求|岗位条件|工作地点|工作地址|"
    r"上班时间|工作时间|温馨提示|公司服务|岗位优势|优势|原标题|薪酬区间|薪资架构|工作魅力|我们能给到您什么|"
    r"我们能为您提供|我们能提供|我们需要你做什么|我们希望你是|我们希望你是什么样的人|"
    r"希望你和我们一起|你未来会掌握的)\s*[:：]"
)
SUFFIX_HEADING_RE = re.compile(
    r"(?P<label>[\u4e00-\u9fffA-Za-z0-9 /、_-]{0,16}?"
    r"(?:岗位描述|岗位职责|工作职责|工作内容|工作要求|岗位要求|职位要求|任职要求|任职资格|职业要求|招聘需求|年龄要求|採用要求|采用要求|必须条件|优先条件))"
    r"\s*[:：]?\s*(?P<rest>.*)$"
)
LEADING_HEADING_RE = re.compile(
    r"^(?:[一二三四五六七八九十0-9]+[、.．]?\s*)?"
    r"(?P<label>工作内容|岗位职责|工作职责|任职要求|岗位要求|职位要求|招聘要求|职业要求|相关要求|年龄要求|薪资待遇|薪酬区间|福利待遇|工作时间)"
    r"\s*(?P<rest>(?:\d+[、.．].*|[：:].*|$))"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。；;])\s*")


def standardize_title(raw_title: str) -> str:
    """将原始标题别名规范到岗位职责、任职要求、福利待遇或其他信息。"""
    cleaned = sanitize_item(raw_title)
    cleaned = PREFIX_ENUM_RE.sub("", cleaned)
    cleaned = re.sub(OPTIONAL_NOTE_RE + r"$", "", cleaned).strip()
    compact = re.sub(r'[\s:：\[\]()"“”]+', '', cleaned)
    compact_lower = compact.lower()
    if compact in {"职责", "工作职责"}:
        return "岗位职责"
    if compact in {"要求", "任职要求", "任职资格"}:
        return "任职要求"
    if compact in {"作职责", "工作总责", "总责"}:
        return "岗位职责"
    if compact in {"具体要求", "专业要求", "工作技能", "技能"}:
        return "任职要求"
    if compact_lower in {
        "responsibilities", "responsibility", "mainresponsibilities", "jobresponsibilities",
        "keyresponsibilities", "rolesandresponsibilities", "roleandresponsibility",
        "primaryresponsibilities", "primaryandsecondaryresponsibilities", "jobdescription",
        "positionobjective", "rolepurpose", "whatyouwilldo", "jobprofile"
    }:
        return "岗位职责"
    if compact_lower in {
        "requirements", "jobrequirements", "qualification", "qualifications", "candidateprofile"
    }:
        return "任职要求"
    if compact_lower in {"benefits", "companyprofile", "aboutthecompany"}:
        return "其他信息" if "company" in compact_lower else "福利待遇"
    if compact_lower in {"whoweare", "brandintroduction", "positiontitle", "jobtitle"}:
        return "其他信息"
    for patt, alias, std in ALIAS_PATTERNS:
        if patt.fullmatch(cleaned) or patt.fullmatch(compact):
            return std
    if ("任职" in compact or "资格" in compact or "岗位" in compact or "职位" in compact) and ("要求" in compact or "条件" in compact):
        return "任职要求"
    if compact.endswith(("任职要求", "任职资格", "岗位要求", "职位要求", "工作要求")):
        return "任职要求"
    if compact.endswith(("要求", "条件", "资格")) and len(compact) <= 10:
        return "任职要求"
    if compact.endswith(("岗位职责", "工作职责", "工作内容", "岗位描述", "职位描述")):
        return "岗位职责"
    if ("职责" in compact or "内容" in compact or "描述" in compact or "范围" in compact or "总责" in compact) and (
        "岗位" in compact or "工作" in compact or "职位" in compact or compact == "职责" or len(compact) <= 8
    ):
        return "岗位职责"
    if "福利" in compact or "待遇" in compact or "薪酬" in compact:
        return "福利待遇"
    if any(k in compact for k in [
        "地点", "地址", "时间", "联系", "方式", "简介", "应聘", "社保", "人数", "日期", "方向", "空间", "路线",
        "信息", "提示", "介绍", "校招岗位", "招聘岗位"
    ]):
        return "其他信息"
    return cleaned


def match_heading(line: str):
    """识别单行是否为 section 标题，并返回标题与同行正文。"""
    original = line.strip()
    if not original:
        return None
    s = remove_noise(original)
    s = re.sub(r'^[?？·•●▪◆★※"\']+\s*', '', s)
    s2 = PREFIX_ENUM_RE.sub("", s)
    s2 = s2.strip()
    if re.fullmatch(r'职责\s*:?', s2):
        return "职责", ""
    if re.fullmatch(r'要求\s*:?', s2):
        return "任职要求", ""
    leading = LEADING_HEADING_RE.match(s2)
    if leading:
        label = leading.group("label").strip()
        rest = leading.group("rest").strip()
        rest = re.sub(r"^[:：]\s*", "", rest)
        return label, rest
    for patt, alias, std in ALIAS_PATTERNS:
        if std in {"其他信息", "福利待遇"}:
            regex = rf'^(?:\[)?{patt.pattern}(?:\])?{OPTIONAL_NOTE_RE}(?=\s*(?::|$))\s*(?::)?\s*(?P<rest>.*)$'
        else:
            regex = rf'^(?:\[)?{patt.pattern}(?:\])?{OPTIONAL_NOTE_RE}\s*(?::)?\s*(?P<rest>.*)$'
        m = re.match(regex, s2, re.IGNORECASE)
        if m:
            return alias, m.group("rest").strip()
    m = re.match(r'^[\[\(【]?\s*([A-Za-z\u4e00-\u9fff /&+、\-]{1,32})\s*[\]\)】]?\s*:\s*(.*)$', s2)
    if m:
        label, rest = m.group(1).strip(), m.group(2).strip()
        title_std = standardize_title(label)
        if title_std != label:
            return label, rest
    m = SUFFIX_HEADING_RE.match(s2)
    if m:
        label = m.group("label").strip()
        title_std = standardize_title(label)
        if title_std in {"岗位职责", "任职要求"}:
            return label, m.group("rest").strip()
    return None


def split_numbered(text: str) -> List[str]:
    """按数字、中文序号或项目符号切分列表文本。"""
    starts = [m.start() for m in ITEM_START_RE.finditer(text)]
    if re.match(rf'^\s*(?:"|“)?{ITEM_TOKEN}', text):
        if not starts or starts[0] != 0:
            starts = [0] + starts
    starts = sorted(set(starts))
    if len(starts) >= 2:
        bounds = starts + [len(text)]
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = text[a:b].strip()
            seg = ITEM_LEAD_RE.sub("", seg).strip()
            seg = re.sub(r"\s*\n\s*", " ", seg)
            seg = sanitize_item(seg)
            if seg:
                parts.append(seg)
        return parts
    if len(starts) == 1 and starts[0] == 0:
        seg = sanitize_item(ITEM_LEAD_RE.sub("", text).strip())
        return [seg] if seg else []
    return []


def split_cn_subheads(text: str):
    """切分“一、标题 二、标题”这类中文子标题结构。"""
    pat = re.compile(r'([一二三四五六七八九十]+)\s*[、.]\s*([^\n:：；;。]{1,20})(?::)?')
    ms = list(pat.finditer(text))
    if len(ms) < 2 or ms[0].start() != 0:
        return None
    labels = [m.group(2).strip() for m in ms]
    if sum(len(lb) <= 12 and not re.search(r'[，,；;。]', lb) for lb in labels) < 2:
        return None
    bounds = [m.start() for m in ms] + [len(text)]
    items = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = text[a:b].strip()
        m = pat.match(seg)
        if not m:
            continue
        label = sanitize_item(m.group(2))
        rest = seg[m.end():].strip()
        nums = split_numbered(rest)
        item = sanitize_item(label + (": " + "；".join(nums) if nums else (": " + rest if rest else "")))
        if item:
            items.append(item)
    return items if len(items) >= 2 else None


def split_items(text: str) -> List[str]:
    """将 section 正文切分为条目，按编号、换行、分号逐级兜底。"""
    text = text.strip()
    if not text:
        return []
    cn = split_cn_subheads(text)
    if cn:
        return cn
    nums = split_numbered(text)
    if nums and len(nums) >= 2:
        return nums
    lines = [sanitize_item(x) for x in re.split(r"\n+", text) if sanitize_item(x)]
    if len(lines) >= 2:
        return lines
    parts = [sanitize_item(x) for x in re.split(r"[；;]", text) if sanitize_item(x)]
    if len(parts) >= 2:
        return parts
    one = sanitize_item(text)
    return [one] if one else []


def likely_duty_text(lines: List[str]) -> bool:
    """用关键词粗判无标题文本更像职责还是要求。"""
    t = " ".join(lines)
    duty_score = len(re.findall(r"负责|参与|开展|执行|制定|完成|跟进|处理|管理|维护|开发|设计|研究|协助|推进|撰写|拓展|走访|下达|调试|改善|培训", t))
    req_score = len(re.findall(r"学历|专业|经验|优先|熟练|能力|责任心|沟通|任职|要求|资格|抗压|本科|大专|硕士|博士|以上学历", t))
    return duty_score >= req_score


def infer_section_title(lines: List[str], fallback: Optional[str] = None) -> Optional[str]:
    """根据关键词得分推断无标题文本所属 section。"""
    text = " ".join(sanitize_item(x) for x in lines if sanitize_item(x))
    if not text:
        return fallback

    scores = {
        "岗位职责": len(re.findall(r"负责|参与|开展|执行|制定|完成|跟进|处理|管理|维护|开发|设计|研究|协助|推进|撰写|拓展|走访|下达|调试|改善|培训|巡察|接待|销售|测试|支持", text)),
        "任职要求": len(re.findall(r"学历|专业|经验|优先|熟练|能力|责任心|沟通|任职|要求|资格|抗压|本科|大专|硕士|博士|以上学历|年龄|CET|GPA|证书|职称|相关专业|工作经验|身体健康|可接受|office|英语", text, re.IGNORECASE)),
        "福利待遇": len(re.findall(r"福利|待遇|薪资|薪酬|补贴|奖金|住宿|餐补|房补|五险一金|社保|公积金|年假|休假|节日|体检|提成|补助", text)),
        "其他信息": len(re.findall(r"公司简介|公司介绍|项目背景|工作地点|上班地点|地址|联系|联系方式|简历投递|校招岗位|招聘岗位|职位信息|基本信息|重要提示|上市|成立于|股票代码|欢迎投递|工作地址|办公地点", text)),
    }

    if "公司主要从事" in text or "以下简称" in text or "欢迎投递" in text:
        scores["其他信息"] += 3
    if "岗位类别" in text or "招聘岗位" in text:
        scores["其他信息"] += 2
    if "工作技能" in text or "专业要求" in text or "具体要求" in text:
        scores["任职要求"] += 2
    if "工作范围" in text or "工作总责" in text:
        scores["岗位职责"] += 2
    if DUTY_ITEM_RE.search(text) and REQUIREMENT_ITEM_RE.search(text) and scores["福利待遇"] >= max(scores["岗位职责"], scores["任职要求"]):
        scores["福利待遇"] = max(0, scores["福利待遇"] - 2)

    best_title, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return fallback
    return best_title


def classify_item(item: str, current_title: str) -> str:
    """按条目关键词对 section 内混杂内容做轻量分流。"""
    text = sanitize_item(item)
    if not text:
        return current_title
    duty_score = len(DUTY_ITEM_RE.findall(text))
    req_score = len(REQUIREMENT_ITEM_RE.findall(text))
    if OTHER_ITEM_RE.search(text) and duty_score == 0 and req_score == 0:
        return "其他信息"
    if re.search(r"服从管理|男女不限|经验不限|学历不限|年龄\d", text):
        req_score += 2
    if re.search(r"无需经验|无经验|接受兼职|接受短期|短期过渡|年龄[:：]?\d|男女不限", text):
        req_score += 2
    if re.search(r"^(?:具有|具备|熟悉|熟练|掌握|精通|能够|能独立|有较强|较强的|良好的|优秀的)", text):
        req_score += 2
    if re.search(r"具有[^。；;]{0,24}(能力|经验|背景|意识)|具备[^。；;]{0,24}(能力|经验|资质|证书)", text):
        req_score += 2
    if re.search(r"负责[^。；;]{0,30}(配送|跑腿|维护|推广|销售|接待)|主要跑|主要负责|日常负责|同城跑腿|小件物品配送", text):
        duty_score += 3
    if current_title == "其他信息" and duty_score > req_score and duty_score > 0:
        return "岗位职责"
    if BENEFIT_ITEM_RE.search(text) and req_score == 0 and duty_score == 0:
        return "福利待遇"
    if current_title == "岗位职责" and req_score >= 2 and req_score > duty_score:
        return "任职要求"
    if current_title == "任职要求" and duty_score >= 2 and duty_score > req_score:
        return "岗位职责"
    return current_title


def split_mixed_item_by_sentence(item: str, current_title: str) -> List[Tuple[str, str]]:
    """把一个混有职责、要求、福利的长条目按短句重新分流。"""
    text = sanitize_item(item)
    if not text:
        return []
    if len(text) < 30:
        return [(classify_item(text, current_title), text)]
    pieces = [sanitize_item(x) for x in SENTENCE_SPLIT_RE.split(text) if sanitize_item(x)]
    if len(pieces) <= 1 and DUTY_ITEM_RE.search(text) and REQUIREMENT_ITEM_RE.search(text):
        pieces = [sanitize_item(x) for x in re.split(r"[，,]", text) if sanitize_item(x)]
    if len(pieces) <= 1:
        return [(classify_item(text, current_title), text)]

    output: List[Tuple[str, str]] = []
    for piece in pieces:
        target_title = classify_item(piece, current_title)
        if output and output[-1][0] == target_title:
            output[-1] = (target_title, sanitize_item(output[-1][1] + " " + piece))
        else:
            output.append((target_title, piece))
    return output


def split_mixed_items(items: List[str], current_title: str) -> List[Tuple[str, str]]:
    """对 section 中的所有条目执行职责/要求/福利轻量重分流。"""
    output: List[Tuple[str, str]] = []
    for item in items:
        output.extend(split_mixed_item_by_sentence(item, current_title))
    return output


def split_embedded_heading_item(item: str, current_title: str) -> List[Tuple[str, str]]:
    """把单个条目中残留的内嵌标题拆成多个 `(标题, 正文)` 片段。"""
    text = sanitize_item(item)
    matches = list(EMBEDDED_HEADING_RE.finditer(text))
    if not matches:
        return split_mixed_item_by_sentence(text, current_title)

    parts: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        prefix = sanitize_item(text[:matches[0].start()])
        if prefix:
            parts.extend(split_mixed_item_by_sentence(prefix, current_title))

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(1)
        body = sanitize_item(text[match.end():next_start])
        if body:
            parts.extend(split_mixed_item_by_sentence(body, standardize_title(heading)))
    return parts


def normalize_final_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """清理最终 sections，把福利、地点和明显错入的条目重新归类。"""
    grouped: List[Dict[str, Any]] = []
    for sec in sections:
        base_title = sec.get("title_std", "")
        for item in sec.get("items", []):
            for embedded_title, embedded_item in split_embedded_heading_item(item, base_title):
                clean_item = sanitize_item(embedded_item)
                if not clean_item:
                    continue
                target_title = classify_item(clean_item, embedded_title)
                if grouped and grouped[-1]["title_std"] == target_title:
                    grouped[-1]["items"].append(clean_item)
                    continue
                grouped.append(
                    {
                        "title_raw": target_title if target_title != base_title else sec.get("title_raw", target_title),
                        "title_std": target_title,
                        "title_inferred": sec.get("title_inferred", False) or target_title != base_title,
                        "items": [clean_item],
                    }
                )
    return grouped


def parse_job_description(text: str) -> Dict[str, Any]:
    """解析单条岗位描述，返回原文、sections 和无法归类文本。"""
    raw = "" if text is None else str(text)
    text = normalize_text(raw)
    if not text:
        return {"岗位描述_raw": raw, "sections": [], "unclassified": []}
    lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
    sections = []
    current = None
    preamble = []
    explicit_seen = False

    def start_section(title_raw: str, inferred: bool = False):
        nonlocal current
        current = {
            "title_raw": sanitize_item(PREFIX_ENUM_RE.sub("", title_raw)),
            "title_std": standardize_title(title_raw),
            "title_inferred": inferred,
            "buffer": [],
        }
        sections.append(current)

    for line in lines:
        mh = match_heading(line)
        if mh:
            title_raw, rest = mh
            explicit_seen = True
            start_section(title_raw, inferred=False)
            if rest:
                current["buffer"].append(rest)
            continue
        if explicit_seen:
            if current is None:
                start_section("岗位职责", inferred=True)
            current["buffer"].append(line)
        else:
            preamble.append(line)

    # ── 前置文本（preamble）清洗 ──────────────────────────────────────────────
    # preamble 收集了"第一个明确标题出现之前"的所有文本行。
    # 先过滤掉纯噪声行（仅含标点/空白/零宽字符），避免干扰后续推断。
    preamble = [ln for ln in preamble if not NOISE_ONLY_RE.fullmatch(sanitize_item(ln) or "")]

    # ── unclassified 初始化 ───────────────────────────────────────────────────
    # unclassified 最终存放"无法归入任何 section 的文本片段"，供人工复查。
    # 以下分两条路径处理 preamble：
    unclassified = []

    if explicit_seen:
        # ── 路径A：文本中存在至少一个明确标题 ──────────────────────────────
        # 此时 preamble 是标题出现"之前"的内容，通常是公司介绍、岗位背景等。
        if preamble:
            inferred_title = None

            # 特殊情形：第一个明确 section 已经是"任职要求"，
            # 但 preamble 语义上更像职责描述（动作词多于要求词）
            # → 补推断一个"岗位职责" section，避免职责内容丢失。
            if sections and sections[0]["title_std"] == "任职要求" and likely_duty_text(preamble):
                inferred_title = "岗位职责"
            else:
                # 通用情形：用关键词打分推断 preamble 最可能属于哪个 section
                # （岗位职责 / 任职要求 / 福利待遇 / 其他信息）。
                # 若各类别得分均为 0，infer_section_title 返回 None。
                inferred_title = infer_section_title(preamble)

            if inferred_title:
                # 能推断出类别 → 在 sections 头部插入一个"推断 section"
                # title_inferred=True 标记此 section 无显式标题行
                sections.insert(0, {
                    "title_raw": inferred_title,
                    "title_std": inferred_title,
                    "title_inferred": True,
                    "buffer": preamble[:],
                })
            else:
                # 无法推断（得分全为 0，内容模糊）→ 归入 unclassified
                # 典型情况：纯公司宣传语、格式乱码、仅含岗位名称等无意义前言
                unclassified.extend(preamble)
    else:
        # ── 路径B：全文没有任何明确标题行 ───────────────────────────────────
        # 整段 preamble 就是全部内容，强制推断一个 section 标题。
        # fallback 保证即使关键词全无命中，也能兜底为"岗位职责"或"任职要求"。
        # 此路径下不会产生 unclassified 内容。
        inferred_title = infer_section_title(
            preamble,
            fallback="岗位职责" if likely_duty_text(preamble) else "任职要求"
        )
        sections = [{
            "title_raw": inferred_title,
            "title_std": inferred_title,
            "title_inferred": True,
            "buffer": preamble[:],
        }]

    merged = []
    for sec in sections:
        if merged and sec["title_raw"] == merged[-1]["title_raw"] and sec["title_inferred"] == merged[-1]["title_inferred"]:
            merged[-1]["buffer"].extend(sec["buffer"])
        elif merged and sec["title_std"] == merged[-1]["title_std"] and sec["title_std"] in {"岗位职责", "任职要求", "福利待遇"} and (sec["title_inferred"] or merged[-1]["title_inferred"]):
            merged[-1]["buffer"].extend(sec["buffer"])
        else:
            merged.append(sec)
    sections = merged

    final_sections = []
    for sec in sections:
        body = "\n".join(sec.pop("buffer")).strip()
        if not body:
            continue
        items = split_items(body)
        if items:
            final_sections.append({**sec, "items": items})

    for idx, sec in enumerate(final_sections[:-1]):
        if sec["title_std"] == "任职要求" and likely_duty_text(sec["items"]) and final_sections[idx + 1]["title_std"] == "任职要求":
            sec["title_raw"] = "岗位职责"
            sec["title_std"] = "岗位职责"
            sec["title_inferred"] = True
    final_sections = normalize_final_sections(final_sections)

    cleaned_unclassified = []
    seen = set()
    for x in unclassified:
        sx = sanitize_item(x)
        if NOISE_ONLY_RE.fullmatch(sx or ""):
            continue
        if sx and sx not in seen:
            cleaned_unclassified.append(sx)
            seen.add(sx)

    return {"岗位描述_raw": raw, "sections": final_sections, "unclassified": cleaned_unclassified}


def _parse_job_description_worker(text: str) -> Dict[str, Any]:
    """供多进程批处理复用的顶层 worker。"""
    return parse_job_description(text)


def _join_section_items(parsed_obj: Dict[str, Any], title_std: str) -> str:
    """汇总指定 section 的 items，保留原有顺序。"""
    items: List[str] = []
    for sec in parsed_obj.get("sections", []):
        if sec.get("title_std") != title_std:
            continue
        for item in sec.get("items", []):
            clean_item = sanitize_item(item)
            if clean_item:
                items.append(clean_item)
    return " | ".join(items)


def _build_sections_brief(parsed_obj: Dict[str, Any]) -> str:
    """构建紧凑的 section 摘要，便于人工排查。"""
    brief_parts = []
    for sec in parsed_obj.get("sections", []):
        title_std = sec.get("title_std", "")
        items = [sanitize_item(x) for x in sec.get("items", []) if sanitize_item(x)]
        if not items:
            continue
        brief_parts.append(f"{title_std}:{' / '.join(items[:2])}")
    return " | ".join(brief_parts)


def _select_default_rag_query(
    requirement_text: str,
    duty_text: str,
    cleaned_desc: str,
) -> Tuple[str, str]:
    """为预处理阶段提供确定性的默认匹配文本。"""
    if requirement_text:
        return requirement_text, "任职要求"
    if duty_text:
        return duty_text, "岗位职责"
    if cleaned_desc:
        return cleaned_desc, "岗位描述_清洗"
    return "", "空文本"


def _parse_batch_texts(
    texts: List[str],
    num_workers: int,
    executor: Optional[ProcessPoolExecutor] = None,
) -> List[Dict[str, Any]]:
    """对一个批次文本执行解析，单进程和多进程共用。"""
    if not texts:
        return []

    if num_workers <= 1:
        return [parse_job_description(text) for text in texts]

    chunksize = max(1, len(texts) // max(1, num_workers * 4))
    if executor is None:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as local_executor:
            return list(local_executor.map(_parse_job_description_worker, texts, chunksize=chunksize))
    return list(executor.map(_parse_job_description_worker, texts, chunksize=chunksize))


def parse_desc_df(
    df: pd.DataFrame,
    desc_col: str = "岗位描述",
    batch_size: int = 2000,
    num_workers: int = 1,
) -> pd.DataFrame:
    """批量切分岗位描述，返回附带解析结果列的新 DataFrame。

    新增列：
    - 岗位描述_清洗
    - 岗位描述_切分JSON
    - 任职要求_items_text
    - 岗位职责_items_text
    - unclassified_text
    - sections_brief
    - RAG匹配文本
    - RAG匹配来源
    """
    if desc_col not in df.columns:
        raise KeyError(f"DataFrame 缺少描述列: {desc_col}")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if num_workers <= 0:
        raise ValueError("num_workers 必须大于 0")

    output_df = df.copy()
    desc_texts = output_df[desc_col].fillna("").astype(str).tolist()
    parsed_results: List[Dict[str, Any]] = []

    executor: Optional[ProcessPoolExecutor] = None
    active_workers = num_workers
    main_file = getattr(sys.modules.get("__main__"), "__file__", "")
    if main_file.endswith("<stdin>"):
        active_workers = 1
    try:
        if active_workers > 1:
            ctx = mp.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=active_workers, mp_context=ctx)

        for start in range(0, len(desc_texts), batch_size):
            batch_texts = desc_texts[start: start + batch_size]
            try:
                parsed_results.extend(
                    _parse_batch_texts(batch_texts, num_workers=active_workers, executor=executor)
                )
            except (BrokenProcessPool, OSError, RuntimeError):
                if active_workers <= 1:
                    raise
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None
                active_workers = 1
                parsed_results.extend(_parse_batch_texts(batch_texts, num_workers=1))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    cleaned_descs: List[str] = []
    parsed_jsons: List[str] = []
    requirement_texts: List[str] = []
    duty_texts: List[str] = []
    unclassified_texts: List[str] = []
    sections_briefs: List[str] = []
    rag_query_texts: List[str] = []
    rag_query_sources: List[str] = []

    for raw_text, parsed_obj in zip(desc_texts, parsed_results):
        cleaned_desc = normalize_text(raw_text)
        requirement_text = _join_section_items(parsed_obj, "任职要求")
        duty_text = _join_section_items(parsed_obj, "岗位职责")
        unclassified_text = " | ".join(parsed_obj.get("unclassified", []))
        sections_brief = _build_sections_brief(parsed_obj)
        rag_query_text, rag_query_source = _select_default_rag_query(
            requirement_text=requirement_text,
            duty_text=duty_text,
            cleaned_desc=cleaned_desc,
        )

        cleaned_descs.append(cleaned_desc)
        parsed_jsons.append(json.dumps(parsed_obj, ensure_ascii=False))
        requirement_texts.append(requirement_text)
        duty_texts.append(duty_text)
        unclassified_texts.append(unclassified_text)
        sections_briefs.append(sections_brief)
        rag_query_texts.append(rag_query_text)
        rag_query_sources.append(rag_query_source)

    output_df[JOB_DESCRIPTION_CLEAN_COL] = cleaned_descs
    output_df[DESCRIPTION_SECTIONS_JSON_COL] = parsed_jsons
    output_df[REQUIREMENTS_TEXT_COL] = requirement_texts
    output_df[DUTIES_TEXT_COL] = duty_texts
    output_df[UNCLASSIFIED_TEXT_COL] = unclassified_texts
    output_df[SECTIONS_BRIEF_COL] = sections_briefs
    output_df[RAG_QUERY_TEXT_COL] = rag_query_texts
    output_df[RAG_QUERY_SOURCE_COL] = rag_query_sources
    return output_df


def parse_and_write_desc_df_to_postgres(
    df: pd.DataFrame,
    source_table: str,
    source_platform: str | None = None,
    table_name: str = DEFAULT_PARSED_TABLE,
    desc_col: str = "岗位描述",
    title_col: str = "岗位名称",
    batch_size: int = 2000,
    num_workers: int = 1,
    parser_version: str = PARSER_VERSION,
) -> int:
    """解析 DataFrame 中的岗位描述，并将结果写入 PostgreSQL。"""
    parsed_df = parse_desc_df(
        df,
        desc_col=desc_col,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    rows = build_parsed_pg_rows(
        parsed_df=parsed_df,
        source_table=source_table,
        source_platform=source_platform,
        parser_version=parser_version,
        title_col=title_col,
        desc_col=desc_col,
    )
    return write_parsed_rows_to_postgres(rows, table_name=table_name)


def read_source_table_from_postgres(
    source_table: str,
    limit_rows: int | None = None,
    offset_rows: int = 0,
    only_risk_rows: bool = False,
    only_unparsed: bool = False,
    risk_parser_version: str = "description_parsing_v3",
    target_table: str = DEFAULT_PARSED_TABLE,
    source_table_filter: List[str] | None = None,
) -> pd.DataFrame:
    """从 PostgreSQL 源表读取岗位数据，并补充稳定到本次查询的行号。"""
    schema_name, table_name = split_table_name(source_table)
    qualified_table = quote_table_name(source_table)
    target_qualified_table = quote_table_name(target_table)
    limit_sql = ""
    params: dict[str, int] = {}
    if limit_rows is not None:
        if limit_rows <= 0:
            raise ValueError("limit_rows 必须大于 0")
        limit_sql = "LIMIT :limit_rows"
        params["limit_rows"] = int(limit_rows)
    offset_sql = ""
    if int(offset_rows) > 0:
        offset_sql = "OFFSET :offset_rows"
        params["offset_rows"] = int(offset_rows)
    params["source_table"] = source_table
    params["risk_parser_version"] = risk_parser_version
    normalized_source_table_filter = [
        _normalize_source_table_filter_value(item)
        for item in (source_table_filter or [])
        if str(item).strip()
    ]
    filter_sql = ""
    if normalized_source_table_filter:
        filter_sql = 'WHERE s.source_table IN :source_table_filter'
        params["source_table_filter"] = normalized_source_table_filter

    is_normalized_source = (
        schema_name == "public"
        and table_name == "recruitment_jobs_normalized"
        and not only_risk_rows
    )

    if only_risk_rows:
        risk_filter_sql = (
            f"AND s.source_table IN :source_table_filter" if normalized_source_table_filter else ""
        )
        risk_where = """
            (
                (coalesce(p.requirements_text, '') = '' and coalesce(p.duties_text, '') = '')
                or (
                    coalesce(p.requirements_text, '') ~ '(岗位职责|工作职责|职责描述|任职要求|岗位要求|任职资格|职位要求|应聘要求|员工要求|司机要求|基本要求|工作内容|职位描述|工作薪资|工作岗位|工作要求|福利待遇|其他福利)\\s*[:：]'
                    or coalesce(p.duties_text, '') ~ '(岗位职责|工作职责|职责描述|任职要求|岗位要求|任职资格|职位要求|应聘要求|员工要求|司机要求|基本要求|工作内容|职位描述|工作薪资|工作岗位|工作要求|福利待遇|其他福利)\\s*[:：]'
                )
                or coalesce(p.requirements_text, '') ~ '(薪资|福利|五险|公积金|年假|奖金|补贴|包吃|包住|住宿|工作时间|双休)'
                or (
                    coalesce(p.requirements_text, '') = ''
                    and coalesce(p.job_description_raw, '') ~ '(学历|经验|熟悉|熟练|优先|本科|大专|硕士|博士|任职|要求|资格|证书|专业|年龄|健康)'
                )
                or (
                    coalesce(p.duties_text, '') = ''
                    and coalesce(p.job_description_raw, '') ~ '(负责|参与|开发|维护|执行|管理|完成|协助|跟进|处理|制定|接待|销售|收集|打包|分拣|配送|组装)'
                )
            )
        """
        if risk_parser_version.lower() == "latest":
            query = text(
                f"""
                WITH latest_parsed AS (
                    SELECT *
                    FROM (
                        SELECT
                            p.*,
                            row_number() OVER (
                                PARTITION BY p.source_table, p.source_row_number
                                ORDER BY p.parser_version DESC, p.parsed_at DESC
                            ) AS parsed_rank
                        FROM {target_qualified_table} p
                        WHERE p.source_table = :source_table
                    ) ranked
                    WHERE parsed_rank = 1
                )
                SELECT s.*
                FROM (
                    SELECT
                        row_number() OVER (ORDER BY ctid) AS "__source_row_number",
                        *
                    FROM {qualified_table}
                ) s
                JOIN latest_parsed p
                  ON p.source_table = :source_table
                 AND p.source_row_number = s."__source_row_number"
                WHERE {risk_where}
                {risk_filter_sql}
                ORDER BY s."__source_row_number"
                {limit_sql}
                {offset_sql}
                """
            )
        else:
            query = text(
                f"""
                SELECT s.*
                FROM (
                    SELECT
                        row_number() OVER (ORDER BY ctid) AS "__source_row_number",
                        *
                    FROM {qualified_table}
                ) s
                JOIN {target_qualified_table} p
                  ON p.source_table = :source_table
                 AND p.source_row_number = s."__source_row_number"
                 AND p.parser_version = :risk_parser_version
                WHERE {risk_where}
                {risk_filter_sql}
                ORDER BY s."__source_row_number"
                {limit_sql}
                {offset_sql}
                """
            )
    elif is_normalized_source:
        parsed_join_sql = ""
        parsed_where_sql = ""
        if only_unparsed:
            parsed_join_sql = (
                f"LEFT JOIN {target_qualified_table} p "
                "ON p.recruitment_record_id = s.recruitment_record_id"
            )
            parsed_where_sql = (
                "AND p.recruitment_record_id IS NULL"
                if filter_sql
                else "WHERE p.recruitment_record_id IS NULL"
            )
        query = text(
            f"""
            SELECT
                s.source_row_number AS "__source_row_number",
                s.recruitment_record_id,
                s.source_platform,
                s.source_table,
                s.source_row_number,
                s.job_title,
                s.job_description_raw
            FROM {qualified_table} s
            {parsed_join_sql}
            {filter_sql}
            {parsed_where_sql}
            ORDER BY s.source_table, s.source_row_number
            {limit_sql}
            {offset_sql}
            """
        )
    else:
        query = text(
            f"""
            SELECT *
            FROM (
                SELECT
                    row_number() OVER (ORDER BY ctid) AS "__source_row_number",
                    *
                FROM {qualified_table}
            ) s
            {filter_sql}
            ORDER BY s."__source_row_number"
            {limit_sql}
            {offset_sql}
            """
        )
    if normalized_source_table_filter:
        query = query.bindparams(bindparam("source_table_filter", expanding=True))
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            if not table_exists(connection, schema_name, table_name):
                raise ValueError(f"PostgreSQL 源表不存在: {source_table}")
            return pd.read_sql_query(query, connection, params=params)
    finally:
        engine.dispose()


def _normalize_source_table_filter_value(value: object) -> str:
    """规范化 source_table 过滤值，兼容 PowerShell native 参数吞掉双引号的情况。"""
    text_value = str(value).strip().replace('\\"', '"')
    if text_value in {
        "51job.cleaned_data",
        "51job.sample",
        "51job.raw_data",
    }:
        schema_name, table_name = text_value.split(".", 1)
        return f'"{schema_name}".{table_name}'
    if text_value in {
        "Liepin.cleaned_data",
        "Liepin.sample",
        "Liepin.raw_data",
        "Zhilian.cleaned_data",
        "Zhilian.sample",
        "Zhilian.raw_data",
    }:
        schema_name, table_name = text_value.split(".", 1)
        return f'"{schema_name}".{table_name}'
    return text_value


def build_issue_dataframe(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """输出 unclassified 不为空的问题行，便于人工检查。"""
    if "unclassified_text" not in parsed_df.columns:
        raise KeyError("请先运行 parse_desc_df，再构建问题行数据。")
    issue_mask = parsed_df["unclassified_text"].fillna("").astype(str).str.strip() != ""
    return parsed_df.loc[issue_mask].copy()


def build_hardcase_dataframe(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """识别仍可能存在切分异常的样本。"""
    if "岗位描述_切分JSON" not in parsed_df.columns:
        raise KeyError("请先运行 parse_desc_df，再构建疑难样本数据。")

    suspicious_idx: List[int] = []
    parsed_json_series = parsed_df["岗位描述_切分JSON"].fillna("").astype(str)
    parsed_objects = parsed_json_series.apply(json.loads)

    for idx, obj in enumerate(parsed_objects):
        joined = " ".join(
            [sec["title_raw"] + " " + sec["title_std"] + " " + " ".join(sec["items"]) for sec in obj.get("sections", [])]
            + obj.get("unclassified", [])
        )
        if re.search(
            r'\[[^\]]*(工作职责|岗位职责|任职要求|任职资格|岗位要求|准入要求|工作地点|办公地|上班时间|福利待遇)[^\]]*\]'
            r'|(?:工作职责|岗位职责|任职要求|任职资格|岗位要求|准入要求|工作地点|办公地|上班时间|福利待遇)\s*:',
            joined,
        ):
            suspicious_idx.append(idx)
            continue
        if any(
            (not item or re.fullmatch(r'[\W_]+', item))
            for sec in obj.get("sections", [])
            for item in sec.get("items", [])
        ):
            suspicious_idx.append(idx)

    return parsed_df.iloc[suspicious_idx].copy() if suspicious_idx else parsed_df.iloc[0:0].copy()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="岗位描述结构化切分")
    parser.add_argument(
        "--input-source",
        choices=["csv", "postgres"],
        default="csv",
        help="输入来源类型：csv 或 postgres",
    )
    parser.add_argument("--input-csv", default=None, help="输入 CSV 文件路径")
    parser.add_argument("--output-csv", default=None, help="可选：保存解析结果 CSV")
    parser.add_argument("--desc-col", default="岗位描述", help="岗位描述列名")
    parser.add_argument("--title-col", default="岗位名称", help="岗位名称列名")
    parser.add_argument(
        "--source-table",
        nargs="+",
        default=[],
        help="PostgreSQL 输入源表名，可一次传入多个；写入时也作为来源表名记录",
    )
    parser.add_argument(
        "--source-platform",
        nargs="*",
        default=None,
        help="可选：来源平台名；多表时按 --source-table 顺序对应，不传则自动推断",
    )
    parser.add_argument("--target-table", default=DEFAULT_PARSED_TABLE, help="PostgreSQL 解析结果表名")
    parser.add_argument("--write-postgres", action="store_true", help="将解析结果写入 PostgreSQL")
    parser.add_argument("--limit-rows", type=int, default=None, help="仅用于调试，限制 PostgreSQL 源表读取行数")
    parser.add_argument(
        "--source-table-filter",
        nargs="*",
        default=None,
        help="可选：按输入表中的 source_table 字段过滤；常用于 public.recruitment_jobs_normalized",
    )
    parser.add_argument("--only-risk-rows", action="store_true", help="仅重跑上一版本解析结果中的高风险行")
    parser.add_argument(
        "--only-unparsed",
        action="store_true",
        help="仅解析目标表中尚不存在 recruitment_record_id 的记录；适合全量任务断点续跑",
    )
    parser.add_argument(
        "--risk-parser-version",
        default="latest",
        help="--only-risk-rows 使用的上一版 parser_version；传 latest 表示按每条源记录当前最新解析结果筛选",
    )
    parser.add_argument("--parse-workers", type=int, default=32, help="岗位描述切分并发数")
    parser.add_argument("--parse-batch-size", type=int, default=20000, help="岗位描述切分批大小")
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=0,
        help="PostgreSQL 分批读取行数；0 表示一次性读取，推荐全量任务设置为较大正整数",
    )
    return parser


def resolve_source_platforms(
    source_tables: List[str],
    source_platforms: Optional[List[str]],
) -> List[Optional[str]]:
    """校验并展开多源表对应的平台名。"""
    if not source_platforms:
        return [None] * len(source_tables)
    if len(source_platforms) != len(source_tables):
        raise ValueError("--source-platform 数量必须为 0 或与 --source-table 数量一致")
    return source_platforms


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    source_tables: List[str] = list(args.source_table or [])
    source_platforms = resolve_source_platforms(source_tables, args.source_platform)

    if args.input_source == "csv":
        if not args.input_csv:
            raise ValueError("--input-source csv 需要提供 --input-csv")
        if args.write_postgres and len(source_tables) != 1:
            raise ValueError("--input-source csv 写入 PostgreSQL 时需要且只能提供一个 --source-table")
        df = pd.read_csv(args.input_csv, encoding="utf-8")
        parsed_df = parse_desc_df(
            df,
            desc_col=args.desc_col,
            batch_size=max(1, int(args.parse_batch_size)),
            num_workers=max(1, int(args.parse_workers)),
        )
        if args.output_csv:
            parsed_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

        if args.write_postgres:
            rows = build_parsed_pg_rows(
                parsed_df=parsed_df,
                source_table=source_tables[0],
                source_platform=source_platforms[0],
                title_col=args.title_col,
                desc_col=args.desc_col,
            )
            written_count = write_parsed_rows_to_postgres(rows, table_name=args.target_table)
            print(f"{source_tables[0]} written rows: {written_count}")
        return

    if not source_tables:
        raise ValueError("--input-source postgres 需要提供 --source-table")
    output_frames: List[pd.DataFrame] = []
    for source_table, source_platform in zip(source_tables, source_platforms):
        read_batch_size = max(0, int(args.read_batch_size))
        total_written_count = 0
        total_read_count = 0
        while True:
            current_limit = args.limit_rows
            if read_batch_size > 0:
                remaining = None if args.limit_rows is None else int(args.limit_rows) - total_read_count
                if remaining is not None and remaining <= 0:
                    break
                current_limit = min(read_batch_size, remaining) if remaining is not None else read_batch_size
            df = read_source_table_from_postgres(
                source_table=source_table,
                limit_rows=current_limit,
                only_risk_rows=bool(args.only_risk_rows),
                only_unparsed=bool(args.only_unparsed),
                risk_parser_version=args.risk_parser_version,
                target_table=args.target_table,
                source_table_filter=args.source_table_filter,
                offset_rows=0 if bool(args.only_unparsed) else (total_read_count if read_batch_size > 0 else 0),
            )
            if df.empty:
                break
            total_read_count += len(df)
            parsed_df = parse_desc_df(
                df,
                desc_col=args.desc_col,
                batch_size=max(1, int(args.parse_batch_size)),
                num_workers=max(1, int(args.parse_workers)),
            )
            if args.output_csv:
                parsed_df = parsed_df.copy()
                parsed_df["__parsed_source_table"] = source_table
                output_frames.append(parsed_df)

            if args.write_postgres:
                rows = build_parsed_pg_rows(
                    parsed_df=parsed_df,
                    source_table=source_table,
                    source_platform=source_platform,
                    title_col=args.title_col,
                    desc_col=args.desc_col,
                )
                written_count = write_parsed_rows_to_postgres(rows, table_name=args.target_table)
                total_written_count += int(written_count)
                print(f"{source_table} batch written rows: {written_count}; total written rows: {total_written_count}")
            if read_batch_size <= 0 or len(df) < int(current_limit):
                break

    if args.output_csv and output_frames:
        pd.concat(output_frames, ignore_index=True).to_csv(
            args.output_csv,
            index=False,
            encoding="utf-8-sig",
        )

if __name__ == "__main__":
    """
    python -m src.data_pipeline.description_parsing `
  --input-source postgres `
  --source-table '"51job".sample' `
  --source-platform 51job `
  --write-postgres `
  --parse-workers 50
    """
    main()
