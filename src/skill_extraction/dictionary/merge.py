"""
merge_similar_skills.py — 使用本地 Qwen3-8B (vLLM) 批量合并技能词典中的相似技能
================================================================================

目标硬件:
    - CPU : 32-Core Intel Xeon Platinum 8358P, 3200 MHz
    - RAM : 130 647 MB (~128 GB)
    - GPU : NVIDIA GeForce RTX 4090 (24 564 MB)
    - 模型: Qwen3-8B (FP16 ≈ 16 GB 显存)

实现思路（五阶段流水线）:
    Phase 1 — 候选生成:
        使用 difflib.SequenceMatcher 对全部技能名称做两两相似度计算，
        保留相似度 ≥ threshold（默认 0.70）的技能对作为候选。
        内部先用 quick_ratio() 做快速剪枝，再用 ratio() 精确计算，
        避免 O(n²) 全量精确匹配的开销。

    Phase 2 — 聚类:
        用 Union-Find（并查集）算法将候选对聚合成连通分量。
        每个连通分量即为一个"待判定合并组"。
        仅保留 size ≥ 2 的组（单技能无需判断）。

    Phase 3 — Prompt 构建:
        为每个候选组生成一条 Chat 消息（system + user）。
        System prompt 定义合并规则（语义等价才合并、保守策略等），
        User prompt 列出该组的全部技能名称，要求输出结构化 JSON。

    Phase 4 — vLLM 批量推理:
        直接使用 vllm.LLM.generate() 进行离线批量推理（非 HTTP Server）。
        这是 vLLM 吞吐最高的模式：所有 prompt 一次性提交，引擎内部做
        continuous batching + PagedAttention，充分利用 RTX 4090 的 24 GB 显存。
        关键参数:
          - gpu_memory_utilization = 0.90  → 留 ~2.4 GB 给 OS/CUDA runtime
          - max_model_len = 4096           → 我们的 prompt 很短，无需长上下文
          - max_num_seqs = 48              → 并发序列数，平衡吞吐与显存
          - enable_prefix_caching = True   → 所有 prompt 共享同一 system prompt

    Phase 5 — 合并执行:
        解析 LLM 返回的 JSON，对每个 merge group：
          - 以 primary 为主名称
          - 将被合并技能的 name 加入 primary 的 aliases
          - 将被合并技能的 aliases 也全部并入 primary 的 aliases
          - 合并 notes（如果有）
          - 从技能列表中删除被合并的技能
        最终输出新版词典 JSON。

使用方式:
    # 1. 确保已安装 vllm (pip install vllm)
    # 2. 运行:
    python merge_similar_skills.py --model D:/model/Qwen3-8B

    # 自定义参数:
    python merge_similar_skills.py \\
        --model D:/model/Qwen3-8B \\
        --input  dicts/occupation_skill_dictionary_v2.4.json \\
        --output dicts/occupation_skill_dictionary_v2.5.json \\
        --threshold 0.70 \\
        --gpu-memory-utilization 0.90

作者: Claude (Anthropic)
日期: 2026-04-12
"""

# ============================================================================
#  0. Windows 兼容性 & 导入
# ============================================================================
import sys
import types
import os

# Stub uvloop on Windows — vLLM 内部会尝试 import uvloop，
# 但 Windows 不支持 uvloop，因此我们提供一个空壳模块。
if sys.platform == "win32":
    _uvloop = types.ModuleType("uvloop")
    _uvloop.run = __import__("asyncio").run  # type: ignore[attr-defined]
    sys.modules["uvloop"] = _uvloop

import json
import re
import time
import logging
import argparse
import difflib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# vLLM 导入（放在 uvloop stub 之后）
from vllm import LLM, SamplingParams

# ============================================================================
#  日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("skill_merger")

# ============================================================================
#  1. 常量 & Prompt 模板
# ============================================================================

# --- vLLM 默认参数（针对 RTX 4090 + Qwen3-8B 优化）---
DEFAULT_GPU_MEM_UTIL = 0.80       # 90% 显存给 vLLM，留 ~2.4 GB 给系统
DEFAULT_MAX_MODEL_LEN = 4096      # prompt 很短，4096 足够
DEFAULT_MAX_NUM_SEQS = 48         # 并发序列数（平衡吞吐 & 显存）
DEFAULT_THRESHOLD = 0.70          # difflib 相似度阈值
DEFAULT_INPUT = "dicts/occupation_skill_dictionary_v2.4.json"
DEFAULT_OUTPUT = "dicts/occupation_skill_dictionary_v2.5.json"

# --- LLM 系统提示词 ---
SYSTEM_PROMPT = """\
你是技能词典合并专家。你的任务是判断一组技能名称中，哪些指代同一个技能概念，应当合并为一条记录。

## 合并规则

1. **仅合并语义完全等价的技能**：即同一技能的不同表述方式。
   - 例如 "3D建模" 与 "3D建模技能" 与 "三维建模" → 合并
   - 例如 "Java开发" 与 "Java编程" → 合并

2. **"XX技能/技术/能力/操作/应用"** 若核心概念相同，则合并。
   - 例如 "CAD制图" 与 "CAD制图技术" → 合并

3. **不合并上下位关系**：范围不同的技能不合并。
   - 例如 "Python" 与 "Python Web开发" → 不合并
   - 例如 "数据库" 与 "MySQL数据库" → 不合并

4. **不合并工具名与方法名**：软件工具和技能方法不混为一谈。
   - 例如 "SolidWorks软件" 与 "三维建模" → 不合并

5. **不合并不同领域的同名后缀技能**：
   - 例如 ".NET开发" 与 "ETL开发" → 不合并

6. **存疑则不合并**（保守策略优先）。

## 输出要求

- 选择最简洁、最规范、最通用的名称作为 primary（主名称）。
- **严格**输出以下 JSON 格式，不要添加任何解释文字、markdown标记或思考过程：

{"groups":[{"primary":"主名称","merge":["要合并的别名1","要合并的别名2"]}],"standalone":["不需要合并的独立技能1","独立技能2"]}
"""

# --- LLM 用户提示词模板 ---
USER_PROMPT_TEMPLATE = """\
请判断以下 {count} 个技能名称中，哪些应该合并为同一技能：

{skill_list}
"""


# ============================================================================
#  2. 数据加载 / 保存
# ============================================================================

def load_skill_dictionary(path: str) -> dict:
    """
    从 JSON 文件加载技能词典。

    参数:
        path (str): JSON 文件的路径，文件应符合 schema_version 3 的格式，
                     包含 "metadata" 和 "skills" 两个顶层字段。

    返回:
        dict: 完整的词典数据，包含 metadata 和 skills 列表。

    异常:
        FileNotFoundError: 文件不存在时抛出。
        json.JSONDecodeError: JSON 格式错误时抛出。

    示例:
        >>> data = load_skill_dictionary("dicts/occupation_skill_dictionary_v2.4.json")
        >>> print(len(data["skills"]))
        4912
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"技能词典文件不存在: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    skills = data.get("skills", [])
    logger.info("已加载技能词典: %d 条技能, 来源: %s", len(skills), path)
    return data


def save_skill_dictionary(data: dict, path: str) -> None:
    """
    将技能词典保存为格式化的 JSON 文件。

    参数:
        data (dict): 完整的词典数据，包含 metadata 和 skills 列表。
        path (str): 输出文件路径。

    行为:
        - 自动创建父目录（如果不存在）。
        - 使用 UTF-8 编码，ensure_ascii=False 以保留中文字符。
        - 使用 2 空格缩进以保持可读性。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("已保存合并后的词典: %s", path)


# ============================================================================
#  3. Phase 1: 候选对生成（difflib 文本相似度）
# ============================================================================

def find_similar_pairs(
    skills: list[dict],
    threshold: float = 0.70,
    min_name_length: int = 2,
) -> list[tuple[int, int, float]]:
    """
    使用 difflib.SequenceMatcher 找出名称相似的技能对。

    算法流程:
        1. 提取所有技能的 name 字段，建立 name → indices 的映射。
        2. 对去重后的名称列表做两两比较。
        3. 先用 quick_ratio() 快速剪枝（O(1) 复杂度），
           再用 ratio() 精确计算（O(n) 复杂度）。
        4. 保留相似度 ≥ threshold 的对。

    参数:
        skills (list[dict]): 技能列表，每个元素至少包含 "name" 字段。
        threshold (float): 相似度阈值，范围 [0, 1]，默认 0.70。
                           值越高越严格（更少误报），值越低越宽松（更多候选）。
        min_name_length (int): 最小名称长度。长度 < 此值的技能跳过比较，
                                避免极短名称（如 "C"）产生大量误匹配。

    返回:
        list[tuple[int, int, float]]:
            每个元素为 (索引i, 索引j, 相似度)，其中 i < j。
            索引对应 skills 列表中的位置。

    性能:
        对于 ~4900 个唯一名称，约需 20-60 秒（取决于 CPU）。
        quick_ratio 剪枝可过滤掉 ~95% 的无效比较。

    示例:
        >>> pairs = find_similar_pairs(skills, threshold=0.70)
        >>> print(pairs[0])  # (12, 45, 0.82)
    """
    # 建立 name → 在 skills 列表中的索引映射
    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, skill in enumerate(skills):
        name = skill["name"]
        if len(name) >= min_name_length:
            name_to_indices[name].append(idx)

    unique_names = sorted(name_to_indices.keys())
    total_unique = len(unique_names)
    logger.info("Phase 1: 开始候选对生成，%d 个唯一技能名称，阈值 %.2f", total_unique, threshold)

    pairs: list[tuple[int, int, float]] = []
    comparisons = 0
    quick_pass = 0
    start_time = time.time()

    for i in range(total_unique):
        name_a = unique_names[i]

        # 进度日志：每处理 500 个名称输出一次
        if (i + 1) % 500 == 0 or i == total_unique - 1:
            elapsed = time.time() - start_time
            logger.info(
                "  进度: %d/%d (%.1f%%), 已找到 %d 对, 耗时 %.1fs",
                i + 1, total_unique, (i + 1) / total_unique * 100,
                len(pairs), elapsed,
            )

        for j in range(i + 1, total_unique):
            name_b = unique_names[j]
            comparisons += 1

            # 快速剪枝：长度差异过大则跳过
            len_a, len_b = len(name_a), len(name_b)
            if len_a > 0 and len_b > 0:
                len_ratio = min(len_a, len_b) / max(len_a, len_b)
                if len_ratio < threshold * 0.8:
                    continue

            # 使用 SequenceMatcher 计算相似度
            matcher = difflib.SequenceMatcher(None, name_a, name_b)

            # 第一层剪枝：quick_ratio (O(1))
            if matcher.quick_ratio() < threshold:
                continue
            quick_pass += 1

            # 第二层：精确 ratio (O(n))
            ratio = matcher.ratio()
            if ratio >= threshold:
                # 将名称对映射回所有对应的 skill 索引
                for idx_a in name_to_indices[name_a]:
                    for idx_b in name_to_indices[name_b]:
                        pairs.append((idx_a, idx_b, ratio))

    elapsed = time.time() - start_time
    logger.info(
        "Phase 1 完成: %d 次比较, %d 次通过快速剪枝, %d 个候选对, 耗时 %.1fs",
        comparisons, quick_pass, len(pairs), elapsed,
    )
    return pairs


# ============================================================================
#  4. Phase 2: 聚类（Union-Find / 并查集）
# ============================================================================

class UnionFind:
    """
    并查集（Union-Find）数据结构，用于高效地将相似技能对聚合成连通分量。

    实现细节:
        - 使用路径压缩（path compression）优化 find 操作。
        - 使用按秩合并（union by rank）优化 union 操作。
        - 均摊时间复杂度接近 O(α(n))，其中 α 是反 Ackermann 函数（极慢增长）。

    属性:
        parent (dict[int, int]): 每个节点的父节点映射。
        rank (dict[int, int]): 每个节点的秩（近似树高），用于按秩合并。

    示例:
        >>> uf = UnionFind()
        >>> uf.union(0, 1)
        >>> uf.union(1, 2)
        >>> uf.find(0) == uf.find(2)  # True — 0, 1, 2 在同一组
    """

    def __init__(self) -> None:
        """初始化空的并查集。"""
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        """
        查找元素 x 所在集合的根节点（代表元素），并执行路径压缩。

        参数:
            x (int): 要查找的元素。

        返回:
            int: x 所在集合的根节点。
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """
        合并元素 x 和 y 所在的两个集合。使用按秩合并策略。

        参数:
            x (int): 第一个元素。
            y (int): 第二个元素。

        行为:
            如果 x 和 y 已在同一集合中，不做任何操作。
            否则将秩较小的树挂到秩较大的树下。
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # 按秩合并：矮树挂到高树下
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def build_clusters(
    pairs: list[tuple[int, int, float]],
    skills: list[dict],
) -> list[list[int]]:
    """
    将候选对通过 Union-Find 聚合成技能聚���。

    算法:
        1. 将每个候选对 (i, j) 执行 union(i, j)。
        2. 遍历所有涉及的节点，按根节点分组。
        3. 过滤掉 size < 2 的单元素组（无需合并判断）。
        4. 按组大小降序排列（大组优先处理）。

    参数:
        pairs (list[tuple[int, int, float]]):
            候选对列表，来自 find_similar_pairs() 的输出。
        skills (list[dict]):
            原始技能列表，用于日志输出技能名称。

    返回:
        list[list[int]]:
            每个子列表是一个聚类，包含技能在 skills 中的索引。
            仅包含 size ≥ 2 的聚类。

    示例:
        >>> clusters = build_clusters(pairs, skills)
        >>> print(len(clusters))  # e.g., 328
        >>> print([skills[i]["name"] for i in clusters[0]])
        ['3D建模', '3D建模技能', '3D建模技术']
    """
    logger.info("Phase 2: 开始聚类 (Union-Find)...")
    uf = UnionFind()

    for idx_a, idx_b, _ in pairs:
        uf.union(idx_a, idx_b)

    # 按根节点分组
    groups: dict[int, list[int]] = defaultdict(list)
    all_nodes = set()
    for idx_a, idx_b, _ in pairs:
        all_nodes.add(idx_a)
        all_nodes.add(idx_b)

    for node in all_nodes:
        root = uf.find(node)
        groups[root].append(node)

    # 去重（一个节点可能被多次添加）并过滤
    clusters = []
    for root, members in groups.items():
        unique_members = sorted(set(members))
        if len(unique_members) >= 2:
            clusters.append(unique_members)

    # 按组大小降序排列
    clusters.sort(key=len, reverse=True)

    logger.info(
        "Phase 2 完成: %d 个聚类 (size≥2), 最大聚类 %d 个技能, 涉及 %d 个技能",
        len(clusters),
        max(len(c) for c in clusters) if clusters else 0,
        sum(len(c) for c in clusters),
    )

    # 输出前 5 个最大聚类的详情
    for i, cluster in enumerate(clusters[:5]):
        names = [skills[idx]["name"] for idx in cluster]
        logger.info("  聚类 #%d (%d 个): %s", i + 1, len(cluster), ", ".join(names))

    return clusters


# ============================================================================
#  5. Phase 3: Prompt 构建
# ============================================================================

def build_prompts_for_clusters(
    clusters: list[list[int]],
    skills: list[dict],
) -> list[tuple[list[int], str]]:
    """
    为每个候选聚类构建 LLM 对话 prompt。

    每个 prompt 由 system message + user message 组成，
    经过 tokenizer.apply_chat_template() 后变为模型可直接处理的文本。

    参数:
        clusters (list[list[int]]): 聚类列表，每个聚类是技能索引的列表。
        skills (list[dict]): 原始技能列表。

    返回:
        list[tuple[list[int], str]]:
            每个元素为 (聚类索引列表, 格式化的 user prompt 文本)。
            system prompt 统一使用全局常量 SYSTEM_PROMPT。

    说明:
        - 超大聚类（>15 个技能）会被拆分为多个子组分别提交，
          避免单条 prompt 过长导致输出质量下降。
        - user prompt 中将技能按编号列出，便于 LLM 逐一判断。
    """
    MAX_CLUSTER_SIZE = 15  # 单个 prompt 最大技能数

    prompt_tasks: list[tuple[list[int], str]] = []

    for cluster in clusters:
        # 如果聚类过大，拆分为子组
        sub_clusters = []
        if len(cluster) > MAX_CLUSTER_SIZE:
            for start in range(0, len(cluster), MAX_CLUSTER_SIZE):
                sub_clusters.append(cluster[start:start + MAX_CLUSTER_SIZE])
        else:
            sub_clusters.append(cluster)

        for sub in sub_clusters:
            # 构建技能列表文本
            skill_lines = []
            for seq, idx in enumerate(sub, 1):
                name = skills[idx]["name"]
                aliases = skills[idx].get("aliases", [])
                if aliases:
                    alias_str = "（已有别名: " + ", ".join(aliases[:5]) + "）"
                else:
                    alias_str = ""
                skill_lines.append(f"{seq}. {name}{alias_str}")

            skill_list_text = "\n".join(skill_lines)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                count=len(sub),
                skill_list=skill_list_text,
            )
            prompt_tasks.append((sub, user_prompt))

    logger.info("Phase 3: 已构建 %d 个 prompt（含拆分后的子组）", len(prompt_tasks))
    return prompt_tasks


# ============================================================================
#  6. Phase 4: vLLM 批量推理
# ============================================================================

def init_vllm_engine(
    model_path: str,
    gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
) -> LLM:
    """
    初始化 vLLM 离线推理引擎，针对 RTX 4090 + Qwen3-8B 的批量处理场景优化。

    参数说明与调优建议:
        model_path (str):
            Qwen3-8B 模型的本地路径（HuggingFace 格式目录）。

        gpu_memory_utilization (float):
            GPU 显存利用率，范围 (0, 1)。
            - 0.90: 推荐值，~21.6 GB 给 vLLM，~2.4 GB 给系统。
            - 0.95: 激进值，显存紧张时可能 OOM。
            - 0.80: 保守值，如果同时运行其他 GPU 程序。

        max_model_len (int):
            最大序列长度（输入 + 输出 token 数之和）。
            - 4096: 推荐值，本任务 prompt 约 200-500 token，输出约 100-300 token。
            - 8192: 如果 prompt 特别长（大聚类）可适当提高。

        max_num_seqs (int):
            同时处理的最大序列数（continuous batching 的并发度）。
            - 48: 推荐值，平衡吞吐和显存。
            - 64: 如果 prompt 很短可以更激进。
            - 32: 如果 OOM 则降低。

    返回:
        LLM: vLLM 引擎实例，可直接调用 .generate() 进行批量推理。

    异常:
        ValueError: 显存不足以支撑 KV cache 时抛出，附带调参建议。

    RTX 4090 显存分配估算:
        Qwen3-8B FP16 权重:        ~15.3 GB
        系统/CUDA runtime 保留:    ~2.4 GB (10%)
        KV cache (剩余显存):       ~4.4 GB @ 0.90 utilization
        每 token KV cache 开销:    ~144 KB (36层 × 8 KV头 × 128维 × 2(K+V) × FP16)
        可缓存总 token 数:         ~31,000 tokens
        48 并发序列 × ~500 token:  ~24,000 tokens → 安全范围
    """
    logger.info("正在初始化 vLLM 引擎...")
    logger.info("  模型路径: %s", model_path)
    logger.info("  GPU 显存利用率: %.2f", gpu_memory_utilization)
    logger.info("  最大序列长度: %d", max_model_len)
    logger.info("  最大并发序列数: %d", max_num_seqs)

    start_time = time.time()

    try:
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=True,     # 共享 system prompt 的 KV cache
            trust_remote_code=True,         # Qwen3 可能需要自定义代码
        )
    except ValueError as exc:
        msg = str(exc)
        if "KV cache" in msg or "max seq len" in msg or "memory" in msg.lower():
            logger.error("显存不足！建议降低参数后重试:")
            logger.error("  1. --max-model-len 2048")
            logger.error("  2. --gpu-memory-utilization 0.92")
            logger.error("  3. --max-num-seqs 16")
            raise SystemExit(f"vLLM 启动失败（显存不足）: {exc}") from exc
        raise

    elapsed = time.time() - start_time
    logger.info("vLLM 引擎初始化完成，耗时 %.1fs", elapsed)
    return llm


def batch_inference(
    llm: LLM,
    prompt_tasks: list[tuple[list[int], str]],
    batch_size: int = 100,
) -> list[tuple[list[int], str]]:
    """
    使用 vLLM 引擎对所有 prompt 执行批量推理。

    采用 vLLM 的离线批量推理模式（LLM.generate()），这是 vLLM 吞吐最高的模式：
    所有 prompt 一次性提交给引擎，引擎内部通过 continuous batching 和
    PagedAttention 机制自动调度，最大化 GPU 利用率。

    参数:
        llm (LLM): 已初始化的 vLLM 引擎。
        prompt_tasks (list[tuple[list[int], str]]):
            来自 build_prompts_for_clusters() 的输出。
            每个元素为 (聚类索引列表, user prompt 文本)。
        batch_size (int):
            每批提交的 prompt 数量，默认 100。
            vLLM 内部会进一步做 continuous batching，此参数主要用于
            控制内存占用和进度日志的粒度。

    返回:
        list[tuple[list[int], str]]:
            每个元素为 (聚类索引列表, LLM 原始输出文本)。

    采样参数说明:
        - temperature=0.1: 极低温度，确保输出确定性高、格式稳定。
        - max_tokens=1024: 足够输出 JSON 结果（通常 100-300 token）。
        - top_p=0.9: 配合低温度，进一步稳定输出。
        - repetition_penalty=1.05: 轻微抑制重复，避免 JSON 结构重复。
    """
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    # 构建完整 prompt（应用 chat template）
    formatted_prompts: list[str] = []
    for _, user_prompt in prompt_tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 回退：简单拼接
            prompt_text = f"system: {SYSTEM_PROMPT}\nuser: {user_prompt}\nassistant:"

        formatted_prompts.append(prompt_text)

    logger.info("Phase 4: 开始批量推理，共 %d 个 prompt", len(formatted_prompts))

    # 分批提交（每批 batch_size 个），方便跟踪进度
    results: list[tuple[list[int], str]] = []
    total_batches = (len(formatted_prompts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(formatted_prompts))
        batch_prompts = formatted_prompts[start:end]
        batch_tasks = prompt_tasks[start:end]

        logger.info(
            "  批次 %d/%d: 处理 prompt %d-%d...",
            batch_idx + 1, total_batches, start + 1, end,
        )
        batch_start_time = time.time()

        outputs = llm.generate(batch_prompts, sampling_params)

        batch_elapsed = time.time() - batch_start_time
        logger.info("  批次 %d 完成，耗时 %.1fs", batch_idx + 1, batch_elapsed)

        for (cluster_indices, _), output in zip(batch_tasks, outputs):
            generated_text = output.outputs[0].text
            results.append((cluster_indices, generated_text))

    logger.info("Phase 4 完成: 共处理 %d 个 prompt", len(results))
    return results


# ============================================================================
#  7. Phase 5: 响应解析（鲁棒 JSON 提取）
# ============================================================================

def extract_json_from_response(text: str) -> dict | None:
    """
    从 LLM 输出文本中鲁棒地提取 JSON 对象。

    LLM 的输出可能包含以下噪声，本函数逐一处理:
        1. Qwen3 的 <think>...</think> 思考标签
        2. Markdown 代码块标记 (```json ... ```)
        3. JSON 前后的额外文字说明
        4. 轻微的 JSON 格式错误（尾逗号等）

    参数:
        text (str): LLM 的原始输出文本。

    返回:
        dict | None: 解析成功返回 dict，失败返回 None。

    解析策略（按优先级）:
        1. 去除 <think>...</think> 标签及其内容。
        2. 去除 markdown 代码块标记。
        3. 尝试直接 json.loads()。
        4. 用正则提取最外层 {...} 并解析。
        5. 修复常见 JSON 错误后重试。
        6. 全部失败则返回 None 并记录警告。

    示例:
        >>> text = '<think>让我分析...</think>```json\\n{"groups":[],"standalone":["A"]}\\n```'
        >>> extract_json_from_response(text)
        {'groups': [], 'standalone': ['A']}
    """
    if not text or not text.strip():
        return None

    # Step 1: 去除 Qwen3 思考标签
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)

    # Step 2: 去除 markdown 代码块标记
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = text.strip()

    # Step 3: 直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4: 提取最外层花括号内容
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Step 5: 尝试修复常见 JSON 错误
        fixed = candidate
        # 修复尾逗号: ,] → ] 和 ,} → }
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        # 修复单引号 → 双引号
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Step 6: 全部失败
    logger.warning("JSON 解析失败，原始文本前 200 字符: %s", text[:200])
    return None


def parse_merge_decisions(
    inference_results: list[tuple[list[int], str]],
    skills: list[dict],
) -> list[dict]:
    """
    解析 LLM 批量推理的结果，提取合并决策。

    参数:
        inference_results (list[tuple[list[int], str]]):
            来自 batch_inference() 的输出。
            每个元素为 (聚类索引列表, LLM 原始输出文本)。
        skills (list[dict]):
            原始技能列表，用于将名称映射回索引。

    返回:
        list[dict]: 合并决策列表，每个元素格式为:
            {
                "primary_idx": int,       # 主技能在 skills 中的索引
                "merge_indices": [int],    # 要合并的技能索引列表
            }

    处理逻辑:
        1. 对每个聚类的 LLM 输出调用 extract_json_from_response() 解析。
        2. 遍历 JSON 中的 "groups" 字段。
        3. 将 primary 和 merge 中的名称映射回 skills 索引。
        4. 跳过解析失败的聚类（保守策略：解析失败则��合并）。
    """
    logger.info("Phase 5: 开始解析 LLM 响应...")

    merge_decisions: list[dict] = []
    success_count = 0
    fail_count = 0

    for cluster_indices, raw_text in inference_results:
        # 建立该聚类内的 name → index 映射
        cluster_name_to_idx: dict[str, int] = {}
        for idx in cluster_indices:
            cluster_name_to_idx[skills[idx]["name"]] = idx

        parsed = extract_json_from_response(raw_text)
        if parsed is None:
            fail_count += 1
            continue

        success_count += 1
        groups = parsed.get("groups", [])

        for group in groups:
            primary_name = group.get("primary", "")
            merge_names = group.get("merge", [])

            if not primary_name or not merge_names:
                continue

            # 查找 primary 的索引
            primary_idx = cluster_name_to_idx.get(primary_name)
            if primary_idx is None:
                # 尝试模糊匹配（LLM 可能轻微修改了名称）
                for cname, cidx in cluster_name_to_idx.items():
                    if cname.strip() == primary_name.strip():
                        primary_idx = cidx
                        break
            if primary_idx is None:
                logger.warning("未找到 primary 技能: '%s'", primary_name)
                continue

            # 查找要合并的技能索引
            merge_indices = []
            for mname in merge_names:
                midx = cluster_name_to_idx.get(mname)
                if midx is None:
                    # 模糊匹配
                    for cname, cidx in cluster_name_to_idx.items():
                        if cname.strip() == mname.strip():
                            midx = cidx
                            break
                if midx is not None and midx != primary_idx:
                    merge_indices.append(midx)

            if merge_indices:
                merge_decisions.append({
                    "primary_idx": primary_idx,
                    "merge_indices": merge_indices,
                })

    logger.info(
        "Phase 5 完成: %d/%d 个聚类解析成功, 生成 %d 个合并决策",
        success_count, success_count + fail_count, len(merge_decisions),
    )
    return merge_decisions


# ============================================================================
#  8. Phase 6: 执行合并
# ============================================================================

def execute_merges(
    skills: list[dict],
    merge_decisions: list[dict],
) -> tuple[list[dict], dict]:
    """
    根据 LLM 的合并决策，执行技能词典的实际合并操作。

    合并策略:
        对于每个 merge decision:
        1. 以 primary 技能为保留对象。
        2. 将每个被合并技能的 name 加入 primary 的 aliases（若不重复）。
        3. 将每个被合并技能的 aliases 也全部并入 primary 的 aliases。
        4. 如果被合并技能有非空 notes，追加到 primary 的 notes 后。
        5. 合并后 aliases 列表去重、排序。
        6. 从技能列表中移除被合并的技能。

    冲突处理:
        - 如果一个技能同时出现在多个 merge decision 的 merge_indices 中，
          以第一次出现为准（先到先得）。
        - 如果 primary 本身被其他决策要求合并，跳过该决策。

    参数:
        skills (list[dict]):
            原始技能列表（会被复制，不修改原始数据）。
        merge_decisions (list[dict]):
            来自 parse_merge_decisions() 的输出。

    返回:
        tuple[list[dict], dict]:
            - list[dict]: 合并后的技能列表。
            - dict: 合并统计信息，包含合并数、删除数等。

    示例:
        >>> merged_skills, stats = execute_merges(skills, decisions)
        >>> print(stats)
        {'skills_before': 4912, 'skills_after': 4650, 'merged_groups': 180, ...}
    """
    logger.info("Phase 6: 开始执行合并...")

    # 深拷贝技能列表，避免修改原始数据
    import copy
    merged_skills = copy.deepcopy(skills)

    # 记录哪些索引已被消费（作为 merge target 或 primary）
    consumed_as_merge: set[int] = set()
    consumed_as_primary: set[int] = set()

    merged_group_count = 0
    merged_skill_count = 0
    alias_added_count = 0

    for decision in merge_decisions:
        primary_idx = decision["primary_idx"]
        merge_indices = decision["merge_indices"]

        # 跳过冲突：primary 已被其他决策消费
        if primary_idx in consumed_as_merge:
            logger.debug("跳过: primary %d 已被合并到其他技能", primary_idx)
            continue

        primary_skill = merged_skills[primary_idx]
        existing_aliases = set(primary_skill.get("aliases", []))
        existing_aliases.add(primary_skill["name"])  # 用于去重

        valid_merge_indices = []
        for midx in merge_indices:
            if midx in consumed_as_merge or midx in consumed_as_primary:
                continue
            valid_merge_indices.append(midx)

        if not valid_merge_indices:
            continue

        consumed_as_primary.add(primary_idx)

        for midx in valid_merge_indices:
            merge_skill = merged_skills[midx]

            # 将被合并技能的 name 加入 aliases
            if merge_skill["name"] not in existing_aliases:
                primary_skill.setdefault("aliases", []).append(merge_skill["name"])
                existing_aliases.add(merge_skill["name"])
                alias_added_count += 1

            # 将被合并技能的 aliases 也并入
            for alias in merge_skill.get("aliases", []):
                if alias not in existing_aliases:
                    primary_skill.setdefault("aliases", []).append(alias)
                    existing_aliases.add(alias)
                    alias_added_count += 1

            # 合并 notes
            merge_notes = merge_skill.get("notes", "").strip()
            primary_notes = primary_skill.get("notes", "").strip()
            if merge_notes and merge_notes not in primary_notes:
                if primary_notes:
                    primary_skill["notes"] = primary_notes + "；" + merge_notes
                else:
                    primary_skill["notes"] = merge_notes

            consumed_as_merge.add(midx)
            merged_skill_count += 1

        # 对 aliases 排序去重
        primary_skill["aliases"] = sorted(set(primary_skill.get("aliases", [])))

        merged_group_count += 1

    # 删除被合并的技能（从后向前删除以保持索引稳定）
    indices_to_remove = sorted(consumed_as_merge, reverse=True)
    for idx in indices_to_remove:
        merged_skills.pop(idx)

    stats = {
        "skills_before": len(skills),
        "skills_after": len(merged_skills),
        "merged_groups": merged_group_count,
        "merged_skills_removed": merged_skill_count,
        "aliases_added": alias_added_count,
    }

    logger.info("Phase 6 完成:")
    logger.info("  合并前技能数: %d", stats["skills_before"])
    logger.info("  合并后技能数: %d", stats["skills_after"])
    logger.info("  合并组数: %d", stats["merged_groups"])
    logger.info("  移除的技能数: %d", stats["merged_skills_removed"])
    logger.info("  新增的别名数: %d", stats["aliases_added"])

    return merged_skills, stats


# ============================================================================
#  9. 元数据更新
# ============================================================================

def update_metadata(data: dict, stats: dict) -> None:
    """
    更新词典的 metadata 字段，记录本次合并的统计信息。

    参数:
        data (dict): 完整的词典数据（将就地修改 metadata 字段）。
        stats (dict): 来自 execute_merges() 的合并统计信息。

    行为:
        在 metadata 中新增 "llm_merge_summary" 字段，记录:
        - 合并时间
        - 使用的模型
        - 合并前后技能数
        - 合并组数、移除数、别名新增数
    """
    metadata = data.setdefault("metadata", {})
    metadata["llm_merge_summary"] = {
        "merged_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "Qwen3-8B",
        "method": "difflib_similarity + LLM_semantic_judgment",
        **stats,
    }
    metadata["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================================
#  10. 合并详情报告
# ============================================================================

def save_merge_report(
    merge_decisions: list[dict],
    skills: list[dict],
    report_path: str,
) -> None:
    """
    将合并决策的详细信息保存为可读的 JSON 报告文件。

    报告内容包括每个合并组的:
        - 主技能名称（primary）
        - 被合并的技能名称列表（merged）
        - 合并后的完整别名列表

    参数:
        merge_decisions (list[dict]): 合并决策列表。
        skills (list[dict]): 原始（合并前的）技能列表。
        report_path (str): 报告文件的输出路径。

    用途:
        方便人工审核合并结果，确认 LLM 的判断是否合理。
        如有误合并，可手动修正后重新运行。
    """
    report = []
    for decision in merge_decisions:
        primary_idx = decision["primary_idx"]
        merge_indices = decision["merge_indices"]
        report.append({
            "primary": skills[primary_idx]["name"],
            "merged": [skills[i]["name"] for i in merge_indices],
            "primary_aliases_before": skills[primary_idx].get("aliases", []),
            "merged_aliases_before": {
                skills[i]["name"]: skills[i].get("aliases", [])
                for i in merge_indices
            },
        })

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("合并详情报告已保存: %s", report_path)


# ============================================================================
#  11. 主函数
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    支持的参数:
        --model: Qwen3-8B 模型路径（必需）
        --input: 输入词典路径
        --output: 输出词典路径
        --threshold: difflib 相似度阈值
        --gpu-memory-utilization: GPU 显存利用率
        --max-model-len: 最大序列长度
        --max-num-seqs: 最大并发序列数
        --report: 合并报告输出路径
        --dry-run: 仅生成候选对，不执行 LLM 推理和合并

    返回:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="使用本地 Qwen3-8B (vLLM) 批量合并技能词典中的相似技能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例用法:
  # 基本用法
  python merge_similar_skills.py --model D:/model/Qwen3-8B

  # 自定义参数
  python merge_similar_skills.py \\
      --model D:/model/Qwen3-8B \\
      --threshold 0.75 \\
      --gpu-memory-utilization 0.92

  # 仅生成候选对（不调用 LLM）
  python merge_similar_skills.py --model D:/model/Qwen3-8B --dry-run
""",
    )
    parser.add_argument(
        "--model", required=True,
        help="Qwen3-8B 模型的本地路径（HuggingFace 格式目录）",
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"输入技能词典 JSON 路径 (默认: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出技能词典 JSON 路径 (默认: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"difflib 文本相似度阈值, 0-1 (默认: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL,
        help=f"GPU 显存利用率 (默认: {DEFAULT_GPU_MEM_UTIL})",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=f"最大序列长度 (默认: {DEFAULT_MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS,
        help=f"最大并发序列数 (默认: {DEFAULT_MAX_NUM_SEQS})",
    )
    parser.add_argument(
        "--report", default=None,
        help="合并详情报告输出路径 (默认: 输出文件同目录下 _merge_report.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅执行 Phase 1-2（候选生成 + 聚类），不调用 LLM，不执行合并",
    )
    return parser.parse_args()


def main() -> None:
    """
    主函数：编排五阶段合并流水线。

    执行流程:
        1. 解析命令行参数。
        2. 加载技能词典。
        3. Phase 1: 使用 difflib 找出名称相似的候选对。
        4. Phase 2: 使用 Union-Find 将候选对聚合成聚类。
        5. (dry-run 在此退出)
        6. Phase 3: 为每个聚类构建 LLM prompt。
        7. Phase 4: 初始化 vLLM 引擎，执行批量推理。
        8. Phase 5: 解析 LLM 响应，提取合并决策。
        9. Phase 6: 执行技能合并，生成新词典。
        10. 保存合并后的词典和合并报告。

    退出码:
        0: 成功完成。
        1: 参数错误或文件不存在。
    """
    args = parse_args()
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  技能词典相似合并工具")
    logger.info("  模型: Qwen3-8B via vLLM")
    logger.info("=" * 60)

    # --- 加载词典 ---
    data = load_skill_dictionary(args.input)
    skills = data["skills"]

    # --- Phase 1: 候选对生成 ---
    pairs = find_similar_pairs(skills, threshold=args.threshold)

    if not pairs:
        logger.info("未找到相似技能对（阈值 %.2f），无需合并。", args.threshold)
        return

    # --- Phase 2: 聚类 ---
    clusters = build_clusters(pairs, skills)

    if not clusters:
        logger.info("未形成有效聚类，无需合并。")
        return

    # --- Dry-run 退出点 ---
    if args.dry_run:
        logger.info("--- Dry-run 模式: 仅展示候选聚类，不执行合并 ---")
        for i, cluster in enumerate(clusters):
            names = [skills[idx]["name"] for idx in cluster]
            logger.info("  聚类 #%d: %s", i + 1, " | ".join(names))
        logger.info("共 %d 个聚类，涉及 %d 个技能。", len(clusters), sum(len(c) for c in clusters))
        return

    # --- Phase 3: 构建 Prompt ---
    prompt_tasks = build_prompts_for_clusters(clusters, skills)

    # --- Phase 4: vLLM 批量推理 ---
    llm = init_vllm_engine(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    inference_results = batch_inference(llm, prompt_tasks)

    # --- Phase 5: 解析合并决策 ---
    merge_decisions = parse_merge_decisions(inference_results, skills)

    if not merge_decisions:
        logger.info("LLM 未建议任何合并操作。")
        return

    # --- Phase 6: 执行合并 ---
    merged_skills, stats = execute_merges(skills, merge_decisions)

    # --- 更新元数据并保存 ---
    data["skills"] = merged_skills
    update_metadata(data, stats)
    save_skill_dictionary(data, args.output)

    # --- 保存合并报告 ---
    report_path = args.report
    if report_path is None:
        out_stem = Path(args.output).stem
        report_path = str(Path(args.output).parent / f"{out_stem}_merge_report.json")
    save_merge_report(merge_decisions, skills, report_path)

    # --- 总结 ---
    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("  全部完成！总耗时: %.1fs", total_elapsed)
    logger.info("  合并前: %d 个技能", stats["skills_before"])
    logger.info("  合并后: %d 个技能 (减少 %d 个)",
                stats["skills_after"], stats["skills_before"] - stats["skills_after"])
    logger.info("  输出文件: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
