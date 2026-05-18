"""
quality_checker.py
质量检查管道：格式校验 → 语义去重 → 事实校验 → 覆盖率分析

使用方式：
    python src/quality_checker.py --input ./data/expanded/*.jsonl
    python src/quality_checker.py --input ./data/seeds/seed_xxx.jsonl --stages format,dedup
"""

import json
import os
import re
import argparse
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import yaml
import asyncio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================
# 元数据枚举约束
# ============================================================
VALID_DIFFICULTIES = {"入门", "进阶", "专业"}
VALID_QUESTION_TYPES = {
    "概念解释", "流程指引", "条款解读", "边界判断",
    "计算说明", "对比区分", "注意事项", "案例分析",
    "赔付判断", "操作指引",
    "保全操作", "健康告知", "争议处理",
    "产品对比", "澄清引导",
}
VALID_BUSINESS_STAGES = {
    # 产品类文档：保险业务流程节点
    "投保咨询", "健康告知", "核保", "承保生效",
    "保全变更", "续保复效", "报案", "理赔材料",
    "理赔审核", "赔付结案", "拒赔争议",
    # 法规类文档专属
    "合规监管", "消费者权益",
    # 通用兜底
    "通用",
}
STAGE_NORMALIZE = {
    # 产品类同义词归一化
    "产品设计": "投保咨询", "精算定价": "投保咨询",
    "销售展业": "投保咨询", "投保告知": "健康告知",
    "承保出单": "承保生效", "续保续费": "续保复效",
    "查勘定损": "理赔审核",
    "理赔": "理赔审核", "投保": "投保咨询",
    "销售": "投保咨询", "承保": "承保生效",
    "保全": "保全变更", "续保": "续保复效",
    "拒赔": "拒赔争议",
    # 法规类同义词归一化
    "监管合规": "合规监管", "法规解读": "合规监管",
    "合规": "合规监管", "监管": "合规监管",
    "消费者保护": "消费者权益", "投诉处理": "消费者权益",
    "权益保障": "消费者权益",
}
VALID_QA_CATEGORIES = {
    "knowledge", "tool_routed", "misconception_correction",
    "product_comparison", "refusal_or_clarification",
}


def normalize_metadata(qa: dict) -> dict:
    """入库前统一归一化元数据字段"""
    # business_stage
    stage = qa.get("business_stage", "通用")
    qa["business_stage"] = STAGE_NORMALIZE.get(stage, stage)
    if qa["business_stage"] not in VALID_BUSINESS_STAGES:
        qa["business_stage"] = "通用"

    # difficulty
    diff = qa.get("difficulty", "入门")
    qa["difficulty"] = {"入门级": "入门", "进阶级": "进阶", "专业级": "专业"}.get(diff, diff)
    if qa["difficulty"] not in VALID_DIFFICULTIES:
        qa["difficulty"] = "入门"

    # question_type
    if qa.get("question_type") not in VALID_QUESTION_TYPES:
        qa["question_type"] = "概念解释"

    # qa_category 推断
    if not qa.get("qa_category"):
        if qa.get("is_tool_routed"):
            qa["qa_category"] = "tool_routed"
        elif qa.get("misconception"):
            qa["qa_category"] = "misconception_correction"
        elif qa.get("comparison_type"):
            qa["qa_category"] = "product_comparison"
        elif qa.get("requires_clarification"):
            qa["qa_category"] = "refusal_or_clarification"
        else:
            qa["qa_category"] = "knowledge"

    if qa.get("qa_category") not in VALID_QA_CATEGORIES:
        qa["qa_category"] = "knowledge"

    # tool_routing 一致性
    if qa.get("tool_routing") and not qa.get("is_tool_routed"):
        qa["is_tool_routed"] = True
    if qa.get("is_tool_routed") and not qa.get("tool_routing"):
        qa["is_tool_routed"] = False

    # product_comparison 跨字段一致性：有 category 但缺 product_a/product_b 时降级
    if qa.get("qa_category") == "product_comparison":
        if not qa.get("product_a") or not qa.get("product_b"):
            logger.warning(
                f"qa_id={qa.get('id', '?')} 声明为 product_comparison 但缺少 product_a/product_b，降级为 knowledge"
            )
            qa["qa_category"] = "knowledge"

    # tool_routed 跨字段一致性：is_tool_routed=True 但 qa_category 不匹配时修正
    if qa.get("is_tool_routed") and qa.get("qa_category") not in ("tool_routed",):
        qa["qa_category"] = "tool_routed"

    # 默认值补全
    defaults = [
        ("product_name", ""), ("tool_routing", ""), ("tool_params", {}),
        ("is_tool_routed", False), ("requires_clarification", False),
        ("misconception", ""), ("qa_category", "knowledge"),
        ("comparison_type", ""), ("product_a", ""), ("product_b", ""),
        ("comparison_dimensions", []), ("comparison_verdict", ""),
        ("comparison_limitations", ""),
    ]
    for field, default in defaults:
        qa.setdefault(field, default)

    return qa


# ============================================================
# 工具路由安全校验
# ============================================================
def validate_tool_routing_safety(qa: dict) -> list[str]:
    """检查工具路由型答案是否包含不应出现的确定性结论"""
    if not qa.get("is_tool_routed"):
        return []
    issues = []
    answer = qa.get("answer", "")
    dangerous_patterns = [
        r"可以(?:投保|购买|买)",
        r"不能(?:投保|购买|买)",
        r"会被拒保",
        r"标准体承保",
        r"加费承保",
        r"除外承保",
        r"保费(?:为|是)\s*\d+",
        r"(?:退保金|现金价值)(?:为|是)\s*\d+",
        r"(?:能|可以)赔\s*\d+",
        r"一定(?:能|可以|不能)",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, answer):
            issues.append(f"工具路由型答案包含确定性结论: {pattern}")
    if qa.get("tool_routing") and not qa.get("tool_params"):
        issues.append("tool_routing 已设置但 tool_params 为空")
    return issues


# ============================================================
# Stage 1: 格式校验
# ============================================================


def validate_format(qa: dict) -> dict:
    """格式校验，返回 {passed, issues, auto_fixable}"""
    issues = []

    # 必填字段
    if not qa.get("question"):
        issues.append("缺少 question")
    if not qa.get("answer"):
        issues.append("缺少 answer")

    q = qa.get("question", "")
    a = qa.get("answer", "")

    # 长度校验
    if len(q) < 8:
        issues.append(f"问题过短({len(q)}字)")
    if len(q) > 500:
        issues.append(f"问题过长({len(q)}字)")
    if len(a) < 20:
        issues.append(f"答案过短({len(a)}字)")
    if len(a) > 2000:
        issues.append(f"答案过长({len(a)}字)")

    # 模型拒绝回答的标记
    refusal_markers = ["作为AI", "我无法", "我不确定", "抱歉，我"]
    if any(m in a and len(a) < 100 for m in refusal_markers):
        issues.append("答案疑似模型拒绝回答")

    # 问答不相关（简单字符重叠检测）
    q_chars = set(q)
    a_chars = set(a)
    overlap = len(q_chars & a_chars) / max(len(q_chars), 1)
    if overlap < 0.05:
        issues.append("问答字符重叠率极低，可能不相关")

    # 枚举值校验（宽松：只警告不拦截）
    diff = qa.get("difficulty", "")
    if diff and diff not in VALID_DIFFICULTIES:
        issues.append(f"非标准难度值: {diff}")

    return {
        "qa_id": qa.get("id", "unknown"),
        "passed": len(issues) == 0,
        "issues": issues,
    }


def run_format_check(qa_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """批量格式校验，返回 (通过列表, 失败列表)。格式校验前先做元数据归一化。"""
    passed, failed = [], []

    def check_one(qa):
        normalize_metadata(qa)
        result = validate_format(qa)
        tool_issues = validate_tool_routing_safety(qa)
        all_issues = result["issues"] + tool_issues
        if not all_issues:
            return "pass", qa
        else:
            qa["_qc_issues"] = all_issues
            return "fail", qa

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as executor:
        futures = [executor.submit(check_one, qa) for qa in qa_list]
        for future in concurrent.futures.as_completed(futures):
            status, qa = future.result()
            if status == "pass":
                passed.append(qa)
            else:
                failed.append(qa)

    logger.info(f"格式校验: {len(passed)} 通过, {len(failed)} 失败")
    return passed, failed


# ============================================================
# Stage 2: 语义去重（基于文本相似度）
# ============================================================
def run_seed_dedup_filter(seed_files: list[str]) -> tuple[set, set]:
    """
    加载多个种子文件，运行语义去重，返回 (unique_ids, duplicate_ids)。
    供 orchestrator 在扩展前调用，将重复种子 ID 写入过滤清单，避免浪费 LLM token。
    """
    all_qa: list[dict] = []
    for f in seed_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        all_qa.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not all_qa:
        return set(), set()

    logger.info(f"种子前置去重: 加载 {len(all_qa)} 条种子，运行语义去重...")
    unique, duplicates = run_dedup(all_qa)

    unique_ids = {qa.get("id", "") for qa in unique if qa.get("id")}
    duplicate_ids = {qa.get("id", "") for qa in duplicates if qa.get("id")}
    logger.info(f"种子前置去重: 保留 {len(unique_ids)} 条唯一，过滤 {len(duplicate_ids)} 条重复")
    return unique_ids, duplicate_ids


def run_dedup(qa_list: list[dict], threshold: float = 0.92) -> tuple[list[dict], list[dict]]:
    """
    语义去重
    优先尝试用 sentence-transformers 做 embedding 去重
    如果依赖不可用，则退化为基于 n-gram 的快速去重
    """
    logger.info(f"去重: 处理 {len(qa_list)} 条...")

    try:
        return _dedup_embedding(qa_list, threshold)
    except ImportError:
        logger.warning("sentence-transformers 不可用，使用 n-gram 去重")
        return _dedup_ngram(qa_list, threshold=0.7)


def _dedup_ngram(qa_list: list[dict], threshold: float = 0.7) -> tuple[list[dict], list[dict]]:
    """基于字符 n-gram 的快速去重（不需要额外依赖）"""
    def ngrams(text: str, n: int = 3) -> set:
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    unique, duplicates = [], []
    seen_ngrams = []

    for qa in qa_list:
        q_ng = ngrams(qa.get("question", ""))
        is_dup = False
        for existing_ng in seen_ngrams:
            if jaccard(q_ng, existing_ng) > threshold:
                is_dup = True
                break
        if is_dup:
            qa["_qc_issues"] = qa.get("_qc_issues", []) + ["语义重复"]
            duplicates.append(qa)
        else:
            unique.append(qa)
            seen_ngrams.append(q_ng)

    logger.info(f"去重: {len(unique)} 唯一, {len(duplicates)} 重复")
    return unique, duplicates


def _dedup_embedding(qa_list: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """
    基于 embedding 的语义去重。
    使用向量化矩阵乘法替代 O(N²) Python 嵌套循环：
    对每个候选向量，用 BLAS 一次性计算它与所有已知唯一向量的相似度（O(N×K) numpy）。
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    questions = [qa.get("question", "") for qa in qa_list]
    # normalize_embeddings=True → cosine sim = dot product
    logger.info(f"去重(embedding): 计算 {len(questions)} 条问题的向量表示（可能需要数分钟）...")
    # 在线程池中运行 encode，避免阻塞（SentenceTransformer 不释放 GIL，但可隔离）
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.encode, questions, normalize_embeddings=True, show_progress_bar=False)
        embeddings = future.result()
    logger.info(f"去重(embedding): 向量计算完成，开始相似度比对...")

    N = len(embeddings)
    unique_mask = np.zeros(N, dtype=bool)
    LOG_EVERY = max(1000, N // 10)

    for i in range(N):
        if i > 0 and i % LOG_EVERY == 0:
            logger.info(f"去重进度: {i}/{N} ({i/N:.0%})，已找到 {unique_mask.sum()} 条唯一")
        if not unique_mask.any():
            unique_mask[i] = True
            continue
        # 向量化：一次性计算与所有唯一向量的相似度
        # unique_vecs = embeddings[unique_mask]  # (K, D)
        # sims = unique_vecs @ embeddings[i]  # (K,)
        # 改用布尔掩码直接索引，避免每次创建新数组
        sims = embeddings[unique_mask] @ embeddings[i]
        if float(sims.max()) > threshold:
            # duplicate
            continue
        else:
            unique_mask[i] = True

    unique_indices = np.where(unique_mask)[0].tolist()
    duplicate_indices = np.where(~unique_mask)[0].tolist()
    unique = [qa_list[i] for i in unique_indices]
    duplicates = [qa_list[i] for i in duplicate_indices]
    for d in duplicates:
        d["_qc_issues"] = d.get("_qc_issues", []) + ["语义重复"]

    logger.info(f"去重(embedding): {len(unique)} 唯一, {len(duplicates)} 重复")
    return unique, duplicates


def run_dedup_incremental(
    new_qa: list[dict],
    baseline_qa: list[dict],
    threshold: float = 0.92,
) -> tuple[list[dict], list[dict]]:
    """增量去重：检查 new_qa 与 baseline_qa 及 new_qa 内部是否重复。
    baseline_qa 已去重，直接作为"已见"基准，不出现在返回值中。
    """
    if not new_qa:
        return [], []
    logger.info(f"增量去重: {len(new_qa)} 条新 Q&A，基准 {len(baseline_qa)} 条...")
    try:
        return _dedup_embedding_incremental(new_qa, baseline_qa, threshold)
    except ImportError:
        logger.warning("sentence-transformers 不可用，使用 n-gram 增量去重")
        return _dedup_ngram_incremental(new_qa, baseline_qa, threshold=0.7)


def _dedup_ngram_incremental(
    new_qa: list[dict],
    baseline_qa: list[dict],
    threshold: float = 0.7,
) -> tuple[list[dict], list[dict]]:
    def ngrams(text: str, n: int = 3) -> set:
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    seen_ngrams = [ngrams(qa.get("question", "")) for qa in baseline_qa]
    unique, duplicates = [], []
    for qa in new_qa:
        q_ng = ngrams(qa.get("question", ""))
        is_dup = any(jaccard(q_ng, existing) > threshold for existing in seen_ngrams)
        if is_dup:
            qa["_qc_issues"] = qa.get("_qc_issues", []) + ["语义重复"]
            duplicates.append(qa)
        else:
            unique.append(qa)
            seen_ngrams.append(q_ng)
    logger.info(f"增量去重(n-gram): {len(unique)} 唯一, {len(duplicates)} 重复")
    return unique, duplicates


def _dedup_embedding_incremental(
    new_qa: list[dict],
    baseline_qa: list[dict],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    new_questions = [qa.get("question", "") for qa in new_qa]

    logger.info(f"增量去重(embedding): 计算 {len(new_questions)} 条新问题向量...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        new_embs = executor.submit(
            model.encode, new_questions, normalize_embeddings=True, show_progress_bar=False,
        ).result()

    if baseline_qa:
        baseline_questions = [qa.get("question", "") for qa in baseline_qa]
        logger.info(f"增量去重(embedding): 计算 {len(baseline_questions)} 条基准问题向量...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            base_embs = executor.submit(
                model.encode, baseline_questions, normalize_embeddings=True, show_progress_bar=False,
            ).result()
    else:
        base_embs = None

    # 预分配矩阵替代 list.append + np.stack，避免 10 万级时的内存抖动
    max_unique = len(baseline_qa) + len(new_qa)
    dim = new_embs.shape[1]
    unique_matrix = np.zeros((max_unique, dim), dtype=np.float32)
    unique_count = 0

    if base_embs is not None and len(base_embs) > 0:
        unique_matrix[:len(base_embs)] = base_embs
        unique_count = len(base_embs)

    unique, duplicates = [], []
    LOG_EVERY = max(500, len(new_qa) // 10)
    for i, (qa, emb) in enumerate(zip(new_qa, new_embs)):
        if i > 0 and i % LOG_EVERY == 0:
            logger.info(f"增量去重进度: {i}/{len(new_qa)} ({i/len(new_qa):.0%})")
        if unique_count == 0:
            unique.append(qa)
            unique_matrix[0] = emb
            unique_count = 1
            continue
        # 切片计算，不重建矩阵
        sims = unique_matrix[:unique_count] @ emb
        if float(sims.max()) > threshold:
            qa["_qc_issues"] = qa.get("_qc_issues", []) + ["语义重复"]
            duplicates.append(qa)
        else:
            unique.append(qa)
            unique_matrix[unique_count] = emb
            unique_count += 1

    logger.info(f"增量去重(embedding): {len(unique)} 唯一, {len(duplicates)} 重复")
    return unique, duplicates


def run_seed_dedup_incremental_with_cache(
    new_qa: list[dict],
    cached_matrix,
    threshold: float = 0.92,
) -> tuple[list[dict], list[dict], object]:
    """
    增量去重（使用缓存 embedding 矩阵）。
    cached_matrix: np.ndarray (N, dim) 或 None（首次运行）。
    返回 (unique_qa, dup_qa, new_unique_embeddings)。
    new_unique_embeddings 是新增唯一种子的 embedding，供调用方追加到缓存。
    """
    if not new_qa:
        return [], [], None
    try:
        return _dedup_embedding_cached(new_qa, cached_matrix, threshold)
    except ImportError:
        logger.warning("sentence-transformers 不可用，使用 n-gram 增量去重（无 embedding 缓存）")
        unique, dups = _dedup_ngram_incremental(new_qa, [], threshold=0.7)
        return unique, dups, None


def _dedup_embedding_cached(
    new_qa: list[dict],
    cached_matrix,
    threshold: float,
) -> tuple[list[dict], list[dict], object]:
    """只对新种子编码，与缓存矩阵比对，避免重新编码历史数据。"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    new_questions = [qa.get("question", "") for qa in new_qa]

    logger.info(f"增量去重(cached): 编码 {len(new_questions)} 条新问题向量...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        new_embs = executor.submit(
            model.encode, new_questions, normalize_embeddings=True, show_progress_bar=False
        ).result()

    N_cached = len(cached_matrix) if cached_matrix is not None else 0
    dim = new_embs.shape[1]
    max_unique = N_cached + len(new_qa)
    unique_matrix = np.zeros((max_unique, dim), dtype=np.float32)

    if cached_matrix is not None and N_cached > 0:
        unique_matrix[:N_cached] = cached_matrix
    unique_count = N_cached

    unique, duplicates, new_unique_embs = [], [], []
    LOG_EVERY = max(500, len(new_qa) // 10)

    for i, (qa, emb) in enumerate(zip(new_qa, new_embs)):
        if i > 0 and i % LOG_EVERY == 0:
            logger.info(f"增量去重进度: {i}/{len(new_qa)} ({i/len(new_qa):.0%})")
        if unique_count == 0:
            unique.append(qa)
            unique_matrix[0] = emb
            new_unique_embs.append(emb)
            unique_count = 1
            continue
        sims = unique_matrix[:unique_count] @ emb
        if float(sims.max()) > threshold:
            qa["_qc_issues"] = qa.get("_qc_issues", []) + ["语义重复"]
            duplicates.append(qa)
        else:
            unique.append(qa)
            unique_matrix[unique_count] = emb
            new_unique_embs.append(emb)
            unique_count += 1

    result_embs = (
        np.array(new_unique_embs, dtype=np.float32)
        if new_unique_embs
        else np.zeros((0, dim), dtype=np.float32)
    )
    logger.info(f"增量去重(cached): {len(unique)} 唯一, {len(duplicates)} 重复")
    return unique, duplicates, result_embs


# ============================================================
# Stage 3: 事实校验（规则引擎）
# ============================================================
# 保险领域可校验的硬性事实
KNOWN_FACTS = {
    "交强险": {
        "死亡伤残限额_现行": 180000,
        "医疗费用限额_现行": 18000,
        "财产损失限额_现行": 2000,
        "无责死亡伤残限额": 18000,
        "无责医疗费用限额": 1800,
        "无责财产损失限额": 100,
    },
    "等待期": {
        "重疾险_常见": [90, 180],
        "医疗险_常见": [30, 90],
        "寿险_常见": [90, 180],
    },
    "犹豫期": {
        "长期险_最短": 15,
    },
    "宽限期": {
        "最短": 60,
    },
    "复效期": {
        "最长": 730,    # 2年 = 730天
    },
    "不可抗辩期": {
        "期限": 730,    # 2年
    },
    "诉讼时效": {
        "人寿险": 1825,   # 5年
        "其他险": 730,    # 2年
    },
    "理赔核定": {
        "核定时限": 30,     # 天
        "支付时限": 10,     # 天
    },
    "法规": {
        "保险法_如实告知": "第十六条",
        "保险法_不可抗辩": "第十六条",
        "保险法_代位求偿": "第六十条",
        "保险法_索赔时效": "第二十六条",
    }
}


def run_fact_check(qa_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """规则引擎事实校验"""
    passed, flagged = [], []

    def check_one(qa):
        issues = _check_facts(qa)
        if issues:
            qa["_fact_issues"] = issues
            return "flag", qa
        return "pass", qa

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as executor:
        futures = [executor.submit(check_one, qa) for qa in qa_list]
        for future in concurrent.futures.as_completed(futures):
            status, qa = future.result()
            if status == "pass":
                passed.append(qa)
            else:
                flagged.append(qa)

    logger.info(f"事实校验: {len(passed)} 通过, {len(flagged)} 存疑")
    return passed, flagged


def _quick_report(qa_list: list[dict], output_path: Path, config: dict) -> dict:
    """为全跳过场景生成快速报告"""
    coverage_report = run_coverage_analysis(qa_list, config)
    return {
        "qc_time": datetime.now().isoformat(),
        "input_total": len(qa_list),
        "passed_total": len(qa_list),
        "rejected_total": 0,
        "pass_rate": "100.0%",
        "stage_stats": {},
        "coverage": coverage_report,
        "skipped_all": True,
    }


def _parse_yuan(text: str) -> list[int]:
    """
    从文本中提取金额（元为单位），正确处理"万元"单位。
    例：'18万元' → 180000，'2000元' → 2000，'18.5万元' → 185000
    """
    amounts = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(万)?\s*元", text):
        value = float(m.group(1))
        if m.group(2) == "万":
            value *= 10000
        amounts.append(int(value))
    return amounts


def _check_facts(qa: dict) -> list[str]:
    """检查单条 Q&A 中的可校验事实"""
    issues = []
    answer = qa.get("answer", "")

    # 检查交强险限额（含万元单位）
    if "交强险" in answer:
        known_limits = {180000, 18000, 2000, 200000, 20000, 2500, 18, 1800, 100}
        for amount in _parse_yuan(answer):
            if amount > 1000 and amount not in known_limits:
                issues.append(f"交强险相关金额 {amount} 元不在已知限额范围内")

    # 检查等待期天数
    if "等待期" in answer or "观察期" in answer:
        for d in re.findall(r"(\d+)\s*(?:天|日|个自然日)", answer):
            d = int(d)
            if d > 0 and d not in {30, 60, 90, 120, 180, 365}:
                issues.append(f"等待期 {d} 天不是常见取值（常见: 30/90/180天）")

    # 检查犹豫期天数
    if "犹豫期" in answer or "冷静期" in answer:
        for d in re.findall(r"(\d+)\s*(?:天|日)", answer):
            d = int(d)
            if d > 0 and d not in {10, 15, 20}:
                issues.append(f"犹豫期 {d} 天不是常见取值（常见: 15/20天）")

    # 检查宽限期天数（保险法规定最低60天）
    if "宽限期" in answer or "缴费宽限" in answer:
        for d in re.findall(r"(\d+)\s*(?:天|日)", answer):
            d = int(d)
            if 0 < d < 60:
                issues.append(f"宽限期 {d} 天低于法定最低60天")

    # 检查复效期（最长2年）
    if "复效" in answer and "2年" not in answer and "两年" not in answer:
        for y in re.findall(r"(\d+)\s*年", answer):
            y = int(y)
            if y > 2:
                issues.append(f"复效期 {y} 年超过法定最长2年")

    # 检查不可抗辩期（2年）
    if "不可抗辩" in answer:
        for y in re.findall(r"(\d+)\s*年", answer):
            y = int(y)
            if y != 2:
                issues.append(f"不可抗辩期应为2年，答案中出现 {y} 年")

    # 检查理赔核定时限（30天内核定，10天内支付）
    if "理赔核定" in answer or "理赔决定" in answer:
        for d in re.findall(r"(\d+)\s*(?:天|日)内.*核定", answer):
            d = int(d)
            if d > 30:
                issues.append(f"理赔核定时限 {d} 天超过法定30天")

    return issues


# ============================================================
# Stage 4: 覆盖率分析（三维 + 工具路由 + qa_category）
# ============================================================
def run_coverage_analysis(qa_list: list[dict], config: dict) -> dict:
    """分析知识坐标三维覆盖情况 + 工具路由覆盖 + qa_category 分布"""
    schema = config.get("knowledge_schema", {})

    all_types = schema.get("insurance_types", [])
    stages = schema.get("business_stages", [
        "投保咨询", "健康告知", "核保", "承保生效",
        "保全变更", "续保复效", "报案", "理赔材料",
        "理赔审核", "赔付结案", "拒赔争议", "通用",
    ])
    qtypes = schema.get("question_types", [])

    # 三维覆盖：insurance_type × business_stage × question_type
    coverage_3d: dict[tuple, int] = defaultdict(int)
    coverage_2d: dict[tuple, int] = defaultdict(int)  # insurance_type × question_type
    by_insurance: dict[str, int] = defaultdict(int)
    by_stage: dict[str, int] = defaultdict(int)
    by_qtype: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    by_tool: dict[str, int] = defaultdict(int)

    for qa in qa_list:
        ins = qa.get("insurance_type", "未知")
        stage = qa.get("business_stage", "通用")
        qtype = qa.get("question_type", "未知")
        category = qa.get("qa_category", "knowledge")
        tool = qa.get("tool_routing", "")

        coverage_3d[(ins, stage, qtype)] += 1
        coverage_2d[(ins, qtype)] += 1
        by_insurance[ins] += 1
        by_stage[stage] += 1
        by_qtype[qtype] += 1
        by_category[category] += 1
        if tool:
            by_tool[tool] += 1

    # 二维空白/薄弱分析（insurance_type × question_type）
    gaps, weak, strong = [], [], []
    for ins in all_types:
        for qtype in qtypes:
            count = coverage_2d.get((ins, qtype), 0)
            coord = {"insurance_type": ins, "question_type": qtype, "count": count}
            if count == 0:
                gaps.append(coord)
            elif count < 5:
                weak.append(coord)
            elif count > 50:
                strong.append(coord)

    # 工具路由覆盖情况
    required_tools = {"underwriting_check", "premium_calculator", "cash_value_query",
                      "claim_calculator", "account_query"}
    tool_coverage = {tool: by_tool.get(tool, 0) for tool in required_tools}
    missing_tools = [t for t, cnt in tool_coverage.items() if cnt == 0]

    # 产品对比覆盖（白名单）
    from param_extractor import COMPARISON_WHITELIST
    comparison_coverage = {}
    for type_a, type_b, comp_type in COMPARISON_WHITELIST:
        key = f"{type_a} vs {type_b}"
        count = sum(
            1 for qa in qa_list
            if qa.get("qa_category") == "product_comparison"
            and type_a in qa.get("insurance_type", "")
            and type_b in qa.get("insurance_type", "")
        )
        comparison_coverage[key] = count

    report = {
        "total_qa": len(qa_list),
        "coverage_by_insurance_type": dict(by_insurance),
        "coverage_by_business_stage": dict(by_stage),
        "coverage_by_question_type": dict(by_qtype),
        "qa_category_distribution": dict(by_category),
        "tool_routing_coverage": tool_coverage,
        "missing_tools": missing_tools,
        "comparison_whitelist_coverage": comparison_coverage,
        "gaps": gaps[:50],
        "weak_spots": weak[:50],
        "strong_spots": strong[:20],
        "gap_count": len(gaps),
        "weak_count": len(weak),
    }

    logger.info(
        f"覆盖率(三维): 总 {len(qa_list)} 条 | "
        f"空白坐标 {len(gaps)} | 薄弱坐标 {len(weak)} | "
        f"qa_category={dict(by_category)}"
    )
    return report


# ============================================================
# Stage 5: 分层答案回溯校验（P2）
# ============================================================

_CHUNKS_INDEX_CACHE: dict[str, tuple[float, dict[str, str]]] = {}  # path → (mtime, index)
_CHUNKS_META_CACHE: dict[str, tuple[float, tuple[dict[str, dict], dict[str, dict]]]] = {}  # path → (mtime, (by_chunk, by_doc))


def _build_chunks_meta_indexes(chunks_dir: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """构建 chunk_id/doc_file 元数据索引，用于 P2 分池前运行时补齐来源类型。"""
    chunks_file = Path(chunks_dir) / "chunks.jsonl"
    if not chunks_file.exists():
        return {}, {}
    mtime = chunks_file.stat().st_mtime
    cache_key = str(chunks_file)
    if cache_key in _CHUNKS_META_CACHE and _CHUNKS_META_CACHE[cache_key][0] == mtime:
        return _CHUNKS_META_CACHE[cache_key][1]

    by_chunk: dict[str, dict] = {}
    by_doc: dict[str, dict] = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id", "")
            doc_file = chunk.get("doc_file", "")
            meta = {
                "chunk_id": chunk_id,
                "doc_file": doc_file,
                "doc_type": chunk.get("doc_type", "未分类"),
                "insurance_type": chunk.get("insurance_type", "其他保险"),
                "product_name": chunk.get("product_name", "") or "",
                "text": chunk.get("text", ""),
            }
            if chunk_id:
                by_chunk[chunk_id] = meta
            if doc_file and doc_file not in by_doc:
                by_doc[doc_file] = {
                    "doc_file": doc_file,
                    "doc_type": meta["doc_type"],
                    "insurance_type": meta["insurance_type"],
                    "product_name": meta["product_name"],
                }

    _CHUNKS_META_CACHE[cache_key] = (mtime, (by_chunk, by_doc))
    return by_chunk, by_doc


def enrich_p2_source_metadata(qa_list: list[dict], chunks_dir: str, config: dict) -> list[dict]:
    """
    运行时补齐 P2 所需的来源元数据，不回写 qa_passed.jsonl。
    优先按 source_chunk_id 回查，缺失时按 source_doc 回查。
    """
    by_chunk, by_doc = _build_chunks_meta_indexes(chunks_dir)
    qcfg = config.get("quality_check", {})
    product_doc_types = set(qcfg.get("p2_product_doc_types", ["保险条款", "理赔指南", "核保手册", "产品说明书"]))
    regulatory_doc_types = set(qcfg.get("p2_regulatory_doc_types", ["法律法规", "监管文件", "行业标准"]))

    enriched = []
    for qa in qa_list:
        qa = dict(qa)
        meta = {}
        chunk_id = qa.get("source_chunk_id", "")
        source_doc = qa.get("source_doc", "")
        if chunk_id and chunk_id in by_chunk:
            meta = by_chunk[chunk_id]
        elif source_doc and source_doc in by_doc:
            meta = by_doc[source_doc]

        doc_type = meta.get("doc_type") or qa.get("doc_type") or qa.get("source_doc_type") or "未分类"
        qa["source_doc_type"] = doc_type
        if not qa.get("doc_type"):
            qa["doc_type"] = doc_type
        if meta.get("doc_file") and not qa.get("source_doc"):
            qa["source_doc"] = meta["doc_file"]
        if meta.get("product_name") and not qa.get("product_name"):
            qa["product_name"] = meta["product_name"]
        if meta.get("insurance_type") and not qa.get("insurance_type"):
            qa["insurance_type"] = meta["insurance_type"]

        if doc_type in product_doc_types:
            qa["p2_pool"] = "product_docs"
        elif doc_type in regulatory_doc_types:
            qa["p2_pool"] = "regulatory_docs"
        else:
            qa["p2_pool"] = "unknown_docs"
        enriched.append(qa)
    return enriched


def split_p2_pools(qa_list: list[dict], config: dict) -> dict[str, list[dict]]:
    pools = {"product_docs": [], "regulatory_docs": [], "unknown_docs": []}
    unknown_policy = config.get("quality_check", {}).get("p2_unknown_policy", "regulatory")
    for qa in qa_list:
        pool = qa.get("p2_pool", "unknown_docs")
        pools.setdefault(pool, []).append(qa)
        if pool == "unknown_docs" and unknown_policy == "regulatory":
            qa["effective_p2_pool"] = "regulatory_docs"
        else:
            qa["effective_p2_pool"] = pool
    return pools

def _build_chunks_index(qa_list: list[dict], chunks_dir: str) -> dict[str, str]:
    """从 chunks.jsonl 构建 chunk_id → text 索引（按 mtime 缓存，避免重复加载）"""
    chunks_file = Path(chunks_dir) / "chunks.jsonl"
    if not chunks_file.exists():
        return {}
    mtime = chunks_file.stat().st_mtime
    cache_key = str(chunks_file)
    if cache_key in _CHUNKS_INDEX_CACHE and _CHUNKS_INDEX_CACHE[cache_key][0] == mtime:
        return _CHUNKS_INDEX_CACHE[cache_key][1]
    index: dict[str, str] = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                index[chunk.get("chunk_id", "")] = chunk.get("text", "")
    _CHUNKS_INDEX_CACHE[cache_key] = (mtime, index)
    return index


def anchor_verify(qa: dict, chunks_index: dict[str, str]) -> list[str]:
    """
    第一级回溯校验：检查答案中的数字是否出现在来源 chunk 中。
    工具路由型条目跳过（答案是模板，无需核验）。
    """
    if qa.get("is_tool_routed"):
        return []
    issues = []
    answer = qa.get("answer", "")
    chunk_id = qa.get("source_chunk_id", "")
    if not chunk_id or chunk_id not in chunks_index:
        return []
    source_text = chunks_index[chunk_id]
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(万)?\s*(元|天|日|年|%|个月)", answer):
        val_str = m.group(0).replace(" ", "")
        if val_str not in source_text and m.group(0) not in source_text:
            issues.append(f"'{val_str}' 未在来源 chunk 中找到")
    return issues


def regulatory_anchor_verify(qa: dict, chunks_index: dict[str, str]) -> list[str]:
    """
    规范性文档校验：只核查法条编号、明确期限/金额与来源 chunk 的显式一致性。
    不对抽象归纳或解释性总结做产品型数字锚定误报。
    """
    if qa.get("is_tool_routed"):
        return []
    issues = []
    chunk_id = qa.get("source_chunk_id", "")
    if not chunk_id or chunk_id not in chunks_index:
        return []

    answer = qa.get("answer", "")
    source_text = chunks_index[chunk_id]

    article_refs = set(re.findall(r"第[一二三四五六七八九十百千万\d]+条", answer))
    for ref in article_refs:
        if ref not in source_text:
            issues.append(f"法条引用 '{ref}' 未在来源 chunk 中找到")

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(万)?\s*(元|天|日|年|%|个月)", answer):
        value = m.group(0).replace(" ", "")
        window_start = max(0, m.start() - 10)
        window = answer[window_start:m.end() + 10]
        if any(token in window for token in ("第", "法", "条例", "规定", "办法", "期限", "时限", "限额")):
            if value not in source_text and m.group(0) not in source_text:
                issues.append(f"规范性文档关键值 '{value}' 未在来源 chunk 中找到")
    return issues


def _is_high_risk_qa(qa: dict, anchor_issues: list[str]) -> bool:
    """判断是否为高风险条目，需要进行 LLM 二级校验"""
    if anchor_issues:
        return True
    answer = qa.get("answer", "")
    num_count = len(re.findall(r"\d+(?:\.\d+)?(?:\s*(?:万|元|天|%|年))", answer))
    if num_count >= 3:
        return True
    gen_method = qa.get("generation_method", "")
    if "followup" in gen_method and qa.get("parent_id"):
        chain_depth = qa.get("chain_depth", 0)
        if chain_depth >= 3:
            return True
    return False


def _is_high_risk_regulatory_qa(qa: dict, anchor_issues: list[str]) -> bool:
    """规范性文档仅在条文号、金额/期限冲突或引用缺失时进入二级校验。"""
    if anchor_issues:
        return True
    answer = qa.get("answer", "")
    if re.search(r"第[一二三四五六七八九十百千万\d]+条", answer) and not qa.get("source_chunk_id"):
        return True
    return False


async def _llm_verify_one(qa: dict, chunks_index: dict[str, str], client, template: str, sem: asyncio.Semaphore) -> dict:
    """对单条高风险 Q&A 做 LLM 二级校验"""
    chunk_id = qa.get("source_chunk_id", "")
    source_text = chunks_index.get(chunk_id, "")
    # If source doc is unavailable, flag for manual review instead of auto-passing
    if not source_text:
        return {
            "qa_id": qa.get("id", ""),
            "verdict": "flag",
            "issues": ["来源文档不可用，无法核验答案准确性，需人工复核"],
            "hallucination_risk": "unknown",
        }
    async with sem:
        prompt = template.format(
            source_text=source_text[:2000],
            question=qa.get("question", ""),
            answer=qa.get("answer", ""),
        )
        try:
            from llm_client import create_client as _create_client
            response = await client.acall(prompt)
            response = re.sub(r"```(?:json)?\s*", "", response).strip()
            result = json.loads(response)
            return {"qa_id": qa.get("id", ""), "verdict": result.get("verdict", "pass"),
                    "issues": result.get("issues", []), "hallucination_risk": result.get("hallucination_risk", "low")}
        except Exception as e:
            return {"qa_id": qa.get("id", ""), "verdict": "pass", "issues": [], "error": str(e)}


def run_anchor_verify(
    qa_list: list[dict],
    chunks_dir: str,
    config: dict,
    llm_verify_rate: float = 0.05,
    checkpoint_file: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    分层答案回溯校验：
    Level 1（零成本）: 锚点数字校验
    Level 2（并发 LLM，~5%）: 高风险条目 LLM 核验，支持断点续传
    返回 (通过列表, 存疑列表)
    """
    chunks_index = _build_chunks_index(qa_list, chunks_dir)
    if not chunks_index:
        logger.info("未找到 chunks 索引，跳过锚点校验")
        return qa_list, []

    level1_issues: dict[str, list[str]] = {}
    for qa in qa_list:
        issues = anchor_verify(qa, chunks_index)
        if issues:
            level1_issues[qa.get("id", "")] = issues

    logger.info(f"锚点校验: {len(level1_issues)} 条发现数字不一致")

    # 识别高风险条目（Level 1 问题 + 其他判断）
    high_risk = [qa for qa in qa_list if _is_high_risk_qa(qa, level1_issues.get(qa.get("id", ""), []))]
    sample_size = max(1, int(len(qa_list) * llm_verify_rate))
    llm_verify_targets = high_risk[:sample_size]

    logger.info(f"LLM 二级校验: {len(llm_verify_targets)}/{len(qa_list)} 条（高风险抽样）")

    # 加载断点：已完成的校验结果
    llm_results: dict[str, dict] = {}
    ckpt_path = Path(checkpoint_file) if checkpoint_file else None
    if ckpt_path and ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        logger.info(f"断点续传: 已加载 {len(llm_results)} 条 Level-2 校验结果")

    pending = [qa for qa in llm_verify_targets if qa.get("id", "") not in llm_results]

    # LLM 二级校验（异步并发）
    if pending:
        try:
            import asyncio as _asyncio
            from llm_client import create_client as _create_client
            verify_template = (Path(__file__).parent.parent / "prompts" / "verify_answer.txt").read_text(encoding="utf-8")
            client = _create_client(config, "quality_check")
            concurrency = config.get("quality_check", {}).get("verify_concurrency", 10)
            sem = _asyncio.Semaphore(concurrency)

            async def _run_llm_verify():
                async def _verify_and_save(qa: dict) -> dict:
                    r = await _llm_verify_one(qa, chunks_index, client, verify_template, sem)
                    llm_results[r["qa_id"]] = r
                    if ckpt_path:
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(ckpt_path, "w", encoding="utf-8") as f:
                            json.dump(llm_results, f, ensure_ascii=False)
                    return r

                tasks = [_verify_and_save(qa) for qa in pending]
                return await _asyncio.gather(*tasks)

            asyncio.run(_run_llm_verify())
        except Exception as e:
            logger.warning(f"LLM 二级校验失败: {e}")

    passed, flagged = [], []
    for qa in qa_list:
        qa_id = qa.get("id", "")
        l1_issues = level1_issues.get(qa_id, [])
        llm_result = llm_results.get(qa_id, {})
        llm_issues = llm_result.get("issues", []) if llm_result.get("verdict") == "flag" else []
        all_issues = l1_issues + llm_issues
        if all_issues:
            qa["_verify_issues"] = all_issues
            flagged.append(qa)
        else:
            passed.append(qa)

    logger.info(f"回溯校验: {len(passed)} 通过, {len(flagged)} 存疑")
    return passed, flagged


def run_regulatory_anchor_verify(
    qa_list: list[dict],
    chunks_dir: str,
    config: dict,
    llm_verify_rate: float = 0.01,
    checkpoint_file: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    规范性文档的简化 P2：
    Level 1: 法条号/明确期限金额校验
    Level 2: 仅对冲突或引用缺失的条目做小样本 LLM 校验
    """
    chunks_index = _build_chunks_index(qa_list, chunks_dir)
    if not chunks_index:
        logger.info("未找到 chunks 索引，跳过规范性文档回溯校验")
        return qa_list, []

    level1_issues: dict[str, list[str]] = {}
    for qa in qa_list:
        issues = regulatory_anchor_verify(qa, chunks_index)
        if issues:
            level1_issues[qa.get("id", "")] = issues

    logger.info(f"规范性文档锚点校验: {len(level1_issues)} 条发现条文/期限/金额不一致")

    high_risk = [qa for qa in qa_list if _is_high_risk_regulatory_qa(qa, level1_issues.get(qa.get('id', ''), []))]
    sample_size = min(len(high_risk), max(1, int(len(qa_list) * llm_verify_rate))) if qa_list else 0
    llm_verify_targets = high_risk[:sample_size]
    logger.info(f"规范性文档 LLM 二级校验: {len(llm_verify_targets)}/{len(qa_list)} 条")

    llm_results: dict[str, dict] = {}
    ckpt_path = Path(checkpoint_file) if checkpoint_file else None
    if ckpt_path and ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        logger.info(f"断点续传: 已加载 {len(llm_results)} 条规范性文档 Level-2 校验结果")

    pending = [qa for qa in llm_verify_targets if qa.get("id", "") not in llm_results]
    if pending:
        try:
            import asyncio as _asyncio
            from llm_client import create_client as _create_client
            verify_template = (Path(__file__).parent.parent / "prompts" / "verify_answer_regulatory.txt").read_text(encoding="utf-8")
            client = _create_client(config, "quality_check")
            concurrency = config.get("quality_check", {}).get("verify_concurrency", 10)
            sem = _asyncio.Semaphore(concurrency)

            async def _run_llm_verify():
                async def _verify_and_save(qa: dict) -> dict:
                    r = await _llm_verify_one(qa, chunks_index, client, verify_template, sem)
                    llm_results[r["qa_id"]] = r
                    if ckpt_path:
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(ckpt_path, "w", encoding="utf-8") as f:
                            json.dump(llm_results, f, ensure_ascii=False)
                    return r

                tasks = [_verify_and_save(qa) for qa in pending]
                return await _asyncio.gather(*tasks)

            asyncio.run(_run_llm_verify())
        except Exception as e:
            logger.warning(f"规范性文档 LLM 二级校验失败: {e}")

    passed, flagged = [], []
    for qa in qa_list:
        qa_id = qa.get("id", "")
        l1_issues = level1_issues.get(qa_id, [])
        llm_result = llm_results.get(qa_id, {})
        llm_issues = llm_result.get("issues", []) if llm_result.get("verdict") == "flag" else []
        all_issues = l1_issues + llm_issues
        if all_issues:
            qa["_verify_issues"] = all_issues
            flagged.append(qa)
        else:
            passed.append(qa)

    logger.info(f"规范性文档回溯校验: {len(passed)} 通过, {len(flagged)} 存疑")
    return passed, flagged


# ============================================================
# Stage 6: LLM 自动评分（P2，抽样层）
# ============================================================
SCORING_PROFILES = {
    "product": {
        "prompt_file": "score_product_qa.txt",
        "dimensions": ["accuracy", "completeness", "practicality", "professionalism", "clarity", "overall"],
    },
    "regulatory": {
        "prompt_file": "score_regulatory_qa.txt",
        "dimensions": ["rule_accuracy", "citation_appropriateness", "overreach_risk", "clarity", "kb_fitness", "overall"],
    },
}


def run_llm_scoring(
    qa_list: list[dict],
    config: dict,
    sample_rate: float = 0.05,
    checkpoint_file: str | None = None,
    scoring_profile: str = "product",
) -> dict:
    """对质检通过的 Q&A 抽样进行 LLM 五维评分（异步并发 + 断点续传）"""
    import random

    profile = SCORING_PROFILES.get(scoring_profile, SCORING_PROFILES["product"])
    prompt_template = (Path(__file__).parent.parent / "prompts" / profile["prompt_file"]).read_text(encoding="utf-8")

    # 跨池一致性校验：product 池中有 product_name 为空的条目，或 regulatory 池中有 product_name 的条目，记录警告
    if scoring_profile == "product":
        no_product = [qa for qa in qa_list if not qa.get("product_name")]
        if no_product:
            logger.warning(
                f"跨池警告: product 评分池中有 {len(no_product)} 条 product_name 为空的条目，"
                "可能来自监管类文档，建议检查 enrich_p2_source_metadata 分池逻辑"
            )
    elif scoring_profile == "regulatory":
        has_product = [qa for qa in qa_list if qa.get("product_name")]
        if has_product:
            logger.warning(
                f"跨池警告: regulatory 评分池中有 {len(has_product)} 条携带 product_name 的条目，"
                f"示例: {has_product[0].get('product_name', '')} ({has_product[0].get('source_doc', '')})"
            )

    sample_size = max(1, int(len(qa_list) * sample_rate))
    sample = random.sample(qa_list, min(sample_size, len(qa_list)))
    logger.info(f"LLM 自动评分: 从 {len(qa_list)} 条中抽取 {len(sample)} 条评分")

    # 加载断点
    ckpt_path = Path(checkpoint_file) if checkpoint_file else None
    scores_by_id: dict[str, dict] = {}
    if ckpt_path and ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            scores_by_id = json.load(f)
        logger.info(f"断点续传: 已加载 {len(scores_by_id)} 条评分结果")

    pending = [qa for qa in sample if qa.get("id", "") not in scores_by_id]

    if pending:
        from llm_client import create_client
        client = create_client(config, "quality_check")
        concurrency = config.get("quality_check", {}).get("scoring_concurrency", 10)

        async def _score_all():
            sem = asyncio.Semaphore(concurrency)

            async def _score_one(qa: dict):
                async with sem:
                    try:
                        prompt = prompt_template.format(
                            question=qa.get("question", ""),
                            answer=qa.get("answer", ""),
                            insurance_type=qa.get("insurance_type", ""),
                            question_type=qa.get("question_type", ""),
                        )
                        response = await client.acall(prompt)
                        response = re.sub(r"```(?:json)?\s*", "", response).strip()
                        result = json.loads(response)
                        result["qa_id"] = qa.get("id", "")
                        scores_by_id[result["qa_id"]] = result
                        if ckpt_path:
                            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(ckpt_path, "w", encoding="utf-8") as f:
                                json.dump(scores_by_id, f, ensure_ascii=False)
                    except Exception as e:
                        logger.warning(f"评分失败 ({qa.get('id', '')}): {e}")

            await asyncio.gather(*[_score_one(qa) for qa in pending])

        asyncio.run(_score_all())

    scores = list(scores_by_id.values())
    if not scores:
        return {"sample_size": 0, "avg_scores": {}}

    dims = profile["dimensions"]
    avg_scores = {}
    for dim in dims:
        vals = [s[dim] for s in scores if isinstance(s.get(dim), (int, float))]
        avg_scores[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

    report = {
        "scoring_profile": scoring_profile,
        "sample_size": len(scores),
        "total_qa": len(qa_list),
        "sample_rate": f"{len(scores)/len(qa_list):.1%}",
        "avg_scores": avg_scores,
        "low_quality_count": sum(1 for s in scores if s.get("overall", 5) < 3),
        "scores": scores,
    }

    logger.info(f"LLM 评分完成: 均分 {avg_scores.get('overall', 0):.2f}/5.0")
    return report


# ============================================================
# 主流程
# ============================================================
def run_full_qc(
    input_files: list[str],
    output_dir: str,
    config: dict,
    stages: str = "format,dedup,fact,coverage",
    progress_callback=None,
) -> dict:
    """增量质检管道：只对新增/变化的文件做 format/dedup/fact，已通过的数据不重复处理。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载已质检文件清单（filepath → mtime）
    qc_manifest_file = output_path / "qc_manifest.json"
    qc_manifest: dict[str, str] = {}
    if qc_manifest_file.exists():
        with open(qc_manifest_file, "r", encoding="utf-8") as f:
            qc_manifest = json.load(f)

    # 找出新增或有变化的文件
    pending_files = []
    for fp in input_files:
        fpath = Path(fp)
        if not fpath.exists():
            continue
        mtime = str(fpath.stat().st_mtime)
        if qc_manifest.get(str(fpath)) == mtime:
            logger.info(f"跳过（已质检无变化）: {fpath.name}")
        else:
            pending_files.append(fp)

    # 加载已通过质检的历史数据（作为去重基准，不再重新校验）
    passed_file = output_path / "qa_passed.jsonl"
    existing_passed: list[dict] = []
    if passed_file.exists():
        with open(passed_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_passed.append(json.loads(line))

    if not pending_files:
        if existing_passed:
            logger.info(f"所有文件已质检且无变化，当前累计通过 {len(existing_passed)} 条")
            return _quick_report(existing_passed, output_path, config)
        logger.info("没有待质检的文件")
        return {"input_total": 0, "passed_total": 0, "rejected_total": 0}

    # 仅加载新文件的条目
    new_qa: list[dict] = []
    for fp in pending_files:
        before = len(new_qa)
        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    new_qa.append(json.loads(line))
        logger.info(f"  载入 {Path(fp).name}: +{len(new_qa) - before} 条")

    logger.info(
        f"加载完成: 新增 {len(new_qa)} 条（{len(pending_files)} 个文件），"
        f"已有历史 {len(existing_passed)} 条，开始增量质检"
    )

    stages_list = [s.strip() for s in stages.split(",")]
    current = new_qa
    new_rejected: list[dict] = []
    stage_stats = {}
    _stage_weights = {"format": 25, "dedup": 50, "fact": 75, "coverage": 90}
    _progress_done = 0

    def _report_stage(name: str):
        nonlocal _progress_done
        _progress_done = _stage_weights.get(name, _progress_done)
        if progress_callback:
            progress_callback(_progress_done, 100)

    # Stage 1: 格式校验（仅对新条目）
    if "format" in stages_list:
        current, rejected = run_format_check(current)
        new_rejected.extend(rejected)
        stage_stats["format"] = {"passed": len(current), "rejected": len(rejected)}
        _report_stage("format")

    # Stage 2: 增量去重（新条目 vs 历史基准 + 新条目内部）
    if "dedup" in stages_list:
        current, rejected = run_dedup_incremental(current, existing_passed)
        new_rejected.extend(rejected)
        stage_stats["dedup"] = {"passed": len(current), "rejected": len(rejected)}
        _report_stage("dedup")

    # Stage 3: 事实校验（仅对新条目）
    if "fact" in stages_list:
        current, flagged = run_fact_check(current)
        new_rejected.extend(flagged)
        stage_stats["fact"] = {"passed": len(current), "flagged": len(flagged)}
        _report_stage("fact")

    # 将新通过的条目追加写入 qa_passed.jsonl
    with open(passed_file, "a", encoding="utf-8") as f:
        for qa in current:
            qa.pop("_qc_issues", None)
            qa.pop("_fact_issues", None)
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    all_passed = existing_passed + current

    # Stage 4: 覆盖率分析（基于全量通过数据）
    coverage_report = {}
    if "coverage" in stages_list:
        coverage_report = run_coverage_analysis(all_passed, config)
        stage_stats["coverage"] = {
            "gaps": coverage_report.get("gap_count", 0),
            "weak": coverage_report.get("weak_count", 0),
        }
        _report_stage("coverage")

    # 更新质检清单
    for fp in pending_files:
        fpath = Path(fp)
        qc_manifest[str(fpath)] = str(fpath.stat().st_mtime)
    with open(qc_manifest_file, "w", encoding="utf-8") as f:
        json.dump(qc_manifest, f, ensure_ascii=False, indent=2)

    # 将本次被拒绝的条目追加写入 qa_rejected.jsonl
    rejected_file = output_path / "qa_rejected.jsonl"
    with open(rejected_file, "a", encoding="utf-8") as f:
        for qa in new_rejected:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    input_total = len(existing_passed) + len(new_qa)
    report = {
        "qc_time": datetime.now().isoformat(),
        "input_total": input_total,
        "new_input": len(new_qa),
        "passed_total": len(all_passed),
        "passed_new": len(current),
        "rejected_total": len(new_rejected),
        "pass_rate": f"{len(all_passed)/max(input_total, 1):.1%}",
        "stage_stats": stage_stats,
        "coverage": coverage_report,
        "output_files": {
            "passed": str(passed_file),
            "rejected": str(rejected_file),
        },
    }

    report_file = output_path / "qc_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info("增量质检完成")
    logger.info(f"  本次新增输入: {len(new_qa)} 条")
    logger.info(f"  本次新增通过: {len(current)} 条")
    logger.info(f"  本次新增拒绝: {len(new_rejected)} 条")
    logger.info(f"  累计通过: {len(all_passed)} 条 ({len(all_passed)/max(input_total, 1):.1%})")
    logger.info(f"  报告: {report_file}")
    logger.info("=" * 50)

    return report


def main():
    parser = argparse.ArgumentParser(description="质量检查管道")
    parser.add_argument("--input", nargs="+", required=True, help="输入文件(可多个)")
    parser.add_argument("--output", default="./data/qc_results")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--stages", default="format,dedup,fact,coverage",
                       help="执行的质检环节，逗号分隔")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_full_qc(args.input, args.output, config, args.stages)


if __name__ == "__main__":
    main()
