"""
quality_checker.py
质量检查管道：格式校验 → 语义去重 → 事实校验 → 覆盖率分析

使用方式：
    python src/quality_checker.py --input ./data/expanded/*.jsonl
    python src/quality_checker.py --input ./data/seeds/seed_xxx.jsonl --stages format,dedup
"""

import json
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
# Stage 1: 格式校验
# ============================================================
VALID_DIFFICULTIES = {"入门", "进阶", "专业"}
VALID_QUESTION_TYPES = {
    "概念解释", "流程指引", "条款解读", "边界判断",
    "计算说明", "对比区分", "注意事项", "案例分析",
    "法规引用", "争议处理", "赔付判断", "产品对比",
    "核保影响", "操作指引",
}


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
    """批量格式校验，返回 (通过列表, 失败列表)"""
    passed, failed = [], []

    def check_one(qa):
        result = validate_format(qa)
        if result["passed"]:
            return "pass", qa
        else:
            qa["_qc_issues"] = result["issues"]
            return "fail", qa

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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
        unique_embs = list(base_embs)
    else:
        unique_embs = []

    unique, duplicates = [], []
    LOG_EVERY = max(500, len(new_qa) // 10)
    for i, (qa, emb) in enumerate(zip(new_qa, new_embs)):
        if i > 0 and i % LOG_EVERY == 0:
            logger.info(f"增量去重进度: {i}/{len(new_qa)} ({i/len(new_qa):.0%})")
        if not unique_embs:
            unique.append(qa)
            unique_embs.append(emb)
            continue
        sims = np.stack(unique_embs) @ emb
        if float(sims.max()) > threshold:
            qa["_qc_issues"] = qa.get("_qc_issues", []) + ["语义重复"]
            duplicates.append(qa)
        else:
            unique.append(qa)
            unique_embs.append(emb)

    logger.info(f"增量去重(embedding): {len(unique)} 唯一, {len(duplicates)} 重复")
    return unique, duplicates


# ============================================================
# Stage 3: 事实校验（规则引擎）
# ============================================================
# 保险领域可校验的硬性事实
KNOWN_FACTS = {
    "交强险": {
        "死亡伤残限额": 180000,
        "医疗费用限额": 18000,
        "财产损失限额": 2000,
    },
    "等待期": {
        "重疾险_常见": [90, 180],        # 天
        "医疗险_常见": [30, 90],
        "寿险_常见": [90, 180],
    },
    "犹豫期": {
        "长期险_最短": 15,                # 天
    },
    "法规": {
        "保险法_如实告知": "第十六条",
        "保险法_不可抗辩": "第十六条第三款",
        "保险法_代位求偿": "第六十条",
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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
        known_limits = {180000, 18000, 2000, 200000, 20000, 2500}  # 现行 + 旧版
        for amount in _parse_yuan(answer):
            if amount > 1000 and amount not in known_limits:
                issues.append(f"交强险相关金额 {amount} 元不在已知限额范围内")

    # 检查等待期天数
    if "等待期" in answer:
        for d in re.findall(r"(\d+)\s*(?:天|日|个自然日)", answer):
            d = int(d)
            if d > 0 and d not in {30, 60, 90, 180, 365}:
                issues.append(f"等待期 {d} 天不是常见取值")

    # 检查犹豫期天数
    if "犹豫期" in answer:
        for d in re.findall(r"(\d+)\s*(?:天|日)", answer):
            d = int(d)
            if d > 0 and d not in {10, 15, 20}:
                issues.append(f"犹豫期 {d} 天不是常见取值")

    return issues


# ============================================================
# Stage 4: 覆盖率分析
# ============================================================
def run_coverage_analysis(qa_list: list[dict], config: dict) -> dict:
    """分析知识坐标覆盖情况"""
    schema = config.get("knowledge_schema", {})

    all_types = schema.get("insurance_types", [])

    stages = schema.get("business_stages", [])
    qtypes = schema.get("question_types", [])

    # 统计
    coverage = defaultdict(int)
    for qa in qa_list:
        ins = qa.get("insurance_type", "未知")
        qtype = qa.get("question_type", "未知")
        coverage[(ins, qtype)] += 1

    # 生成报告
    gaps = []
    weak = []
    strong = []

    for ins in all_types:
        for qtype in qtypes:
            count = coverage.get((ins, qtype), 0)
            coord = {"insurance_type": ins, "question_type": qtype, "count": count}
            if count == 0:
                gaps.append(coord)
            elif count < 5:
                weak.append(coord)
            elif count > 50:
                strong.append(coord)

    # 按险种聚合
    by_insurance = defaultdict(int)
    for qa in qa_list:
        by_insurance[qa.get("insurance_type", "未知")] += 1

    # 按问题类型聚合
    by_qtype = defaultdict(int)
    for qa in qa_list:
        by_qtype[qa.get("question_type", "未知")] += 1

    report = {
        "total_qa": len(qa_list),
        "coverage_by_insurance_type": dict(by_insurance),
        "coverage_by_question_type": dict(by_qtype),
        "gaps": gaps[:50],      # 空白坐标（最多显示 50 个）
        "weak_spots": weak[:50],
        "strong_spots": strong[:20],
        "gap_count": len(gaps),
        "weak_count": len(weak),
    }

    logger.info(
        f"覆盖率: 总 {len(qa_list)} 条 | "
        f"空白坐标 {len(gaps)} | 薄弱坐标 {len(weak)}"
    )
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
