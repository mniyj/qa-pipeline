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
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import yaml

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
    for qa in qa_list:
        result = validate_format(qa)
        if result["passed"]:
            passed.append(qa)
        else:
            qa["_qc_issues"] = result["issues"]
            failed.append(qa)

    logger.info(f"格式校验: {len(passed)} 通过, {len(failed)} 失败")
    return passed, failed


# ============================================================
# Stage 2: 语义去重（基于文本相似度）
# ============================================================
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

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    questions = [qa.get("question", "") for qa in qa_list]
    # normalize_embeddings=True → cosine sim = dot product
    embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    unique_indices: list[int] = []
    duplicate_indices: list[int] = []

    for i in range(len(embeddings)):
        if not unique_indices:
            unique_indices.append(i)
            continue
        # 一次 BLAS 矩阵-向量乘法替代 Python for-loop
        unique_vecs = embeddings[unique_indices]          # (K, D) numpy slice
        sims = unique_vecs @ embeddings[i]               # (K,) 向量化点积
        if float(sims.max()) > threshold:
            duplicate_indices.append(i)
        else:
            unique_indices.append(i)

    unique = [qa_list[i] for i in unique_indices]
    duplicates = [qa_list[i] for i in duplicate_indices]
    for d in duplicates:
        d["_qc_issues"] = d.get("_qc_issues", []) + ["语义重复"]

    logger.info(f"去重(embedding): {len(unique)} 唯一, {len(duplicates)} 重复")
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

    for qa in qa_list:
        issues = _check_facts(qa)
        if issues:
            qa["_fact_issues"] = issues
            flagged.append(qa)
        else:
            passed.append(qa)

    logger.info(f"事实校验: {len(passed)} 通过, {len(flagged)} 存疑")
    return passed, flagged


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

    # 收集所有险种（展平嵌套结构）
    all_types = []
    for category, subtypes in schema.get("insurance_types", {}).items():
        if isinstance(subtypes, list):
            all_types.extend(subtypes)
        else:
            all_types.append(category)

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
) -> dict:
    """运行完整的质检管道"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载所有 Q&A
    all_qa = []
    for f in input_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_qa.append(json.loads(line))

    logger.info(f"加载 {len(all_qa)} 条 Q&A，开始质检")
    stages_list = [s.strip() for s in stages.split(",")]

    current = all_qa
    all_rejected = []
    stage_stats = {}

    # Stage 1: 格式校验
    if "format" in stages_list:
        current, rejected = run_format_check(current)
        all_rejected.extend(rejected)
        stage_stats["format"] = {"passed": len(current), "rejected": len(rejected)}

    # Stage 2: 去重
    if "dedup" in stages_list:
        current, rejected = run_dedup(current)
        all_rejected.extend(rejected)
        stage_stats["dedup"] = {"passed": len(current), "rejected": len(rejected)}

    # Stage 3: 事实校验
    if "fact" in stages_list:
        current, flagged = run_fact_check(current)
        all_rejected.extend(flagged)
        stage_stats["fact"] = {"passed": len(current), "flagged": len(flagged)}

    # Stage 4: 覆盖率分析
    coverage_report = {}
    if "coverage" in stages_list:
        coverage_report = run_coverage_analysis(current, config)
        stage_stats["coverage"] = {
            "gaps": coverage_report.get("gap_count", 0),
            "weak": coverage_report.get("weak_count", 0),
        }

    # 保存通过质检的 Q&A
    passed_file = output_path / "qa_passed.jsonl"
    with open(passed_file, "w", encoding="utf-8") as f:
        for qa in current:
            # 清除内部标记
            qa.pop("_qc_issues", None)
            qa.pop("_fact_issues", None)
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # 保存被拒绝的
    rejected_file = output_path / "qa_rejected.jsonl"
    with open(rejected_file, "w", encoding="utf-8") as f:
        for qa in all_rejected:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # 综合报告
    report = {
        "qc_time": datetime.now().isoformat(),
        "input_total": len(all_qa),
        "passed_total": len(current),
        "rejected_total": len(all_rejected),
        "pass_rate": f"{len(current)/max(len(all_qa),1):.1%}",
        "stage_stats": stage_stats,
        "coverage": coverage_report,
        "output_files": {
            "passed": str(passed_file),
            "rejected": str(rejected_file),
        }
    }

    report_file = output_path / "qc_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info("质检完成")
    logger.info(f"  输入: {len(all_qa)} 条")
    logger.info(f"  通过: {len(current)} 条 ({len(current)/max(len(all_qa),1):.1%})")
    logger.info(f"  拒绝: {len(all_rejected)} 条")
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
