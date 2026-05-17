"""
orchestrator.py
主调度器：串联 预处理 → 种子生成 → 批量扩展 → 质检 全流程

使用方式：
    # 全流程运行
    python src/orchestrator.py run

    # 只跑预处理
    python src/orchestrator.py preprocess

    # 只跑种子生成（先试跑 5 个 chunk）
    python src/orchestrator.py seed --limit 5

    # 只跑扩展
    python src/orchestrator.py expand --seed-file ./data/seeds/seed_xxx.jsonl

    # 只跑质检
    python src/orchestrator.py qc --input ./data/expanded/*.jsonl

    # 查看当前状态
    python src/orchestrator.py status
"""

import argparse
import json
import logging
import glob
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from doc_preprocessor import process_all_documents
from seed_generator import generate_seeds_from_chunks
from expander import (
    expand_rephrase, expand_followup, expand_template, load_seeds,
    expand_tool_routing, expand_policy_service,
    expand_product_comparison, expand_misconception_batch,
    expand_tool_routing_rephrase, expand_clarification_refusal,
)
from quality_checker import (
    run_full_qc, run_anchor_verify, run_regulatory_anchor_verify, run_llm_scoring,
    enrich_p2_source_metadata, split_p2_pools,
)
from param_extractor import extract_comparison_profiles
from llm_client import usage_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path: str = "./config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Step 1: 文档预处理
# ============================================================
def cmd_preprocess(config: dict, progress_callback=None):
    paths = config["storage"]["paths"]
    report = process_all_documents(
        paths["raw_documents"],
        paths["chunks"],
        config,
        progress_callback=progress_callback,
    )
    return report


# ============================================================
# Step 2: 种子生成
# ============================================================
def cmd_seed(config: dict, limit: int = None, pairs: int = 8, progress_callback=None):
    paths = config["storage"]["paths"]
    chunks_file = Path(paths["chunks"]) / "chunks.jsonl"

    if not chunks_file.exists():
        logger.error(f"chunks 文件不存在: {chunks_file}，请先运行 preprocess")
        return

    report = generate_seeds_from_chunks(
        str(chunks_file),
        paths["seeds"],
        config,
        limit=limit,
        pairs_per_chunk=pairs,
        progress_callback=progress_callback,
    )
    return report


# ============================================================
# Step 2b: 种子前置去重（在扩展前消除重复，节省 LLM token）
# ============================================================
def cmd_seed_dedup(config: dict, progress_callback=None) -> dict:
    """
    对所有种子文件做语义去重（embedding，无 LLM 调用），
    将重复种子 ID 写入 expanded/seed_dedup_filter.json。
    cmd_expand() 读取此文件跳过重复种子，避免浪费 token。
    """
    paths = config["storage"]["paths"]
    from quality_checker import run_seed_dedup_filter

    def _progress(pct: int):
        if progress_callback:
            progress_callback(pct, 100)

    seed_files = sorted(glob.glob(f"{paths['seeds']}/*.jsonl"))
    seed_files = [f for f in seed_files if not f.endswith("_report.json")]
    if not seed_files:
        logger.warning("没有种子文件可去重，跳过")
        return {}

    expanded_dir = Path(paths["expanded"])
    expanded_dir.mkdir(parents=True, exist_ok=True)

    # 检查种子文件自上次去重后是否有变化
    state_file  = expanded_dir / "seed_dedup_state.json"
    filter_file = expanded_dir / "seed_dedup_filter.json"
    current_state = {str(Path(f).resolve()): str(Path(f).stat().st_mtime) for f in seed_files}
    if state_file.exists() and filter_file.exists():
        with open(state_file, encoding="utf-8") as f:
            prev_state = json.load(f)
        if prev_state == current_state:
            with open(filter_file, encoding="utf-8") as f:
                skip_ids = json.load(f)
            logger.info(f"种子前置去重: 种子文件无变化，复用已有过滤清单（{len(skip_ids)} 条重复）")
            return {"skipped": True, "duplicate_count": len(skip_ids)}

    _progress(20)  # 文件状态检查完成
    unique_ids, duplicate_ids = run_seed_dedup_filter(seed_files)
    _progress(80)  # embedding 计算完成

    with open(filter_file, "w", encoding="utf-8") as f:
        json.dump(list(duplicate_ids), f, ensure_ascii=False)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(current_state, f, ensure_ascii=False, indent=2)

    logger.info(
        f"种子前置去重完成: 共 {len(unique_ids)+len(duplicate_ids)} 条 → "
        f"保留 {len(unique_ids)} 条，跳过 {len(duplicate_ids)} 条重复"
    )
    return {
        "total": len(unique_ids) + len(duplicate_ids),
        "unique": len(unique_ids),
        "duplicates": len(duplicate_ids),
    }


# ============================================================
# Step 3: 批量扩展
# ============================================================
def _stop_requested() -> bool:
    try:
        from src.pipeline_state import pipeline_state
        return pipeline_state.is_stop_requested()
    except Exception:
        return False


def cmd_expand(config: dict, seed_file: str = None, limit: int = None, progress_callback=None):
    paths = config["storage"]["paths"]

    # 找到种子文件
    if seed_file:
        seed_files = [seed_file]
    else:
        seed_files = sorted(glob.glob(f"{paths['seeds']}/*.jsonl"))
        seed_files = [f for f in seed_files if not f.endswith("_report.json")]

    if not seed_files:
        logger.error("没有找到种子文件，请先运行 seed")
        return

    # 加载种子前置去重过滤器（由 cmd_seed_dedup 生成）
    seed_filter_file = Path(paths["expanded"]) / "seed_dedup_filter.json"
    seed_skip_ids: set[str] = set()
    if seed_filter_file.exists():
        with open(seed_filter_file, encoding="utf-8") as f:
            seed_skip_ids = set(json.load(f))
        logger.info(f"已加载种子去重过滤器: {len(seed_skip_ids)} 条重复种子将跳过（节省约 {len(seed_skip_ids)*5} 次 LLM 调用）")

    # 加载已扩展的种子文件清单
    expanded_manifest_file = Path(paths["expanded"]) / "expanded_seeds.json"
    expanded_manifest: list[str] = []
    if expanded_manifest_file.exists():
        with open(expanded_manifest_file, "r", encoding="utf-8") as f:
            expanded_manifest = json.load(f)

    new_seed_files = [sf for sf in seed_files if sf not in expanded_manifest]
    logger.info(
        f"找到 {len(seed_files)} 个种子文件，"
        f"跳过已扩展 {len(seed_files) - len(new_seed_files)} 个，"
        f"待扩展 {len(new_seed_files)} 个"
    )

    # 预计算总种子数（用于进度计算）
    _total_expand_seeds = 0
    if progress_callback:
        for sf in new_seed_files:
            s = load_seeds(sf)
            if seed_skip_ids:
                s = [x for x in s if x.get("id", "") not in seed_skip_ids]
            _total_expand_seeds += len(s)
    _expand_offset = 0

    for sf in new_seed_files:
        if _stop_requested():
            logger.warning("⏹️ 收到停止请求，扩展步骤终止")
            break

        seeds = load_seeds(sf)
        if not seeds:
            expanded_manifest.append(sf)
            continue

        # 应用前置去重过滤器
        if seed_skip_ids:
            before = len(seeds)
            seeds = [s for s in seeds if s.get("id", "") not in seed_skip_ids]
            filtered = before - len(seeds)
            if filtered:
                logger.info(f"  前置去重过滤: {Path(sf).name} 原 {before} 条 → 保留 {len(seeds)} 条（跳过 {filtered} 条）")

        if not seeds:
            expanded_manifest.append(sf)
            continue

        logger.info(f"处理种子文件: {sf} ({len(seeds)} 条)")
        n = len(seeds)

        # 3.1 + 3.2 改述变体 & 追问链并行执行（各自内部有 asyncio 事件循环）
        logger.info("--- 并行启动改述扩展 + 追问链扩展 ---")
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=2) as _pool:
            _f_rephrase = _pool.submit(
                expand_rephrase,
                seeds, config, paths["expanded"],
                num_variants=config["generation"]["expansion"]["rephrase_variants"],
                limit=limit,
            )
            _f_followup = _pool.submit(
                expand_followup,
                seeds, config, paths["expanded"],
                chain_length=config["generation"]["expansion"]["follow_up_depth"],
                limit=limit,
            )
            _f_rephrase.result()
            _f_followup.result()

        if _stop_requested():
            logger.warning("⏹️ 收到停止请求，扩展步骤终止")
            break

        _expand_offset += n

        expanded_manifest.append(sf)
        # 每完成一个文件立即写入清单，防止中途崩溃丢失进度
        Path(paths["expanded"]).mkdir(parents=True, exist_ok=True)
        with open(expanded_manifest_file, "w", encoding="utf-8") as f:
            json.dump(expanded_manifest, f, ensure_ascii=False, indent=2)

    # 确保最终清单存在（循环提前退出时也要写入）
    Path(paths["expanded"]).mkdir(parents=True, exist_ok=True)
    with open(expanded_manifest_file, "w", encoding="utf-8") as f:
        json.dump(expanded_manifest, f, ensure_ascii=False, indent=2)


# ============================================================
# Step 3c: 工具路由型 Q&A 生成（纯模板，零 LLM token）
# ============================================================
def cmd_tool_routing(config: dict, progress_callback=None) -> dict:
    """生成工具路由型 Q&A（核保/保费/现金价值/理赔/账户查询）"""
    paths = config["storage"]["paths"]
    expanded_dir = paths["expanded"]

    if progress_callback:
        progress_callback(10, 100)

    results = expand_tool_routing(expanded_dir)

    if progress_callback:
        progress_callback(70, 100)

    # 同时生成保全变更场景
    schema = config.get("knowledge_schema", {})
    ins_types = schema.get("insurance_types", [])
    long_term_types = [
        t for t in ins_types
        if any(k in t for k in ["寿险", "重疾", "年金", "两全", "医疗", "意外"])
    ]
    policy_results = expand_policy_service(long_term_types or ins_types[:20], expanded_dir)

    if progress_callback:
        progress_callback(100, 100)

    total = len(results) + len(policy_results)
    logger.info(f"工具路由步骤完成: 工具路由 {len(results)} 条 + 保全变更 {len(policy_results)} 条 = {total} 条")
    return {"tool_routing_count": len(results), "policy_service_count": len(policy_results), "total": total}


# ============================================================
# Step 4: 质检
# ============================================================
def cmd_qc(config: dict, input_files: list[str] = None, progress_callback=None):
    paths = config["storage"]["paths"]

    if input_files:
        files = input_files
    else:
        files = sorted(glob.glob(f"{paths['seeds']}/*.jsonl"))
        files += sorted(glob.glob(f"{paths['expanded']}/*.jsonl"))
        files = [f for f in files if not f.endswith("_report.json")]

    if not files:
        logger.error("没有找到待质检的文件")
        return

    if _stop_requested():
        logger.warning("⏹️ 收到停止请求，跳过质检")
        return

    report = run_full_qc(
        files,
        paths.get("qc_results", "./data/qc_results"),
        config,
        progress_callback=progress_callback,
    )
    return report


# ============================================================
# Step 3d: 产品对比画像提取 + 对比 Q&A 生成
# ============================================================
def cmd_comparison(config: dict, progress_callback=None) -> dict:
    """提取文档对比画像，生成产品对比型 Q&A"""
    paths = config["storage"]["paths"]
    chunks_file = Path(paths["chunks"]) / "chunks.jsonl"

    if not chunks_file.exists():
        logger.warning("chunks.jsonl 不存在，跳过对比画像提取")
        return {}

    if progress_callback:
        progress_callback(10, 100)

    # Step 1: 提取对比画像
    profiles = extract_comparison_profiles(str(chunks_file), paths["chunks"], config)
    logger.info(f"对比画像提取完成: {len(profiles)} 个文档")

    if progress_callback:
        progress_callback(40, 100)

    # Step 2: 生成产品对比 Q&A
    profiles_file = Path(paths["chunks"]) / "doc_comparison_profiles.jsonl"
    comparison_qa = expand_product_comparison(str(profiles_file), config, paths["expanded"])

    if progress_callback:
        progress_callback(80, 100)

    # Step 3: 生成误解纠正型 Q&A
    misconception_qa = expand_misconception_batch(config, paths["expanded"])

    # Step 4: 生成澄清/拒答 Q&A
    clarification_qa = expand_clarification_refusal(paths["expanded"])

    if progress_callback:
        progress_callback(100, 100)

    total = len(comparison_qa) + len(misconception_qa) + len(clarification_qa)
    logger.info(
        f"P1 扩展完成: 对比 {len(comparison_qa)} + 误解纠正 {len(misconception_qa)} "
        f"+ 澄清/拒答 {len(clarification_qa)} = {total} 条"
    )
    return {
        "comparison_count": len(comparison_qa),
        "misconception_count": len(misconception_qa),
        "clarification_count": len(clarification_qa),
        "total": total,
    }


# ============================================================
# Step 3e: P2 精细质量增强
# ============================================================
def cmd_p2_quality(config: dict, progress_callback=None) -> dict:
    """P2 质量增强：工具路由改述扩展 + 分层答案回溯校验 + LLM 评分"""
    paths = config["storage"]["paths"]
    expanded_dir = paths["expanded"]
    qc_dir = paths.get("qc_results", "./data/qc_results")
    qcfg = config.get("quality_check", {})

    if progress_callback:
        progress_callback(10, 100)

    # Step 1: 工具路由问题改述扩展
    tool_routing_files = sorted(
        [str(f) for f in Path(expanded_dir).glob("tool_routing_*.jsonl")]
    )
    rephrase_count = 0
    for tr_file in tool_routing_files:
        results = expand_tool_routing_rephrase(tr_file, config, expanded_dir)
        rephrase_count += len(results)
    logger.info(f"工具路由改述: 生成 {rephrase_count} 条")

    if progress_callback:
        progress_callback(40, 100)

    # Step 2: 分层答案回溯校验（对已通过质检的数据）
    passed_file = Path(qc_dir) / "qa_passed.jsonl"
    if passed_file.exists():
        passed_qa = []
        with open(passed_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    passed_qa.append(json.loads(line))

        enriched_qa = enrich_p2_source_metadata(passed_qa, paths["chunks"], config)
        pools = split_p2_pools(enriched_qa, config)
        pool_counts = {k: len(v) for k, v in pools.items()}
        unknown_effective = sum(1 for qa in pools["unknown_docs"] if qa.get("effective_p2_pool") == "regulatory_docs")

        product_pool = pools["product_docs"]
        regulatory_pool = pools["regulatory_docs"] + [
            qa for qa in pools["unknown_docs"] if qa.get("effective_p2_pool") == "regulatory_docs"
        ]

        verify_ckpt_product = str(Path(qc_dir) / "anchor_verify_product_checkpoint.json")
        verify_product_passed, verify_product_flagged = run_anchor_verify(
            product_pool,
            paths["chunks"],
            config,
            llm_verify_rate=qcfg.get("product_verify_rate", 0.05),
            checkpoint_file=verify_ckpt_product,
        )
        verify_ckpt_reg = str(Path(qc_dir) / "anchor_verify_regulatory_checkpoint.json")
        verify_reg_passed, verify_reg_flagged = run_regulatory_anchor_verify(
            regulatory_pool,
            paths["chunks"],
            config,
            llm_verify_rate=qcfg.get("regulatory_verify_rate", 0.01),
            checkpoint_file=verify_ckpt_reg,
        )

        verify_passed = verify_product_passed + verify_reg_passed
        verify_flagged = verify_product_flagged + verify_reg_flagged
        verify_flagged_by_pool = {
            "product_docs": len(verify_product_flagged),
            "regulatory_docs": len(verify_reg_flagged),
            "unknown_docs": sum(1 for qa in verify_reg_flagged if qa.get("p2_pool") == "unknown_docs"),
        }
        logger.info(
            f"P2 分池回溯校验: 产品类 {len(verify_product_passed)} 通过/{len(verify_product_flagged)} 存疑 | "
            f"规范类 {len(verify_reg_passed)} 通过/{len(verify_reg_flagged)} 存疑"
        )

        if progress_callback:
            progress_callback(70, 100)

        # Step 3: LLM 自动评分（按 pool / profile 抽样）
        scoring_product_ckpt = str(Path(qc_dir) / "llm_scoring_product_checkpoint.json")
        scoring_reg_ckpt = str(Path(qc_dir) / "llm_scoring_regulatory_checkpoint.json")
        scoring_product_report = run_llm_scoring(
            verify_product_passed,
            config,
            sample_rate=qcfg.get("product_scoring_rate", 0.05),
            checkpoint_file=scoring_product_ckpt,
            scoring_profile="product",
        ) if verify_product_passed else {"sample_size": 0, "avg_scores": {}, "scoring_profile": "product"}
        scoring_reg_report = run_llm_scoring(
            verify_reg_passed,
            config,
            sample_rate=qcfg.get("regulatory_scoring_rate", 0.03),
            checkpoint_file=scoring_reg_ckpt,
            scoring_profile="regulatory",
        ) if verify_reg_passed else {"sample_size": 0, "avg_scores": {}, "scoring_profile": "regulatory"}
        scoring_report = {
            "by_pool": {
                "product_docs": scoring_product_report,
                "regulatory_docs": scoring_reg_report,
            },
            "avg_scores_by_pool": {
                "product_docs": scoring_product_report.get("avg_scores", {}),
                "regulatory_docs": scoring_reg_report.get("avg_scores", {}),
            },
        }

        # 保存验证标记
        verify_file = Path(qc_dir) / "anchor_verify_flagged.jsonl"
        with open(verify_file, "a", encoding="utf-8") as f:
            for qa in verify_flagged:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")

        scoring_file = Path(qc_dir) / "llm_scoring_report.json"
        with open(scoring_file, "w", encoding="utf-8") as f:
            json.dump(scoring_report, f, ensure_ascii=False, indent=2)

        p2_report = {
            "p2_time": datetime.now().isoformat(),
            "pool_counts": pool_counts,
            "verify_flagged_by_pool": verify_flagged_by_pool,
            "avg_scores_by_pool": scoring_report["avg_scores_by_pool"],
            "unknown_doc_count": pool_counts.get("unknown_docs", 0),
            "unknown_docs_routed_to_regulatory": unknown_effective,
            "tool_routing_rephrase": rephrase_count,
            "verify_passed_total": len(verify_passed),
            "verify_flagged_total": len(verify_flagged),
        }
        with open(Path(qc_dir) / "p2_quality_report.json", "w", encoding="utf-8") as f:
            json.dump(p2_report, f, ensure_ascii=False, indent=2)
    else:
        verify_flagged = []
        scoring_report = {}
        pool_counts = {"product_docs": 0, "regulatory_docs": 0, "unknown_docs": 0}

    if progress_callback:
        progress_callback(100, 100)

    return {
        "tool_routing_rephrase": rephrase_count,
        "anchor_verify_flagged": len(verify_flagged),
        "pool_counts": pool_counts,
        "avg_scores_by_pool": scoring_report.get("avg_scores_by_pool", {}),
    }


# ============================================================
# 状态查看
# ============================================================
def cmd_status(config: dict):
    paths = config["storage"]["paths"]

    print("\n" + "=" * 60)
    print("📊 保险知识问答生成管道 - 当前状态")
    print("=" * 60)

    # 原始文档
    raw_dir = Path(paths["raw_documents"])
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("**/*"))
        raw_files = [f for f in raw_files if f.is_file()]
        print(f"\n📁 原始文档: {len(raw_files)} 个文件")
    else:
        print(f"\n📁 原始文档: 目录不存在 ({raw_dir})")

    # Chunks
    chunks_dir = Path(paths["chunks"])
    chunks_file = chunks_dir / "chunks.jsonl"
    if chunks_file.exists():
        chunk_count = sum(1 for _ in open(chunks_file))
        print(f"📦 文档 Chunks: {chunk_count} 个")
    else:
        print("📦 文档 Chunks: 未生成")

    # 种子
    seeds_dir = Path(paths["seeds"])
    if seeds_dir.exists():
        seed_files = list(seeds_dir.glob("*.jsonl"))
        seed_count = sum(
            sum(1 for _ in open(f)) for f in seed_files
        ) if seed_files else 0
        print(f"🌱 种子 Q&A: {seed_count} 条 ({len(seed_files)} 个批次)")
    else:
        print("🌱 种子 Q&A: 未生成")

    # 扩展
    expanded_dir = Path(paths["expanded"])
    if expanded_dir.exists():
        exp_files = list(expanded_dir.glob("*.jsonl"))
        exp_count = sum(
            sum(1 for _ in open(f)) for f in exp_files
        ) if exp_files else 0
        print(f"📈 扩展 Q&A: {exp_count} 条 ({len(exp_files)} 个批次)")
    else:
        print("📈 扩展 Q&A: 未生成")

    # 质检
    qc_dir = Path(paths.get("qc_results", "./data/qc_results"))
    qc_report = qc_dir / "qc_report.json"
    if qc_report.exists():
        with open(qc_report) as f:
            report = json.load(f)
        print(f"✅ 质检结果: {report.get('passed_total', 0)} 条通过 "
              f"(通过率 {report.get('pass_rate', 'N/A')})")
    else:
        print("✅ 质检结果: 未执行")

    # 最终输出
    final_dir = Path(paths.get("final", "./data/final"))
    if final_dir.exists():
        final_files = list(final_dir.glob("*.jsonl"))
        final_count = sum(
            sum(1 for _ in open(f)) for f in final_files
        ) if final_files else 0
        print(f"🎯 最终入库: {final_count} 条")
    else:
        print("🎯 最终入库: 未执行")

    target = config.get("project", {}).get("target_total", 100000)
    print(f"\n🎯 目标: {target:,} 条")
    print("=" * 60 + "\n")


# ============================================================
# 全流程运行
# ============================================================
def cmd_run(config: dict, seed_limit: int = None, expand_limit: int = None):
    """端到端运行全流程"""
    logger.info("🚀 启动全流程管道")

    # Step 1
    logger.info("=" * 40 + " Step 1: 文档预处理 " + "=" * 40)
    cmd_preprocess(config)

    # Step 2
    logger.info("=" * 40 + " Step 2: 种子生成 " + "=" * 40)
    cmd_seed(config, limit=seed_limit)

    # Step 2b: 种子前置去重（节省扩展 token）
    logger.info("=" * 40 + " Step 2b: 种子前置去重 " + "=" * 40)
    cmd_seed_dedup(config)

    # Step 3a: 改述 + 追问链扩展（基于种子文件）
    logger.info("=" * 40 + " Step 3a: 改述/追问链扩展 " + "=" * 40)
    cmd_expand(config, limit=expand_limit)

    # Step 3b: 模板驱动扩展（基于文档参数）
    logger.info("=" * 40 + " Step 3b: 模板扩展 " + "=" * 40)
    expand_template(
        config,
        config["storage"]["paths"]["expanded"],
        chunks_dir=config["storage"]["paths"]["chunks"],
    )

    # Step 3c: 工具路由型 Q&A 生成（纯模板，零 LLM）
    logger.info("=" * 40 + " Step 3c: 工具路由型 Q&A 生成 " + "=" * 40)
    cmd_tool_routing(config)

    # Step 3d: 产品对比 + 误解纠正 + 澄清/拒答
    logger.info("=" * 40 + " Step 3d: 产品对比/误解纠正/澄清拒答 " + "=" * 40)
    cmd_comparison(config)

    # Step 4
    logger.info("=" * 40 + " Step 4: 质量检查 " + "=" * 40)
    cmd_qc(config)

    # 最终状态
    cmd_status(config)
    logger.info(f"📊 全流程 API 用量: {usage_stats.summary()}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="保险知识问答生成管道 - 主调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/orchestrator.py status              # 查看当前状态
  python src/orchestrator.py preprocess           # 预处理文档
  python src/orchestrator.py seed --limit 5       # 试跑5个chunk的种子生成
  python src/orchestrator.py seed                 # 全量种子生成
  python src/orchestrator.py seed-dedup           # 种子前置去重（节省扩展 token）
  python src/orchestrator.py expand               # 批量扩展（自动使用去重过滤器）
  python src/orchestrator.py qc                   # 质检
  python src/orchestrator.py run                  # 全流程
  python src/orchestrator.py run --seed-limit 3   # 全流程试跑
        """
    )
    parser.add_argument("command", choices=[
        "run", "preprocess", "seed", "seed-dedup", "expand", "expand-template",
        "tool-routing", "comparison", "p2-quality", "qc", "status",
    ])
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument("--seed-limit", type=int, default=None, help="种子阶段限制chunk数")
    parser.add_argument("--expand-limit", type=int, default=None, help="扩展阶段限制种子数")
    parser.add_argument("--seed-file", default=None, help="指定种子文件")
    parser.add_argument("--input", nargs="*", default=None, help="质检输入文件")
    parser.add_argument("--pairs", type=int, default=8, help="每chunk生成Q&A数量")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == "status":
        cmd_status(config)
    elif args.command == "preprocess":
        cmd_preprocess(config)
    elif args.command == "seed":
        cmd_seed(config, limit=args.limit or args.seed_limit, pairs=args.pairs)
    elif args.command == "seed-dedup":
        cmd_seed_dedup(config)
    elif args.command == "expand":
        cmd_expand(config, seed_file=args.seed_file, limit=args.limit or args.expand_limit)
    elif args.command == "expand-template":
        expand_template(
            config,
            config["storage"]["paths"]["expanded"],
            chunks_dir=config["storage"]["paths"]["chunks"],
        )
    elif args.command == "tool-routing":
        cmd_tool_routing(config)
    elif args.command == "comparison":
        cmd_comparison(config)
    elif args.command == "p2-quality":
        cmd_p2_quality(config)
    elif args.command == "qc":
        cmd_qc(config, input_files=args.input)
    elif args.command == "run":
        cmd_run(config, seed_limit=args.seed_limit, expand_limit=args.expand_limit)


if __name__ == "__main__":
    main()
