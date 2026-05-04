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
from expander import expand_rephrase, expand_followup, expand_template, load_seeds
from quality_checker import run_full_qc
from llm_client import usage_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path: str = "./config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Step 1: 文档预处理
# ============================================================
def cmd_preprocess(config: dict):
    paths = config["storage"]["paths"]
    report = process_all_documents(
        paths["raw_documents"],
        paths["chunks"],
        config,
    )
    return report


# ============================================================
# Step 2: 种子生成
# ============================================================
def cmd_seed(config: dict, limit: int = None, pairs: int = 8):
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
    )
    return report


# ============================================================
# Step 3: 批量扩展
# ============================================================
def cmd_expand(config: dict, seed_file: str = None, limit: int = None):
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

    for sf in new_seed_files:
        seeds = load_seeds(sf)
        if not seeds:
            expanded_manifest.append(sf)
            continue

        logger.info(f"处理种子文件: {sf} ({len(seeds)} 条)")

        # 3.1 改述变体
        logger.info("--- 开始改述扩展 ---")
        expand_rephrase(
            seeds, config, paths["expanded"],
            num_variants=config["generation"]["expansion"]["rephrase_variants"],
            limit=limit,
        )

        # 3.2 追问链
        logger.info("--- 开始追问链扩展 ---")
        expand_followup(
            seeds, config, paths["expanded"],
            chain_length=config["generation"]["expansion"]["follow_up_depth"],
            limit=limit,
        )

        expanded_manifest.append(sf)

    # 更新清单
    Path(paths["expanded"]).mkdir(parents=True, exist_ok=True)
    with open(expanded_manifest_file, "w", encoding="utf-8") as f:
        json.dump(expanded_manifest, f, ensure_ascii=False, indent=2)


# ============================================================
# Step 4: 质检
# ============================================================
def cmd_qc(config: dict, input_files: list[str] = None):
    paths = config["storage"]["paths"]

    if input_files:
        files = input_files
    else:
        # 收集所有种子 + 扩展的 JSONL 文件
        files = sorted(glob.glob(f"{paths['seeds']}/*.jsonl"))
        files += sorted(glob.glob(f"{paths['expanded']}/*.jsonl"))
        # 排除报告文件
        files = [f for f in files if not f.endswith("_report.json")]

    if not files:
        logger.error("没有找到待质检的文件")
        return

    report = run_full_qc(
        files,
        paths.get("qc_results", "./data/qc_results"),
        config,
    )
    return report


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
  python src/orchestrator.py expand               # 批量扩展
  python src/orchestrator.py qc                   # 质检
  python src/orchestrator.py run                  # 全流程
  python src/orchestrator.py run --seed-limit 3   # 全流程试跑
        """
    )
    parser.add_argument("command", choices=["run", "preprocess", "seed", "expand", "expand-template", "qc", "status"])
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
    elif args.command == "expand":
        cmd_expand(config, seed_file=args.seed_file, limit=args.limit or args.expand_limit)
    elif args.command == "expand-template":
        expand_template(
            config,
            config["storage"]["paths"]["expanded"],
            chunks_dir=config["storage"]["paths"]["chunks"],
        )
    elif args.command == "qc":
        cmd_qc(config, input_files=args.input)
    elif args.command == "run":
        cmd_run(config, seed_limit=args.seed_limit, expand_limit=args.expand_limit)


if __name__ == "__main__":
    main()
