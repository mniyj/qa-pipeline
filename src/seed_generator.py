"""
seed_generator.py
种子问答生成器：读取文档 chunks → 调用高质量 LLM → 输出种子 Q&A

使用方式：
    python src/seed_generator.py
    python src/seed_generator.py --chunks ./data/chunks/chunks.jsonl --output ./data/seeds
    python src/seed_generator.py --limit 10  # 只处理前 10 个 chunk（试跑）
"""

import json
import re
import asyncio
import argparse
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
from llm_client import create_client, usage_stats
from dedup import DedupTracker

_stop_event: threading.Event | None = None


def _check_stop() -> bool:
    global _stop_event
    if _stop_event is None:
        try:
            from src.pipeline_state import pipeline_state
            _stop_event = pipeline_state._cancel_event
        except Exception:
            return False
    return _stop_event.is_set()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_DIFFICULTY_NORMALIZE = {"入门级": "入门", "进阶级": "进阶", "专业级": "专业"}


def _normalize_qa(qa: dict) -> dict:
    diff = qa.get("difficulty", "")
    if diff in _DIFFICULTY_NORMALIZE:
        qa["difficulty"] = _DIFFICULTY_NORMALIZE[diff]
    return qa


def _is_valid_qa(qa: dict) -> bool:
    return bool(qa.get("question")) and len(qa.get("answer", "")) >= 20


def load_prompt_template(template_name: str) -> str:
    """加载 prompt 模板"""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{template_name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_json_response(response: str) -> list[dict]:
    """从 LLM 响应中提取 JSON 数组，兼容截断/markdown/多余文字等常见格式问题"""
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*", "", response).strip()

    # 尝试直接解析
    try:
        result = json.loads(response)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # 提取最外层数组（允许末尾截断）
    start = response.find("[")
    if start != -1:
        end = response.rfind("]")
        if end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass
        # 末尾截断：剥离最后不完整元素后补 "]"
        fragment = response[start:]
        last_complete = fragment.rfind("},")
        if last_complete != -1:
            try:
                return json.loads(fragment[:last_complete + 1] + "]")
            except json.JSONDecodeError:
                pass

    # 单个 JSON 对象
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

    logger.warning(f"JSON 解析失败，响应前 200 字: {response[:200]}")
    return []


def generate_seeds_from_chunks(
    chunks_file: str,
    output_dir: str,
    config: dict,
    limit: Optional[int] = None,
    pairs_per_chunk: int = 8,
    progress_callback=None,
) -> dict:
    return asyncio.run(_generate_seeds_async(chunks_file, output_dir, config, limit, pairs_per_chunk, progress_callback))


async def _generate_seeds_async(
    chunks_file: str,
    output_dir: str,
    config: dict,
    limit: Optional[int] = None,
    pairs_per_chunk: int = 8,
    progress_callback=None,
) -> dict:
    """
    从 chunks 批量生成种子 Q&A

    Args:
        chunks_file: chunks.jsonl 文件路径
        output_dir: 输出目录
        config: 项目配置
        limit: 限制处理的 chunk 数量（用于试跑）
        pairs_per_chunk: 每个 chunk 生成的问答对数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载 chunks
    all_chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_chunks.append(json.loads(line))

    if limit:
        all_chunks = all_chunks[:limit]

    # 加载已生成种子的 chunk_id 清单
    seeded_manifest_file = Path(output_dir) / "seeded_chunks.json"
    seeded_ids: set[str] = set()
    if seeded_manifest_file.exists():
        with open(seeded_manifest_file, "r", encoding="utf-8") as f:
            seeded_ids = set(json.load(f))

    original_count = len(all_chunks)
    chunks = [c for c in all_chunks if c["chunk_id"] not in seeded_ids]
    logger.info(
        f"加载了 {original_count} 个 chunk，"
        f"跳过已处理 {original_count - len(chunks)} 个，"
        f"待生成 {len(chunks)} 个"
    )

    if not chunks:
        return {"total_seeds": 0, "skipped_chunks": original_count}

    # ── 构建去重指纹：收集已有种子的 chunk 文本，避免调用 LLM 前发现内容重复 ──
    chunk_dedup = DedupTracker(threshold=0.70)
    # 从已有种子文件中反向查找已处理 chunk 的文本
    for f in sorted(Path(output_dir).glob("seed_*.jsonl")):
        if not f.exists() or f.stat().st_size == 0:
            continue
        seen_ids_in_file: set[str] = set()
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        item = json.loads(line)
                        cid = item.get("source_chunk_id", "")
                        if cid and cid not in seen_ids_in_file:
                            seen_ids_in_file.add(cid)
                    except json.JSONDecodeError:
                        continue
        # 用这些 chunk_id 从 all_chunks 中找文本
        for c in all_chunks:
            if c["chunk_id"] in seen_ids_in_file:
                chunk_dedup.add(c.get("text", ""))
    if chunk_dedup.count:
        logger.info(f"重复校验: 已加载 {chunk_dedup.count} 个已处理 chunk 的文本指纹")

    gen_config = config.get("generation", {}).get("seed", {})
    pairs_per_chunk = gen_config.get("pairs_per_chunk", pairs_per_chunk)
    concurrency = gen_config.get("concurrency", 3)

    client = create_client(config, "seed_generation")
    prompt_template = load_prompt_template("seed_from_document")
    batch_id = f"seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sem = asyncio.Semaphore(concurrency)

    seeds_file = output_path / f"{batch_id}.jsonl"
    write_lock = asyncio.Lock()
    total_seeds_written = 0
    skipped_dedup = 0
    chunks_done = 0
    failed_chunks: list[dict] = []

    async def process_chunk(i: int, chunk: dict):
        nonlocal total_seeds_written, skipped_dedup, chunks_done
        async with sem:
            try:
                if _check_stop():
                    return i, {"chunk_id": chunk["chunk_id"], "reason": "用户停止"}

                # 调用 LLM 前：检查 chunk 文本是否与已处理的 chunk 高度相似
                chunk_text = chunk.get("text", "")
                if chunk_dedup.is_duplicate(chunk_text):
                    skipped_dedup += 1
                    logger.info(
                        f"[{i+1}/{len(chunks)}] 跳过（内容重复）: {chunk['chunk_id']} "
                        f"({chunk['doc_type']} / {chunk['insurance_type']})"
                    )
                    # 也标记为已处理，避免下次再检查
                    async with write_lock:
                        seeded_ids.add(chunk["chunk_id"])
                        with open(seeded_manifest_file, "w", encoding="utf-8") as mf:
                            json.dump(list(seeded_ids), mf, ensure_ascii=False)
                    return i, {"chunk_id": chunk["chunk_id"], "reason": "去重跳过"}

                logger.info(
                    f"[{i+1}/{len(chunks)}] 处理 chunk: {chunk['chunk_id']} "
                    f"({chunk['doc_type']} / {chunk['insurance_type']})"
                )
                doc_file = chunk.get("doc_file", "")
                doc_type = chunk["doc_type"]
                product_header = f"【来源产品：{doc_file}】\n\n" if doc_file else ""
                if doc_type in ("法律法规", "监管文件", "行业标准"):
                    disclaimer = (
                        "答案末尾必须附加一句免责提示："
                        f"以上内容依据{doc_file}，法规条款可能随监管政策调整而变化，"
                        "请以最新发布的官方文件为准。"
                    )
                else:
                    disclaimer = "答案末尾无需附加免责提示。"
                prompt = prompt_template.format(
                    num_pairs=pairs_per_chunk,
                    doc_type=doc_type,
                    insurance_type=chunk["insurance_type"],
                    section_title=chunk.get("section_title", ""),
                    doc_file=doc_file,
                    document_content=product_header + chunk_text,
                    disclaimer=disclaimer,
                )
                try:
                    response = await client.acall(prompt)
                    qa_pairs = parse_json_response(response)
                    if not qa_pairs:
                        return i, {"chunk_id": chunk["chunk_id"], "reason": "JSON 解析失败"}
                    for j, qa in enumerate(qa_pairs):
                        qa["id"] = f"{batch_id}_{chunk['chunk_id']}_{j:03d}"
                        qa["source_chunk_id"] = chunk["chunk_id"]
                        qa["source_doc"] = chunk["doc_file"]
                        qa["doc_type"] = doc_type
                        qa["insurance_type"] = chunk["insurance_type"]
                        qa["product_name"] = chunk.get("product_name", "")
                        qa["generation_method"] = "seed_from_document"
                        qa["batch_id"] = batch_id
                        qa["created_at"] = datetime.now().isoformat()
                    # 归一化难度值，过滤空答案
                    qa_pairs = [_normalize_qa(qa) for qa in qa_pairs]
                    qa_pairs = [qa for qa in qa_pairs if _is_valid_qa(qa)]

                    async with write_lock:
                        with open(seeds_file, "a", encoding="utf-8") as f:
                            for qa in qa_pairs:
                                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                        total_seeds_written += len(qa_pairs)
                        chunk_dedup.add(chunk_text)
                        seeded_ids.add(chunk["chunk_id"])
                        with open(seeded_manifest_file, "w", encoding="utf-8") as mf:
                            json.dump(list(seeded_ids), mf, ensure_ascii=False)

                    logger.info(f"  → 生成 {len(qa_pairs)} 条种子 Q&A（累计 {total_seeds_written}）")
                    return i, None
                except Exception as e:
                    logger.error(f"  → 失败: {e}")
                    return i, {"chunk_id": chunk["chunk_id"], "reason": str(e)}
            finally:
                chunks_done += 1
                if progress_callback:
                    progress_callback(chunks_done, len(chunks))

    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    raw = await asyncio.gather(*tasks)

    for _, err in sorted(raw):
        if err:
            failed_chunks.append(err)

    # 从已写入文件中读取统计（避免全量加载到内存）
    all_seeds_for_stats: list[dict] = []
    if seeds_file.exists():
        with open(seeds_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_seeds_for_stats.append(json.loads(line))

    report = {
        "batch_id": batch_id,
        "process_time": datetime.now().isoformat(),
        "chunks_processed": len(chunks),
        "chunks_failed": len(failed_chunks),
        "chunks_skipped_dedup": skipped_dedup,
        "total_seeds": total_seeds_written,
        "difficulty_distribution": _count_by(all_seeds_for_stats, "difficulty"),
        "question_type_distribution": _count_by(all_seeds_for_stats, "question_type"),
        "insurance_type_distribution": _count_by(all_seeds_for_stats, "insurance_type"),
        "failures": failed_chunks,
        "api_usage": usage_stats.summary(),
    }

    report_file = output_path / f"{batch_id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info("种子生成完成")
    logger.info(f"  总计: {total_seeds_written} 条种子 Q&A")
    logger.info(f"  重复跳过: {skipped_dedup} 个 chunk")
    logger.info(f"  失败 chunk: {len(failed_chunks)}")
    logger.info(f"  {usage_stats.summary()}")
    logger.info(f"  输出: {seeds_file}")
    logger.info("=" * 50)

    return report


def _count_by(items: list[dict], key: str) -> dict:
    counts = {}
    for item in items:
        val = item.get(key, "未知")
        counts[val] = counts.get(val, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="种子问答生成器")
    parser.add_argument("--chunks", default="./data/chunks/chunks.jsonl")
    parser.add_argument("--output", default="./data/seeds")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="限制处理 chunk 数")
    parser.add_argument("--pairs", type=int, default=8, help="每 chunk 生成数量")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generate_seeds_from_chunks(
        args.chunks, args.output, config,
        limit=args.limit,
        pairs_per_chunk=args.pairs,
    )


if __name__ == "__main__":
    main()
