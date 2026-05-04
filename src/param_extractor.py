"""
param_extractor.py
从文档 chunks 中提取结构化参数，供模板扩展使用

使用方式：
    python src/param_extractor.py
"""

import json
import re
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import yaml
from llm_client import create_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_prompt(name: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_json(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def merge_params(param_list: list[dict]) -> dict:
    """合并同一文档多个 chunk 提取的参数，去重合并列表字段"""
    merged = {
        "insurance_type": "",
        "specific_amounts": [],
        "time_periods": [],
        "conditions": [],
        "exclusions": [],
        "procedures": [],
        "key_clauses": [],
        "user_scenarios": [],
    }
    for params in param_list:
        if not merged["insurance_type"] and params.get("insurance_type"):
            merged["insurance_type"] = params["insurance_type"]
        for key in ["specific_amounts", "time_periods", "conditions",
                    "exclusions", "procedures", "key_clauses", "user_scenarios"]:
            merged[key].extend(params.get(key, []))

    # 去重，保留顺序
    for key in merged:
        if isinstance(merged[key], list):
            seen = set()
            deduped = []
            for item in merged[key]:
                if item and item not in seen:
                    seen.add(item)
                    deduped.append(item)
            merged[key] = deduped

    return merged


async def extract_params_for_doc(
    doc_file: str,
    chunks: list[dict],
    client,
    template: str,
    sem: asyncio.Semaphore,
) -> dict:
    """对单个文档的所有 chunk 提取参数并合并"""
    async def extract_chunk(chunk: dict) -> dict:
        async with sem:
            prompt = template.format(
                doc_file=chunk.get("doc_file", doc_file),
                doc_type=chunk.get("doc_type", "未分类"),
                insurance_type=chunk.get("insurance_type", "通用"),
                document_content=chunk["text"],
            )
            try:
                response = await client.acall(prompt)
                return parse_json(response)
            except Exception as e:
                logger.warning(f"  chunk 提取失败: {e}")
                return {}

    tasks = [extract_chunk(c) for c in chunks]
    results = await asyncio.gather(*tasks)
    merged = merge_params([r for r in results if r])
    merged["doc_file"] = doc_file
    merged["doc_type"] = chunks[0].get("doc_type", "未分类") if chunks else "未分类"
    return merged


def extract_all_params(chunks_file: str, output_dir: str, config: dict) -> list[dict]:
    return asyncio.run(_extract_all_async(chunks_file, output_dir, config))


async def _extract_all_async(chunks_file: str, output_dir: str, config: dict) -> list[dict]:
    output_path = Path(output_dir)
    params_file = output_path / "doc_params.jsonl"

    # 加载已提取的文档清单
    already_extracted: set[str] = set()
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    already_extracted.add(doc.get("doc_file", ""))

    # 加载 chunks，按文档分组
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunks_by_doc[chunk["doc_file"]].append(chunk)

    new_docs = {doc: chunks for doc, chunks in chunks_by_doc.items()
                if doc not in already_extracted}

    logger.info(
        f"共 {len(chunks_by_doc)} 个文档，"
        f"跳过已提取 {len(already_extracted)} 个，"
        f"待提取 {len(new_docs)} 个"
    )

    if not new_docs:
        # 返回已有数据
        all_params = []
        if params_file.exists():
            with open(params_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_params.append(json.loads(line))
        return all_params

    client = create_client(config, "seed_generation")
    template = load_prompt("extract_params")
    concurrency = config.get("generation", {}).get("seed", {}).get("concurrency", 3)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        extract_params_for_doc(doc_file, chunks, client, template, sem)
        for doc_file, chunks in new_docs.items()
    ]
    new_params = await asyncio.gather(*tasks)

    # 追加写入
    with open(params_file, "a", encoding="utf-8") as f:
        for params in new_params:
            params["extracted_at"] = datetime.now().isoformat()
            f.write(json.dumps(params, ensure_ascii=False) + "\n")
            logger.info(f"  提取完成: {params['doc_file']} → "
                        f"金额{len(params['specific_amounts'])}条，"
                        f"时限{len(params['time_periods'])}条，"
                        f"场景{len(params['user_scenarios'])}条")

    # 返回全量
    all_params = []
    with open(params_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_params.append(json.loads(line))
    return all_params
