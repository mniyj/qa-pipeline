"""
expander.py
批量扩展器：种子 Q&A → 变体扩展 → 10 万+

支持的扩展方式：
- rephrase: 改述变体（同知识点、不同问法）
- followup: 追问链（多轮对话）
- template: 模板化批量生成

使用方式：
    python src/expander.py --mode rephrase --input ./data/seeds/seed_xxx.jsonl
    python src/expander.py --mode followup --input ./data/seeds/seed_xxx.jsonl
    python src/expander.py --mode template --config ./config/config.yaml
    python src/expander.py --mode all --input ./data/seeds/seed_xxx.jsonl
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


def _safe_format(template: str, **kwargs) -> str:
    """
    安全地格式化模板字符串，将参数值中的花括号转义，防止 LLM 返回的文档内容
    （如"第{1}条"）被 str.format() 误解析而抛出 ValueError/KeyError。
    """
    escaped = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
    return template.format(**escaped)


def _retrieve_chunks(question: str, chunks: list[str], k: int = 5) -> list[str]:
    """
    从 chunks 中召回与 question 最相关的 k 个，使用字符 bigram 重叠率打分。
    无需外部依赖；当 chunks 总量 <= k 时直接返回全部。
    """
    if len(chunks) <= k:
        return chunks
    q_bigrams = {question[i:i + 2] for i in range(len(question) - 1)}
    if not q_bigrams:
        return chunks[:k]
    scored = []
    for chunk in chunks:
        c_bigrams = {chunk[i:i + 2] for i in range(len(chunk) - 1)}
        score = len(q_bigrams & c_bigrams) / len(q_bigrams)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def load_prompt(name: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_json(text: str) -> list[dict]:
    # 清除 markdown 代码块标记
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text).strip()

    # 尝试直接解析
    try:
        r = json.loads(text)
        return r if isinstance(r, list) else [r]
    except json.JSONDecodeError:
        pass

    # 提取最外层 JSON 数组（允许末尾被截断）
    start = text.find("[")
    if start != -1:
        # 先尝试完整数组
        end = text.rfind("]")
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        # 末尾截断时，逐步剥离最后一个不完整元素后补 "]"
        fragment = text[start:]
        last_complete = fragment.rfind("},")
        if last_complete != -1:
            try:
                return json.loads(fragment[:last_complete + 1] + "]")
            except json.JSONDecodeError:
                pass

    # 提取单个 JSON 对象
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

    logger.warning(f"JSON 解析失败，响应前200字: {text[:200]}")
    return []


def load_seeds(file_path: str) -> list[dict]:
    seeds = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    return seeds


# ============================================================
# 扩展方式 1: 改述变体
# ============================================================
def score_seed(seed: dict) -> float:
    """对种子 Q&A 质量评分（0.0-1.0），用于差异化扩展倍率。"""
    score = 0.5
    answer = seed.get("answer", "")
    question = seed.get("question", "")
    if len(answer) < 50:
        score -= 0.2
    elif len(answer) > 150:
        score += 0.1
    if re.search(r"\d+(?:元|万元|天|%|年)", answer):
        score += 0.15
    if re.search(r"第[一二三四五六七八九十\d]+条", answer):
        score += 0.1
    if seed.get("product_name") and seed["product_name"] in question:
        score += 0.1
    if seed.get("difficulty") == "专业":
        score += 0.1
    return min(1.0, max(0.0, score))


def get_expansion_multiplier(seed: dict) -> tuple[int, int]:
    """根据种子质量评分返回 (改述数, 追问深度)"""
    s = score_seed(seed)
    if s >= 0.8:
        return 5, 4
    elif s >= 0.5:
        return 3, 2
    else:
        return 1, 0


def expand_rephrase(
    seeds: list[dict],
    config: dict,
    output_dir: str,
    num_variants: int = 5,
    limit: Optional[int] = None,
    progress_callback=None,
) -> list[dict]:
    """对每条种子生成改述变体（并发）"""
    return asyncio.run(_expand_rephrase_async(seeds, config, output_dir, num_variants, limit, progress_callback))


async def _expand_rephrase_async(
    seeds: list[dict],
    config: dict,
    output_dir: str,
    num_variants: int,
    limit: Optional[int],
    progress_callback=None,
) -> list[dict]:
    client = create_client(config, "expansion")
    template = load_prompt("expand_rephrase")
    batch_id = f"rephrase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    concurrency = config.get("generation", {}).get("expansion", {}).get("concurrency", 5)
    sem = asyncio.Semaphore(concurrency)

    if limit:
        seeds = seeds[:limit]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"{batch_id}.jsonl"

    # 断点续传：加载已处理的 seed_id 清单
    checkpoint_file = output_path / "rephrase_checkpoint.json"
    processed_ids: set[str] = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            processed_ids = set(json.load(f))

    pending = [s for s in seeds if s.get("id", "") not in processed_ids]
    skipped = len(seeds) - len(pending)
    if skipped:
        logger.info(f"改述跳过已处理 {skipped} 条种子，待处理 {len(pending)} 条")

    if not pending:
        return []

    # ── 构建种子问题去重指纹：已有扩展的种子问题不再生成相似变体 ──
    dedup = DedupTracker(threshold=0.65)
    for s in seeds:
        if s.get("id", "") in processed_ids:
            dedup.add(s.get("question", ""))
    if dedup.count:
        logger.info(f"重复校验: 已加载 {dedup.count} 个已改述种子的问题指纹")

    write_lock = asyncio.Lock()
    total_written = 0
    skipped_dedup = 0
    seeds_done = 0

    async def process_one(i: int, seed: dict):
        nonlocal total_written, skipped_dedup, seeds_done
        seed_id = seed.get("id", "")
        async with sem:
            try:
                if _check_stop():
                    return i, []

                # 调用 LLM 前：检查该种子问题是否与已改述的种子高度相似
                question = seed.get("question", "")
                if not dedup.deduped_add(question):
                    skipped_dedup += 1
                    logger.info(f"  [{i+1}/{len(pending)}] 跳过（问题重复）: {question[:50]}...")
                    return i, []

                logger.info(f"[改述 {i+1}/{len(pending)}] {question[:50]}...")
                prompt = _safe_format(
                    template,
                    num_variants=num_variants,
                    original_question=seed["question"],
                    original_answer=seed.get("answer", seed.get("summary", seed.get("description", ""))),
                    insurance_type=seed.get("insurance_type", "其他保险"),
                    difficulty=seed.get("difficulty", "入门"),
                    parent_id=seed_id,
                )
                try:
                    response = await client.acall(prompt)
                    variants = parse_json(response)
                    for j, v in enumerate(variants):
                        v["id"] = f"{batch_id}_{i:04d}_{j:02d}"
                        v["generation_method"] = "rephrase"
                        v["parent_id"] = seed_id
                        v["insurance_type"] = seed.get("insurance_type", "其他保险")
                        v["batch_id"] = batch_id
                        v["created_at"] = datetime.now().isoformat()
                    variants = [_normalize_qa(v) for v in variants]
                    variants = [v for v in variants if _is_valid_qa(v)]
                    async with write_lock:
                        with open(results_file, "a", encoding="utf-8") as f:
                            for v in variants:
                                f.write(json.dumps(v, ensure_ascii=False) + "\n")
                        total_written += len(variants)
                        processed_ids.add(seed_id)
                        with open(checkpoint_file, "w", encoding="utf-8") as mf:
                            json.dump(list(processed_ids), mf, ensure_ascii=False)
                    logger.info(f"  → {len(variants)} 个变体（累计 {total_written}）")
                    return i, variants
                except Exception as e:
                    logger.error(f"  → 失败: {e}")
                    return i, []
            finally:
                seeds_done += 1
                if progress_callback:
                    progress_callback(seeds_done, len(pending))

    tasks = [process_one(i, seed) for i, seed in enumerate(pending)]
    raw = await asyncio.gather(*tasks)
    all_expanded = [item for _, variants in sorted(raw) for item in variants]

    # 保存报告（不覆盖 checkpoint 数据文件）
    report = {
        "batch_id": batch_id,
        "method": "rephrase",
        "total_generated": total_written,
        "total_from_prev": skipped,
        "skipped_dedup": skipped_dedup,
        "time": datetime.now().isoformat(),
        "api_usage": usage_stats.summary(),
    }
    report_file = output_path / f"{batch_id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"改述扩展完成，本批生成 {total_written} 条（跳过 {skipped} 已处理 + {skipped_dedup} 去重），输出至 {results_file}")
    return all_expanded


# ============================================================
# 扩展方式 2: 追问链
# ============================================================
def expand_followup(
    seeds: list[dict],
    config: dict,
    output_dir: str,
    chain_length: int = 4,
    limit: Optional[int] = None,
    progress_callback=None,
) -> list[dict]:
    """对每条种子生成追问链（并发）"""
    return asyncio.run(_expand_followup_async(seeds, config, output_dir, chain_length, limit, progress_callback))


async def _expand_followup_async(
    seeds: list[dict],
    config: dict,
    output_dir: str,
    chain_length: int,
    limit: Optional[int],
    progress_callback=None,
) -> list[dict]:
    client = create_client(config, "expansion")
    template = load_prompt("expand_followup")
    batch_id = f"followup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    concurrency = config.get("generation", {}).get("expansion", {}).get("concurrency", 5)
    sem = asyncio.Semaphore(concurrency)

    if limit:
        seeds = seeds[:limit]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"{batch_id}.jsonl"

    checkpoint_file = output_path / "followup_checkpoint.json"
    processed_ids: set[str] = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            processed_ids = set(json.load(f))

    pending = [s for s in seeds if s.get("id", "") not in processed_ids]
    skipped = len(seeds) - len(pending)
    if skipped:
        logger.info(f"追问跳过已处理 {skipped} 条种子，待处理 {len(pending)} 条")

    if not pending:
        return []

    dedup = DedupTracker(threshold=0.65)
    for s in seeds:
        if s.get("id", "") in processed_ids:
            dedup.add(s.get("question", ""))
    if dedup.count:
        logger.info(f"重复校验: 已加载 {dedup.count} 个已追问种子的问题指纹")

    write_lock = asyncio.Lock()
    total_written = 0
    skipped_dedup = 0
    seeds_done = 0

    async def process_one(i: int, seed: dict):
        nonlocal total_written, skipped_dedup, seeds_done
        seed_id = seed.get("id", "")
        async with sem:
            try:
                if _check_stop():
                    return i, []

                question = seed.get("question", "")
                if not dedup.deduped_add(question):
                    skipped_dedup += 1
                    logger.info(f"  [{i+1}/{len(pending)}] 跳过（问题重复）: {question[:50]}...")
                    return i, []

                logger.info(f"[追问 {i+1}/{len(pending)}] {question[:50]}...")
                prompt = _safe_format(
                    template,
                    chain_length=chain_length,
                    original_question=seed["question"],
                    original_answer=seed.get("answer", seed.get("summary", seed.get("description", ""))),
                    insurance_type=seed.get("insurance_type", "其他保险"),
                    parent_id=seed_id,
                )
                try:
                    response = await client.acall(prompt)
                    turns = parse_json(response)
                    for j, t in enumerate(turns):
                        t["id"] = f"{batch_id}_{i:04d}_t{j+1:02d}"
                        t["generation_method"] = "followup"
                        t["parent_id"] = seed_id
                        t["insurance_type"] = seed.get("insurance_type", "其他保险")
                        t["batch_id"] = batch_id
                        t["created_at"] = datetime.now().isoformat()
                    turns = [_normalize_qa(t) for t in turns]
                    turns = [t for t in turns if _is_valid_qa(t)]
                    async with write_lock:
                        with open(results_file, "a", encoding="utf-8") as f:
                            for t in turns:
                                f.write(json.dumps(t, ensure_ascii=False) + "\n")
                        total_written += len(turns)
                        processed_ids.add(seed_id)
                        with open(checkpoint_file, "w", encoding="utf-8") as mf:
                            json.dump(list(processed_ids), mf, ensure_ascii=False)
                    logger.info(f"  → {len(turns)} 轮追问（累计 {total_written}）")
                    return i, turns
                except Exception as e:
                    logger.error(f"  → 失败: {e}")
                    return i, []
            finally:
                seeds_done += 1
                if progress_callback:
                    progress_callback(seeds_done, len(pending))

    tasks = [process_one(i, seed) for i, seed in enumerate(pending)]
    raw = await asyncio.gather(*tasks)
    all_expanded = [item for _, turns in sorted(raw) for item in turns]

    report = {
        "batch_id": batch_id,
        "method": "followup",
        "total_generated": total_written,
        "total_from_prev": skipped,
        "skipped_dedup": skipped_dedup,
        "time": datetime.now().isoformat(),
        "api_usage": usage_stats.summary(),
    }
    report_file = output_path / f"{batch_id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"追问扩展完成，本批生成 {total_written} 条（跳过 {skipped} 已处理 + {skipped_dedup} 去重），输出至 {results_file}")
    return all_expanded


# ============================================================
# 扩展方式 3: 文档参数驱动的模板生成
# ============================================================

# 固定句式模板（占位符从文档中提取的参数填入）
TEMPLATES = {
    "赔付判断": "{insurance_type}中，{scenario}，能赔吗？",
    "条款解读": '{insurance_type}里提到"{clause}"，这具体是什么意思？',
    "时限说明": "{insurance_type}的{time_period}是怎么计算的？超过了怎么办？",
    "金额计算": "{insurance_type}约定的{amount}，实际理赔时怎么算？",
    "操作指引": "办理{insurance_type}的{procedure}需要准备什么？流程是怎样的？",
    "除外责任": '{insurance_type}中"{exclusion}"属于除外责任吗？有没有例外情况？',
}

# 险种 → 大类映射，用于模板过滤和专属模板选择
INSURANCE_CATEGORY: dict[str, str] = {
    "百万医疗险": "人身险_医疗",
    "小额医疗险": "人身险_医疗",
    "重疾险":     "人身险_重疾",
    "防癌险":     "人身险_重疾",
    "惠民保":     "人身险_医疗",
    "长期护理险": "人身险_医疗",
    "定期寿险":   "人身险_寿险",
    "终身寿险":   "人身险_寿险",
    "增额终身寿": "人身险_寿险",
    "年金险":     "人身险_年金",
    "综合意外险": "人身险_意外",
    "旅游意外险": "人身险_意外",
    "交强险":         "财产险_车",
    "车损险":         "财产险_车",
    "第三者责任险":   "财产险_车",
    "车上人员责任险": "财产险_车",
    "新能源车险":     "财产险_车",
    "企业财产险": "财产险_企业",
    "雇主责任险": "财产险_企业",
    "公众责任险": "财产险_企业",
}

_KNOWN_INSURANCE_TYPES: frozenset[str] = frozenset(INSURANCE_CATEGORY.keys())


def _normalize_insurance_type(raw: str) -> str:
    """
    将 LLM 可能返回的产品级名称（如"尊享一生2024"）标准化为已知险种名。
    已知险种名直接返回；否则用 doc_preprocessor 的正则规则重新匹配。
    匹配不到则返回 "通用"。
    """
    if raw in _KNOWN_INSURANCE_TYPES:
        return raw
    from doc_preprocessor import INSURANCE_TYPE_PATTERNS
    for std_type, patterns in INSURANCE_TYPE_PATTERNS.items():
        if any(re.search(p, raw) for p in patterns):
            return std_type
    return "通用"


# 通用模板在某些险种大类下不适用，生成时跳过
# key: template_name → value: 应跳过的 category 集合
TEMPLATE_EXCLUSIONS: dict[str, set[str]] = {
    # 财产险没有等待期/犹豫期/宽限期概念
    "时限说明": {"财产险_车", "财产险_企业"},
}

# 险种大类专属模板
# 每条 = (template_name, template_str, [需要的 doc_params 字段名])
CATEGORY_TEMPLATES: dict[str, list[tuple[str, str, list[str]]]] = {
    "人身险_重疾": [
        (
            "轻重症区分",
            '{insurance_type}里"{clause}"算轻症还是重症？两者赔付比例有何区别？',
            ["key_clauses"],
        ),
        (
            "带病投保",
            "{insurance_type}投保时{scenario}，能正常核保吗？会有哪些限制或除外？",
            ["user_scenarios"],
        ),
        (
            "产品对比",
            "{insurance_type}和防癌险的保障范围有什么核心差异？分别适合哪类人群？",
            [],
        ),
    ],
    "人身险_医疗": [
        (
            "社保衔接",
            "{insurance_type}中{amount}，社保报销后剩余部分商业险能继续报销吗？",
            ["specific_amounts"],
        ),
        (
            "免赔额规则",
            "{insurance_type}的{amount}免赔额，每次就医都重新起算吗？有没有累计方式？",
            ["specific_amounts"],
        ),
        (
            "产品对比",
            "百万医疗险、小额医疗险和惠民保有什么区别？{insurance_type}适合哪类人群？",
            [],
        ),
    ],
    "人身险_寿险": [
        (
            "受益人设置",
            "{insurance_type}的受益人可以随时更改吗？受益人先于被保险人去世怎么处理？",
            [],
        ),
        (
            "保额调整",
            "{insurance_type}的保额{amount}，投保后能申请增加保额吗？需要重新核保吗？",
            ["specific_amounts"],
        ),
        (
            "产品对比",
            "定期寿险、终身寿险和增额终身寿有什么本质区别？{insurance_type}适合什么情况下购买？",
            [],
        ),
    ],
    "人身险_年金": [
        (
            "产品对比",
            "年金险和增额终身寿险有什么核心差异？{insurance_type}更适合哪类养老规划需求？",
            [],
        ),
        (
            "收益计算",
            "{insurance_type}中{amount}，实际收益是怎么计算的？和银行定期存款比哪个合适？",
            ["specific_amounts"],
        ),
    ],
    "人身险_意外": [
        (
            "意外认定标准",
            "{insurance_type}中，{scenario}能否被认定为意外伤害？认定的核心标准是什么？",
            ["user_scenarios"],
        ),
        (
            "职业类别影响",
            "{insurance_type}中，{procedure}职业类别是如何影响保费和保障范围的？",
            ["procedures"],
        ),
        (
            "产品对比",
            "综合意外险和旅游意外险有什么区别？{insurance_type}在境外出险能赔吗？",
            [],
        ),
    ],
    "财产险_车": [
        (
            "定损流程",
            "{insurance_type}出险后，定损的具体流程是什么？车主如何保障自己的定损权益？",
            [],
        ),
        (
            "不计免赔",
            '{insurance_type}的"{exclusion}"不计免赔附加险，具体能免除哪些免赔责任？',
            ["exclusions"],
        ),
        (
            "NCD折扣",
            "{insurance_type}续保时，无赔款优待系数是怎么计算的？出险一次会影响多少？",
            [],
        ),
        (
            "产品对比",
            "交强险、第三者责任险和车损险分别保什么？{insurance_type}和其他车险险种怎么搭配？",
            [],
        ),
    ],
    "财产险_企业": [
        (
            "责任认定",
            "{insurance_type}中{scenario}，企业与保险公司之间的赔偿责任是如何划分的？",
            ["user_scenarios"],
        ),
        (
            "理赔材料",
            "申请{insurance_type}理赔需要提交哪些{procedure}材料？有没有申请时间限制？",
            ["procedures"],
        ),
        (
            "产品对比",
            "雇主责任险和公众责任险有什么区别？{insurance_type}是否能覆盖员工工伤以外的第三方责任？",
            [],
        ),
    ],
}


# 跨险种对比问题对（仅当两边险种文档都在库中时才生成）
CROSS_COMPARISON_PAIRS: list[tuple[str, str, str]] = [
    # 健康险横向
    ("重疾险", "百万医疗险",
     "{type_a}和{type_b}都能覆盖大病费用，两者有什么本质区别？可以互相替代吗？"),
    ("重疾险", "防癌险",
     "{type_a}已包含癌症保障，还有必要单独购买{type_b}吗？保障范围有何差异？"),
    ("百万医疗险", "惠民保",
     "{type_a}和{type_b}都是报销型医疗险，价格相差这么大，保障有什么实质不同？"),
    ("重疾险", "长期护理险",
     "{type_a}理赔后{type_b}还能继续赔付吗？两者的保障侧重点有什么不同？"),
    ("百万医疗险", "长期护理险",
     "{type_a}和{type_b}在老年阶段的保障有什么区别？是否有必要同时配置？"),
    ("惠民保", "重疾险",
     "{type_a}保费这么低，能替代{type_b}吗？两者核心差距在哪里？"),
    # 寿险横向
    ("定期寿险", "终身寿险",
     "{type_a}和{type_b}都提供身故保障，哪个更划算？什么情况下选哪种？"),
    ("终身寿险", "增额终身寿",
     "{type_a}和{type_b}有什么区别？{type_b}的「增额」体现在哪里？"),
    ("定期寿险", "增额终身寿",
     "{type_a}保障期有限但保费低，{type_b}终身有效但保费高，如何根据需求选择？"),
    # 储蓄/养老横向
    ("增额终身寿", "年金险",
     "{type_a}和{type_b}都能用于长期储蓄和养老规划，核心差异和选择逻辑是什么？"),
    ("年金险", "定期寿险",
     "{type_a}侧重生存给付，{type_b}侧重身故保障，两者能否互相补充？"),
    # 寿险 vs 意外险
    ("定期寿险", "综合意外险",
     "{type_a}和{type_b}都有身故赔付，区别是什么？是否需要同时购买？"),
    ("综合意外险", "百万医疗险",
     "{type_a}和{type_b}在意外住院方面有保障重叠吗？如何合理搭配？"),
    # 健康险 vs 意外险跨类
    ("综合意外险", "重疾险",
     "意外导致重疾时，{type_a}和{type_b}会重复赔付吗？两者如何协调理赔？"),
    # 车险横向
    ("交强险", "第三者责任险",
     "{type_a}是强制险，{type_b}是商业险，保障范围如何区分？出险时赔付顺序怎么排？"),
    ("车损险", "第三者责任险",
     "{type_a}和{type_b}分别保什么损失？单方事故和双方事故下各自如何赔付？"),
    ("交强险", "车损险",
     "{type_a}和{type_b}能叠加赔付吗？只投了{type_a}，车辆自身损失能报销吗？"),
    # 企业险横向
    ("雇主责任险", "公众责任险",
     "{type_a}和{type_b}都是企业责任险，分别保什么风险？企业应如何搭配投保？"),
]


def _build_cross_comparisons(available_types: set[str]) -> list[dict]:
    """根据当前文档库已有险种，生成跨险种对比问题（两边文档均在库中才生成）"""
    combinations = []
    for type_a, type_b, tpl in CROSS_COMPARISON_PAIRS:
        if type_a in available_types and type_b in available_types:
            combinations.append({
                "template_name": "产品对比",
                "question": _safe_format(tpl, type_a=type_a, type_b=type_b),
                "insurance_type": f"{type_a}/{type_b}",
                "doc_file": "",
                "_cross_types": [type_a, type_b],  # 供 answer_one 取 chunks，不写入输出
            })
    return combinations


def _build_combinations_from_doc(doc_params: dict) -> list[dict]:
    """根据单个文档的提取参数构建模板组合（含险种专属模板）"""
    insurance_type = _normalize_insurance_type(doc_params.get("insurance_type") or "通用")
    doc_file = doc_params.get("doc_file", "")
    category = INSURANCE_CATEGORY.get(insurance_type, "通用")

    combinations = []

    def pick(lst: list, n: int = 3) -> list:
        # 移除 FALLBACK 兜底。如果没有从文档提取到真实的参数，直接返回空列表
        if not lst:
            return []
        return lst[:n]

    def combo(template_name: str, question: str) -> dict:
        return {
            "template_name": template_name,
            "question": question,
            "insurance_type": insurance_type,
            "doc_file": doc_file,
        }

    # ── 通用模板（按险种大类过滤不适用的，参数为空则不生成） ────────
    if category not in TEMPLATE_EXCLUSIONS.get("赔付判断", set()):
        for scenario in pick(doc_params.get("user_scenarios", [])):
            combinations.append(combo("赔付判断", _safe_format(
                TEMPLATES["赔付判断"], insurance_type=insurance_type, scenario=scenario)))

    if category not in TEMPLATE_EXCLUSIONS.get("条款解读", set()):
        for clause in pick(doc_params.get("key_clauses", [])):
            combinations.append(combo("条款解读", _safe_format(
                TEMPLATES["条款解读"], insurance_type=insurance_type, clause=clause)))

    if category not in TEMPLATE_EXCLUSIONS.get("时限说明", set()):
        for period in pick(doc_params.get("time_periods", [])):
            combinations.append(combo("时限说明", _safe_format(
                TEMPLATES["时限说明"], insurance_type=insurance_type, time_period=period)))

    if category not in TEMPLATE_EXCLUSIONS.get("金额计算", set()):
        for amount in pick(doc_params.get("specific_amounts", [])):
            combinations.append(combo("金额计算", _safe_format(
                TEMPLATES["金额计算"], insurance_type=insurance_type, amount=amount)))

    if category not in TEMPLATE_EXCLUSIONS.get("操作指引", set()):
        for procedure in pick(doc_params.get("procedures", [])):
            combinations.append(combo("操作指引", _safe_format(
                TEMPLATES["操作指引"], insurance_type=insurance_type, procedure=procedure)))

    if category not in TEMPLATE_EXCLUSIONS.get("除外责任", set()):
        for exclusion in pick(doc_params.get("exclusions", [])):
            combinations.append(combo("除外责任", _safe_format(
                TEMPLATES["除外责任"], insurance_type=insurance_type, exclusion=exclusion)))

    # ── 险种大类专属模板 ─────────────────────────────────────────────
    for tpl_name, tpl_str, param_fields in CATEGORY_TEMPLATES.get(category, []):
        if not param_fields:
            combinations.append(combo(tpl_name, _safe_format(
                tpl_str, insurance_type=insurance_type)))
            continue

        # 只用文档中真实提取的参数（pick 返回空则不生成）
        primary_field = param_fields[0]
        for val in pick(doc_params.get(primary_field, []), n=2):
            filled = _safe_format(
                tpl_str,
                insurance_type=insurance_type,
                clause=val, scenario=val, amount=val,
                procedure=val, exclusion=val,
            )
            combinations.append(combo(tpl_name, filled))

    return combinations


def expand_template(
    config: dict,
    output_dir: str,
    chunks_dir: str = "./data/chunks",
) -> list[dict]:
    """从文档提取参数后生成模板 Q&A（文档驱动，并发）"""
    return asyncio.run(_expand_template_async(config, output_dir, chunks_dir))


async def _expand_template_async(
    config: dict,
    output_dir: str,
    chunks_dir: str,
) -> list[dict]:
    import itertools
    from param_extractor import extract_all_params

    chunks_file = Path(chunks_dir) / "chunks.jsonl"
    if not chunks_file.exists():
        logger.error(f"chunks 文件不存在: {chunks_file}")
        return []

    # Step 1: 提取文档参数
    logger.info("--- 提取文档参数 ---")
    all_doc_params = extract_all_params(str(chunks_file), chunks_dir, config)

    if not all_doc_params:
        logger.error("未提取到文档参数，终止模板扩展")
        return []

    # Step 2: 构建 chunks 双索引（doc_file → texts，insurance_type → texts）
    # 同时构建 chunk 文本到来源文档的反向映射，用于跨险种对比的数据溯源
    chunks_by_doc: dict[str, list[str]] = {}
    chunks_by_type: dict[str, list[str]] = {}
    chunk_to_doc: dict[str, str] = {}  # chunk 文本 → doc_file，用于溯源
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                doc = chunk["doc_file"]
                ins_type = chunk.get("insurance_type", "")
                text = chunk["text"]
                chunks_by_doc.setdefault(doc, []).append(text)
                chunk_to_doc[text] = doc
                if ins_type:
                    chunks_by_type.setdefault(ins_type, []).append(text)

    # Step 3: 按文档构建单险种模板问题
    all_combinations = []
    for doc_params in all_doc_params:
        combos = _build_combinations_from_doc(doc_params)
        all_combinations.extend(combos)

    # Step 4: 跨险种对比问题（仅当两边险种均在文档库中）
    available_types = set(chunks_by_type.keys())
    cross_combos = _build_cross_comparisons(available_types)
    all_combinations.extend(cross_combos)

    logger.info(
        f"共生成 {len(all_combinations)} 个模板问题"
        f"（单险种 {len(all_combinations) - len(cross_combos)}，"
        f"跨险种对比 {len(cross_combos)}，来自 {len(all_doc_params)} 个文档）"
    )

    # 断点续传：加载已处理问题的清单（以问题文本作为幂等 key）
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    template_manifest_file = output_path / "template_processed.json"
    processed_questions: set[str] = set()
    if template_manifest_file.exists():
        with open(template_manifest_file, "r", encoding="utf-8") as f:
            processed_questions = set(json.load(f))

    pending_combos = [(i, c) for i, c in enumerate(all_combinations)
                      if c["question"] not in processed_questions]
    skipped = len(all_combinations) - len(pending_combos)
    if skipped:
        logger.info(f"跳过已处理 {skipped} 个模板问题，待生成 {len(pending_combos)} 个")

    client = create_client(config, "expansion")
    concurrency = config.get("generation", {}).get("expansion", {}).get("concurrency", 5)
    sem = asyncio.Semaphore(concurrency)
    batch_id = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results_file = output_path / f"{batch_id}.jsonl"
    write_lock = asyncio.Lock()
    total_written = 0

    # 单险种问题：严格基于文档，避免幻觉
    doc_system_prompt = (
        "你是资深保险顾问。请严格基于提供的文档内容回答问题，"
        "不得添加文档未提及的信息。答案100-300字，引用具体条款时注明出处。"
        "末尾附一句：'以上内容依据所提供文档，请以最新官方文件为准。'"
    )
    # 跨险种对比：允许结合专业知识，不强制锁定单一文档
    cross_system_prompt = (
        "你是资深保险顾问，拥有15年保险行业经验。"
        "回答跨险种对比问题时，请综合运用保险专业知识和所提供的文档片段，"
        "从功能定位、保障范围、适用人群三个维度进行对比，条理清晰，150-300字。"
        "末尾附一句：'以上内容仅供参考，请结合自身情况咨询专业顾问后决策。'"
    )

    async def answer_one(i: int, combo: dict) -> tuple[int, dict | None]:
        nonlocal total_written
        async with sem:
            doc_file = combo.get("doc_file", "")
            cross_types = combo.get("_cross_types")
            question = combo["question"]

            if cross_types:
                # 跨险种对比：每个险种取最相关的 2 个 chunk，并记录实际来源文档
                context_parts = []
                source_docs_set: set[str] = set()
                for t in cross_types:
                    type_chunks = chunks_by_type.get(t, [])
                    selected = _retrieve_chunks(question, type_chunks, k=2)
                    context_parts.extend(selected)
                    for chunk_text in selected:
                        doc = chunk_to_doc.get(chunk_text)
                        if doc:
                            source_docs_set.add(doc)
                context = "\n\n".join(context_parts)
                # 优先记录实际使用的文档文件名；无法溯源时降级为险种名
                source_doc = "|".join(sorted(source_docs_set)) if source_docs_set else "/".join(cross_types)
                system_prompt = cross_system_prompt
            else:
                # 单险种：从该文档的所有 chunk 中检索最相关的 5 个
                all_chunks = chunks_by_doc.get(doc_file, [])
                context = "\n\n".join(_retrieve_chunks(question, all_chunks, k=5))
                source_doc = doc_file
                system_prompt = doc_system_prompt

            prompt = f"""请基于以下文档内容回答问题。

## 文档内容
{context}

## 问题
{question}"""

            try:
                response = await client.acall(prompt, system=system_prompt)
                qa = {
                    "id": f"{batch_id}_{i:05d}",
                    "question": question,
                    "answer": response.strip(),
                    "difficulty": "入门",
                    "question_type": combo["template_name"],
                    "insurance_type": combo["insurance_type"],
                    "source_doc": source_doc,
                    "generation_method": "template_cross_compare" if cross_types else "template_from_doc",
                    "batch_id": batch_id,
                    "created_at": datetime.now().isoformat(),
                }
                # 立即追加写入，并更新断点清单
                async with write_lock:
                    with open(results_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total_written += 1
                    processed_questions.add(question)
                    with open(template_manifest_file, "w", encoding="utf-8") as mf:
                        json.dump(list(processed_questions), mf, ensure_ascii=False)
                logger.info(f"[模板 {i+1}/{len(all_combinations)}] {question[:40]}... (累计 {total_written})")
                return i, qa
            except Exception as e:
                logger.error(f"[模板 {i+1}] 失败: {e}")
                return i, None

    tasks = [answer_one(i, combo) for i, combo in pending_combos]
    raw = await asyncio.gather(*tasks)
    all_expanded = [qa for _, qa in sorted(raw) if qa is not None]

    # 生成报告（不重写已有数据文件）
    report = {
        "batch_id": batch_id,
        "method": "template_from_doc",
        "total_generated": total_written,
        "time": datetime.now().isoformat(),
        "api_usage": usage_stats.summary(),
    }
    report_file = output_path / f"{batch_id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"模板扩展完成，本批生成 {total_written} 条，输出至 {results_file}")
    return all_expanded


# ============================================================
# 工具函数
# ============================================================
# ============================================================
# 工具路由型 Q&A 生成（纯模板，零 LLM token）
# ============================================================

# 核保场景：高频疾病 × 险种 × 问法
_UNDERWRITING_DISEASES = [
    "乙肝", "高血压", "糖尿病", "甲状腺结节", "乳腺结节",
    "肺结节", "脂肪肝", "痛风", "心脏病", "脑梗", "肾病",
    "哮喘", "抑郁症", "癫痫", "甲亢", "宫颈病变", "子宫肌瘤",
    "卵巢囊肿", "前列腺增生", "腰椎间盘突出", "颈椎病",
    "过敏性鼻炎", "湿疹", "银屑病", "类风湿关节炎",
    "贫血", "肝囊肿", "肾囊肿", "胆囊息肉", "胃溃疡",
]
_UNDERWRITING_INSURANCE_TYPES = [
    "重疾险", "百万医疗险", "定期寿险", "终身寿险", "意外险", "医疗险",
]
_UNDERWRITING_QUESTION_TEMPLATES = [
    "有{disease}病史，能买{insurance_type}吗？",
    "{disease}患者购买{insurance_type}需要注意什么？",
    "我之前得过{disease}，现在{insurance_type}能正常核保吗？",
    "{disease}算{insurance_type}的健康告知范围吗？",
    "投保{insurance_type}时，{disease}需要如实告知吗？",
    "有{disease}记录，{insurance_type}会被拒保还是加费？",
    "{disease}手术后，{insurance_type}可以正常投保吗？",
    "体检发现{disease}，还能买{insurance_type}吗？",
]
_UNDERWRITING_ANSWER_TEMPLATE = (
    "您好！{disease}是否影响{insurance_type}的投保，需要核保师根据您的具体"
    "病情、治疗情况、最近体检结果等综合评估。不同保险公司的核保规则不同，"
    "可能的结果包括：标准体承保、加费承保、除外承保或拒保。"
    "建议您如实填写健康告知，并通过我们的核保工具进行智能测评，"
    "我们将为您匹配最合适的方案。"
)

# 保费计算场景
_PREMIUM_TEMPLATES = [
    "{}岁，想买{}万保额的{}，大概要多少钱？",
    "{}岁男性购买{}，保额{}万，保期{}年，每年保费是多少？",
    "{}岁女性，想了解{}保费，保额{}万怎么算？",
    "{}岁，计划买{}，预算{}元/年，保额大概能买多少？",
]
_PREMIUM_ANSWER_TEMPLATE = (
    "保费金额受年龄、性别、健康状况、保额、保期、缴费方式等多种因素影响，"
    "无法直接给出精确报价。建议使用我们的保费测算工具，"
    "输入您的具体信息即可获得实时报价，并可对比多款产品。"
)

# 现金价值/退保场景
_CASH_VALUE_TEMPLATES = [
    "买了{}的{}，交了{}年保费，退保能退多少钱？",
    "{}现在的现金价值是多少？",
    "{}交了{}年，退保金怎么计算？",
    "{}减额缴清后，现金价值有什么变化？",
]
_CASH_VALUE_ANSWER_TEMPLATE = (
    "退保金（现金价值）根据保单具体条款、已缴保费年数、险种类型等综合计算，"
    "每个时间节点的现金价值均有所不同。建议通过保单查询工具，"
    "输入您的保单号即可获取精确的当前现金价值和退保金信息。"
    "请注意，退保可能导致保障终止，且早期退保通常损失较大。"
)

# 理赔金额计算场景
_CLAIM_TEMPLATES = [
    "社保报销后，{}还能报多少？",
    "我住院花了{}元，{}能报销多少？",
    "理赔时，{}和社保怎么协调赔付？",
    "{}的理赔金额怎么计算？有没有上限？",
    "发生意外后，{}能赔多少钱？",
]
_CLAIM_ANSWER_TEMPLATE = (
    "理赔金额取决于实际发生费用、保险责任范围、免赔额、社保报销情况、"
    "赔付比例等多个因素。建议您通过理赔计算工具，"
    "填写实际费用明细即可获取预估赔付金额。"
    "如有疑问，也可联系我们的理赔专员进行详细说明。"
)

# 账户查询场景（万能险/分红险）
_ACCOUNT_TEMPLATES = [
    "我的{}万能账户里现在有多少钱？",
    "{}的分红今年是多少？",
    "{}账户价值怎么查询？",
    "{}万能账户收益率是多少？",
]
_ACCOUNT_ANSWER_TEMPLATE = (
    "账户价值和分红数据需要通过保单查询系统实时获取，"
    "金额受市场利率、结算利率调整等因素影响，会定期更新。"
    "建议通过保单账户查询工具，输入保单号即可查看最新账户价值、"
    "历史结算利率及分红信息。"
)

TOOL_ROUTING_SCENARIOS = {
    "underwriting_check": {
        "description": "核保/健康告知查询",
        "tool_params_template": {"disease": "{disease}", "insurance_type": "{insurance_type}"},
    },
    "premium_calculator": {
        "description": "保费计算",
        "tool_params_template": {"insurance_type": "{insurance_type}"},
    },
    "cash_value_query": {
        "description": "现金价值/退保金查询",
        "tool_params_template": {"policy_type": "{insurance_type}"},
    },
    "claim_calculator": {
        "description": "理赔金额计算",
        "tool_params_template": {"insurance_type": "{insurance_type}"},
    },
    "account_query": {
        "description": "分红/万能账户查询",
        "tool_params_template": {"account_type": "万能账户"},
    },
}


def expand_tool_routing(output_dir: str) -> list[dict]:
    """
    纯模板生成工具路由型 Q&A，零 LLM token。
    覆盖核保/保费计算/现金价值/理赔计算/账户查询五类高频场景。
    """
    from datetime import datetime
    import uuid

    results = []
    batch_id = f"tool_routing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    created_at = datetime.now().isoformat()

    def _make_qa(question, answer, tool_routing, tool_params, insurance_type, question_type):
        return {
            "id": f"{batch_id}_{len(results):05d}",
            "question": question,
            "answer": answer,
            "qa_category": "tool_routed",
            "insurance_type": insurance_type,
            "product_name": "",
            "business_stage": _tool_to_stage(tool_routing),
            "question_type": question_type,
            "difficulty": "入门",
            "tool_routing": tool_routing,
            "tool_params": tool_params,
            "is_tool_routed": True,
            "requires_clarification": False,
            "misconception": "",
            "source_chunk_id": "",
            "source_doc": "",
            "source_reference": "",
            "parent_id": "",
            "generation_method": "tool_routing_template",
            "batch_id": batch_id,
            "created_at": created_at,
            "tags": ["工具路由", tool_routing],
        }

    # 1. 核保场景（疾病 × 险种 × 问法）
    for disease in _UNDERWRITING_DISEASES:
        for ins_type in _UNDERWRITING_INSURANCE_TYPES:
            for tpl in _UNDERWRITING_QUESTION_TEMPLATES:
                question = tpl.format(disease=disease, insurance_type=ins_type)
                answer = _UNDERWRITING_ANSWER_TEMPLATE.format(
                    disease=disease, insurance_type=ins_type
                )
                results.append(_make_qa(
                    question=question,
                    answer=answer,
                    tool_routing="underwriting_check",
                    tool_params={"disease": disease, "insurance_type": ins_type},
                    insurance_type=ins_type,
                    question_type="健康告知",
                ))

    # 2. 保费计算场景
    _premium_combos = [
        ("30", "50", "重疾险"), ("35", "100", "重疾险"), ("40", "50", "定期寿险"),
        ("25", "100", "百万医疗险"), ("45", "50", "终身寿险"), ("30", "30", "意外险"),
    ]
    for age, coverage, ins_type in _premium_combos:
        for tpl in _PREMIUM_TEMPLATES[:2]:
            try:
                if "{}" in tpl:
                    parts = tpl.count("{}")
                    if parts == 3:
                        question = tpl.format(age, coverage, ins_type)
                    elif parts == 4:
                        question = tpl.format(age, ins_type, coverage, "20")
                    else:
                        continue
                    results.append(_make_qa(
                        question=question,
                        answer=_PREMIUM_ANSWER_TEMPLATE,
                        tool_routing="premium_calculator",
                        tool_params={"age": age, "coverage": coverage, "insurance_type": ins_type},
                        insurance_type=ins_type,
                        question_type="计算说明",
                    ))
            except (IndexError, KeyError):
                continue

    # 3. 现金价值/退保场景
    _cash_combos = [
        ("重疾险", "3"), ("终身寿险", "5"), ("增额终身寿", "10"), ("年金险", "7"),
        ("两全险", "5"), ("万能险", "3"),
    ]
    for ins_type, years in _cash_combos:
        for tpl in _CASH_VALUE_TEMPLATES[:3]:
            try:
                parts = tpl.count("{}")
                if parts == 3:
                    question = tpl.format("某保险公司", ins_type, years)
                elif parts == 2:
                    question = tpl.format(ins_type, years)
                elif parts == 1:
                    question = tpl.format(ins_type)
                else:
                    continue
                results.append(_make_qa(
                    question=question,
                    answer=_CASH_VALUE_ANSWER_TEMPLATE,
                    tool_routing="cash_value_query",
                    tool_params={"insurance_type": ins_type, "years_paid": years},
                    insurance_type=ins_type,
                    question_type="计算说明",
                ))
            except (IndexError, KeyError):
                continue

    # 4. 理赔计算场景
    _claim_combos = [
        "百万医疗险", "重疾险", "意外险", "交强险", "第三者责任险",
    ]
    for ins_type in _claim_combos:
        for tpl in _CLAIM_TEMPLATES:
            try:
                parts = tpl.count("{}")
                if parts == 2:
                    question = tpl.format("5万元", ins_type)
                elif parts == 1:
                    question = tpl.format(ins_type)
                else:
                    continue
                results.append(_make_qa(
                    question=question,
                    answer=_CLAIM_ANSWER_TEMPLATE,
                    tool_routing="claim_calculator",
                    tool_params={"insurance_type": ins_type},
                    insurance_type=ins_type,
                    question_type="赔付判断",
                ))
            except (IndexError, KeyError):
                continue

    # 5. 账户查询场景
    _account_types = ["万能险", "分红险", "增额终身寿", "年金险"]
    for ins_type in _account_types:
        for tpl in _ACCOUNT_TEMPLATES:
            try:
                question = tpl.format(ins_type)
                results.append(_make_qa(
                    question=question,
                    answer=_ACCOUNT_ANSWER_TEMPLATE,
                    tool_routing="account_query",
                    tool_params={"insurance_type": ins_type},
                    insurance_type=ins_type,
                    question_type="操作指引",
                ))
            except (IndexError, KeyError):
                continue

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{batch_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for qa in results:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    logger.info(f"工具路由型 Q&A 生成完成: {len(results)} 条 → {out_file}")
    return results


# ============================================================
# 误解纠正型 Q&A 生成（P1）
# ============================================================

# 高频误解清单
COMMON_MISCONCEPTIONS = [
    {
        "misconception": "百万医疗险什么费用都能报销",
        "correct_understanding": "百万医疗险有免赔额（通常1万元）、除外责任（特定疾病/治疗方式）、医院级别限制，并非什么都报",
        "trigger_context": "百万医疗险理赔问题",
        "insurance_type": "医疗保险",
        "tags": ["百万医疗", "免赔额", "除外责任"],
    },
    {
        "misconception": "重疾险确诊即可获得赔付",
        "correct_understanding": "部分重疾需要满足约定的疾病状态（如心肌梗塞需达到特定诊断标准）或实施了约定手术，而非仅凭确诊报告",
        "trigger_context": "重疾险理赔条件",
        "insurance_type": "疾病保险",
        "tags": ["重疾险", "确诊标准", "理赔条件"],
    },
    {
        "misconception": "交强险可以全额赔偿对方损失",
        "correct_understanding": "交强险有有责/无责限额之分，且各项目限额固定：有责时死亡伤残18万、医疗1.8万、财产2000元；无责限额更低",
        "trigger_context": "交强险赔付问题",
        "insurance_type": "交强险",
        "tags": ["交强险", "限额", "有责无责"],
    },
    {
        "misconception": "犹豫期内退保可以全额退款没有任何损失",
        "correct_understanding": "犹豫期退保可退回大部分保费，但保险公司可能扣除体检费、工本费等少量费用（通常不超过10元），并非完全无损失",
        "trigger_context": "犹豫期退保",
        "insurance_type": "疾病保险",
        "tags": ["犹豫期", "退保", "费用扣除"],
    },
    {
        "misconception": "买了保险立刻就能获得理赔保障",
        "correct_understanding": "大多数健康险产品设有等待期（重疾险通常90-180天，医疗险30-90天），等待期内发生的疾病通常不赔",
        "trigger_context": "保险生效时间",
        "insurance_type": "医疗保险",
        "tags": ["等待期", "保险生效", "观察期"],
    },
    {
        "misconception": "免赔额每年只需满足一次",
        "correct_understanding": "不同产品设计不同：有的是年度累计免赔（满足一次后当年剩余就诊不再扣），有的是每次就医单独计算免赔额，需看具体条款",
        "trigger_context": "医疗险免赔额计算",
        "insurance_type": "医疗保险",
        "tags": ["免赔额", "年度累计", "逐次计算"],
    },
    {
        "misconception": "社保报销后医疗险会按剩余全额赔付",
        "correct_understanding": "医疗险通常对社保报销后的自付部分进行赔付，且还需扣除免赔额，并按约定比例（如80%-100%）赔付，不一定全赔",
        "trigger_context": "社保与商业医疗险协调赔付",
        "insurance_type": "医疗保险",
        "tags": ["社保统筹", "商业医疗险", "赔付比例"],
    },
    {
        "misconception": "定期寿险保满期后可以拿回保费",
        "correct_understanding": "定期寿险是消费型保险，保期内未出险则保费不退还，到期保障自动终止，与储蓄型保险不同",
        "trigger_context": "定期寿险满期处理",
        "insurance_type": "定期寿险",
        "tags": ["定期寿险", "消费型", "满期退保"],
    },
    {
        "misconception": "有医保就不需要再买商业医疗险",
        "correct_understanding": "医保报销比例有限（通常50-80%），且有起付线、目录外费用限制。商业医疗险可补充覆盖自付部分、目录外药品和高端医疗，建议搭配",
        "trigger_context": "医保与商业险关系",
        "insurance_type": "医疗保险",
        "tags": ["医保", "商业医疗险", "互补"],
    },
    {
        "misconception": "保险公司理赔故意拖延、找理由拒赔",
        "correct_understanding": "保险法规定保险公司需在30天内完成理赔核定，10天内支付赔款。拒赔必须说明理由。消费者可通过投诉或仲裁维权",
        "trigger_context": "理赔时效和争议处理",
        "insurance_type": "疾病保险",
        "tags": ["理赔时效", "拒赔争议", "消费者权益"],
    },
    {
        "misconception": "被保险人死亡后，保险金肯定给直系亲属",
        "correct_understanding": "保险金优先按保单指定受益人给付；如未指定受益人，才按法定继承顺序分配；指定受益人可以是任何人，不限于直系亲属",
        "trigger_context": "受益人和保险金给付",
        "insurance_type": "定期寿险",
        "tags": ["受益人", "法定继承", "保险金给付"],
    },
    {
        "misconception": "投保时隐瞒既往病史，出险时保险公司也无法查到",
        "correct_understanding": "保险公司理赔时会调取医疗记录，若发现未如实告知，可解除合同并不赔。保险法规定2年后不可因此解约，但故意欺诈除外",
        "trigger_context": "健康告知诚信原则",
        "insurance_type": "疾病保险",
        "tags": ["如实告知", "健康告知", "不可抗辩"],
    },
]


def expand_misconception_batch(config: dict, output_dir: str) -> list[dict]:
    """
    基于预定义误解清单，通过 LLM 生成误解纠正型问答对。
    每条误解生成1条高质量问答。
    """
    results = []
    batch_id = f"misconception_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    created_at = datetime.now().isoformat()

    prompt_path = Path(__file__).parent.parent / "prompts" / "expand_misconception.txt"
    prompt_tpl = prompt_path.read_text(encoding="utf-8")

    from llm_client import create_client
    client = create_client(config, "expansion")

    for misconception_def in COMMON_MISCONCEPTIONS:
        try:
            prompt = prompt_tpl.format(
                misconception=misconception_def["misconception"],
                insurance_type=misconception_def["insurance_type"],
                correct_understanding=misconception_def["correct_understanding"],
                trigger_context=misconception_def["trigger_context"],
            )
            response = client.call(prompt)
            qa_list = parse_json(response)
            if not qa_list:
                logger.warning(f"误解纠正生成失败: {misconception_def['misconception'][:20]}")
                continue
            qa = qa_list[0] if isinstance(qa_list, list) else qa_list

            qa.update({
                "id": f"{batch_id}_{len(results):05d}",
                "qa_category": "misconception_correction",
                "insurance_type": misconception_def["insurance_type"],
                "product_name": "",
                "business_stage": "通用",
                "misconception": misconception_def["misconception"],
                "is_tool_routed": False,
                "tool_routing": "",
                "tool_params": {},
                "requires_clarification": False,
                "source_chunk_id": "",
                "source_doc": "",
                "source_reference": "",
                "parent_id": "",
                "generation_method": "misconception_correction",
                "batch_id": batch_id,
                "created_at": created_at,
            })
            if not qa.get("tags"):
                qa["tags"] = misconception_def.get("tags", ["误解纠正"])
            results.append(qa)
        except Exception as e:
            logger.warning(f"误解纠正生成异常: {e}")
            continue

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{batch_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for qa in results:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    logger.info(f"误解纠正型 Q&A 生成完成: {len(results)} 条 → {out_file}")
    return results


def _tool_to_stage(tool_routing: str) -> str:
    mapping = {
        "underwriting_check": "健康告知",
        "premium_calculator": "投保咨询",
        "cash_value_query": "保全变更",
        "claim_calculator": "理赔审核",
        "account_query": "通用",
    }
    return mapping.get(tool_routing, "通用")


# ============================================================
# 保全变更场景模板（P1）
# ============================================================
POLICY_SERVICE_TEMPLATES = [
    "{insurance_type}买了之后可以退保吗？流程是怎样的？",
    "{insurance_type}要变更受益人需要什么手续？",
    "{insurance_type}的投保人可以变更吗？需要被保人同意吗？",
    "{insurance_type}忘记缴费了，过了宽限期怎么办？",
    "{insurance_type}可以减额缴清吗？减额后保障会有什么变化？",
    "{insurance_type}的缴费方式能从年缴改成月缴吗？",
    "我搬家了，{insurance_type}的联系地址怎么变更？",
    "{insurance_type}能不能附加其他险种？怎么加？",
    "{insurance_type}如何申请保单贷款？最多能贷多少？",
    "{insurance_type}保单复效需要什么条件和材料？",
]

_POLICY_SERVICE_ANSWER_TEMPLATES = {
    "退保": (
        "退保流程一般包括：填写退保申请书、提供身份证明、保单原件、银行账户信息等材料，"
        "提交至保险公司柜面或线上渠道处理。请注意，退保后保障立即终止，"
        "且早期退保现金价值通常远低于已缴保费，建议谨慎决定。"
        "具体退保金额请通过现金价值查询工具获取。"
    ),
    "受益人": (
        "变更受益人通常需要：投保人填写《变更申请书》、提供投保人身份证明、"
        "原保单（如需）等材料。如涉及主被保险人变更，"
        "还需被保险人书面同意。可通过线上APP或线下柜台办理，"
        "具体所需材料以各公司规定为准。"
    ),
    "复效": (
        "保单失效（宽限期后未缴费）后，通常可在2年内申请复效。"
        "复效需缴清欠费及利息，并可能需要重新进行健康告知/体检。"
        "如超过2年，则无法复效，需重新投保。"
    ),
    "default": (
        "该业务可通过线上APP、官网或线下柜台办理。"
        "具体所需材料和流程以保险公司实际规定为准，"
        "建议联系客服获取详细指引。"
    ),
}


def expand_policy_service(insurance_types: list[str], output_dir: str) -> list[dict]:
    """生成保全变更场景 Q&A（P1）"""
    results = []
    batch_id = f"policy_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    created_at = datetime.now().isoformat()

    long_term_types = {
        "重疾险", "终身寿险", "定期寿险", "增额终身寿",
        "年金险", "两全保险", "万能险", "分红险",
    }

    for ins_type in insurance_types:
        for tpl in POLICY_SERVICE_TEMPLATES:
            # 涉及退保金额的问法路由到工具
            if "退保" in tpl and "金额" in tpl or "多少钱" in tpl:
                continue  # 由 expand_tool_routing 中 cash_value_query 覆盖
            question = tpl.format(insurance_type=ins_type)
            # 选择答案模板
            answer_key = "default"
            for key in _POLICY_SERVICE_ANSWER_TEMPLATES:
                if key != "default" and key in tpl:
                    answer_key = key
                    break
            answer = _POLICY_SERVICE_ANSWER_TEMPLATES[answer_key]

            results.append({
                "id": f"{batch_id}_{len(results):05d}",
                "question": question,
                "answer": answer,
                "qa_category": "knowledge",
                "insurance_type": ins_type,
                "product_name": "",
                "business_stage": "保全变更",
                "question_type": "保全操作",
                "difficulty": "入门",
                "is_tool_routed": False,
                "tool_routing": "",
                "tool_params": {},
                "requires_clarification": False,
                "misconception": "",
                "source_chunk_id": "",
                "source_doc": "",
                "source_reference": "",
                "parent_id": "",
                "generation_method": "policy_service_template",
                "batch_id": batch_id,
                "created_at": created_at,
                "tags": ["保全变更", ins_type],
            })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{batch_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for qa in results:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    logger.info(f"保全变更 Q&A 生成完成: {len(results)} 条 → {out_file}")
    return results


def _save_results(
    qa_list: list[dict],
    output_dir: str,
    batch_id: str,
    method: str,
):
    """保存扩展结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / f"{batch_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    report = {
        "batch_id": batch_id,
        "method": method,
        "total_generated": len(qa_list),
        "time": datetime.now().isoformat(),
        "api_usage": usage_stats.summary(),
    }
    report_file = output_path / f"{batch_id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"保存 {len(qa_list)} 条到 {out_file}")


def main():
    parser = argparse.ArgumentParser(description="批量扩展器")
    parser.add_argument("--mode", choices=["rephrase", "followup", "template", "all"], required=True)
    parser.add_argument("--input", default=None, help="种子文件路径 (rephrase/followup 必需)")
    parser.add_argument("--output", default="./data/expanded")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode in ("rephrase", "followup", "all") and not args.input:
        parser.error("--input is required for rephrase/followup/all mode")

    if args.mode == "rephrase" or args.mode == "all":
        seeds = load_seeds(args.input)
        expand_rephrase(seeds, config, args.output, args.variants, args.limit)

    if args.mode == "followup" or args.mode == "all":
        seeds = load_seeds(args.input)
        expand_followup(seeds, config, args.output, limit=args.limit)

    if args.mode == "template" or args.mode == "all":
        expand_template(config, args.output)

    logger.info(f"最终 API 用量: {usage_stats.summary()}")


if __name__ == "__main__":
    main()
