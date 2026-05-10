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
