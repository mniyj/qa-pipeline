# 问答生成管道提速诊断与方案

## Context

用户反馈：很多知识库系统做切块和自动生成问答对非常快，本项目却很慢。需要分析根因并给出提速手段。

本文档先**诊断为什么慢**（区分"架构问题"与"业务设计问题"），再给出**分层提速方案**（按性价比排序，便于挑选落地）。

---

## 一、为什么"看起来很慢"——五层归因

### 1. 你的对比对象不对等

**"快"的知识库系统通常只做这两件事之一：**
- **切块 + 嵌入入库**（无 LLM 生成，纯 chunk → embedding → 向量库，1 万 chunk 几分钟完成）
- **检索时按需生成**（RAG，用户提问才调一次 LLM）

**你的系统在做的事完全不同：**
- 离线**批量预生成** 10 万+ 高质量 Q&A
- 每条 Q&A 都经历 生成 → 改述/追问扩展 → 多级质检
- 这本质上是**离线数据合成 pipeline**，不是检索系统

**量级估算**（[config/config.yaml:170-191](config/config.yaml)）：
- 假设 1000 chunk → 8000 种子（`pairs_per_chunk: 8`）
- 扩展倍数 = `rephrase_variants:5 + scenario_transfer:4 + follow_up_depth:4` ≈ **每条种子 13 次额外 LLM 调用**
- 总 LLM 调用 ≈ 1000（种子）+ 8000 × 13 ≈ **10.5 万次**
- DeepSeek 单次响应 ~3–8 秒 ⟹ 串行 70 万秒（约 8 天），并发 50 ⟹ 约 4 小时
- 加上 P2 质检（`verify_concurrency:30`、`scoring_concurrency:30`，[config/config.yaml:213-214](config/config.yaml)）还有数万次额外 LLM 调用

**结论：你的耗时不是"慢"，是"工作量×30倍 + 串行依赖"的必然结果。**

---

### 2. 模型选型：DeepSeek 在批量生成场景里偏慢

[src/llm_client.py:320-407](src/llm_client.py) 中默认使用 `deepseek-chat`：
- 单次响应延迟 3–8 秒（OpenAI gpt-4o-mini 通常 1–2 秒，gpt-4.1-nano 更快）
- 单账户 RPM 限制相对严格（默认 [src/llm_client.py:124-129](src/llm_client.py) 设为 30 RPM 时是节流主因）
- **官方支持 prompt caching**（context cache），但代码未启用 → 同一份 chunk 文本被反复发送

### 3. 架构层有 3 个具体可优化点

#### (a) httpx.AsyncClient 每次调用都新建（[src/llm_client.py:387](src/llm_client.py)）
```python
async with httpx.AsyncClient(timeout=120.0) as client:  # ← 每次重新建连
    response = await client.post(...)
```
TCP/TLS 握手开销在高并发下累积可观。应改为**模块级共享 client**。

#### (b) 扩展阶段不批量
[src/expander.py:266-403](src/expander.py)（rephrase）、[src/expander.py:421-559](src/expander.py)（followup）：
- 每条种子 × 5 改述 = 5 次 LLM 调用
- 实际上 1 次 LLM 调用就能在一个 prompt 里返回 5 条改述（JSON 数组）
- **节省 ~75% 的扩展调用次数**

#### (c) 没有响应级缓存
- 断点续传只跳过"已完成的 chunk/seed"
- 同样的 prompt 重跑（比如调试改 prompt 后想小范围验证）会全量重算
- 加一层 **磁盘 KV 缓存（key=hash(prompt+model+temperature)）** 即可

### 4. 质检阶段的 O(N²) embedding 比对

[src/quality_checker.py:323-371](src/quality_checker.py)：全量 Q&A embedding 一次性计算 + numpy 矩阵相似度
- 5 万条 × 5 万条相似度 = 2.5e9 次浮点 → 即使 numpy 也得几十秒
- **应使用 FAISS / HNSW** 索引做近邻查询（O(N log N)）

### 5. 业务流程是"分阶段"而非"流式"

[src/pipeline_runner.py](src/pipeline_runner.py) / [src/orchestrator.py](src/orchestrator.py)：
- 必须等"所有 chunk 切完"才开始生成种子
- 必须等"所有种子生成完"才开始扩展
- 必须等"所有扩展完"才开始质检

**改成流式后**：chunk 1 切完 → 立即去生成种子 → 立即去扩展。三个阶段重叠执行，总时长 ≈ max(各阶段时长) 而不是 sum。

---

## 二、用户决策（已锁定）

1. **落地范围**：Day 1-5 全套（一周，目标 5-10×）
2. **质量取舍**：**质量第一，不接受任何质量让步**。
   - ❌ 不降低扩展倍数（rephrase 仍 5、follow_up 仍 4）
   - ⏸ 换模型作为 A/B 实验项，**不默认切换**，要先有质量对比数据
3. 所有提速手段必须是"架构/工程层面"——只优化"怎么算"，不改"算什么"

## 三、提速方案（按性价比排序，⭐表示推荐）

| # | 方案 | 预期加速 | 改动量 | 风险 | 是否纳入本周 |
|---|------|----------|--------|------|------|
| 1 | **httpx.AsyncClient 复用** | 1.2–1.5× | 半天 | 低 | ✅ Day 1 |
| 2 | **扩展阶段批量 prompt**（5 改述/4 追问 → 1 次调用返回多条） | 3–5× 扩展阶段 | 1–2 天 | 中（prompt 设计） | ✅ Day 2-3 |
| 3 | **启用 DeepSeek context cache**（cache_control headers） | 30–50% token 成本，10–20% 延迟 | 半天 | 低 | ✅ Day 1 |
| 4 | **响应级磁盘缓存**（prompt hash → response） | 重跑/调试时 100× | 1 天 | 低 | ✅ Day 4 |
| 5 | **质检 embedding 改 FAISS** | 质检阶段 10–50× | 1 天 | 低 | ✅ Day 5 |
| 6 | **并发度上调**（50 → 100–200） | 1.5–2× | 改 config | 中（RPM 限制） | ✅ Day 1 |
| 7 | **流式管道**（chunk → seed → expand 重叠） | 1.5–2× | 3–5 天（重构 orchestrator） | 中高 | ⏸ 第二阶段 |
| 8 | **OpenAI/Anthropic Batch API**（50% 折扣，24h 异步） | 成本减半，吞吐翻倍 | 2–3 天 | 中（异步轮询） | ⏸ 第二阶段 |
| **A/B** | **换更快的模型** A/B 测试（gpt-4o-mini / qwen-plus / qwen-turbo） | 2–4× | 半天 | 中（质量需评估） | 🔬 Day 6 A/B 评测 |
| ❌ | ~~降低扩展倍数~~（rephrase 5→3、follow_up 4→2） | — | — | — | 用户明确拒绝 |

**质量第一原则下的红线**：
- 批量 prompt（方案 2）必须做 **质量 A/B**：抽样 200 条对比"5 次单调"vs"1 次返回 5 条"的产出质量
- 任何改造在合并前都必须通过 `data/qc_results/` 的现有 P2 质检评分，且分数不低于改造前

---

## 四、推荐落地次序（一周内）

**Day 0（前置，半天）**：基线测量
- 选定一份代表性输入（建议 100 chunk）作为标准回归集，跑完整流程并记录 `data/qc_results/` 评分作为质量基线
- 聚合 [src/llm_logger.py](src/llm_logger.py) 的 duration_ms 日志，画出"切块/种子/扩展/质检"四阶段耗时占比
- 输出文件：`data/benchmark/baseline.json`（含每阶段秒数、token 数、QC 评分）

**Day 1（半天）**：方案 1 + 3 + 6 —— 低风险地基
- [src/llm_client.py:387](src/llm_client.py)：把 `httpx.AsyncClient` 提到模块级单例（lifespan 跟随进程）
- [src/llm_client.py:386-396](src/llm_client.py)：DeepSeek 调用加 prompt caching 标头（DeepSeek API 自动 cache，但要保证 system prompt 在前、长 chunk 在前的顺序）
- [config/config.yaml:182,189,213,214](config/config.yaml)：concurrency 50 → 100（监控 429，若触发回退到 80）
- **验证**：跑同一份 100 chunk，对比 baseline 总耗时，预期 1.5–2×；QC 评分不降

**Day 2–3（1–2 天）**：方案 2 —— 扩展阶段批量化
- 重写 [prompts/expand_rephrase.txt](prompts/expand_rephrase.txt) / [prompts/expand_followup.txt](prompts/expand_followup.txt)，让模型一次返回 JSON 数组（n 条改述 / n 步追问链）
- 改 [src/expander.py:_expand_rephrase_async](src/expander.py)、[src/expander.py:_expand_followup_async](src/expander.py)：移除外层循环，单次调用解析多条
- **质量 A/B**：抽 200 条种子分别跑"旧版（5 次单调）"和"新版（1 次返回 5 条）"，人工审 + 用现有 P2 评分对比，**评分不降才合并**
- **预期**：扩展阶段（占总耗时 ~70%）提速 3–5×

**Day 4（1 天）**：方案 4 —— 响应级磁盘缓存
- 新增 `src/llm_cache.py`：用 `diskcache` 或 sqlite，key = `sha256(model + temperature + system + prompt)`
- 在 [src/llm_client.py:acall](src/llm_client.py) 入口先 lookup，命中直接返回，未命中调 LLM 后写入
- 增加 `--no-cache` CLI 开关方便强制重跑
- **预期**：调试/重跑 prompt 时 10–100× 提速；首次跑无影响

**Day 5（1 天）**：方案 5 —— 质检向量去重换 FAISS
- 改 [src/quality_checker.py:_dedup_via_embedding](src/quality_checker.py) 用 `faiss-cpu` HNSW 索引（M=16, efSearch=64）
- 保留旧 numpy 路径作为 fallback，用 config flag 切换便于回归
- **预期**：质检阶段（5 万条 Q&A）从分钟级 → 秒级

**Day 6（1 天）**：A/B 实验 —— 模型替代评测（不强制落地）
- 实验组：gpt-4o-mini、qwen-plus、qwen-turbo（如有 vLLM 环境也可加本地模型）
- 用 Day 0 的同一份 100 chunk，跑同样流程，对比：
  - 总耗时
  - P2 QC 评分均值与方差
  - 抽样 100 条人工审
- 输出 `data/benchmark/model_ab.md`，**用户最终决定是否切换**

**累计预期**：架构层面 **5–10× 总体提速**（约 4 小时 → 25–50 分钟）；模型 A/B 若通过可叠加 2–4×

---

## 五、关键文件清单

修改对象：
- [src/llm_client.py](src/llm_client.py) — 共享 client、context cache、缓存层
- [src/expander.py](src/expander.py) — 批量扩展 prompt
- [src/quality_checker.py](src/quality_checker.py) — FAISS 替换 numpy
- [config/config.yaml](config/config.yaml) — 并发度、扩展倍数
- [prompts/expand_rephrase.txt](prompts/) / [prompts/expand_followup.txt](prompts/) — 一次返回多条
- 新增 `src/llm_cache.py` — 响应级缓存

参考已有实现：
- [src/llm_logger.py](src/llm_logger.py) — 已有 duration_ms 日志，可用来量化每个阶段的实际耗时分布

---

## 六、验证方案（每一档都必须跑）

每个 Day 完成后必须做：

1. **耗时回归**：用 Day 0 的 100-chunk 标准集跑完整流程，对比 baseline 总耗时与各阶段耗时
2. **质量回归（红线）**：
   - 用现有 P2 质检（[src/quality_checker.py](src/quality_checker.py)）对产出打分
   - **评分均值不得低于 baseline**；若低于，回滚改动
3. **抽样人工审**（仅 Day 2-3 批量 prompt 改造时强制）：随机抽 200 条 Q&A 人工对比新旧版本
4. **成本对比**：用 [src/llm_client.py:usage_stats](src/llm_client.py) 的 token 计数确认未异常上升
5. **断点续传烟囱测试**：中途 Ctrl-C 一次，重启确认能续上，避免缓存/批量改造破坏断点逻辑
