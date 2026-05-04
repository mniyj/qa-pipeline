# 保险知识问答生成管道 🏥

从保险原始文档出发，自动生成 10 万+ 条高质量知识问答对。

## 快速开始

### 1. 环境准备

```bash
cd insurance-qa-pipeline
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 至少配置一个（推荐先用 DashScope，成本低）
export DASHSCOPE_API_KEY="sk-xxxx"      # 通义千问
export ANTHROPIC_API_KEY="sk-ant-xxxx"  # Claude（种子生成用）
```

### 3. 放入原始文档

把保险 PDF、DOCX 文件放入 `data/raw_documents/` 目录：

```
data/raw_documents/
├── 某某百万医疗险条款.pdf
├── 重疾险理赔指南.pdf
├── 车险示范条款.docx
├── 银保监会通知xxx号.pdf
└── ...
```

### 4. 查看状态

```bash
python src/orchestrator.py status
```

### 5. 试跑（强烈建议先试跑！）

```bash
# 只处理 3 个 chunk，看看种子质量
python src/orchestrator.py preprocess
python src/orchestrator.py seed --limit 3

# 检查 data/seeds/ 目录下的输出
cat data/seeds/*.jsonl | python -m json.tool | head -50
```

### 6. 全量运行

```bash
# 确认种子质量满意后，全流程运行
python src/orchestrator.py run

# 或者分步执行：
python src/orchestrator.py preprocess    # Step 1: 文档预处理
python src/orchestrator.py seed          # Step 2: 种子生成
python src/orchestrator.py expand        # Step 3: 批量扩展
python src/orchestrator.py qc            # Step 4: 质检
```

## 核心特性 (100k Scale Ready)

本管道经过专门针对 10 万级数据量生成任务的工程化优化：
- **无卡顿去重 (O(N) 性能优化)**：摒弃传统嵌套循环去重，质检环节引入 Numpy 的 BLAS 底层矩阵乘法与向量化相似度计算。
- **流式写入与断点续传 (Streaming IO)**：所有批量生成均带有 Async 线程安全锁，并采用 Append 模式实时落盘，随时可以中断重启而不会丢失昂贵的生成数据。
- **基于关键词的精确召回**：在跨险种和同险种模板化问题生成中，利用内容特征动态召回 Chunk，避免“强行拼凑前三个 Chunk”而引发的大模型幻觉。
- **鲁棒的事实校验引擎**：支持带有复杂单位（例如 `18万元` 或 `5000元`）的数据抽取与规则比对，极大降低质检的误杀率。
- **强类型参数萃取**：结构化提取险种内的时限、特定金额与操作流程，摒弃会引发幻觉的兜底占位符。

## 项目结构

```
insurance-qa-pipeline/
├── config/
│   └── config.yaml          # 项目配置（险种、模型、参数）
├── data/
│   ├── raw_documents/        # 📁 放入你的保险 PDF/DOCX
│   ├── chunks/               # 预处理后的文档块
│   ├── seeds/                # 种子 Q&A
│   ├── expanded/             # 扩展后的 Q&A
│   ├── qc_results/           # 质检结果
│   └── final/                # 最终入库数据
├── prompts/
│   ├── seed_from_document.txt    # 种子生成 prompt
│   ├── expand_rephrase.txt       # 改述扩展 prompt
│   └── expand_followup.txt       # 追问链 prompt
├── src/
│   ├── orchestrator.py       # 主调度器
│   ├── doc_preprocessor.py   # 文档预处理
│   ├── seed_generator.py     # 种子生成
│   ├── expander.py           # 批量扩展
│   ├── quality_checker.py    # 质检管道
│   └── llm_client.py         # 统一 LLM 客户端
└── requirements.txt
```

## 关键配置

编辑 `config/config.yaml`：

- **knowledge_schema**: 根据你的业务范围调整险种和环节
- **models**: 选择种子/扩展/质检各阶段的模型
- **generation**: 调整每个 chunk 的生成数量、扩展倍数等

## Q&A 输出格式

每条 Q&A 以 JSONL 格式存储：

```json
{
  "id": "seed_20260429_120000_a1b2c3d4_0000_001",
  "question": "百万医疗险的免赔额是什么意思？1万免赔额是不是说1万以下的费用都不赔？",
  "answer": "免赔额是指保险公司不予赔付的金额门槛...",
  "difficulty": "入门",
  "question_type": "概念解释",
  "insurance_type": "百万医疗险",
  "business_stage": "理赔审核",
  "source_reference": "条款第五条",
  "tags": ["免赔额", "百万医疗", "理赔"],
  "generation_method": "seed_from_document",
  "batch_id": "seed_20260429_120000",
  "created_at": "2026-04-29T12:00:00"
}
```

## 成本估算

全量 10 万条，使用通义千问为主力模型：约 ¥800-1200 元。
