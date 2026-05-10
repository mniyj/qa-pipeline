# 文档分类 LLM 化 + Schema 约束

## Context

当前 `doc_type` 和 `insurance_type` 通过正则关键词匹配推断，存在两个问题：
1. 正则覆盖不全，边缘案例（非标文件名、内容混合型文档）频繁落入"未分类"/"通用"
2. 没有 schema 强约束，LLM 生成的 Q&A 中险种名称可能出现"医疗"、"意外"等非标写法

**解决方案：** 以 LLM 为主分类器，将可选类型以枚举形式注入 prompt，强制从预定义列表中选取；对 LLM 输出做校验，校验失败时回退到正则推断结果，不丢弃原有兜底逻辑。对每个文件按 MD5 缓存分类结果，避免重复调用。

险种 schema 以监管标准为准，共 47 种，兜底值改为"其他保险"（原"通用"全局替换）。

---

## 关键文件

| 文件 | 操作 |
|------|------|
| `config/config.yaml` | 替换 `knowledge_schema.insurance_types` 为平铺列表；新增 `preprocessing.doc_types`；新增 `models.classification` |
| `prompts/classify_document.txt` | 新建分类 prompt 模板 |
| `src/doc_preprocessor.py` | 新增 3 个辅助函数；修改 `process_single_document`、`process_all_documents`；默认值"通用"→"其他保险" |
| `src/expander.py` | `seed.get("insurance_type", "通用")` → `"其他保险"`（共 2 处） |
| `src/quality_checker.py` | `run_coverage_analysis` 中对 `insurance_types` 的遍历方式从嵌套 dict 改为平铺 list |

---

## Task 1：config.yaml — 更新险种 schema、新增 doc_types 和 classification 模型

**Step 1：** 将 `knowledge_schema.insurance_types` 由嵌套 dict 替换为平铺列表：

```yaml
knowledge_schema:
  insurance_types:
    - 机动车商业保险
    - 交强险
    - 其他机动车辆保险
    - 企业财产保险
    - 家庭财产保险
    - 货物运输保险
    - 船舶保险
    - 建筑、安装工程保险
    - 航空保险
    - 航天保险
    - 石油保险
    - 核能保险
    - 其他特殊风险财产保险
    - 种植保险
    - 养殖保险
    - 林业保险
    - 其他农业保险
    - 农房保险
    - 农机保险
    - 渔船保险
    - 涉农意外保险
    - 温室大棚保险
    - 其他涉农保险
    - 公众责任保险
    - 产品责任保险
    - 雇主责任保险
    - 职业责任保险
    - 新型责任保险
    - 其他责任保险
    - 定期寿险
    - 终身寿险
    - 两全保险
    - 普通年金保险
    - 养老年金保险
    - 医疗保险
    - 疾病保险
    - 护理保险
    - 失能保险
    - 医疗意外保险
    - 意外伤害保险
    - 健康委托管理产品
    - 养老委托管理产品
    - 个人类信用保险
    - 企业类信用保险
    - 融资性保证保险
    - 非融资性保证保险
    - 其他保险
```

**Step 2：** 在 `preprocessing` 节下新增 `doc_types`：

```yaml
preprocessing:
  doc_types:
    - 法律法规
    - 保险条款
    - 产品说明书
    - 理赔指南
    - 核保手册
    - 监管文件
    - 行业标准
    - 未分类
```

**Step 3：** 在 `models` 节下新增：

```yaml
  classification:
    provider: deepseek
    model: deepseek-chat
    max_tokens: 200
    temperature: 0.1
```

---

## Task 2：新建 `prompts/classify_document.txt`

```
你是保险文档分类专家。请根据文件名和内容预览，从给定选项中选择最合适的文档类型和险种。

文件名：{file_name}

内容预览（前1500字）：
{text_preview}

【文档类型】必须从以下选项中精确选择一个：
{doc_types}

【险种】必须从以下选项中精确选择一个：
{insurance_types}

规则：
- 如果文档涉及多个险种，选最主要的一个
- 险种无法判断时选"其他保险"
- 文档类型无法判断时选"未分类"

只返回 JSON，不要任何其他说明：
{{"doc_type": "...", "insurance_type": "..."}}
```

---

## Task 3：`src/doc_preprocessor.py`

### 3.1 工具函数（模块级）

`_flatten_insurance_types` 不再需要展平嵌套 dict，直接读平铺列表：

```python
def _get_insurance_types(config: dict) -> list[str]:
    return config.get("knowledge_schema", {}).get("insurance_types", [])
```

缓存读写：

```python
def _load_classification_cache(chunks_dir: str) -> dict:
    cache_file = Path(chunks_dir) / "classification_cache.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_classification_cache(cache: dict, chunks_dir: str):
    cache_file = Path(chunks_dir) / "classification_cache.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
```

### 3.2 核心分类函数

```python
def classify_document_with_llm(
    file_name: str,
    text: str,
    md5: str,
    config: dict,
    cache: dict,
) -> tuple[str, str]:
    if md5 in cache:
        entry = cache[md5]
        return entry["doc_type"], entry["insurance_type"]

    doc_types_list      = config.get("preprocessing", {}).get("doc_types", [])
    insurance_types_list = _get_insurance_types(config)

    # 正则结果作为兜底（LLM 失败时使用）
    fallback_doc = infer_doc_type(file_name, text)
    fallback_ins = infer_insurance_type(file_name, text)   # 其默认返回值改为"其他保险"（见 3.3）

    try:
        from llm_client import create_client
        client = create_client(config, "classification")
        prompt_tpl = load_prompt_template("classify_document")
        prompt = prompt_tpl.format(
            file_name=file_name,
            text_preview=text[:1500],
            doc_types="\n".join(f"- {t}" for t in doc_types_list),
            insurance_types="\n".join(f"- {t}" for t in insurance_types_list),
        )
        response = client.call(prompt)
        response = re.sub(r"```(?:json)?\s*", "", response).strip()
        result   = json.loads(response)

        doc_type = result.get("doc_type", "").strip()
        ins_type = result.get("insurance_type", "").strip()

        if doc_type not in doc_types_list:
            logger.warning(f"LLM doc_type '{doc_type}' 不在 schema，回退: {fallback_doc}")
            doc_type = fallback_doc
        if ins_type not in insurance_types_list:
            logger.warning(f"LLM insurance_type '{ins_type}' 不在 schema，回退: {fallback_ins}")
            ins_type = fallback_ins

    except Exception as e:
        logger.warning(f"LLM 分类失败 ({file_name}): {e}，使用正则结果")
        doc_type, ins_type = fallback_doc, fallback_ins

    cache[md5] = {"doc_type": doc_type, "insurance_type": ins_type}
    return doc_type, ins_type
```

### 3.3 默认值"通用"→"其他保险"

- `infer_insurance_type` 末尾 `return "通用"` → `return "其他保险"`
- `DocumentMeta` dataclass：`insurance_type: str = "其他保险"`

### 3.4 修改 `process_single_document`

新增 `cache: dict` 参数，替换两行正则推断：

```python
# 删除：
doc_type = infer_doc_type(file_name, text)
insurance_type = infer_insurance_type(file_name, text)

# 替换为：
doc_type, insurance_type = classify_document_with_llm(
    file_name, text, md5, config, cache
)
```

### 3.5 修改 `process_all_documents`

```python
# 文件扫描循环前加载缓存：
cache = _load_classification_cache(output_dir)

# 循环体内调用改为：
meta, chunks = process_single_document(str(file_path), config, cache)

# 所有文件处理完后（manifest 写入之后）保存缓存：
_save_classification_cache(cache, output_dir)
```

---

## Task 4：`src/expander.py` — 默认值替换

将 2 处 `seed.get("insurance_type", "通用")` 改为 `seed.get("insurance_type", "其他保险")`：
- rephrase 路径（`process_one` 中赋值给 `v["insurance_type"]`）
- followup 路径（`process_one` 中赋值给 `t["insurance_type"]`）

---

## Task 5：`src/quality_checker.py` — coverage 遍历方式更新

`run_coverage_analysis` 中原来对嵌套 dict 遍历：

```python
# 原：
for category, subtypes in schema.get("insurance_types", {}).items():
    for ins_type in subtypes:
        ...

# 改为平铺列表遍历：
for ins_type in schema.get("insurance_types", []):
    ...
```

需读取 `run_coverage_analysis` 完整实现后确认具体改法，其他逻辑不变。

---

## 验证

```bash
# 单文件验证：确认 LLM 调用、缓存写入、schema 校验
python -c "
import yaml, sys
sys.path.insert(0, 'src')
from doc_preprocessor import classify_document_with_llm, _load_classification_cache
with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
cache = {}
# 用一个已有文件测试
from pathlib import Path
f = next(Path('data/raw_documents').rglob('*.md'))
text = f.read_text(encoding='utf-8')
import hashlib
md5 = hashlib.md5(f.read_bytes()).hexdigest()
dt, it = classify_document_with_llm(f.name, text, md5, cfg, cache)
print(f'doc_type={dt}, insurance_type={it}')
print('cache entry:', cache[md5])
"

# 全量验证（运行完整预处理后）
python -c "
import json; from collections import Counter
dt, it = Counter(), Counter()
for line in open('data/chunks/chunks.jsonl'):
    c = json.loads(line); dt[c['doc_type']] += 1; it[c['insurance_type']] += 1
print('doc_type:', dict(dt))
print('insurance_type:', dict(it))
"
```

期望：
- 所有 `doc_type` 值均在 `preprocessing.doc_types` 8 个值内
- 所有 `insurance_type` 值均在 `knowledge_schema.insurance_types` 47 个值内
- 无"通用"出现（已全局替换为"其他保险"）
