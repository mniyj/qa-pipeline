"""
doc_preprocessor.py
保险文档预处理器：PDF/DOCX → 结构化 JSON chunks

功能：
1. 批量扫描 raw_documents 目录
2. 提取文本（支持 PDF / DOCX / TXT）
3. 智能分块（按章节/条款边界切分，而非固定长度）
4. 输出结构化 chunk JSON，供后续种子生成使用

使用方式：
    python src/doc_preprocessor.py
    python src/doc_preprocessor.py --input ./data/raw_documents --output ./data/chunks
"""

import json
import re
import os
import hashlib
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import yaml

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------- 数据结构 ----------
@dataclass
class DocumentMeta:
    """文档元信息"""
    file_name: str
    file_path: str
    file_type: str                          # pdf / docx / txt
    doc_type: str = "未分类"                  # 条款 / 理赔指南 / 监管文件 / ...
    insurance_type: str = "通用"              # 从文件名或内容推断
    page_count: int = 0
    char_count: int = 0
    process_time: str = ""
    md5: str = ""


@dataclass
class TextChunk:
    """文本块"""
    chunk_id: str                           # doc_md5_前8位 + chunk序号
    doc_file: str                           # 来源文件名
    doc_type: str
    insurance_type: str
    chunk_index: int                        # 在文档中的序号
    total_chunks: int
    text: str
    char_count: int
    page_range: str = ""                    # "p3-p5"
    section_title: str = ""                 # 所属章节标题
    has_table: bool = False
    has_numbers: bool = False               # 是否包含金额/比例等数字
    metadata: dict = field(default_factory=dict)


# ---------- 配置加载 ----------
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- PDF 文本提取 ----------
def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    从 PDF 提取文本，返回 (全文文本, 页数)
    优先用 pdfplumber（对中文保险条款效果更好），失败则 fallback 到 pypdf
    """
    try:
        import pdfplumber
        full_text = []
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                # 尝试提取表格并转为文本
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            row_text = " | ".join(
                                [cell or "" for cell in row]
                            )
                            text += "\n" + row_text
                full_text.append(text)
        return "\n\n".join(full_text), page_count
    except Exception as e:
        logger.warning(f"pdfplumber 失败，尝试 pypdf: {e}")

    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        page_count = len(reader.pages)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text() or "")
        return "\n\n".join(full_text), page_count
    except Exception as e:
        logger.error(f"PDF 提取完全失败: {file_path}, {e}")
        return "", 0


def extract_text_from_docx(file_path: str) -> tuple[str, int]:
    """从 DOCX 提取文本"""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # DOCX 没有页码概念，用段落数估算
        estimated_pages = max(1, len(paragraphs) // 20)
        return "\n\n".join(paragraphs), estimated_pages
    except Exception as e:
        logger.error(f"DOCX 提取失败: {file_path}, {e}")
        return "", 0


def extract_text(file_path: str) -> tuple[str, int]:
    """统一的文本提取入口"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, max(1, len(text) // 1500)
    else:
        logger.warning(f"不支持的文件格式: {ext}")
        return "", 0


# ---------- 文档类型 & 险种推断 ----------

# 从文件名 / 内容中推断文档类型
DOC_TYPE_PATTERNS = {
    # 法律法规优先匹配，避免被"保险条款"的宽泛关键词误抢
    "法律法规": [
        r"中华人民共和国.{0,10}法(?:\s|$|[（(])",  # 保险法、道路交通安全法等
        r"(?:保险|农业|交强).{0,8}条例",             # 农业保险条例、交强险条例
        r"保险法",                                    # 直接命中
    ],
    "保险条款": [r"条款", r"保险合同", r"保险责任", r"责任免除"],
    "产品说明书": [r"产品说明", r"产品介绍", r"费率表"],
    "理赔指南": [r"理赔", r"报案", r"索赔", r"赔付"],
    "核保手册": [r"核保", r"健康告知", r"投保规则"],
    "监管文件": [
        r"银保监", r"金监总局", r"通知",
        r"管理办法", r"管理规定", r"监管规定",
        r"实施细则", r"服务规定", r"关联交易",
        r"消费者权益保护", r"销售行为",
    ],
    "行业标准": [r"ICD", r"重疾定义", r"示范条款", r"行业规范"],
}

INSURANCE_TYPE_PATTERNS = {
    "百万医疗险": [r"百万医疗", r"住院医疗"],
    "重疾险": [r"重大疾病", r"重疾"],
    "意外险": [r"意外伤害", r"意外险"],
    "定期寿险": [r"定期寿险", r"定寿"],
    "终身寿险": [r"终身寿险"],
    "增额终身寿": [r"增额终身寿", r"增额寿"],
    "年金险": [r"年金", r"养老金"],
    "车损险": [r"车损", r"机动车损失"],
    "交强险": [r"交强", r"机动车交通事故责任强制"],
    "第三者责任险": [r"第三者", r"三者险"],
    "企业财产险": [r"企财", r"财产一切险", r"企业财产"],
    "雇主责任险": [r"雇主责任"],
    "惠民保": [r"惠民保", r"城市定制"],
}


def infer_doc_type(file_name: str, text_preview: str) -> str:
    """从文件名和内容前 500 字推断文档类型"""
    combined = file_name + " " + text_preview[:500]
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        if any(re.search(p, combined) for p in patterns):
            return doc_type
    return "未分类"


def infer_insurance_type(file_name: str, text_preview: str) -> str:
    """从文件名和内容前 500 字推断险种"""
    combined = file_name + " " + text_preview[:500]
    for ins_type, patterns in INSURANCE_TYPE_PATTERNS.items():
        if any(re.search(p, combined) for p in patterns):
            return ins_type
    return "通用"


# ---------- 智能分块 ----------

# 保险条款常见的章节标题模式
SECTION_PATTERNS = [
    r"第[一二三四五六七八九十百]+[章节条]",          # 第一章、第二节、第三条
    r"第\d+[章节条]",                                # 第1章、第2节
    r"[一二三四五六七八九十]+[、.．]",                # 一、二、三、
    r"\d+[、.．]\d*\s",                              # 1、 2. 3．
    r"(?:保险责任|责任免除|免责|投保须知|理赔流程|"
    r"保险金额|保险期间|等待期|犹豫期|退保|续保)",    # 保险条款核心章节关键词
]


def find_section_boundaries(text: str) -> list[int]:
    """找到文本中的章节边界位置"""
    boundaries = [0]
    for pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            pos = match.start()
            # 确保边界在行首附近（前面最多 2 个字符是换行/空格）
            line_start = text.rfind("\n", max(0, pos - 3), pos)
            if line_start != -1 and pos - line_start <= 3:
                boundaries.append(pos)
            elif pos < 3:  # 文档开头
                boundaries.append(pos)
    boundaries = sorted(set(boundaries))
    return boundaries


def smart_chunk(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_size: int = 300,
) -> list[dict]:
    """
    智能分块：优先按章节/条款边界切分，其次按段落，最后按固定长度
    返回 chunk 字典列表
    """
    if not text.strip():
        return []

    # Step 1: 找章节边界
    boundaries = find_section_boundaries(text)

    # Step 2: 按边界初步切分
    raw_sections = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            raw_sections.append(section_text)

    # Step 3: 合并过短的 section，拆分过长的 section
    chunks = []
    buffer = ""

    for section in raw_sections:
        if len(buffer) + len(section) <= chunk_size:
            buffer += ("\n\n" + section) if buffer else section
        else:
            if buffer:
                chunks.append(buffer)
            # 如果单个 section 超过 chunk_size，按段落拆分
            if len(section) > chunk_size:
                paragraphs = section.split("\n\n")
                sub_buffer = ""
                for para in paragraphs:
                    if len(sub_buffer) + len(para) <= chunk_size:
                        sub_buffer += ("\n\n" + para) if sub_buffer else para
                    else:
                        if sub_buffer:
                            chunks.append(sub_buffer)
                        # 单段落超长，强制按字数切
                        if len(para) > chunk_size:
                            for j in range(0, len(para), chunk_size - chunk_overlap):
                                chunks.append(para[j : j + chunk_size])
                        else:
                            sub_buffer = para
                if sub_buffer:
                    buffer = sub_buffer
                else:
                    buffer = ""
            else:
                buffer = section

    if buffer and len(buffer) >= min_chunk_size:
        chunks.append(buffer)
    elif buffer and chunks:
        # 过短的尾部合并到最后一个 chunk
        chunks[-1] += "\n\n" + buffer

    # Step 4: 提取每个 chunk 的章节标题
    result = []
    for i, chunk_text in enumerate(chunks):
        # 尝试提取第一行作为标题
        first_line = chunk_text.split("\n")[0].strip()
        section_title = ""
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, first_line):
                section_title = first_line[:50]
                break

        # 检测特征
        has_table = bool(re.search(r"[|│┃].*[|│┃]", chunk_text))
        has_numbers = bool(
            re.search(r"\d+(?:\.\d+)?%|(?:元|万元|亿元)", chunk_text)
        )

        result.append({
            "chunk_index": i,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "section_title": section_title,
            "has_table": has_table,
            "has_numbers": has_numbers,
        })

    return result


# ---------- 主处理流程 ----------
def process_single_document(
    file_path: str,
    config: dict,
) -> tuple[DocumentMeta, list[TextChunk]]:
    """处理单个文档，返回元信息和 chunk 列表"""
    file_path = str(file_path)
    file_name = Path(file_path).name

    logger.info(f"处理文档: {file_name}")

    # 1. 计算文件 MD5
    with open(file_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    # 2. 提取文本
    text, page_count = extract_text(file_path)
    if not text.strip():
        logger.warning(f"文档无法提取文本，跳过: {file_name}")
        return None, []

    # 3. 推断文档类型和险种
    doc_type = infer_doc_type(file_name, text)
    insurance_type = infer_insurance_type(file_name, text)

    # 4. 智能分块
    prep_config = config.get("preprocessing", {})
    raw_chunks = smart_chunk(
        text,
        chunk_size=prep_config.get("chunk_size", 1200),
        chunk_overlap=prep_config.get("chunk_overlap", 200),
        min_chunk_size=prep_config.get("min_chunk_size", 300),
    )

    # 5. 构建结构化输出
    doc_meta = DocumentMeta(
        file_name=file_name,
        file_path=file_path,
        file_type=Path(file_path).suffix.lower(),
        doc_type=doc_type,
        insurance_type=insurance_type,
        page_count=page_count,
        char_count=len(text),
        process_time=datetime.now().isoformat(),
        md5=md5,
    )

    chunks = []
    for raw_chunk in raw_chunks:
        chunk_id = f"{md5[:8]}_{raw_chunk['chunk_index']:04d}"
        chunk = TextChunk(
            chunk_id=chunk_id,
            doc_file=file_name,
            doc_type=doc_type,
            insurance_type=insurance_type,
            chunk_index=raw_chunk["chunk_index"],
            total_chunks=len(raw_chunks),
            text=raw_chunk["text"],
            char_count=raw_chunk["char_count"],
            section_title=raw_chunk.get("section_title", ""),
            has_table=raw_chunk.get("has_table", False),
            has_numbers=raw_chunk.get("has_numbers", False),
        )
        chunks.append(chunk)

    logger.info(
        f"  → {doc_type} | {insurance_type} | "
        f"{page_count}页 | {len(text)}字 | {len(chunks)}个chunk"
    )
    return doc_meta, chunks


def process_all_documents(
    input_dir: str,
    output_dir: str,
    config: dict,
) -> dict:
    """批量处理所有文档（增量：跳过已处理文件）"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported = config.get("preprocessing", {}).get(
        "supported_formats", [".pdf", ".docx", ".txt", ".md"]
    )

    # 加载已处理文档的清单（md5 → file_name）
    manifest_file = output_path / "processed_docs.json"
    manifest: dict[str, str] = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # 扫描文件
    files = []
    for ext in supported:
        files.extend(input_path.glob(f"*{ext}"))
        files.extend(input_path.glob(f"**/*{ext}"))  # 递归子目录
    files = sorted(set(files))

    # 过滤已处理文件（按 md5）
    new_files = []
    for fp in files:
        with open(fp, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 in manifest:
            logger.info(f"跳过（已处理）: {fp.name}")
        else:
            new_files.append((fp, md5))

    logger.info(f"发现 {len(files)} 个文档，{len(new_files)} 个待处理，{len(files)-len(new_files)} 个已跳过")

    if not new_files:
        return {"total_files": len(files), "new_files": 0, "total_chunks": 0}

    all_metas = []
    all_chunks = []
    failed_files = []

    for file_path, md5 in new_files:
        try:
            meta, chunks = process_single_document(str(file_path), config)
            if meta:
                all_metas.append(asdict(meta))
                all_chunks.extend([asdict(c) for c in chunks])
                manifest[md5] = meta.file_name
        except Exception as e:
            logger.error(f"处理失败: {file_path}, 错误: {e}")
            failed_files.append({"file": str(file_path), "error": str(e)})

    # 追加新 chunks（不覆盖旧数据）
    chunks_file = output_path / "chunks.jsonl"
    with open(chunks_file, "a", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 更新清单
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 追加文档元信息
    meta_file = output_path / "doc_metadata.json"
    existing_metas = []
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            existing_metas = json.load(f)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(existing_metas + all_metas, f, ensure_ascii=False, indent=2)

    # 保存处理报告
    report = {
        "process_time": datetime.now().isoformat(),
        "total_files": len(files),
        "success_files": len(all_metas),
        "failed_files": len(failed_files),
        "total_chunks": len(all_chunks),
        "total_chars": sum(c["char_count"] for c in all_chunks),
        "doc_type_distribution": _count_by(all_metas, "doc_type"),
        "insurance_type_distribution": _count_by(all_metas, "insurance_type"),
        "failures": failed_files,
    }
    report_file = output_path / "preprocess_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    logger.info("=" * 50)
    logger.info("文档预处理完成")
    logger.info(f"  成功: {len(all_metas)} 个文档")
    logger.info(f"  失败: {len(failed_files)} 个文档")
    logger.info(f"  总 chunk 数: {len(all_chunks)}")
    logger.info(f"  总字数: {sum(c['char_count'] for c in all_chunks):,}")
    logger.info(f"  文档类型分布: {report['doc_type_distribution']}")
    logger.info(f"  险种分布: {report['insurance_type_distribution']}")
    logger.info(f"  输出目录: {output_path}")
    logger.info("=" * 50)

    return report


def _count_by(items: list[dict], key: str) -> dict:
    counts = {}
    for item in items:
        val = item.get(key, "未知")
        counts[val] = counts.get(val, 0) + 1
    return counts


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="保险文档预处理器")
    parser.add_argument(
        "--input", "-i",
        default="./data/raw_documents",
        help="原始文档目录"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/chunks",
        help="输出目录"
    )
    parser.add_argument(
        "--config", "-c",
        default="./config/config.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    process_all_documents(args.input, args.output, config)


if __name__ == "__main__":
    main()
