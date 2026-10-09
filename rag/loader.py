"""读取 data/ 目录下的 .md / .txt 文件，返回带元数据的文档列表。"""
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass
class Document:
    text: str      # 文件全文
    source: str    # 相对路径，如 "book1/characters.md"
    work: str      # 作品名 = 第一层子目录名


def load_documents(data_dir: Path) -> list[Document]:
    docs = []
    if not data_dir.exists():
        return docs
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="gbk", errors="ignore")
            rel = path.relative_to(data_dir).as_posix()
            work = rel.split("/")[0] if "/" in rel else "(root)"
            docs.append(Document(text=text, source=rel, work=work))
    return docs
