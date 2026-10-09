# 小说设定 RAG（阶段一）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline execution with checkpoints is preferred for this learning project). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭出一个可本地运行的"小说设定 RAG"最小版——把 `data/` 里的设定切成片段、向量化、存进 Chroma，用户提问后检索 top-k 片段交给 DeepSeek 生成带来源引用的回答，并在 Streamlit 网页上可视化检索详情。

**Architecture:** 单一 Python 包 `rag/`，按职责拆成 loader（读文件）→ splitter（切段）→ embedder（向量化）→ store（Chroma 存取）→ retriever（检索）→ generator（组装 prompt + 调 DeepSeek）→ pipeline（串联）→ app（Streamlit 网页）。核心流程手写，LangChain 只用于切分；embedder/generator 做成可替换接口，测试用假实现（Fake），真实模型只在冒烟测试时用到。

**Tech Stack:** Python 3.10+、langchain-text-splitters、sentence-transformers（bge-m3）、chromadb、requests、python-dotenv、streamlit、pytest。

**项目根目录：** `G:\deepseek harness\novel-rag`（下文中所有相对路径都相对于它）

---

## 前置准备（一次性，手动）

在 PowerShell 中执行：

```powershell
cd "G:\deepseek harness\novel-rag"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- 装 `sentence-transformers` 会连带装 `torch`，体积较大，耐心等。
- 需要准备 DeepSeek 的 API key（https://platform.deepseek.com），填入 `.env`（见 Task 0）。

## 文件结构（全部文件）

```
novel-rag/
  requirements.txt      # 依赖
  .gitignore            # 忽略 .env / 缓存
  .env.example          # 配置模板
  config.py             # 读取 .env，暴露全局配置
  conftest.py           # 让 pytest 能找到 rag 包
  rag/
    __init__.py
    loader.py           # 读 data/ 下的 .md/.txt
    splitter.py         # 切分文本
    embedder.py         # Embedder 接口 + BgeM3 + Fake
    store.py            # Chroma 向量库封装（Hit 数据类）
    retriever.py        # 检索 top-k
    generator.py        # 组装 prompt + 调 DeepSeek
    pipeline.py         # 串联 build_index / query
  app.py                # Streamlit 网页
  data/example/*.md     # 示例设定
  tests/                # pytest 测试
  README.md
```

---

## Task 0: 项目脚手架与配置

**Files:**
- Create: `requirements.txt`, `.gitignore`, `.env.example`, `config.py`, `conftest.py`, `rag/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写 `requirements.txt`**

```text
langchain-text-splitters>=0.3.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
requests>=2.31.0
python-dotenv>=1.0.0
streamlit>=1.35.0
pytest>=8.0.0
```

- [ ] **Step 2: 写 `.gitignore`**

```text
__pycache__/
*.pyc
.env
.venv/
venv/
.chroma/
*.db
.pytest_cache/
.streamlit/secrets.toml
```

- [ ] **Step 3: 写 `.env.example`**

```text
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
EMBED_MODEL_NAME=BAAI/bge-m3
DATA_DIR=data
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

- [ ] **Step 4: 写 `rag/__init__.py`**（空文件即可，标记这是 Python 包）

```python
```

- [ ] **Step 5: 写 `conftest.py`**（让 pytest 从项目根目录导入 `rag` 包）

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 6: 写 `config.py`**

```python
"""全局配置：从 .env 读取，未设置时使用默认值。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent

# 生成模型（DeepSeek）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 向量化模型（本地 bge-m3）
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

# 数据与向量库路径
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
CHROMA_DIR = ROOT / ".chroma"

# 检索与切分参数
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
```

- [ ] **Step 7: 写失败测试 `tests/test_config.py`**

```python
from pathlib import Path
import config


def test_config_defaults():
    assert isinstance(config.DATA_DIR, Path)
    assert config.DATA_DIR.is_absolute()
    assert config.TOP_K >= 1
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP >= 0
    assert config.DEEPSEEK_MODEL
    assert config.EMBED_MODEL_NAME


def test_api_key_is_string():
    assert isinstance(config.DEEPSEEK_API_KEY, str)
```

- [ ] **Step 8: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'config'` 或找不到包）

- [ ] **Step 9: 安装依赖后运行测试确认通过**

Run: `python -m pytest tests/test_config.py -q`
Expected: `2 passed`

- [ ] **Step 10: 提交**

```bash
git add -A
git commit -m "chore: scaffold project and config"
```

---

## Task 1: 数据加载 loader.py

**Files:**
- Create: `rag/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: 写失败测试 `tests/test_loader.py`**

```python
from pathlib import Path
from rag.loader import load_documents


def test_loads_md_and_txt(tmp_path):
    work = tmp_path / "book1"
    work.mkdir()
    (work / "characters.md").write_text("# 角色\n林澈，十七岁。", encoding="utf-8")
    (work / "notes.txt").write_text("旧稿：林澈十八岁。", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("ignore", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    by_name = {d.source: d for d in docs}
    assert by_name["book1/characters.md"].work == "book1"
    assert "十七岁" in by_name["book1/characters.md"].text
    assert by_name["book1/notes.txt"].work == "book1"


def test_empty_dir_returns_empty(tmp_path):
    assert load_documents(tmp_path) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_loader.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'rag.loader'`）

- [ ] **Step 3: 写实现 `rag/loader.py`**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_loader.py -q`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/loader.py tests/test_loader.py
git commit -m "feat: add data loader for md/txt settings"
```

---

## Task 2: 文本切分 splitter.py

**Files:**
- Create: `rag/splitter.py`
- Test: `tests/test_splitter.py`

- [ ] **Step 1: 写失败测试 `tests/test_splitter.py`**

```python
from rag.splitter import split_text


def test_splits_long_text():
    text = "这是一段测试内容。\n" * 200
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 100 for c in chunks)


def test_short_text_single_chunk():
    chunks = split_text("很短。", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "很短。"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_splitter.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现 `rag/splitter.py`**

```python
"""把长文本切成带重叠的小段（chunk）。借 LangChain 的切分器。"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先按段落/换行切，再按中英文标点，最后按空格/单字
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_splitter.py -q`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/splitter.py tests/test_splitter.py
git commit -m "feat: add recursive character text splitter"
```

---

## Task 3: 向量化 embedder.py

**Files:**
- Create: `rag/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: 写失败测试 `tests/test_embedder.py`**

```python
from rag.embedder import FakeEmbedder


def test_fake_embedder_shape_and_determinism():
    emb = FakeEmbedder(dim=16)
    vecs = emb.embed(["林澈 剑修", "林澈 剑修", "苏晚 药师"])
    assert len(vecs) == 3
    assert all(len(v) == 16 for v in vecs)
    assert vecs[0] == vecs[1]   # 相同输入 -> 相同向量
    assert vecs[0] != vecs[2]   # 不同输入 -> 不同向量


def test_fake_embedder_similarity():
    emb = FakeEmbedder(dim=64)
    a = emb.embed(["林澈 剑修 十七岁"])[0]
    b = emb.embed(["林澈 剑修"])[0]
    c = emb.embed(["苏晚 药师 温和"])[0]

    def dot(x, y):
        return sum(i * j for i, j in zip(x, y))

    assert dot(a, b) > dot(a, c)  # 共享词越多，越相似
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_embedder.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现 `rag/embedder.py`**

```python
"""向量化：把文本变成数字向量。定义可替换接口，真实模型 + 测试假实现。"""
from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成一批向量，每个向量是 list[float]。"""


class BgeM3Embedder(Embedder):
    """真实实现：本地 bge-m3 模型（首次运行会联网下载权重）。"""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class FakeEmbedder(Embedder):
    """测试用假实现：按"词袋"生成稳定向量，共享词越多越相似，无需下载模型。"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            h = sum(ord(c) for c in word) % self.dim
            vec[h] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_embedder.py -q`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/embedder.py tests/test_embedder.py
git commit -m "feat: add embedder interface with bge-m3 and fake"
```

---

## Task 4: 向量库 store.py

**Files:**
- Create: `rag/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: 写失败测试 `tests/test_store.py`**

```python
from pathlib import Path
from rag.store import VectorStore
from rag.embedder import FakeEmbedder


def test_add_and_query(tmp_path):
    emb = FakeEmbedder(dim=64)
    store = VectorStore(Path(tmp_path))
    texts = ["林澈 剑修 十七岁", "苏晚 药师 温和", "灵气 复苏 世界"]
    store.add(
        ids=["0", "1", "2"],
        texts=texts,
        embeddings=emb.embed(texts),
        metadatas=[{"source": "book1/a.md", "work": "book1"}] * 3,
    )
    assert store.count() == 3

    qvec = emb.embed(["林澈 剑修"])[0]
    hits = store.query(qvec, top_k=2)
    assert len(hits) == 2
    assert hits[0].text == "林澈 剑修 十七岁"  # 最相似的那条排在前面
    assert hits[0].source == "book1/a.md"
    assert hits[0].score > 0.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_store.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现 `rag/store.py`**

```python
"""Chroma 向量库封装：写入向量、按相似度查询。"""
from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class Hit:
    text: str
    source: str
    work: str
    score: float  # 余弦相似度，越接近 1 越像


class VectorStore:
    def __init__(self, persist_dir: Path):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="novel_settings", metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, texts, embeddings, metadatas):
        self._collection.add(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[Hit]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        hits = []
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                Hit(
                    text=doc,
                    source=meta.get("source", ""),
                    work=meta.get("work", ""),
                    score=1.0 - dist,  # cosine distance -> similarity
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self):
        try:
            self._client.delete_collection("novel_settings")
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name="novel_settings", metadata={"hnsw:space": "cosine"}
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_store.py -q`
Expected: `1 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: add chroma vector store wrapper"
```

---

## Task 5: 检索 retriever.py

**Files:**
- Create: `rag/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: 写失败测试 `tests/test_retriever.py`**

```python
from pathlib import Path
from rag.store import VectorStore
from rag.embedder import FakeEmbedder
from rag.retriever import Retriever


def test_retriever_returns_top_k(tmp_path):
    emb = FakeEmbedder(dim=64)
    store = VectorStore(Path(tmp_path))
    texts = ["林澈 剑修 十七岁", "苏晚 药师", "灵气 复苏"]
    store.add(
        ids=["0", "1", "2"],
        texts=texts,
        embeddings=emb.embed(texts),
        metadatas=[{"source": f"book1/{i}.md", "work": "book1"} for i in range(3)],
    )
    retriever = Retriever(emb, store, top_k=2)
    hits = retriever.retrieve("林澈 剑修")
    assert len(hits) == 2
    assert hits[0].text == "林澈 剑修 十七岁"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_retriever.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现 `rag/retriever.py`**

```python
"""检索：把问题向量化，再从向量库取最相近的 top-k 个片段。"""
from .embedder import Embedder
from .store import Hit, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Hit]:
        qvec = self.embedder.embed([question])[0]
        return self.store.query(qvec, self.top_k)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_retriever.py -q`
Expected: `1 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever for top-k similarity search"
```

---

## Task 6: 生成 generator.py

**Files:**
- Create: `rag/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: 写失败测试 `tests/test_generator.py`**

```python
from rag.store import Hit
from rag.generator import Generator, build_messages


def make_hits():
    return [
        Hit(text="林澈，十七岁。", source="book1/a.md", work="book1", score=0.9),
        Hit(text="林澈，十八岁（旧稿）。", source="book1/old.md", work="book1", score=0.8),
    ]


def test_build_messages_contains_context_and_question():
    msgs = build_messages("林澈几岁？", make_hits())
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "林澈几岁？" in msgs[1]["content"]
    assert "十七岁" in msgs[1]["content"]
    assert "book1/a.md" in msgs[1]["content"]
    assert "冲突" in msgs[0]["content"]  # 系统提示要求标出冲突


def test_generate_calls_api(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "答案是十七岁。"}}]}

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr("rag.generator.requests.post", fake_post)
    gen = Generator(api_key="sk-test", base_url="https://api.deepseek.com", model="deepseek-chat")
    out = gen.generate("林澈几岁？", make_hits())
    assert out == "答案是十七岁。"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["json"]["model"] == "deepseek-chat"
    assert "十七岁" in captured["json"]["messages"][1]["content"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_generator.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现 `rag/generator.py`**

```python
"""生成：把命中的片段组装成 prompt，交给 DeepSeek 写回答。"""
import requests

from .store import Hit

SYSTEM_PROMPT = (
    "你是作者的小说设定助手。只依据下面提供的【设定片段】回答，不要编造片段里没有的信息。"
    "要求：1) 直接回答问题；2) 在结尾用【来源】列出你引用的片段编号；"
    "3) 如果多条设定互相冲突，明确指出冲突，不要擅自决定哪个正确，把裁决权留给作者。"
)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context_parts = []
    for i, h in enumerate(hits, 1):
        context_parts.append(f"[片段{i}]（来源：{h.work}/{h.source}）\n{h.text}")
    context = "\n\n".join(context_parts)
    user = f"【设定片段】\n{context}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, question: str, hits: list[Hit]) -> str:
        messages = build_messages(question, hits)
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_generator.py -q`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/generator.py tests/test_generator.py
git commit -m "feat: add prompt builder and deepseek generator"
```

---

## Task 7: 组装 pipeline.py

**Files:**
- Create: `rag/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试 `tests/test_pipeline.py`**

```python
from pathlib import Path
from rag.embedder import FakeEmbedder
from rag.store import VectorStore
from rag.retriever import Retriever
from rag.pipeline import Pipeline


class FakeGenerator:
    def __init__(self):
        self.last_question = None
        self.last_hit_count = 0

    def generate(self, question, hits):
        self.last_question = question
        self.last_hit_count = len(hits)
        return "测试回答"


def test_pipeline_build_and_query(tmp_path):
    data_dir = tmp_path / "data"
    book = data_dir / "book1"
    book.mkdir(parents=True)
    (book / "c.md").write_text("# 角色\n林澈，剑修，十七岁。", encoding="utf-8")

    emb = FakeEmbedder(dim=64)
    store = VectorStore(tmp_path / "chroma")
    retriever = Retriever(emb, store, top_k=3)
    gen = FakeGenerator()
    pipe = Pipeline(data_dir, tmp_path / "chroma", emb, store, retriever, gen,
                    chunk_size=200, chunk_overlap=20)

    n = pipe.build_index()
    assert n >= 1

    answer, hits = pipe.query("林澈是谁")
    assert answer == "测试回答"
    assert gen.last_question == "林澈是谁"
    assert len(hits) >= 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现 `rag/pipeline.py`**

```python
"""把 loader/splitter/embedder/store/retriever/generator 串起来。"""
from pathlib import Path

from .loader import load_documents
from .splitter import split_text
from .store import Hit


class Pipeline:
    def __init__(self, data_dir, chroma_dir, embedder, store, retriever, generator,
                 chunk_size=500, chunk_overlap=50):
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.generator = generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_index(self) -> int:
        """读取设定 → 切分 → 向量化 → 写入向量库。返回片段总数。"""
        docs = load_documents(self.data_dir)
        ids, texts, embeddings, metadatas = [], [], [], []
        for d in docs:
            chunks = split_text(d.text, self.chunk_size, self.chunk_overlap)
            if not chunks:
                continue
            vecs = self.embedder.embed(chunks)
            for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                ids.append(f"{d.source}:{i}")
                texts.append(chunk)
                embeddings.append(vec)
                metadatas.append({"source": d.source, "work": d.work})
        if ids:
            self.store.add(ids, texts, embeddings, metadatas)
        return len(ids)

    def query(self, question: str) -> tuple[str, list[Hit]]:
        hits = self.retriever.retrieve(question)
        answer = self.generator.generate(question, hits)
        return answer, hits
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: `1 passed`

- [ ] **Step 5: 提交**

```bash
git add rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrating index and query"
```

---

## Task 8: 网页 app.py

**Files:**
- Create: `app.py`

> Streamlit 脚本不做单元测试，用"启动 + 手动点一遍"来验证。

- [ ] **Step 1: 写 `app.py`**

```python
import streamlit as st

import config
from rag.embedder import BgeM3Embedder
from rag.generator import Generator, build_messages
from rag.pipeline import Pipeline
from rag.retriever import Retriever
from rag.store import VectorStore

st.set_page_config(page_title="小说设定 RAG", page_icon="📚", layout="wide")


@st.cache_resource  # 模型加载很慢，缓存起来只加载一次
def get_pipeline():
    embedder = BgeM3Embedder(config.EMBED_MODEL_NAME)
    store = VectorStore(config.CHROMA_DIR)
    retriever = Retriever(embedder, store, config.TOP_K)
    generator = Generator(config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)
    return Pipeline(config.DATA_DIR, config.CHROMA_DIR, embedder, store, retriever,
                    generator, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


pipe = get_pipeline()

st.title("📚 小说设定 RAG")
st.caption("基于你的设定库问答，回答会引用来源并标出冲突。")

with st.sidebar:
    st.header("索引管理")
    if st.button("重建索引"):
        with st.spinner("正在读取设定 → 切分 → 向量化…"):
            n = pipe.build_index()
        st.success(f"已建立 {n} 个片段")
    st.info(f"数据目录：`{config.DATA_DIR}`")
    if not config.DEEPSEEK_API_KEY:
        st.warning("未检测到 DEEPSEEK_API_KEY，请在 `.env` 中配置后重启。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("问一个关于你小说设定的问题…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            answer, hits = pipe.query(question)
        st.markdown(answer)

        with st.expander("🔍 查看检索详情"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**片段{i}** · 来源 `{h.work}/{h.source}` · 相似度 `{h.score:.3f}`")
                st.text(h.text)

        with st.expander("🧠 查看组装后的 prompt"):
            for m in build_messages(question, hits):
                st.markdown(f"**{m['role']}**")
                st.code(m["content"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
```

- [ ] **Step 2: 启动验证**

Run: `streamlit run app.py`
Expected: 浏览器打开网页，能看到标题、侧边栏"重建索引"按钮。此时模型首次加载（下载 bge-m3）可能较慢。

- [ ] **Step 3: 提交**

```bash
git add app.py
git commit -m "feat: add streamlit web ui"
```

---

## Task 9: 示例数据与 README

**Files:**
- Create: `data/example/characters.md`, `data/example/world.md`, `data/example/timeline.md`, `data/example/old_draft.md`, `README.md`

- [ ] **Step 1: 写示例设定 `data/example/characters.md`**

```markdown
# 角色设定

## 林澈
林澈是《山海烬》的主角，十七岁，剑修，性格冷淡但重情义。
他出身青崖剑宗，佩剑名"听雨"。三年前被逐出师门，原因不明。

## 苏晚
苏晚是《山海烬》的女主角，药师，性格温和。
她是林澈的青梅竹马，一直寻找失踪的林澈。
```

- [ ] **Step 2: 写 `data/example/world.md`**

```markdown
# 世界观

## 山海烬世界
这是一个灵气复苏的世界。修行者分为练气、筑基、金丹、元婴四个境界。

## 力量体系
剑修以剑意为核心，剑意越纯粹，战力越强。
青崖剑宗位于东荒，是四大宗门之一。
```

- [ ] **Step 3: 写 `data/example/timeline.md`**

```markdown
# 时间线

- 三年前：林澈被逐出青崖剑宗。
- 一年前：林澈在临江城与苏晚重逢。
- 现在：灵气潮汐爆发，宗门震动。
```

- [ ] **Step 4: 写 `data/example/old_draft.md`**（故意放一条与角色卡冲突的旧稿，用于演示"冲突"特性）

```markdown
# 早期草稿（已废弃）

林澈的年龄在初稿里写的是十八岁，与角色卡（十七岁）冲突，此处保留旧稿供检索对比。
```

- [ ] **Step 5: 写 `README.md`**

```markdown
# 小说设定 RAG（阶段一）

把自己的小说设定做成可问答的知识库：把设定切成片段、向量化、存入 Chroma，
提问时检索最相关片段，交给 DeepSeek 生成带来源引用的回答，并标出前后冲突。

## 功能
- 多本小说隔离：`data/<作品名>/` 下各自放 `.md`/`.txt`
- 问答 + 来源引用 + 冲突提示
- 网页可视化：展开查看命中的片段、相似度、组装后的 prompt

## 快速开始
1. 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
2. `python -m venv .venv` → 激活 → `pip install -r requirements.txt`
3. 把设定放进 `data/`（每本小说一个子文件夹）
4. `streamlit run app.py` → 点侧边栏"重建索引" → 提问

## 架构
loader → splitter → embedder(bge-m3) → store(Chroma) → retriever → generator(DeepSeek) → Streamlit

## 测试
`python -m pytest -q`
```

- [ ] **Step 6: 提交**

```bash
git add data/example README.md
git commit -m "docs: add sample settings and readme"
```

---

## Task 10: 端到端冒烟测试（真实模型）

> 用真实 bge-m3 + DeepSeek 跑一遍完整流程，确认真能出结果。

- [ ] **Step 1: 确认 `.env` 已配置**（`DEEPSEEK_API_KEY` 非空）

Run: `python -c "import config; print('key set:', bool(config.DEEPSEEK_API_KEY))"`
Expected: `key set: True`

- [ ] **Step 2: 重建索引并计数**

Run: `python -c "from rag.pipeline import Pipeline; import config; from rag.embedder import BgeM3Embedder; from rag.store import VectorStore; from rag.retriever import Retriever; from rag.generator import Generator; e=BgeM3Embedder(config.EMBED_MODEL_NAME); s=VectorStore(config.CHROMA_DIR); r=Retriever(e,s,config.TOP_K); g=Generator(config.DEEPSEEK_API_KEY,config.DEEPSEEK_BASE_URL,config.DEEPSEEK_MODEL); p=Pipeline(config.DATA_DIR,config.CHROMA_DIR,e,s,r,g,config.CHUNK_SIZE,config.CHUNK_OVERLAP); print('chunks:', p.build_index())"`
Expected: 打印 `chunks: N`（N ≥ 1；示例数据应切出若干片段）

- [ ] **Step 3: 提一个能触发冲突的问题**

Run: `python -c "import config; from rag.embedder import BgeM3Embedder; from rag.store import VectorStore; from rag.retriever import Retriever; from rag.generator import Generator; from rag.pipeline import Pipeline; e=BgeM3Embedder(config.EMBED_MODEL_NAME); s=VectorStore(config.CHROMA_DIR); r=Retriever(e,s,config.TOP_K); g=Generator(config.DEEPSEEK_API_KEY,config.DEEPSEEK_BASE_URL,config.DEEPSEEK_MODEL); p=Pipeline(config.DATA_DIR,config.CHROMA_DIR,e,s,r,g,config.CHUNK_SIZE,config.CHUNK_OVERLAP); a,h=p.query('林澈几岁？'); print(a); print('--- 命中的片段 ---'); [print(x.source, round(x.score,3), x.text[:30]) for x in h]"`
Expected: 回答里出现"十七岁"，且命中的片段里应包含 `characters.md` 与 `old_draft.md`（若两者都进 top-k），回答会提示冲突。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "test: document end-to-end smoke test"
```

---

## 自检结论

- 规格覆盖：设计文档里阶段一的每个组件（loader/splitter/embedder/store/retriever/generator/web）都有对应 Task；"引用来源 + 标出冲突"写入 system prompt 并有测试；"可视化检索详情"落在 app.py 的 expander。
- 无占位符：每个代码步骤都给了完整代码。
- 类型/命名一致：`Hit(text,source,work,score)`、`Embedder.embed`、`Retriever.retrieve`、`Pipeline.build_index/query` 在各 Task 间一致。
