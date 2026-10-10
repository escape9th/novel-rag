from pathlib import Path

from rag.store import VectorStore
from rag.embedder import FakeEmbedder
from rag.retriever import Retriever


def test_retriever_returns_top_k(tmp_path):
    emb = FakeEmbedder(dim=64)
    store = VectorStore(Path(tmp_path), ephemeral=True)
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
