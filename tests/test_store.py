from pathlib import Path

from rag.store import VectorStore
from rag.embedder import FakeEmbedder


def test_add_and_query(tmp_path):
    emb = FakeEmbedder(dim=64)
    store = VectorStore(Path(tmp_path), ephemeral=True)
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
