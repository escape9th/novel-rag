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
    store = VectorStore(tmp_path / "chroma", ephemeral=True)
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
