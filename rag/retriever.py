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
