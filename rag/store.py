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
