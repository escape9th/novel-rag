"""Chroma 向量库封装：写入向量、按相似度查询。"""
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb

COLLECTION_NAME = "novel_settings"


@dataclass
class Hit:
    text: str
    source: str
    work: str
    score: float  # 余弦相似度，越接近 1 越像


class VectorStore:
    def __init__(self, persist_dir: Path, ephemeral: bool = False):
        settings = chromadb.Settings(anonymized_telemetry=False)
        if ephemeral:
            # 内存模式：测试用，不落盘；用唯一集合名避免同进程内多个实例共享同一内存库
            self._client = chromadb.EphemeralClient(settings=settings)
            self._name = f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}"
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
            self._name = COLLECTION_NAME
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
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
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
