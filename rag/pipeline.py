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
