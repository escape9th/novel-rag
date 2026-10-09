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
