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
