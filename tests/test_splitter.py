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
