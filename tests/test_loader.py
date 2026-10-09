from pathlib import Path

from rag.loader import load_documents


def test_loads_md_and_txt(tmp_path):
    work = tmp_path / "book1"
    work.mkdir()
    (work / "characters.md").write_text("# 角色\n林澈，十七岁。", encoding="utf-8")
    (work / "notes.txt").write_text("旧稿：林澈十八岁。", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("ignore", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    by_name = {d.source: d for d in docs}
    assert by_name["book1/characters.md"].work == "book1"
    assert "十七岁" in by_name["book1/characters.md"].text
    assert by_name["book1/notes.txt"].work == "book1"


def test_empty_dir_returns_empty(tmp_path):
    assert load_documents(tmp_path) == []
