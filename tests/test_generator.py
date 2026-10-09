from rag.store import Hit
from rag.generator import Generator, build_messages


def make_hits():
    return [
        Hit(text="林澈，十七岁。", source="book1/a.md", work="book1", score=0.9),
        Hit(text="林澈，十八岁（旧稿）。", source="book1/old.md", work="book1", score=0.8),
    ]


def test_build_messages_contains_context_and_question():
    msgs = build_messages("林澈几岁？", make_hits())
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "林澈几岁？" in msgs[1]["content"]
    assert "十七岁" in msgs[1]["content"]
    assert "book1/a.md" in msgs[1]["content"]
    assert "冲突" in msgs[0]["content"]  # 系统提示要求标出冲突


def test_generate_calls_api(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "答案是十七岁。"}}]}

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr("rag.generator.requests.post", fake_post)
    gen = Generator(api_key="sk-test", base_url="https://api.deepseek.com", model="deepseek-chat")
    out = gen.generate("林澈几岁？", make_hits())
    assert out == "答案是十七岁。"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["json"]["model"] == "deepseek-chat"
    assert "十七岁" in captured["json"]["messages"][1]["content"]
