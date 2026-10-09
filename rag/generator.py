"""生成：把命中的片段组装成 prompt，交给 DeepSeek 写回答。"""
import requests

from .store import Hit

SYSTEM_PROMPT = (
    "你是作者的小说设定助手。只依据下面提供的【设定片段】回答，不要编造片段里没有的信息。"
    "要求：1) 直接回答问题；2) 在结尾用【来源】列出你引用的片段编号；"
    "3) 如果多条设定互相冲突，明确指出冲突，不要擅自决定哪个正确，把裁决权留给作者。"
)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context_parts = []
    for i, h in enumerate(hits, 1):
        context_parts.append(f"[片段{i}]（来源：{h.work}/{h.source}）\n{h.text}")
    context = "\n\n".join(context_parts)
    user = f"【设定片段】\n{context}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, question: str, hits: list[Hit]) -> str:
        messages = build_messages(question, hits)
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
