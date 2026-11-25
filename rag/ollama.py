import json
from collections.abc import Iterator

import httpx

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Ollama:
    def __init__(self, base_url: str, embed_model: str, chat_model: str):
        self.client = httpx.Client(base_url=base_url, timeout=300)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.post("/api/embed", json={
            "model": self.embed_model, "input": texts, "truncate": True,
        })
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed([DOC_PREFIX + t for t in texts[i:i + batch_size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    def chat_stream(self, messages: list[dict], num_ctx: int = 8192) -> Iterator[str]:
        with self.client.stream("POST", "/api/chat", json={
            "model": self.chat_model, "messages": messages, "stream": True,
            "options": {"num_ctx": num_ctx, "temperature": 0.2},
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("message", {}).get("content", "")
