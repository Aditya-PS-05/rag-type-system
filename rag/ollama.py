import json
import os
from collections.abc import Iterator

import httpx

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


class Ollama:
    def __init__(self, base_url: str, embed_model: str, chat_model: str):
        self.client = httpx.Client(base_url=base_url, timeout=300)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.post("/api/embed", json={
            "model": self.embed_model, "input": texts, "truncate": True, "keep_alive": KEEP_ALIVE,
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

    def warm(self):
        """Load both models now so the first question doesn't pay for it."""
        self.embed_query("warm up")
        self.client.post("/api/generate", json={"model": self.chat_model, "keep_alive": KEEP_ALIVE}).raise_for_status()

    def chat_stream(self, messages: list[dict], num_ctx: int = 8192, max_tokens: int | None = None) -> Iterator[str]:
        options = {"num_ctx": num_ctx, "temperature": 0} | ({"num_predict": max_tokens} if max_tokens else {})
        with self.client.stream("POST", "/api/chat", json={
            "model": self.chat_model, "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
            "options": options,
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("message", {}).get("content", "")

    def chat_json(self, messages: list[dict], schema: dict, num_ctx: int = 8192) -> dict:
        r = self.client.post("/api/chat", json={
            "model": self.chat_model, "messages": messages, "stream": False, "format": schema, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": num_ctx, "temperature": 0},
        })
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
