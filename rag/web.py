"""Local web app: the same ask() pipeline as the terminal chat, in a browser.

Standard library only; nothing is fetched from the internet. One request is answered at a time
(the database connection and the laptop's model are shared).

Usage: uv run python -m rag.web [--port 8000] [--host 127.0.0.1] [--data-dir data/raw]
Then open http://127.0.0.1:8000
"""

import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from rag import calibrate
from rag.config import Settings
from rag.ollama import Ollama
from rag.qa import Answer, ask
from rag.rerank import Reranker
from rag.store import VectorStore

STATIC = Path(__file__).resolve().parent / "static"
MAX_HISTORY_MESSAGES = 8
DOC_TYPES = {"text", "pdf", "image"}


def hit_url(hit, data_dir: Path) -> str | None:
    if hit.doc_type in ("image", "pdf") and (data_dir / hit.path).is_file():
        return f"/files/{hit.path}" + (f"#page={hit.page}" if hit.doc_type == "pdf" and hit.page else "")
    return hit.source or None


def answer_to_json(result: Answer, threshold: float, data_dir: Path) -> dict:
    removed = {id(c) for c in result.removed}
    return {
        "text": result.text,
        "refused": result.refused,
        "gated": result.gated,
        "search_query": result.search_query,
        "best_score": max((h.score for h in result.hits), default=0.0),
        "threshold": threshold,
        "hits": [{"n": i, "doc_type": h.doc_type, "title": h.title, "source": h.source, "page": h.page,
                  "score": round(h.score, 3), "url": hit_url(h, data_dir), "content": h.content,
                  "image": f"/files/{h.path}" if h.doc_type == "image" and (data_dir / h.path).is_file() else None}
                 for i, h in enumerate(result.hits, 1)],
        "quotes": [{"source": q.source, "text": q.text, "verified": q.verified} for q in result.quotes],
        "claims": [{"text": c.text, "citations": c.citations, "supported": c.supported, "issue": c.issue,
                    "reason": c.reason, "removed": id(c) in removed} for c in result.grounding.claims],
        "cited": result.cited_sources,
    }


def clean_history(raw) -> list[dict]:
    history = [{"role": m["role"], "content": str(m["content"])} for m in raw or []
               if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    return history[-MAX_HISTORY_MESSAGES:]


class App:
    def __init__(self, data_dir: Path):
        cfg = Settings()
        self.store = VectorStore(cfg.database_url, cfg.embed_dim)
        self.llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
        self.reranker = Reranker(cfg.rerank_model) if cfg.use_reranker else None
        self.cfg, self.gate_note = calibrate.apply(cfg, self.store, self.reranker is not None)
        self.threshold = self.cfg.min_rerank_score if self.reranker else self.cfg.min_score
        self.data_dir = data_dir.resolve()
        self.lock = threading.Lock()

    def info(self) -> dict:
        with self.lock:
            stats = self.store.stats(self.cfg.team)
        return {"chat_model": self.cfg.chat_model, "embed_model": self.cfg.embed_model,
                "reranker": self.cfg.rerank_model if self.reranker else None, "gate": self.gate_note,
                "threshold": self.threshold, "team": self.cfg.team,
                "stats": [{"doc_type": t, "documents": d, "chunks": c} for t, d, c in stats]}

    def documents(self) -> list[dict]:
        with self.lock:
            return [{"doc_type": t, "title": title} for t, title in self.store.list_documents(self.cfg.team)]

    def ask(self, body: dict, emit):
        question = str(body.get("question", "")).strip()
        if not question:
            raise ValueError("Type a question first.")
        types = [t for t in body.get("types") or [] if t in DOC_TYPES] or None
        k = min(max(int(body.get("k", 8)), 1), 20)
        strict = bool(body.get("strict", True))
        with self.lock:
            result = ask(self.store, self.llm, self.cfg, question, history=clean_history(body.get("history")),
                         k=k, doc_types=types, strict=strict, reranker=self.reranker,
                         on_token=lambda t: emit({"type": "token", "text": t}),
                         on_stage=lambda s: emit({"type": "stage", "name": s}))
        emit({"type": "result", "answer": answer_to_json(result, self.threshold, self.data_dir)})

    def file_path(self, rel: str) -> Path | None:
        path = (self.data_dir / rel).resolve()
        if path.is_file() and path.is_relative_to(self.data_dir) and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf"}:
            return path
        return None


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send_body(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status: int = 200):
            self.send_body(status, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_body(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self.send_json(app.info())
            elif path == "/api/documents":
                self.send_json(app.documents())
            elif path.startswith("/files/") and (file := app.file_path(unquote(path[len("/files/"):]))):
                self.send_body(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/ask":
                return self.send_json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self.send_json({"error": "Request body must be JSON."}, 400)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(event: dict):
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()

            try:
                app.ask(body, emit)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                emit({"type": "error", "message": str(exc) or exc.__class__.__name__})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args()
    app = App(Path(args.data_dir))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"RAG web app on http://{args.host}:{args.port}  ({app.gate_note})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
