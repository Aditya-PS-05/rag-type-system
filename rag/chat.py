"""Terminal chatbot over the indexed corpus.

Usage: uv run python -m rag.chat [--type text,pdf,image] [-k 6]
"""

import argparse
import re
import sys
from pathlib import Path

from rag.config import Settings
from rag.ollama import Ollama
from rag.store import SearchHit, VectorStore

SYSTEM_PROMPT = """You are a helpful assistant answering questions about space exploration \
using ONLY the numbered sources provided in each message.
- Cite sources inline like [1] or [2][3].
- If the sources do not contain the answer, say you don't know based on the available documents.
- When a source is an image, describe what the image shows based on its description and mention it can be viewed.
- Be concise and accurate."""

HISTORY_TURNS = 4
IMAGES_PER_VISUAL_QUERY = 3
VISUAL_INTENT = re.compile(r"\b(images?|pictures?|photos?|photographs?|show me|look like|looks like)\b", re.I)


def retrieve(store: VectorStore, llm: Ollama, cfg: Settings, question: str, k: int,
             doc_types: list[str] | None) -> list[SearchHit]:
    embedding = llm.embed_query(question)
    hits = store.search(embedding, team=cfg.team, embed_model=cfg.embed_model, k=k, doc_types=doc_types)
    if doc_types is None and VISUAL_INTENT.search(question):
        images = store.search(embedding, team=cfg.team, embed_model=cfg.embed_model,
                              k=IMAGES_PER_VISUAL_QUERY, doc_types=["image"])
        others = [h for h in hits if h.doc_type != "image"]
        hits = images + others[:max(k - len(images), 0)]
    return hits


def format_sources(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f", page {h.page}" if h.page else ""
        blocks.append(f"[{i}] ({h.doc_type}) {h.title}{where}\n{h.content}")
    return "\n\n".join(blocks)


def citation_line(i: int, h: SearchHit, data_dir: Path) -> str:
    where = f" p.{h.page}" if h.page else ""
    loc = data_dir / h.path if h.doc_type == "image" else h.source
    return f"  [{i}] {h.doc_type:5} {h.score:.2f}  {h.title[:60]}{where}  {loc}"


def answer(llm: Ollama, history: list[dict], question: str, hits: list[SearchHit]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[-2 * HISTORY_TURNS:],
                {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"}]
    parts = []
    for token in llm.chat_stream(messages):
        parts.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()
    print()
    return "".join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", help="comma-separated doc types to search: text,pdf,image")
    ap.add_argument("-k", type=int, default=6, help="number of chunks to retrieve")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    doc_types = args.type.split(",") if args.type else None
    data_dir = Path(args.data_dir).resolve()
    history: list[dict] = []

    def ask(question: str):
        hits = retrieve(store, llm, cfg, question, args.k, doc_types)
        if not hits:
            print("No indexed documents found. Run: uv run python -m rag.ingest")
            return
        reply = answer(llm, history, question, hits)
        print("\nSources:")
        for i, h in enumerate(hits, start=1):
            print(citation_line(i, h, data_dir))
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": reply}])

    if args.question:
        ask(args.question)
        return

    print(f"Local RAG chat ({cfg.chat_model} + {cfg.embed_model}). Type 'exit' to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("bot> ", end="")
            ask(question)
            print()


if __name__ == "__main__":
    main()
