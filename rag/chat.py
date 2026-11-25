"""Terminal chatbot over the indexed corpus.

Usage: uv run python -m rag.chat [--type text,pdf,image] [-k 6] [--verify]
Commands inside the chat: /sources, /verify (toggle), /help, exit
"""

import argparse
import sys
from pathlib import Path

from rag.config import Settings
from rag.grounding import verify_claims
from rag.ollama import Ollama
from rag.qa import Answer, ask
from rag.store import SearchHit, VectorStore

HELP = """Commands:
  /sources   list the indexed documents
  /verify    toggle fact-checking of each answer against its cited sources
  /help      show this help
  exit       quit"""


def citation_line(i: int, h: SearchHit, data_dir: Path) -> str:
    where = f" p.{h.page}" if h.page else ""
    loc = data_dir / h.path if h.doc_type == "image" else h.source
    return f"  [{i}] {h.doc_type:5} {h.score:.2f}  {h.title[:60]}{where}  {loc}"


def print_report(result: Answer, data_dir: Path, min_score: float):
    if result.gated:
        best = max((h.score for h in result.hits), default=0)
        print(f"\n(no source was relevant enough: best match {best:.2f} < threshold {min_score:.2f}; "
              "the model was not called)")
        return
    if not result.refused:
        print("\nSources:")
        for i, h in enumerate(result.hits, start=1):
            print(citation_line(i, h, data_dir))
    claims = result.grounding.claims
    if not claims:
        return
    checked = [c for c in claims if c.supported is not None]
    problems = result.grounding.problems
    if checked:
        print(f"\nFact check: {sum(c.supported for c in checked)}/{len(checked)} cited claims supported by their sources")
    if problems:
        print("Warnings:")
        for c in problems:
            why = c.issue or f"not supported by its source: {c.reason}"
            print(f"  ⚠ {why}\n      \"{c.text[:160]}\"")
    elif not checked:
        print("\nCitations: every claim cites a source (use /verify to also fact-check them)")


def print_sources(store: VectorStore, team: str):
    docs = store.list_documents(team)
    for doc_type in ("text", "pdf", "image"):
        titles = [t for d, t in docs if d == doc_type]
        if titles:
            print(f"\n{doc_type} ({len(titles)}):")
            for t in titles:
                print(f"  - {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", help="comma-separated doc types to search: text,pdf,image")
    ap.add_argument("-k", type=int, default=6, help="number of chunks to retrieve")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--verify", action="store_true", help="fact-check each answer against its sources")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    doc_types = args.type.split(",") if args.type else None
    data_dir = Path(args.data_dir).resolve()
    history: list[dict] = []
    verify = args.verify

    def stream(token: str):
        sys.stdout.write(token)
        sys.stdout.flush()

    def handle(question: str):
        result = ask(store, llm, cfg, question, history=history, k=args.k, doc_types=doc_types,
                     on_token=stream)
        print()
        if not result.hits:
            print("No indexed documents found. Run: uv run python -m rag.ingest")
            return
        if verify and result.grounding.claims:
            print("\n(fact-checking each claim against its cited source...)")
            verify_claims(llm, result.grounding, result.hits)
        print_report(result, data_dir, cfg.min_score)
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result.text}])

    if args.question:
        handle(args.question)
        return

    print(f"Local RAG chat ({cfg.chat_model} + {cfg.embed_model}). Type /help for commands.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question == "/help":
            print(HELP)
        elif question == "/sources":
            print_sources(store, cfg.team)
        elif question == "/verify":
            verify = not verify
            print(f"fact-checking {'on' if verify else 'off'}")
        elif question:
            print("bot> ", end="")
            handle(question)
        print()


if __name__ == "__main__":
    main()
