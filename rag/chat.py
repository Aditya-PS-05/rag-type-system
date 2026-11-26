"""Terminal chatbot over the indexed corpus.

Usage: uv run python -m rag.chat [--type text,pdf,image] [-k 6] [--fast] [--show-evidence]
Commands inside the chat: /sources, /strict (toggle), /evidence (toggle), /help, exit
"""

import argparse
import sys
from pathlib import Path

from rag import calibrate
from rag.config import Settings
from rag.ollama import Ollama
from rag.qa import Answer, ask
from rag.rerank import Reranker
from rag.store import SearchHit, VectorStore

HELP = """Commands:
  /sources   list the indexed documents
  /strict    toggle strict mode (fact-check and remove unsupported claims; on by default)
  /evidence  toggle showing the verified quotes each answer was built from
  /help      show this help
  exit       quit"""


def citation_line(i: int, h: SearchHit, data_dir: Path) -> str:
    where = f" p.{h.page}" if h.page else ""
    loc = data_dir / h.path if h.doc_type == "image" else h.source
    return f"  [{i}] {h.doc_type:5} {h.title[:60]}{where}  {loc}"


def print_report(result: Answer, data_dir: Path, min_score: float, strict: bool, show_evidence: bool):
    if result.gated:
        best = max((h.score for h in result.hits), default=0)
        print(f"\n(no source was relevant enough: best match {best:.2f} < threshold {min_score:.2f}; "
              "the model was not called)")
        return
    if result.rejected_quotes:
        print(f"\n({len(result.rejected_quotes)} evidence ID(s) discarded: not present in the sources)")
    if not result.verified_quotes:
        print("(no evidence for this question was found in the retrieved sources)")
        return
    if show_evidence:
        print("\nEvidence (verified quotes):")
        for q in result.verified_quotes:
            print(f"  [{q.source}] \"{q.text[:220]}\"")
    if not result.refused and result.cited_sources:
        print("\nSources:")
        for n in result.cited_sources:
            print(citation_line(n, result.hits[n - 1], data_dir))
    if result.removed:
        print(f"\nRemoved {len(result.removed)} unsupported sentence(s):")
        for c in result.removed:
            why = c.issue or c.reason
            print(f"  ✗ \"{c.text[:150]}\"\n      {why}")
    elif not strict and result.grounding.problems:
        print("\nWarnings:")
        for c in result.grounding.problems:
            print(f"  ⚠ {c.issue}: \"{c.text[:150]}\"")
    elif strict and result.grounding.claims:
        print(f"\nFact check: all {len(result.grounding.claims)} claim(s) supported by the quoted sources")


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
    ap.add_argument("-k", type=int, default=8, help="number of chunks to retrieve")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--fast", action="store_true", help="stream answers without fact-checking each claim")
    ap.add_argument("--show-evidence", action="store_true", help="print the verified quotes behind each answer")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    reranker = Reranker(cfg.rerank_model) if cfg.use_reranker else None
    cfg, gate_note = calibrate.apply(cfg, store, reranker is not None)
    if "STALE" in gate_note:
        print(f"warning: {gate_note}", file=sys.stderr)
    doc_types = args.type.split(",") if args.type else None
    data_dir = Path(args.data_dir).resolve()
    history: list[dict] = []
    strict, show_evidence = not args.fast, args.show_evidence

    def stream(token: str):
        sys.stdout.write(token)
        sys.stdout.flush()

    def handle(question: str):
        result = ask(store, llm, cfg, question, history=history, k=args.k, doc_types=doc_types,
                     strict=strict, reranker=reranker, on_token=stream)
        if strict or not result.verified_quotes:
            print(result.text)
        else:
            print()
        if result.search_query != question:
            print(f"\n(searched for: {result.search_query})")
        if not result.hits:
            print("No indexed documents found. Run: uv run python -m rag.ingest")
            return
        print_report(result, data_dir, cfg.min_rerank_score if reranker else cfg.min_score, strict, show_evidence)
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result.text}])

    if args.question:
        handle(args.question)
        return

    mode = "strict" if strict else "fast"
    rerank = "reranker on" if reranker else "reranker off"
    print(f"Local RAG chat ({cfg.chat_model} + {cfg.embed_model}, {rerank}, {mode} mode, {gate_note}). "
          f"Type /help for commands.\n")
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
        elif question == "/strict":
            strict = not strict
            print(f"strict mode {'on' if strict else 'off'}")
        elif question == "/evidence":
            show_evidence = not show_evidence
            print(f"showing evidence {'on' if show_evidence else 'off'}")
        elif question:
            print("bot> ", end="", flush=True)
            handle(question)
        print()


if __name__ == "__main__":
    main()
