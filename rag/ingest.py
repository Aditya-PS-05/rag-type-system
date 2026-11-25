"""Chunk, embed and store every document listed in a dataset manifest.

Usage: uv run python -m rag.ingest [data_dir] [--force]
"""

import argparse
import time
from pathlib import Path

from rag.chunking import chunk_text
from rag.config import Settings
from rag.loaders import Document, load_manifest
from rag.ollama import Ollama
from rag.store import Chunk, VectorStore, content_hash


def build_chunks(doc: Document) -> list[tuple[int | None, str]]:
    return [(page, piece) for page, text in doc.sections for piece in chunk_text(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data/raw")
    ap.add_argument("--force", action="store_true", help="re-embed documents that are unchanged")
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    store.init_schema()
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)

    started, done, skipped = time.time(), 0, 0
    for doc in load_manifest(Path(args.data_dir)):
        chash = content_hash(doc.sections)
        if not args.force and store.is_current(cfg.team, doc.path, chash, cfg.embed_model):
            skipped += 1
            continue
        pieces = build_chunks(doc)
        if not pieces:
            print(f"  [{doc.doc_type}] no text extracted: {doc.path}")
            continue
        vectors = llm.embed_documents([f"{doc.title}\n{text}" for _, text in pieces])
        store.upsert_document(
            team=cfg.team, doc_type=doc.doc_type, path=doc.path, title=doc.title,
            source=doc.source, license=doc.license, meta=doc.meta, chash=chash,
            embed_model=cfg.embed_model,
            chunks=[Chunk(i, page, text, vec) for i, ((page, text), vec) in enumerate(zip(pieces, vectors))],
        )
        done += 1
        print(f"  [{doc.doc_type}] {doc.title[:70]} -> {len(pieces)} chunks")

    print(f"\nIngested {done}, unchanged {skipped} in {time.time() - started:.0f}s (team={cfg.team})")
    for doc_type, docs, chunks in store.stats(cfg.team):
        print(f"  {doc_type:6} {docs:4} documents {chunks:6} chunks")


if __name__ == "__main__":
    main()
