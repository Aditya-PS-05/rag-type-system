# Local multimodal RAG

Offline question answering over a space-exploration corpus of text, PDFs and images.
Everything runs on your machine: Postgres + pgvector stores the embeddings, Ollama runs the models.

```
data/raw/ (txt, pdf, jpg + manifest.jsonl)
   │  rag.ingest: load → chunk → embed (nomic-embed-text) → upsert
   ▼
Postgres + pgvector (documents, chunks, HNSW index)
   ▲
   │  rag.chat: rewrite follow-up → vector + keyword search → rerank → gate
   │            → select evidence sentences → answer from quotes → fact-check → remove unsupported
you
```

## Setup

Requirements: Docker, [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/).

```bash
brew install ollama && brew services start ollama
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct

docker compose up -d          # Postgres 17 + pgvector on 127.0.0.1:5432
uv sync
```

## Usage

```bash
python3 scripts/download_dataset.py data/raw     # optional: re-download (text,pdf,image subset as 2nd arg)
uv run python -m rag.ingest data/raw              # unchanged documents are skipped; --force re-embeds
uv run python -m rag.chat                         # interactive chat
uv run python -m rag.chat -q "What does PIXL do?" # one-shot
uv run python -m rag.chat --type image            # search only images (or text, pdf)
uv run python -m rag.chat --show-evidence         # also print the quotes each answer was built from
uv run python -m rag.chat --fast                  # stream answers, skip claim fact-checking
uv run python -m rag.eval [--fast]                # grounding evaluation (evals/cases.jsonl)
uv run pytest                                     # tests (store tests need the database running)
```

In the chat: `/sources` lists indexed documents, `/strict` toggles fact-checking, `/evidence` toggles quotes, `/help` shows commands.

The first run downloads the reranker (~150 MB) into `models/`; after that everything works offline.

## Keeping answers grounded in the documents

| Step | Where | What it does |
|---|---|---|
| Follow-up rewriting | `rag/qa.py` | "When did it launch?" becomes a standalone question before searching; chat history is never used as a source. |
| Hybrid retrieval | `rag/retrieval.py` | Vector search + Postgres full-text search, merged with reciprocal rank fusion, reranked by a local cross-encoder, neighbouring chunks added. |
| Relevance gate | `rag/qa.py` | If the best reranker score is below `RAG_MIN_RERANK_SCORE`, the bot refuses **without calling the LLM**. |
| Evidence by sentence ID | `rag/qa.py`, `rag/grounding.py` | Sources are split into numbered sentences; the LLM picks IDs and code looks up the exact text, so quotes cannot be invented. No valid evidence → refusal. |
| Answer from quotes | `rag/qa.py` | The answering LLM sees only the verified quotes, never whole chunks. |
| Citation check | `rag/grounding.py` | Every factual sentence must cite a source that has a verified quote. |
| Fact check + removal | `rag/grounding.py` | One batched LLM check of every claim against its cited quotes; unsupported sentences are removed (strict mode, the default). |
| Evaluation | `rag/eval.py` | Facts, images, follow-ups, and trap questions (general knowledge or on-topic but absent) that must be refused. |

Re-calibrate the gate with `RAG_MIN_RERANK_SCORE=0 uv run python -m rag.eval` when the corpus or models change.
Architecture history and diagrams: [`docs/architecture`](docs/architecture/README.md).

## Dataset

| Type | Count | Source | License |
|---|---|---|---|
| Text | 35 | Wikipedia articles | CC BY-SA 4.0 |
| PDF | 20 | arXiv papers | per paper, see source link |
| Image | 100 | NASA Image and Video Library | Public domain |

Images are retrieved through their NASA title, description, keywords and date.

## Configuration

| Env var | Default |
|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `EMBED_MODEL` / `EMBED_DIM` | `nomic-embed-text` / `768` |
| `CHAT_MODEL` | `qwen3:4b-instruct` |
| `RAG_TEAM` | `default` |
| `RAG_RERANK` / `RERANK_MODEL` | `1` / `jinaai/jina-reranker-v1-turbo-en` |
| `RAG_MIN_RERANK_SCORE` | `0.1` (gate with reranker) |
| `RAG_MIN_SCORE` | `0.68` (gate with `RAG_RERANK=0`) |

## Sharing across teams

- Point `DATABASE_URL` at a shared Postgres with the pgvector extension (same `pgvector/pgvector` image) and `OLLAMA_URL` at a shared Ollama server.
- Every document row carries `team` and `embed_model`; searches filter on both. Vectors from different embedding models are never mixed.
- Change the embedding model by setting `EMBED_MODEL`/`EMBED_DIM` and re-ingesting. A different dimension needs a fresh `chunks` table.
- All database access lives in `rag/store.py`, so swapping to another vector database means changing that file only.
