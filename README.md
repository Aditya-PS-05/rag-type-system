# Local multimodal RAG

Offline question answering over a space-exploration corpus of text, PDFs and images.
Everything runs on your machine: Postgres + pgvector stores the embeddings, Ollama runs the models.

```
data/raw/ (txt, pdf, jpg + manifest.jsonl)
   │  rag.ingest: load → chunk → embed (nomic-embed-text) → upsert
   ▼
Postgres + pgvector (documents, chunks, HNSW index)
   ▲
   │  rag.chat: embed question → vector search → prompt → qwen3:4b-instruct
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
uv run python -m rag.chat --verify                # fact-check every answer (about 2x slower)
uv run python -m rag.eval [--verify]              # grounding evaluation (evals/cases.jsonl)
uv run pytest                                     # tests (store tests need the database running)
```

In the chat: `/sources` lists indexed documents, `/verify` toggles fact-checking, `/help` shows commands.

## Keeping answers grounded in the documents

| Layer | Where | What it does |
|---|---|---|
| Relevance gate | `rag/qa.py` | If no retrieved chunk scores ≥ `RAG_MIN_SCORE` (0.68), the bot refuses **without calling the LLM**. |
| Grounding prompt | `rag/qa.py` | Sources only, a citation on every factual sentence, a fixed refusal sentence, temperature 0. |
| Citation check | `rag/grounding.py` | Flags sentences without a citation or citing a source number that doesn't exist. Always on. |
| Fact check | `rag/grounding.py` | `--verify`: the LLM judges each cited sentence against only its cited chunks and flags unsupported claims. Strict: true facts whose evidence sits in a chunk the sentence doesn't cite are flagged too. |
| Evaluation | `rag/eval.py` | Answerable questions with expected answers, image questions, and trap questions (general knowledge not in the corpus) that must be refused. Re-run after changing the model, prompt, chunking or threshold. |

The threshold was calibrated with the evaluation: answerable questions scored ≥ 0.76, trap questions ≤ 0.63.
Re-check it with `RAG_MIN_SCORE=0 uv run python -m rag.eval` when the corpus or embedding model changes.

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
| `RAG_MIN_SCORE` | `0.68` |

## Sharing across teams

- Point `DATABASE_URL` at a shared Postgres with the pgvector extension (same `pgvector/pgvector` image) and `OLLAMA_URL` at a shared Ollama server.
- Every document row carries `team` and `embed_model`; searches filter on both. Vectors from different embedding models are never mixed.
- Change the embedding model by setting `EMBED_MODEL`/`EMBED_DIM` and re-ingesting. A different dimension needs a fresh `chunks` table.
- All database access lives in `rag/store.py`, so swapping to another vector database means changing that file only.
