# Sharing across teams

## What changes
1. **Database:** run `pgvector/pgvector:pg17` (or managed Postgres with the `vector` extension) on a server; point `DATABASE_URL` at it. Change the default password.
2. **Models:** run Ollama on a GPU server; point `OLLAMA_URL` at it. Everyone must use the **same embedding model and version**; vectors from different models are meaningless to each other. Each row stores `embed_model` and searches filter on it.
3. **Teams:** set `RAG_TEAM` per team for ingest and chat. Every search filters on `team`. For hard isolation add Postgres row-level security on `documents.team`.
4. **Raw files:** keep originals in object storage (MinIO/S3/NAS); the database holds text, vectors and a pointer.
5. **Service:** wrap `rag.qa.ask()` in an HTTP API so apps don't each embed the pipeline.

## Sizing
One 768-d float vector is 3 KB.

| Chunks | Vectors | With HNSW + text |
|---|---|---|
| 1 M | ~3 GB | ~6–8 GB |
| 10 M | ~30 GB | ~60–80 GB |

Shrink with `halfvec` (half the size), or truncate nomic vectors to 256-d (Matryoshka, small quality loss).

## Scaling path
- Up to tens of millions of chunks: Postgres + pgvector, read replicas for search.
- Beyond, or heavy filtered search: Qdrant (quantization cuts memory 4–32×). Only `rag/store.py` changes.
- Throughput: the LLM is the bottleneck (4 calls per strict answer). Use a bigger GPU server, vLLM, or `--fast` mode for low-risk uses.

## Changing models
- Embedding model: set `EMBED_MODEL`/`EMBED_DIM`, re-ingest. A different dimension needs a new `chunks` table.
- Chat or reranker model: no re-ingest; re-run `rag.calibrate` (chat warns that the gate is stale) and `rag.eval`.

For a full GPU-cluster plan (H100 model choices, placement, sizing, rollout), see [production.html](production.html) or https://rag.adityaps.work/production.
