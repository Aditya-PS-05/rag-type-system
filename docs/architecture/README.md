# Architecture versions

Each `.eraser` file is Eraser diagram-as-code: in eraser.io create a diagram (Flow chart), open the code panel and paste the file.

## V1 — Basic local RAG
Wikipedia + arXiv PDFs + NASA images → loaders → 1,200-char chunks → nomic-embed-text → Postgres/pgvector (HNSW).
Question → embed → top-6 cosine search → (image boost for picture questions) → qwen3-4b-instruct → answer with citations.
**Weakness:** nothing stops the model answering from its own knowledge or misreading a source.

## V2 — Grounding guardrails
V1 + relevance gate (best cosine < 0.68 → refuse, LLM never called) + strict prompt at temperature 0
+ citation check on every answer + optional LLM fact check + evaluation set with trap questions.
**Weakness:** warnings only; the model still reads whole chunks and can misread them; follow-ups break.

## V3 — Evidence-first answering
Follow-up rewriting → retrieve → gate → sources split into numbered sentences → LLM picks sentence IDs
→ code looks up the exact sentences (quotes cannot be invented) → LLM answers from those quotes only
→ citation check → one batched fact check → unsupported sentences removed; nothing left → refusal.
**Weakness:** retrieval misses chunks that use different words than the question.

## V4 — Hybrid retrieval with reranking
Vector search (top 30) + Postgres full-text search (top 30) → reciprocal rank fusion → jina-turbo cross-encoder
reranks 24 candidates → gate on reranker score → neighbouring chunks added for context → V3 evidence pipeline.

## V5 — Shared team deployment (proposed)
Document watcher → ingest workers → MinIO for raw files + replicated Postgres/pgvector with team ACL and embedding-model tags.
Team apps → RAG API service → shared embedding server, reranker and LLM server → grounded answers filtered by team.
