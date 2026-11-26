# Design decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Vector store | Postgres + pgvector | Chroma, Qdrant, Milvus | A real database teams already run: SQL, backups, replication, row-level security, full-text search in the same place. Qdrant if vectors pass ~50M. |
| Embeddings | `nomic-embed-text` (768-d, 137M params) | larger embedders | Open weights, 274 MB, fast on a laptop, 8k context. Needs `search_document:`/`search_query:` prefixes. |
| Chat model | `qwen3:4b-instruct` | `qwen3:4b` (thinking) | The thinking variant ignores `think: false`, prints its reasoning, takes 30–60 s. Instruct: same size, 3–8 s. |
| Evidence | LLM returns sentence IDs | LLM copies quotes | Copying cost ~150 tokens (6–11 s at 19 tok/s) and needed fuzzy matching. IDs: ~15 tokens, exact by construction. |
| Fact check | one batched call | one call per claim | Per-claim was slower and over-strict (rejected "space telescope" vs "observatory"). Batched with source titles and explicit paraphrase rules. |
| Unsupported claims | remove them | show warnings | Users act on what they read; a warning under a wrong sentence still misleads. |
| Reranker | `jina-reranker-v1-turbo-en` | `bge-reranker-v2-m3` (int8) | Same top results in testing; 0.7 s vs 6.3 s on CPU. |
| Gate threshold | calibrated from labelled questions, saved with a fingerprint | hand-picked constant | A constant is only right for the corpus and models it was picked on, and goes wrong silently. Midpoint between the lowest answerable and highest trap below it gives the widest margin; refusing real questions is treated as worse than letting a trap reach evidence selection. |
| Fusion | reciprocal rank fusion | weighted score blend | Cosine and `ts_rank` scores aren't comparable; RRF uses ranks only, no tuning. |
| Keyword query | terms OR-ed | `websearch_to_tsquery` (AND) | AND drops chunks missing one word of a long question; the reranker sorts out precision. |
| PDF parsing | `pypdf` | PyMuPDF | Permissive license (PyMuPDF is AGPL); good enough for text. |
| Images | text metadata | vision-language captions, CLIP | NASA descriptions are rich; no extra model needed. A captioning model is the next step for images without metadata. |
| Runtime | Ollama native, Postgres in Docker | all in Docker | Docker on macOS can't use the GPU; Postgres image matches a team server. |
| Download tool | `curl` | Python `urllib` | Wikipedia returned HTTP 429 to urllib but not to curl. |
