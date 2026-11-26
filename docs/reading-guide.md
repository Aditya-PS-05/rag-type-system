# How to read this code

About 1,100 lines of Python in `rag/`. Read in this order and you will see every idea once, in the order data flows. Budget: about an hour.

Before you start, look at the picture: open [architecture/viewer.html](architecture/viewer.html) in a browser (or the Mermaid diagrams in [architecture/README.md](architecture/README.md)) and click through V1 → V4.

## 0. Run it first (10 min)
Follow the README quickstart, then ask a question with `uv run python -m rag.chat --show-evidence`. Seeing the evidence, citations and fact-check lines makes the code much easier to follow.

## 1. Settings and the model client (5 min)
| File | Read for |
|---|---|
| `rag/config.py` | Every knob, read from environment variables. Note the two gate thresholds. |
| `rag/ollama.py` | Three calls are all the LLM work: `embed_documents`/`embed_query` (note the `search_document:`/`search_query:` prefixes), `chat_stream`, and `chat_json` (JSON-schema-constrained output). |

## 2. Getting documents in (15 min)
Follow `rag/ingest.py: main()` top to bottom; it calls everything else:
1. `rag/loaders.py`: manifest → `Document`. See `clean_wikipedia`, `clean_pdf_page` (why NUL bytes are stripped) and `load_image` (images become text).
2. `rag/chunking.py: chunk_text`: paragraph packing, 1,200 chars, 200 overlap.
3. `rag/ingest.py: build_chunks`: the title is embedded with each chunk.
4. `rag/store.py`: `init_schema` (tables, HNSW + GIN indexes), `is_current` (skip unchanged docs), `upsert_document`.

`rag/store.py` is the **only** module that talks to Postgres. Every query filters by `team` and `embed_model`.

## 3. Answering a question (25 min). This is the core.
Start at `rag/qa.py: ask()` and step into each call:

| Step in `ask()` | Go to | Key idea |
|---|---|---|
| Rewrite follow-up | `qa.rewrite_question` | History is used only to rewrite, never as a source |
| Retrieve | `retrieval.retrieve` | vector `store.search` + keyword `store.keyword_search` → `rrf_fuse` → `rerank.Reranker.score` → `with_neighbors` |
| Gate | back in `ask()` | Refuse if the best score is below the threshold, before any LLM call |
| Select evidence | `qa.select_evidence` → `grounding.number_sentences`, `quotes_from_ids` | The LLM returns sentence IDs like `2.5`; code looks up the exact text. Invented IDs are dropped. |
| Answer | `ask()` with `ANSWER_PROMPT` + `format_evidence` | The model sees only verified quotes, not chunks |
| Check | `grounding.check_citations`, `verify_claims` | Every sentence must cite a quoted source; one batched LLM fact check |
| Remove | `grounding.remove_claims` | Unsupported sentences are deleted; nothing left → `REFUSAL` |

Read the three prompts at the top of `rag/qa.py` and `VERIFY_PROMPT` in `rag/grounding.py`; they are short and explain intent.

## 4. The interfaces (10 min)
- `rag/chat.py`: `handle()` calls `ask()`; `print_report` shows everything `Answer` carries (gate score, quotes, removed sentences).
- `rag/eval.py`: the same `ask()` over `evals/cases.jsonl`; see how facts, traps and follow-ups are scored.

## 5. Tests as executable examples
| Test | Shows |
|---|---|
| `tests/test_grounding.py` | Sentence IDs, citation checks, claim removal with no LLM |
| `tests/test_qa.py` | The whole `ask()` pipeline with a `FakeLLM` that answers by JSON schema |
| `tests/test_retrieval.py` | Fusion, neighbours, image boost |
| `tests/test_store.py` | Real Postgres integration (needs `docker compose up`) |

## Then read the why
[versions.md](versions.md) (what each version fixed) → [hallucination.md](hallucination.md) (which step stops which failure) → [decisions.md](decisions.md) (rejected alternatives) → [evaluation.md](evaluation.md).

## Where to make common changes
| Want to… | Change |
|---|---|
| Use another LLM or embedding model | env vars in `rag/config.py`; re-ingest after changing the embedder |
| Tune refusals | `RAG_MIN_RERANK_SCORE`, then re-calibrate with `rag.eval` |
| Add a document type | a loader in `rag/loaders.py` + a branch in `load_manifest` |
| Change retrieval | constants at the top of `rag/retrieval.py` |
| Update a diagram | edit `docs/architecture/*.eraser`, run `python3 scripts/eraser_to_mermaid.py` |
