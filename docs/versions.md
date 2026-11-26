# Versions

Each version fixed the biggest weakness of the previous one. Diagrams: [architecture/](architecture/) (paste a `.eraser` file into eraser.io).

| | V1 Basic | V2 Guardrails | V3 Evidence-first | V4 Hybrid retrieval | V5 Team (proposed) |
|---|---|---|---|---|---|
| Retrieval | vector top-6 | vector top-6 | vector top-6 | vector + keyword → rerank → top-8 | same, shared services |
| Off-topic questions | model decides | refused before LLM | refused before LLM | refused before LLM (reranker score) | same |
| What the LLM reads | whole chunks | whole chunks | verified quotes only | verified quotes only | same |
| Checking | none | citations; optional judge (warn) | citations + judge, **removes** bad claims | same | same |
| Follow-ups | not handled | not handled | rewritten to standalone | same | same |
| Speed per answer | 3–8 s | 3–8 s | ~11 s | ~11 s | depends on servers |

## V1: Basic local RAG
**Built:** downloader → loaders → 1,200-char chunks → `nomic-embed-text` → Postgres/pgvector. Question → embed → top-6 cosine → `qwen3:4b-instruct` answers with `[n]` citations. Picture questions also pull the top 3 images.
**Problem found:** the prompt was the only control. In testing the model said Voyager 1 "entered interstellar space in 2013" when its source said August 2012 (2013 was the announcement). Off-topic questions reached the model.

## V2: Grounding guardrails
**Added:** relevance gate (best cosine < 0.68 → refuse without calling the LLM), stricter prompt at temperature 0, citation check on every answer, optional LLM fact check (`--verify`), evaluation set with trap questions, `/sources`.
**Fixed:** all 8 general-knowledge traps now stopped by the gate (they scored ≤ 0.63; real questions ≥ 0.76).
**Problem left:** the fact check only *warned*; the model still read whole chunks and could misread them; follow-ups ("when did *it* launch?") searched badly; on-topic questions with no answer in the documents scored above the gate.

## V3: Evidence-first answering
**Added:**
- *Follow-up rewriting* into a standalone question; history is used only for that, never as a source.
- *Evidence by sentence ID:* sources are split into numbered sentences (`2.5` = source 2, sentence 5). The LLM returns IDs; code looks up the text. Quotes are exact by construction.
- *Answer from quotes only:* the answering call never sees whole chunks.
- *Batch fact check + removal:* one LLM call judges every claim against the quotes it cites; unsupported sentences are deleted; nothing left → refusal.

**Fixed:** all 5 on-topic traps refused (e.g. Europa Clipper cost "in euros", scoring 0.85 but with no evidence); 0 of 74 evidence IDs invented; the checker caught a misquoted number (4.7 vs 4.8 mi). Score 41/44.
**Cost:** ~11 s per answer (4 LLM calls on a laptop at ~19 tokens/s).
**Problem left:** 2 of 3 failures were retrieval misses: the right chunk was never in the top 6 (Kepler launch date, Jezero crater).

**Rejected along the way:** having the LLM copy quotes word for word (150 output tokens, 6–11 s, needed fuzzy matching) → replaced by IDs (~15 tokens, exact). A per-claim checker → too strict and slow → one batched call with source titles and explicit "synonyms and simple comparisons are fine" rules.

## V4: Hybrid retrieval with reranking
**Added:**
- *Keyword search* (Postgres full-text, title + content, any term may match) next to vector search; good for exact terms like `PIXL`, `M87`, `GW150914`.
- *Reciprocal rank fusion* of both top-30 lists.
- *Cross-encoder reranker* (`jina-reranker-v1-turbo-en`, 150 MB, ONNX, offline) scores the best 24 against the question; its score drives the gate.
- *Neighbouring chunks:* up to 400 characters before and after each hit, so pronouns like "it" have their subject.
- *Top 8 instead of 6:* the Hubble mirror chunk ranked 7th.
- *Sentence context:* each selected evidence sentence also brings the sentence before and after. The model had picked "Apollo 11 was the first crewed landing…" but not the next sentence naming the commander.
- *Gate on reranker score at 0.3:* general-knowledge traps score 0.01–0.11, the weakest real question 0.66 (cosine separated them by only 0.13).

**Fixed:** Kepler and Perseverance chunks now rank first (scores 0.97–1.0); the Hubble mirror and Apollo commander questions answer correctly. Score 42/44, 9 of 13 traps stopped before the LLM, 0/99 invented evidence IDs.
**Cost:** +0.7 s per question. (`bge-reranker-v2-m3` was tried: same ranking, 6.3 s → rejected.)

## V4.1: Calibrated gate
**Added:** `rag.calibrate` picks the threshold from labelled questions on a tune split, reports it on a held-out test split, and saves it with a fingerprint of the corpus, embedder and reranker. Chat and eval load it and warn when it is stale.
**Fixed:** the hand-picked 0.3 → calibrated 0.501. No answerable question refused on either split; 10 of 13 traps now stopped before the LLM (was 9). The threshold no longer silently goes wrong when the documents or models change.
**Problem left:** the test split is only 7 questions, and on-topic traps still overlap real questions. Next: a larger labelled set, then a gate that combines several signals.

## V4.2: Faster, fewer false removals
**Added:** evidence reads the top 6 sources with repeated neighbour text removed (prompt ~3,550 → ~2,000 tokens); the evidence schema caps IDs at 10 (one question had produced 74 IDs, 15 s); answers capped at 4 sentences / 400 tokens; the rewrite call only runs for questions that refer back; exact excerpts skip the fact check; models stay loaded; the web app streams a draft in strict mode.
**Fixed:** uncited but correct sentences were deleted (e.g. "Chandrayaan-3 is the third mission in the Chandrayaan programme"); they are now judged against the closest quote and cited. New questions: 12–27 s → 8–16 s, first text after ~5 s. Evaluation 42/44 → **44/44**, claims removed 7 → 2.
**Problem left:** reading the evidence prompt (~600 tokens/s on this laptop) is the floor. See [performance.md](performance.md).

## V5: Shared team deployment (proposed, not built)
Document watcher → ingest workers → MinIO for raw files + replicated Postgres/pgvector with team and embedding-model tags. Team apps → one RAG API service → shared embedding server, reranker, LLM server. See [team-deployment.md](team-deployment.md).
