# How it works

## 1. Dataset (`scripts/download_dataset.py`)
Downloads a themed corpus and writes `data/raw/manifest.jsonl`, one line per file with type, path, title, source URL and license (plus authors/abstract for papers, description/keywords/date for images). Everything downstream reads the manifest, so any dataset in the same format works. Uses `curl` because Wikipedia rate-limits Python's HTTP client.

## 2. Ingestion (`rag/ingest.py`)
**Load** (`rag/loaders.py`)
- *Text:* Wikipedia reference sections ("See also", "References"…) are cut; heading markup removed.
- *PDF:* text per page with `pypdf`; hyphenation and soft line-wraps joined; control characters stripped (Postgres rejects NUL). The arXiv abstract becomes page 0.
- *Image:* a text document: `Image: <title>`, date, keywords, NASA description.

**Chunk** (`rag/chunking.py`): paragraphs packed greedily up to 1,200 characters; long paragraphs split on sentences; each chunk starts with ~200 characters of the previous one so facts on a boundary aren't lost. PDF chunks keep their page number.

**Embed** (`rag/ollama.py`): `nomic-embed-text` (768 dimensions). Text is embedded as `search_document: <title>\n<chunk>`; questions as `search_query: <question>`. These prefixes are required by the model; without them retrieval quality drops.

**Store** (`rag/store.py`): unchanged documents (same content hash and embedding model) are skipped; changed ones are replaced in one transaction.

## 3. Storage schema
```
documents  id, team, doc_type, path, title, source, license, meta(jsonb),
           content_hash, embed_model, ingested_at          UNIQUE(team, path)
chunks     id, document_id → documents (cascade delete), chunk_index, page,
           content, embedding vector(768), tsv tsvector
indexes    HNSW on embedding (cosine) · GIN on tsv · (team, doc_type)
```
Every search filters on `team` and `embed_model`, so teams never see each other's documents and vectors from different models are never compared. `rag/store.py` is the only module that touches the database.

## 4. Retrieval (`rag/retrieval.py`)
1. **Vector search:** top 30 by cosine similarity (HNSW, with pgvector's iterative scan so filters don't starve results).
2. **Keyword search:** top 30 by `ts_rank_cd`; the question's terms are OR-ed so one missing word doesn't kill the match.
3. **Fusion:** reciprocal rank fusion (`Σ 1/(60 + rank)`). Rewards chunks either method ranks high; no score normalisation needed.
4. **Rerank:** a cross-encoder reads question and chunk *together* (much more accurate than comparing two separate vectors) and scores the best 24 from 0 to 1. Top 8 kept.
5. **Images:** if the question asks for pictures ("show me", "photos", "image of"), the top 3 images are always included.
6. **Neighbours:** text/PDF hits get up to 400 characters of the previous and next chunk (overlap removed).

With `RAG_RERANK=0` steps 2–4 are skipped (vector top 8, cosine score).

## 5. Answering (`rag/qa.py`)
| Step | LLM call | Output | Guard |
|---|---|---|---|
| Rewrite follow-up | JSON | standalone question | only when there is history |
| Gate | none | refuse / continue | best score < threshold → refuse |
| Select evidence | JSON | sentence IDs (max 10) | each ID also brings the sentence before and after; IDs that don't exist are discarded; none valid → refuse |
| Answer | streamed | text with `[n]` citations | sees only verified quotes |
| Citation check | none | issues per sentence | uncited / cites unquoted source |
| Fact check | JSON, one call | verdict per claim | strict mode only |
| Remove | none | cleaned answer | nothing supported left → refuse |

All calls use temperature 0. JSON calls use Ollama's schema-constrained output, so responses always parse.
Strict mode (default) waits for the checks before printing. `--fast` streams the answer and only reports citation problems.

## 6. Interfaces
- `rag.chat`: terminal chat; shows sources actually cited, removed sentences with reasons, the rewritten query, and the gate score when refusing.
- `rag.eval`: runs `evals/cases.jsonl` through the same `ask()` function the chat uses. See [evaluation.md](evaluation.md).
