# Local multimodal RAG

Ask questions about a collection of **text, PDFs and images** and get answers **only from those documents**, with citations.
Runs fully offline on a laptop: Postgres + pgvector for storage, Ollama for the models, a local reranker.

- **Grounded:** the model can only answer from sentences that code has verified exist in the sources; unsupported sentences are removed, and off-topic questions are refused without calling the model.
- **Multimodal:** PDFs are parsed per page, images are searchable through their titles, descriptions and keywords.
- **Team-ready:** one Postgres database with per-team filtering; point `DATABASE_URL`/`OLLAMA_URL` at shared servers.

```
you> When did Voyager 1 enter interstellar space?
bot> Voyager 1 entered interstellar space on August 25, 2012 [2][4][5][7].
Sources:
  [2] text  Voyager program  https://en.wikipedia.org/wiki/Voyager_program
  [4] text  Voyager 1  https://en.wikipedia.org/wiki/Voyager_1
  ...
Fact check: all 1 claim(s) supported by the quoted sources

you> When was it launched?
bot> Voyager 1 was launched on September 5, 1977 [2].
(searched for: When was Voyager 1 launched?)

you> What is the capital of France?
bot> I don't know based on the available documents.
(no source was relevant enough: best match 0.06 < threshold 0.50; the model was not called)
```

## Quickstart

Requirements: macOS/Linux, Docker, [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/), ~6 GB disk.

```bash
brew install ollama && brew services start ollama
ollama pull nomic-embed-text           # embeddings (274 MB)
ollama pull qwen3:4b-instruct          # answers (2.5 GB)

docker compose up -d                   # Postgres 17 + pgvector on 127.0.0.1:5432
uv sync

python3 scripts/download_dataset.py data/raw   # 35 Wikipedia articles, 20 arXiv PDFs, 100 NASA images (~100 MB)
uv run python -m rag.ingest data/raw           # chunk + embed + store (~1-2 min)
uv run python -m rag.web                       # web app on http://127.0.0.1:8000 (or rag.chat for the terminal)
                                               # first run downloads the reranker (150 MB) into models/
```

After the first run nothing needs the internet.

## Usage

| Command | What it does |
|---|---|
| `uv run python -m rag.web` | **Web app** at http://127.0.0.1:8000: chat, live pipeline steps, clickable citations, evidence and fact-check inspector, image thumbnails, document library |
| `uv run python -m rag.chat` | Terminal chat (strict mode: every claim fact-checked) |
| `uv run python -m rag.chat -q "..."` | One question, then exit |
| `uv run python -m rag.chat --fast` | Stream answers, skip claim fact-checking (warnings only) |
| `uv run python -m rag.chat --show-evidence` | Also print the exact quotes each answer was built from |
| `uv run python -m rag.chat --type pdf,image` | Search only some document types |
| `uv run python -m rag.ingest data/raw [--force]` | Index new/changed documents (unchanged ones are skipped) |
| `uv run python -m rag.eval [--fast] [--split test]` | Run the grounding evaluation (`evals/cases.jsonl`); `--split test` = held-out questions only |
| `uv run python -m rag.calibrate` | Pick the relevance-gate threshold from the eval questions and save it to `calibration.json` |
| `uv run pytest` | Unit + integration tests (integration tests need the database) |

Chat commands: `/sources` (list indexed documents), `/strict`, `/evidence`, `/help`, `exit`.

## How an answer is produced

1. **Rewrite** a follow-up ("when did it launch?") into a standalone question using the chat history. Skipped when the question doesn't refer back to anything.
2. **Retrieve**: vector search + keyword search → merged → reranked by a cross-encoder → top 8, plus neighbouring text.
3. **Gate**: if nothing is relevant enough, refuse. The LLM is never called.
4. **Select evidence**: the top 6 sources are split into numbered sentences; the LLM returns up to 10 sentence IDs; code looks up the exact text (plus one sentence either side).
5. **Answer** from those quotes only, citing them, in at most 4 sentences (8 for lists or details).
6. **Check**: a sentence copied word for word from its quote is accepted; one LLM pass judges every other sentence against the quotes it cites. An uncited sentence is judged against the quote it most resembles.
7. **Remove** unsupported sentences and add the citation to uncited ones that passed. If nothing supported is left, refuse.

Details: [docs/how-it-works.md](docs/how-it-works.md) · Why each step exists: [docs/hallucination.md](docs/hallucination.md)

## Results

Evaluation set: 29 factual questions (incl. 4 follow-ups and a comparison), 2 image requests, 13 trap questions (8 general-knowledge, 5 on-topic but not in the documents).

| Version | Questions | Factual correct | Traps refused | Notes |
|---|---|---|---|---|
| V1 basic RAG | n/a | n/a | n/a | not evaluated; misread a date in manual testing |
| V2 guardrails | 34 | 24/24 | 8/8 (all by the gate) | |
| V3 evidence-first | 44 | 26/29 | 13/13 | 0/74 invented evidence IDs; ~11 s per answer |
| V4 hybrid retrieval | 44 | 27/29 | 13/13 (9 by the gate) | 0/99 invented IDs; ~11 s per answer |
| V4.1 calibrated gate | 44 | 27/29 | 13/13 (10 by the gate) | gate 0.501 chosen on 37 questions; 0/4 answerable refused on 7 held out |
| V4.2 faster, auto-cite | 44 | **29/29** | **13/13** (10 by the gate) | 0/77 invented IDs; 8–16 s per new question |

Version history: [docs/versions.md](docs/versions.md) · Full results: [docs/evaluation.md](docs/evaluation.md)

## Project layout

```
scripts/download_dataset.py   dataset downloader (Wikipedia, arXiv, NASA)
scripts/eraser_to_mermaid.py  diagram converter (.eraser → .mmd, README, viewer)
rag/config.py                 settings from environment variables
rag/loaders.py                text / PDF / image loaders and cleaning
rag/chunking.py               paragraph-packing chunker with overlap
rag/ollama.py                 embeddings, streaming chat, JSON-constrained chat
rag/store.py                  Postgres + pgvector: schema, vector + keyword search (only DB module)
rag/rerank.py                 local ONNX cross-encoder reranker
rag/retrieval.py              hybrid retrieval, fusion, reranking, neighbouring chunks
rag/grounding.py              sentence IDs, citation check, claim fact-check, removal
rag/qa.py                     the answer pipeline (steps 1-7 above)
rag/calibrate.py              gate calibration: tune/test split, threshold choice, staleness check
rag/web.py, rag/static/       local web app (standard library server + one offline HTML page)
rag/ingest.py, chat.py, eval.py   command-line entry points
calibration.json              calibrated gate thresholds + what they depend on
evals/cases.jsonl             evaluation questions
docs/                         design documentation and architecture diagrams
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Postgres with pgvector |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `EMBED_MODEL` / `EMBED_DIM` | `nomic-embed-text` / `768` | Embedding model and vector size |
| `CHAT_MODEL` | `qwen3:4b-instruct` | Model for rewriting, evidence selection, answers, fact checks |
| `RAG_RERANK` / `RERANK_MODEL` | `1` / `jinaai/jina-reranker-v1-turbo-en` | Reranker on/off and model |
| `RAG_MIN_RERANK_SCORE` | from `calibration.json` (0.501), else `0.3` | Gate threshold on reranker score; setting it overrides calibration |
| `RAG_MIN_SCORE` | from `calibration.json`, else `0.68` | Gate threshold on cosine similarity (used when `RAG_RERANK=0`) |
| `RAG_CALIBRATION` | `calibration.json` | Where calibrated thresholds are stored |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the models loaded after a request |
| `RAG_TEAM` | `default` | Team whose documents are indexed and searched |

## Known limitations

**Thresholds and tuning**
- **The gate threshold is only valid for this setup.** It is calibrated (`rag.calibrate`) for one corpus, embedder and reranker, and saved with a fingerprint of them; chat and eval warn when any changes. A new corpus needs labelled questions and a re-run.
- **The evaluation set is small.** 44 questions, 37 to tune and 7 held out. The held-out score is a sanity check, not a reliable estimate.
- **The gate cannot catch on-topic traps.** Questions on-topic but not answered by the documents score 0.15–0.87, overlapping real questions (0.66–0.99). Only evidence selection by the 4B model stops them, which is likely but not guaranteed.
- **The gate looks at one number:** the best chunk's score, not the gap to the others or agreement between keyword and vector search.
- **Other parameters were set by hand,** some to fix a single question: top-8 chunks, ±1 sentence context, 1,200-char chunks, 10 evidence sentences, rerank pool 24. Only the first 40 sentences of each source can be cited.

**Answer quality**
- **The fact checker is the same 4B model that wrote the answer,** so their mistakes are correlated. It reduces unsupported claims but does not prove their absence, and it sometimes removes correct ones (e.g. a comparison citing only one side).
- **Comparisons and multi-step questions need every fact cited,** and the 4B model's evidence choice varies between similar questions.

**System**
- **Speed: 8–16 s per new question on an M5 laptop;** the web app shows a draft after ~5 s. The floor is reading the ~2,000-token evidence prompt: this machine processes prompts at ~600 tokens/s and writes at ~40 tokens/s. Repeated questions are faster because Ollama caches prompts. See [docs/performance.md](docs/performance.md).
- Images are understood only through their text metadata, not pixels; PDF tables and figures become rough text.
- Keyword search is English-only.
- Teams are separated by a `team` column with no authentication; the default database password is for local use only. The shared deployment (V5) is not built.

Planned fixes, in order: a larger labelled set (~150 questions) → a gate that combines several signals → an independent checker model. See [docs/evaluation.md](docs/evaluation.md).

## Documentation

| Doc | Contents |
|---|---|
| [reading-guide.md](docs/reading-guide.md) | Where to start reading the code, in what order |
| [versions.md](docs/versions.md) | V1 → V5: what changed, why, what it fixed, what it cost |
| [how-it-works.md](docs/how-it-works.md) | Ingestion, storage schema, retrieval, answering, in detail |
| [hallucination.md](docs/hallucination.md) | Where hallucinations come from and which step stops each |
| [evaluation.md](docs/evaluation.md) | Test set, metrics, calibrating thresholds, results |
| [performance.md](docs/performance.md) | Where the time goes per step, and what was done about it |
| [decisions.md](docs/decisions.md) | Key technical choices and the alternatives rejected |
| [team-deployment.md](docs/team-deployment.md) | Sharing across teams, sizing, scaling |
| [architecture/](docs/architecture/) | Diagrams per version: Eraser sources, Mermaid (renders on GitHub), interactive viewer |

## Data licenses

Wikipedia text: CC BY-SA 4.0 · arXiv papers: per-paper licenses (see source links) · NASA images: public domain.
