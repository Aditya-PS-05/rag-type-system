# Evaluation

`uv run python -m rag.eval` runs every case in `evals/cases.jsonl` through the same pipeline as the chat.
Exit code is non-zero if fewer than 90% pass (`--min-pass`), so it can gate CI.

## Case kinds
```jsonl
{"q": "When did Juno arrive at Jupiter?", "expect": ["2016"]}                      # fact: answer must contain each item
{"q": "Which galaxy ...?", "expect": [["M87", "Messier 87"]]}                     # inner list = any of these
{"q": "Show me pictures of the ISS", "expect_image": true}                         # an image must be retrieved, not refused
{"q": "Who wrote Hamlet?", "trap": true}                                           # must be refused
{"q": "When did it arrive?", "history": ["Tell me about Juno"], "expect": ["2016"]} # follow-up
```
Traps come in two kinds: **general knowledge** the model knows but the documents don't contain, and **near misses**: on-topic but absent (Armstrong's dog, a cost in euros). Near misses are the real test: they pass the relevance gate.

## Metrics printed
- factual correct, image questions, traps refused (and how many by the gate)
- **invented evidence IDs:** sentence IDs the model made up (should be 0)
- **claims removed:** sentences deleted by the fact check
- **top score:** lowest score among answerable questions vs highest among traps, used to set the gate

## Calibrating the gate
```bash
RAG_MIN_RERANK_SCORE=0 uv run python -m rag.eval --min-pass 0   # gate off: see raw scores
```
Pick a threshold below the lowest answerable score and above most general-knowledge traps.
Current calibration (reranker): general-knowledge traps 0.01–0.11, near-miss traps 0.15–0.87, answerable 0.66–0.99 → threshold 0.3. Near-miss traps will overlap; they are handled by evidence selection, not the gate. Redo this when the documents, embedding model or reranker change.

## Results history
| Version | Set | Facts | Images | Traps | Other |
|---|---|---|---|---|---|
| V1 | none | n/a | n/a | n/a | manual testing only; misread the Voyager date |
| V2, gate off | 34 | 23/24 | 1/2 | 8/8 (model refused) | every claim cited; both misses were test errors |
| V2 | 34 | 24/24 | 2/2 | 8/8 (gate) | 1 claim flagged unsupported |
| V3 | 44 | 26/29 | 2/2 | 13/13 | 0/74 invented IDs, 10 claims removed, 535 s |
| V4, gate off | 44 | 25/29 | 2/2 | 13/13 (model refused) | 0/87 invented IDs; used for calibration |
| V4 | 44 | 27/29 | 2/2 | 13/13 (9 by gate) | 0/99 invented IDs, 7 claims removed, 536 s |

Remaining V4 failures (42/44):
- *"Hubble or Kepler first?"*: correct answer removed. It cited only the Hubble quote, so the checker saw no Kepler evidence. Comparisons need every compared fact cited.
- *Perseverance follow-up*: answered with coordinates instead of the crater name (the non-follow-up version passed). The 4B model's evidence choice varies.
