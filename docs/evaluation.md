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
uv run python -m rag.calibrate              # score every case, choose, save to calibration.json
uv run python -m rag.calibrate --dry-run    # report only
uv run python -m rag.eval --split test      # full pipeline on the held-out questions
```
- **Only retrieval runs**, so calibration takes about 25 s, not 9 minutes.
- **Split:** each case is assigned to *tune* (~70%) or *test* (~30%) by a hash of its question, so the split is stable. Add `"split": "test"` to a case to pin it.
- **Rule:** on the tune split, keep every answerable question above the gate (`--min-recall 1.0`). Then place the threshold halfway between the lowest answerable score and the highest trap score below it. That gives the widest margin on both sides.
- **Saved with a fingerprint:** team, embedding model, scorer (reranker or cosine) and corpus (document hashes and chunk counts). Chat and eval print `STALE` when any of these differ. A threshold set in an environment variable always wins.

Current calibration (reranker, 155 docs / 3,537 chunks):

| | Scores |
|---|---|
| Answerable (tune) | 0.66–0.99 |
| General-knowledge traps | 0.01–0.11 |
| On-topic traps | 0.15–0.87 |
| **Threshold** | **0.501** (hand-set before: 0.3) |

| Split | Answerable wrongly refused | Traps stopped by the gate |
|---|---|---|
| Tune | 0/27 | 7/10 |
| Test | 0/4 | 3/3 |

On-topic traps above the gate (Gagarin's breakfast 0.69, NASA tweets 0.79, Europa Clipper cost in euros 0.87) still need evidence selection to refuse them. The test split has only 7 questions; growing the set is the next step.

## Results history
| Version | Set | Facts | Images | Traps | Other |
|---|---|---|---|---|---|
| V1 | none | n/a | n/a | n/a | manual testing only; misread the Voyager date |
| V2, gate off | 34 | 23/24 | 1/2 | 8/8 (model refused) | every claim cited; both misses were test errors |
| V2 | 34 | 24/24 | 2/2 | 8/8 (gate) | 1 claim flagged unsupported |
| V3 | 44 | 26/29 | 2/2 | 13/13 | 0/74 invented IDs, 10 claims removed, 535 s |
| V4, gate off | 44 | 25/29 | 2/2 | 13/13 (model refused) | 0/87 invented IDs; used for calibration |
| V4 | 44 | 27/29 | 2/2 | 13/13 (9 by gate) | 0/99 invented IDs, 7 claims removed, 536 s |
| V4.1 (gate 0.501) | 44 | 27/29 | 2/2 | 13/13 (10 by gate) | 0/99 invented IDs, 7 claims removed, 602 s; held-out split 7/7 |

Remaining V4/V4.1 failures (42/44, same two):
- *"Hubble or Kepler first?"*: correct answer removed. It cited only the Hubble quote, so the checker saw no Kepler evidence. Comparisons need every compared fact cited.
- *Perseverance follow-up*: answered with coordinates instead of the crater name (the non-follow-up version passed). The 4B model's evidence choice varies.
