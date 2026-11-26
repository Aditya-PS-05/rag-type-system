# Performance

Measured on an Apple M5 laptop (16 GB) with `qwen3:4b-instruct` fully on the GPU in Ollama.

## The two speeds that matter
| | Speed | Consequence |
|---|---|---|
| Reading a prompt | ~600 tokens/s | 2,000 tokens of sources ≈ 3.5 s before the model writes anything |
| Writing output | ~40 tokens/s | a 100-token answer ≈ 2.5 s |

Ollama caches the start of recent prompts, so asking the same question again skips most of the reading. **Always measure with new questions.** Flash attention (`OLLAMA_FLASH_ATTENTION=1`) was tested: about +3% on reading, so not worth configuring.

## Where the time goes (new questions, strict mode)
| Step | Before (V4.1) | Now (V4.2) | What changed |
|---|---|---|---|
| Rewrite | ~1 s with any history | 0 s unless the question refers back | `needs_rewrite()` checks for "it", "they", "the rover"… |
| Retrieve + rerank | ~0.5 s | ~0.5 s | |
| Select evidence | 5–7 s; 15 s when the model listed 74 IDs | 3.5–5.5 s | top 6 sources instead of 8, repeated neighbour text removed, 250 instead of 400 neighbour characters (prompt ~3,550 → ~2,000 tokens); schema `maxItems: 10` |
| Answer | 1–7 s, long answers that restated every quote | 1–5 s | "at most 4 sentences", 400-token cap |
| Fact check | 1–3 s | 1–4 s | exact excerpts skip it, but uncited sentences are now checked instead of deleted |
| **Total** | **12–27 s** | **8–16 s** | |
| First text in the web app | at the end | ~5 s | strict mode shows the draft while checking |

## Checked before shipping
- Every expected answer in the eval set is still present in the top 6 sources (31/31; top 5 also 31/31).
- The full evaluation after the changes: 44/44 (was 42/44), 2 claims removed (was 7), 443 s for 44 questions (was 602 s). See [evaluation.md](evaluation.md).

## Further options, by expected gain
1. **Faster hardware or a smaller/faster model.** Prompt reading scales with GPU compute.
2. **Pre-filter sentences** before evidence selection (e.g. keep sentences sharing words with the question plus their neighbours). Fewer tokens, but this risks dropping sentences whose subject is "it".
3. **Run the fact check per sentence while the answer streams.** This overlaps the check with writing, but costs more total LLM time.
4. **A separate, smaller model for the rewrite and fact check.** The two models would compete for the same GPU on a laptop; better on a server.
