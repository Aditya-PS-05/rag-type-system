# Controlling hallucination

The goal: every claim the user sees either traces to source text that code verified, or is removed.
This cannot be *proven* with an LLM in the loop. The design makes failures rare, visible and measurable.

## Where hallucinations come from

| Cause | Example | What stops it |
|---|---|---|
| Model answers from its own knowledge | "Capital of France?" | **Gate**: weak retrieval → refuse before the LLM is called |
| On-topic question the documents don't answer | "Europa Clipper's cost in euros?" | **Evidence selection**: no relevant sentence → refuse |
| Wrong chunks retrieved; model fills the gap | Kepler question gets Hubble chunks | **Hybrid search + reranker**; then evidence selection finds nothing → refuse rather than guess |
| Model invents a quote | | **Sentence IDs**: code supplies the text; invented IDs are discarded |
| Model misreads or merges what it read | "entered in 2013" (source: entered 2012, confirmed 2013) | **Answer from quotes only** + **fact check** → sentence removed |
| Uncited filler | "It is the most distant object…" | **Citation check** → judged against the closest quote; kept and cited if supported, removed otherwise |
| Follow-up loses its subject or reuses old answers | "When did *it* launch?" | **Rewriting**; history is never a source |

## Guarantees vs best effort
- **Hard (code):** an off-topic question never reaches the LLM; every quote the answer is built from exists word-for-word in a retrieved chunk; every remaining sentence cites a source that has a verified quote.
- **Best effort (LLM):** that the answer says what the quotes say. The fact checker is the same 4B model; it catches number and date errors and sometimes rejects true paraphrases.

## Tuning the trade-off
- **Too many refusals:** lower `RAG_MIN_RERANK_SCORE`, raise `MAX_EVIDENCE_SENTENCES` (`rag/qa.py`), relax `VERIFY_PROMPT` (`rag/grounding.py`).
- **Too many unsupported claims:** keep strict mode on, raise the gate, use a larger `CHAT_MODEL` (e.g. `qwen3:8b`) at least for the fact check.
- Always measure with `rag.eval` before and after. A change that "feels" better can break the traps.

## Signals shown to the user
- `(no source was relevant enough: best match X < threshold Y; the model was not called)`
- `(N evidence ID(s) discarded: not present in the sources)`
- `Removed N unsupported sentence(s)` with the checker's reason
- `(searched for: ...)` when a follow-up was rewritten
- `--show-evidence` prints the exact quotes behind the answer
