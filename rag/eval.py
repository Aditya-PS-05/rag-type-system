"""Grounding evaluation over evals/cases.jsonl.

Case kinds:
  fact      {"q", "expect": [str | [alternatives]]}   answer must contain every expected item
  image     {"q", "expect_image": true}                an image must be retrieved and the answer not refused
  trap      {"q", "trap": true}                        must be refused (not answerable from the documents)
Any case may add "history": [earlier questions] to test follow-up handling.

Usage: uv run python -m rag.eval [--cases evals/cases.jsonl] [--fast] [--min-pass 0.9] [--only fact,trap]
                                [--split all|tune|test]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rag import calibrate
from rag.config import Settings
from rag.ollama import Ollama
from rag.qa import ask
from rag.rerank import Reranker
from rag.store import VectorStore


def kind_of(case: dict) -> str:
    return "trap" if case.get("trap") else "image" if case.get("expect_image") else "fact"


def grade(case: dict, result) -> tuple[bool, str]:
    if case.get("trap"):
        return result.refused, "refused" if result.refused else f"answered: {result.text[:90]!r}"
    if result.refused:
        return False, "refused an answerable question"
    if case.get("expect_image"):
        ok = any(h.doc_type == "image" for h in result.hits)
        return ok, "image retrieved" if ok else "no image retrieved"
    text = result.text.lower()
    alternatives = [[e] if isinstance(e, str) else e for e in case["expect"]]
    missing = [alts for alts in alternatives if not any(a.lower() in text for a in alts)]
    return not missing, "correct" if not missing else f"missing {missing}: {result.text[:90]!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evals/cases.jsonl")
    ap.add_argument("--fast", action="store_true", help="skip claim fact-checking")
    ap.add_argument("--min-pass", type=float, default=0.9)
    ap.add_argument("--only", help="comma-separated case kinds to run: fact,image,trap")
    ap.add_argument("--split", choices=["all", "tune", "test"], default="all",
                    help="run only the calibration tune set or the held-out test set")
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    reranker = Reranker(cfg.rerank_model) if cfg.use_reranker else None
    cfg, gate_note = calibrate.apply(cfg, store, reranker is not None)
    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    if args.only:
        cases = [c for c in cases if kind_of(c) in args.only.split(",")]
    if args.split != "all":
        cases = [c for c in cases if calibrate.split_of(c) == args.split]

    rows, started = [], time.time()
    for case in cases:
        history: list[dict] = []
        for earlier in case.get("history", []):
            prior = ask(store, llm, cfg, earlier, history=history, strict=not args.fast, reranker=reranker)
            history.extend([{"role": "user", "content": earlier}, {"role": "assistant", "content": prior.text}])
        result = ask(store, llm, cfg, case["q"], history=history, strict=not args.fast, reranker=reranker)
        passed, note = grade(case, result)
        top = max((h.score for h in result.hits), default=0)
        rows.append({"case": case, "passed": passed, "top": top, "gated": result.gated,
                     "rejected_quotes": len(result.rejected_quotes), "quotes": len(result.quotes),
                     "claims": len(result.grounding.claims), "removed": len(result.removed)})
        tags = "".join([" gated" if result.gated else "",
                        f" -{len(result.removed)}claim" if result.removed else "",
                        f" -{len(result.rejected_quotes)}id" if result.rejected_quotes else ""])
        q = (f"[follow-up] {case['q']}" if case.get("history") else case["q"])[:60]
        print(f"{'PASS' if passed else 'FAIL'}  {kind_of(case):5} top={top:.2f} {tags:16} {q:60}  {note}")
        if case.get("history") and result.search_query != case["q"]:
            print(f"        rewritten -> {result.search_query}")
        for c in result.removed:
            print(f"        ✗ removed: \"{c.text[:100]}\" — {c.issue or c.reason}")

    def rate(kind):
        sel = [r for r in rows if kind_of(r["case"]) == kind]
        return f"{sum(r['passed'] for r in sel)}/{len(sel)}"

    total = lambda key: sum(r[key] for r in rows)
    traps = [r for r in rows if kind_of(r["case"]) == "trap"]
    answerable_tops = [r["top"] for r in rows if kind_of(r["case"]) != "trap"]
    passed = total("passed")

    gate = f"reranker {'on' if reranker else 'off'}, {gate_note}"
    print(f"\nModel {cfg.chat_model}, {gate}, {'fast' if args.fast else 'strict'} mode, {args.split} cases, "
          f"{time.time() - started:.0f}s")
    print(f"  factual answers correct : {rate('fact')}")
    print(f"  image questions         : {rate('image')}")
    print(f"  traps refused           : {rate('trap')}  ({sum(r['gated'] for r in traps)} by relevance gate)")
    print(f"  invented evidence IDs   : {total('rejected_quotes')}/{total('quotes')}")
    print(f"  claims removed          : {total('removed')}/{total('claims')}")
    if answerable_tops and traps:
        print(f"  top score  answerable min {min(answerable_tops):.2f} | traps max {max(r['top'] for r in traps):.2f}")
    print(f"  overall                 : {passed}/{len(rows)}")
    sys.exit(0 if passed / max(len(rows), 1) >= args.min_pass else 1)


if __name__ == "__main__":
    main()
