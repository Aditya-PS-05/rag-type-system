"""Grounding evaluation: answerable questions, image questions, and trap questions the corpus can't answer.

Usage: uv run python -m rag.eval [--cases evals/cases.jsonl] [--verify] [--min-pass 0.9]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rag.config import Settings
from rag.ollama import Ollama
from rag.qa import ask
from rag.store import VectorStore


def grade(case: dict, result) -> tuple[bool, str]:
    if case.get("trap"):
        return result.refused, "refused" if result.refused else "answered a question the documents can't answer"
    if result.refused:
        return False, "refused an answerable question"
    if case.get("expect_image"):
        ok = any(h.doc_type == "image" for h in result.hits)
        return ok, "image retrieved" if ok else "no image retrieved"
    text = result.text.lower()
    alternatives = [[e] if isinstance(e, str) else e for e in case["expect"]]
    missing = [alts for alts in alternatives if not any(a.lower() in text for a in alts)]
    return not missing, "correct" if not missing else f"missing {missing}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evals/cases.jsonl")
    ap.add_argument("--verify", action="store_true", help="also fact-check every cited claim")
    ap.add_argument("--min-pass", type=float, default=0.9)
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]

    rows, started = [], time.time()
    for case in cases:
        result = ask(store, llm, cfg, case["q"], verify=args.verify)
        passed, note = grade(case, result)
        top = max((h.score for h in result.hits), default=0)
        report = result.grounding
        checked = [c for c in report.claims if c.supported is not None]
        rows.append({"case": case, "passed": passed, "note": note, "top": top, "gated": result.gated,
                     "uncited": sum(1 for c in report.claims if c.issue), "claims": len(report.claims),
                     "checked": len(checked), "unsupported": [c for c in checked if not c.supported]})
        kind = "trap" if case.get("trap") else "image" if case.get("expect_image") else "fact"
        gate = " gated" if result.gated else ""
        print(f"{'PASS' if passed else 'FAIL'}  {kind:5} top={top:.2f}{gate:6}  {case['q'][:62]:62}  {note}")
        for c in rows[-1]["unsupported"]:
            print(f"        ⚠ unsupported: \"{c.text[:110]}\" — {c.reason}")

    def rate(kind):
        sel = [r for r in rows if kind(r["case"])]
        return sum(r["passed"] for r in sel), len(sel)

    facts = rate(lambda c: not c.get("trap") and not c.get("expect_image"))
    images = rate(lambda c: c.get("expect_image"))
    traps = rate(lambda c: c.get("trap"))
    claims = sum(r["claims"] for r in rows)
    uncited = sum(r["uncited"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    answerable_tops = [r["top"] for r in rows if not r["case"].get("trap")]
    trap_tops = [r["top"] for r in rows if r["case"].get("trap")]

    print(f"\nModel {cfg.chat_model}, min_score {cfg.min_score}, {time.time() - started:.0f}s")
    print(f"  factual answers correct : {facts[0]}/{facts[1]}")
    print(f"  image questions         : {images[0]}/{images[1]}")
    print(f"  traps refused           : {traps[0]}/{traps[1]}  "
          f"({sum(r['gated'] for r in rows if r['case'].get('trap'))} by relevance gate, rest by the model)")
    print(f"  claims with citations   : {claims - uncited}/{claims}")
    if args.verify:
        checked = sum(r["checked"] for r in rows)
        unsupported = sum(len(r["unsupported"]) for r in rows)
        print(f"  claims supported        : {checked - unsupported}/{checked}")
    print(f"  top score  answerable min {min(answerable_tops):.2f} | traps max {max(trap_tops):.2f}")
    print(f"  overall                 : {passed}/{len(rows)}")
    sys.exit(0 if passed / len(rows) >= args.min_pass else 1)


if __name__ == "__main__":
    main()
