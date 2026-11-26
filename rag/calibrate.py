"""Calibrate the relevance gate from labelled questions instead of hand-picking it.

Every case in evals/cases.jsonl is retrieved (no answer is generated) and its best score recorded.
The threshold is chosen on the "tune" split and reported on the held-out "test" split. It is saved in
calibration.json with a fingerprint of what the scores depend on (team, embedding model, scorer,
corpus), and chat/eval warn when that fingerprint no longer matches.

Choice rule: keep at least --min-recall of answerable questions above the gate, then put the threshold
--position of the way from the highest trap score below them (0) to the lowest kept answerable score (1).
The default 0.25 leans towards letting questions through: a real question refused by the gate is lost,
while a trap that passes it still has to survive evidence selection and the fact check.

Usage: uv run python -m rag.calibrate [--cases evals/cases.jsonl] [--min-recall 1.0] [--position 0.25] [--dry-run]
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rag.config import Settings

CALIBRATION_FILE = Path(os.getenv("RAG_CALIBRATION", Path(__file__).resolve().parent.parent / "calibration.json"))
TEST_PERCENT = 30
ENV_VAR = {"min_rerank_score": "RAG_MIN_RERANK_SCORE", "min_score": "RAG_MIN_SCORE"}


def split_of(case: dict) -> str:
    """Stable tune/test assignment: an explicit "split" field, otherwise a hash of the question."""
    if case.get("split") in ("tune", "test"):
        return case["split"]
    bucket = int(hashlib.sha1(case["q"].encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < TEST_PERCENT else "tune"


POSITION = 0.25


def choose_threshold(answerable: list[float], traps: list[float], min_recall: float = 1.0,
                     position: float = POSITION) -> float:
    if not answerable:
        raise ValueError("need at least one answerable question to calibrate")
    ranked = sorted(answerable)
    floor = ranked[min(int((1 - min_recall) * len(ranked)), len(ranked) - 1)]
    ceiling = max((t for t in traps if t < floor), default=0.0)
    return round(ceiling + (floor - ceiling) * position, 3)


def gate_stats(rows: list[dict], threshold: float) -> dict:
    answerable = [r for r in rows if not r["trap"]]
    traps = [r for r in rows if r["trap"]]
    return {
        "answerable_refused": f"{sum(r['score'] < threshold for r in answerable)}/{len(answerable)}",
        "traps_gated": f"{sum(r['score'] < threshold for r in traps)}/{len(traps)}",
    }


def setting_name(cfg: Settings, reranked: bool) -> str:
    return "min_rerank_score" if reranked else "min_score"


def fingerprint(store, cfg: Settings, reranked: bool) -> dict:
    return {"team": cfg.team, "embed_model": cfg.embed_model,
            "scorer": cfg.rerank_model if reranked else "cosine",
            "corpus": store.corpus_fingerprint(cfg.team, cfg.embed_model)}


def entry_key(fp: dict) -> str:
    return f"{fp['team']}/{fp['scorer']}"


def load(path: Path = CALIBRATION_FILE) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def apply(cfg: Settings, store, reranked: bool, *, path: Path = CALIBRATION_FILE,
          env: dict | None = None) -> tuple[Settings, str]:
    """Return settings with the calibrated threshold and a one-line note on where it came from."""
    env = os.environ if env is None else env
    name = setting_name(cfg, reranked)
    if ENV_VAR[name] in env:
        return cfg, f"gate {getattr(cfg, name)} (set by {ENV_VAR[name]})"
    fp = fingerprint(store, cfg, reranked)
    entry = load(path).get(entry_key(fp))
    if not entry:
        return cfg, f"gate {getattr(cfg, name)} (default, not calibrated: run python -m rag.calibrate)"
    cfg = dataclasses.replace(cfg, **{name: entry["threshold"]})
    changed = [k for k in fp if entry["fingerprint"].get(k) != fp[k]]
    if changed:
        return cfg, (f"gate {entry['threshold']} (STALE: {', '.join(changed)} changed since calibration; "
                     f"run python -m rag.calibrate)")
    return cfg, f"gate {entry['threshold']} (calibrated {entry['calibrated_at'][:10]})"


def save(fp: dict, record: dict, path: Path = CALIBRATION_FILE):
    data = load(path)
    data[entry_key(fp)] = {**record, "fingerprint": fp}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def gate_score(store, llm, cfg: Settings, case: dict, reranker) -> float:
    from rag.qa import rewrite_question
    from rag.retrieval import retrieve

    history = [{"role": "user", "content": q} for q in case.get("history", [])]
    query = rewrite_question(llm, history, case["q"])
    hits = retrieve(store, llm, cfg, query, 8, None, reranker=reranker)
    return max((h.score for h in hits), default=0.0)


def main():
    from rag.ollama import Ollama
    from rag.rerank import Reranker
    from rag.store import VectorStore

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evals/cases.jsonl")
    ap.add_argument("--position", type=float, default=POSITION,
                    help="0 = just above the highest trap, 1 = at the lowest answerable score")
    ap.add_argument("--min-recall", type=float, default=1.0,
                    help="share of answerable tune questions that must pass the gate")
    ap.add_argument("--dry-run", action="store_true", help="report without saving")
    args = ap.parse_args()

    cfg = Settings()
    store = VectorStore(cfg.database_url, cfg.embed_dim)
    llm = Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    reranker = Reranker(cfg.rerank_model) if cfg.use_reranker else None
    reranked = reranker is not None
    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]

    rows = []
    for case in cases:
        score = gate_score(store, llm, cfg, case, reranker)
        rows.append({"q": case["q"], "trap": bool(case.get("trap")), "split": split_of(case), "score": score})
        print(f"{score:5.2f}  {rows[-1]['split']:4}  {'trap ' if rows[-1]['trap'] else 'answ '} {case['q'][:70]}")

    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    threshold = choose_threshold([r["score"] for r in tune if not r["trap"]],
                                 [r["score"] for r in tune if r["trap"]], args.min_recall, args.position)
    fp = fingerprint(store, cfg, reranked)
    record = {
        "threshold": threshold, "setting": setting_name(cfg, reranked), "min_recall": args.min_recall, "position": args.position,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": {"tune": len(tune), "test": len(test)},
        "tune": gate_stats(tune, threshold), "test": gate_stats(test, threshold),
    }

    print(f"\nScorer {fp['scorer']}, corpus {fp['corpus']}")
    answerable_tune = [r["score"] for r in tune if not r["trap"]]
    print(f"  tune: answerable {min(answerable_tune):.2f}–{max(answerable_tune):.2f}, "
          f"traps {', '.join(f'{r['score']:.2f}' for r in sorted(tune, key=lambda r: r['score']) if r['trap'])}")
    print(f"  threshold ({record['setting']}): {threshold}")
    for split in ("tune", "test"):
        s = record[split]
        print(f"  {split:4}: answerable wrongly refused {s['answerable_refused']:>6}   traps stopped by gate {s['traps_gated']:>6}")
    if args.dry_run:
        print("  (dry run: not saved)")
    else:
        save(fp, record)
        print(f"  saved to {CALIBRATION_FILE.name}")
    test_refused = int(record["test"]["answerable_refused"].split("/")[0])
    sys.exit(1 if test_refused else 0)


if __name__ == "__main__":
    main()
