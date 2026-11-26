import json

from rag import calibrate
from rag.config import Settings


class FakeStore:
    def __init__(self, corpus="3 docs, 9 chunks, abc"):
        self.corpus = corpus

    def corpus_fingerprint(self, team, embed_model):
        return self.corpus


def test_threshold_sits_a_quarter_of_the_way_from_highest_trap_to_lowest_answerable():
    assert calibrate.choose_threshold([0.9, 0.7, 0.95], [0.05, 0.4, 0.85]) == 0.475
    assert calibrate.choose_threshold([0.9, 0.7, 0.95], [0.05, 0.4, 0.85], position=0.5) == 0.55


def test_threshold_without_traps_below_answerable_is_measured_from_zero():
    assert calibrate.choose_threshold([0.6, 0.9], [0.95]) == 0.15


def test_min_recall_lets_the_lowest_answerable_fall_below_the_gate():
    t = calibrate.choose_threshold([0.2, 0.8, 0.9, 0.95], [0.1, 0.6], min_recall=0.75)
    assert 0.6 < t < 0.8 and t == 0.65


def test_split_is_stable_and_respects_explicit_field():
    case = {"q": "When did Juno arrive at Jupiter?"}
    assert calibrate.split_of(case) == calibrate.split_of(dict(case))
    assert calibrate.split_of({**case, "split": "test"}) == "test"
    splits = {calibrate.split_of({"q": f"question {i}"}) for i in range(50)}
    assert splits == {"tune", "test"}


def test_gate_stats_counts_scores_below_threshold():
    rows = [{"trap": False, "score": 0.2}, {"trap": False, "score": 0.9}, {"trap": True, "score": 0.1}]
    assert calibrate.gate_stats(rows, 0.5) == {"answerable_refused": "1/2", "traps_gated": "1/1"}


def test_apply_uses_saved_threshold_and_flags_a_changed_corpus(tmp_path):
    path = tmp_path / "calibration.json"
    cfg = Settings()
    fp = calibrate.fingerprint(FakeStore(), cfg, reranked=True)
    calibrate.save(fp, {"threshold": 0.42, "calibrated_at": "2026-11-26T00:00:00+00:00"}, path)

    fresh, note = calibrate.apply(cfg, FakeStore(), True, path=path, env={})
    assert fresh.min_rerank_score == 0.42 and "calibrated 2026-11-26" in note

    stale, note = calibrate.apply(cfg, FakeStore("4 docs, 12 chunks, def"), True, path=path, env={})
    assert stale.min_rerank_score == 0.42 and "STALE" in note and "corpus" in note


def test_apply_without_calibration_or_with_env_override_keeps_settings(tmp_path):
    cfg = Settings()
    same, note = calibrate.apply(cfg, FakeStore(), True, path=tmp_path / "none.json", env={})
    assert same == cfg and "not calibrated" in note

    path = tmp_path / "calibration.json"
    calibrate.save(calibrate.fingerprint(FakeStore(), cfg, False), {"threshold": 0.9, "calibrated_at": "x"}, path)
    kept, note = calibrate.apply(cfg, FakeStore(), False, path=path, env={"RAG_MIN_SCORE": "0.5"})
    assert kept == cfg and "RAG_MIN_SCORE" in note
    assert json.loads(path.read_text())["default/cosine"]["threshold"] == 0.9
