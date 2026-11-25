from rag.grounding import REFUSAL, GroundingReport, check_citations, split_claims, verify_claims
from rag.store import SearchHit


def test_split_claims_attaches_trailing_citations_and_drops_fragments():
    answer = "Voyager 1 launched in 1977 from Cape Canaveral. [1][2]\n\nKey facts:\n- It visited Jupiter and Saturn [3]."
    assert split_claims(answer) == [
        "Voyager 1 launched in 1977 from Cape Canaveral. [1][2]",
        "It visited Jupiter and Saturn [3].",
    ]


def test_check_citations_flags_missing_and_invalid():
    answer = ("Hubble launched in 1990 aboard STS-31 [1]. It has a 2.4 metre primary mirror. "
              "It will operate until about 2040 [7].")
    report = check_citations(answer, num_sources=6)
    assert [c.issue for c in report.claims] == [None, "no citation", "cites non-existent source [7]"]
    assert len(report.problems) == 2


def test_refusal_has_no_claims():
    assert check_citations(REFUSAL, 6).claims == []


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.prompts = []

    def chat_json(self, messages, schema):
        self.prompts.append(messages[-1]["content"])
        return next(self.verdicts)


def test_verify_claims_checks_only_cited_claims_against_their_sources():
    hits = [SearchHit(f"source text {i}", 0.8, "text", f"T{i}", "s", "p", None) for i in range(1, 4)]
    report = check_citations("Entered interstellar space in 2013 per NASA [2]. An uncited statement about Saturn.", 3)
    judge = FakeJudge([{"supported": False, "reason": "source says 2012"}])
    verify_claims(judge, report, hits)
    assert len(judge.prompts) == 1
    assert "source text 2" in judge.prompts[0] and "source text 1" not in judge.prompts[0]
    assert "[2]" not in judge.prompts[0].split("CLAIM:")[1]
    assert report.claims[0].supported is False and report.claims[0].reason == "source says 2012"
    assert report.claims[1].issue == "no citation"
    assert isinstance(report, GroundingReport) and len(report.problems) == 2
