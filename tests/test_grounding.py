from rag.grounding import (REFUSAL, Claim, check_citations, number_sentences, quotes_from_ids, remove_claims,
                           split_claims, verify_claims)
from rag.store import SearchHit

VOYAGER = ("In March 2013, it was announced that Voyager 1 might have become the first spacecraft to enter "
           "interstellar space, having detected a marked change in the plasma environment on August 25, 2012. "
           "However, until September 12, 2013, it was still an open question.")


def test_number_sentences_ids_each_source_sentence():
    hits = [SearchHit("Hubble launched in 1990. It has a 2.4 m mirror.", 0.8, "text", "Hubble", "s", "p", None),
            SearchHit(VOYAGER, 0.8, "text", "Voyager 1", "s", "p", None)]
    numbered = number_sentences(hits)
    assert numbered["1.1"] == (1, "Hubble launched in 1990.")
    assert numbered["1.2"] == (1, "It has a 2.4 m mirror.")
    assert numbered["2.2"][1].startswith("However, until September 12, 2013")


def test_quotes_from_ids_merges_consecutive_and_flags_invented_ids():
    numbered = {"1.1": (1, "A."), "1.2": (1, "B."), "1.4": (1, "D."), "2.1": (2, "X.")}
    quotes = quotes_from_ids(["1.1", "1.2", "2.1", "1.4", "9.9", "1.1"], numbered, context=0)
    assert [(q.source, q.text, q.verified) for q in quotes] == [
        (1, "A. B.", True), (2, "X.", True), (1, "D.", True), (0, "9.9", False)]


def test_split_claims_attaches_trailing_citations_and_drops_fragments():
    answer = "Voyager 1 launched in 1977 from Cape Canaveral. [1][2]\n\nKey facts:\n- It visited Jupiter and Saturn [3]."
    assert split_claims(answer) == [
        "Voyager 1 launched in 1977 from Cape Canaveral. [1][2]",
        "It visited Jupiter and Saturn [3].",
    ]


def test_check_citations_flags_missing_and_unquoted_sources():
    answer = ("Hubble launched in 1990 aboard STS-31 [1]. It has a 2.4 metre primary mirror. "
              "It will operate until about 2040 [4].")
    report = check_citations(answer, valid_sources={1, 2})
    assert [c.issue for c in report.claims] == [None, "no citation", "cites source(s) [4] with no verified quote"]
    assert len(report.problems) == 2


def test_refusal_has_no_claims():
    assert check_citations(REFUSAL, {1}).claims == []


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.prompts = []

    def chat_json(self, messages, schema):
        self.prompts.append(messages[-1]["content"])
        return next(self.verdicts)


def test_verify_claims_judges_all_cited_claims_in_one_call():
    report = check_citations("Entered interstellar space in 2013 per NASA [2]. An uncited statement about Saturn. "
                             "Voyager 1 launched in the year 1977 [1].", {1, 2, 3})
    judge = FakeJudge([{"verdicts": [{"claim": 1, "supported": False, "reason": "evidence says 2012"},
                                     {"claim": 2, "supported": True, "reason": "stated"}]}])
    verify_claims(judge, report, {1: "evidence one", 2: "evidence two", 3: "unused evidence"})
    prompt = judge.prompts[0]
    assert len(judge.prompts) == 1
    assert "evidence one" in prompt and "evidence two" in prompt and "unused evidence" not in prompt
    assert "Saturn" not in prompt
    assert [c.supported for c in report.claims] == [False, None, True]
    assert report.claims[0].reason == "evidence says 2012" and report.claims[1].issue == "no citation"


def test_claim_without_verdict_is_unsupported():
    report = check_citations("Voyager 1 launched in the year 1977 [1].", {1})
    verify_claims(FakeJudge([{"verdicts": []}]), report, {1: "e"})
    assert report.claims[0].supported is False


def test_remove_claims_cleans_up_bullets_and_blank_lines():
    answer = "Intro sentence about Juno here [1].\n\n- Bad claim that is unsupported [2].\n- Good claim that is fine [1]."
    out = remove_claims(answer, [Claim("Bad claim that is unsupported [2].", [2])])
    assert out == "Intro sentence about Juno here [1].\n\n- Good claim that is fine [1]."


def test_image_source_is_a_single_evidence_unit():
    hits = [SearchHit("Image: ISS\n\nDate: 2001-01-01\n\nThe station orbits Earth. It is large.", 0.8, "image", "ISS", "s", "p", None)]
    assert list(number_sentences(hits).items()) == [
        ("1.1", (1, "Image: ISS Date: 2001-01-01 The station orbits Earth. It is large."))]


def test_quotes_from_ids_adds_neighbouring_sentences():
    numbered = {f"1.{i}": (1, f"S{i}.") for i in range(1, 6)}
    quotes = quotes_from_ids(["1.1", "1.5"], numbered)
    assert [q.text for q in quotes] == ["S1. S2.", "S4. S5."]


def test_exact_excerpt_detection_ignores_case_spacing_and_citations():
    from rag.grounding import copies_evidence
    assert copies_evidence("voyager 1  might have become the first spacecraft [2].", VOYAGER)
    assert not copies_evidence("Voyager 1 entered interstellar space on August 25, 2013 [2].", VOYAGER)
    assert not copies_evidence("Voyager 1 [2].", VOYAGER)


def test_add_citations_inserts_before_final_punctuation():
    from rag.grounding import add_citations
    claim = Claim("Juno is powered by solar arrays.", [3], auto_cited=True)
    assert add_citations("Intro [1]. Juno is powered by solar arrays.", [claim]) == "Intro [1]. Juno is powered by solar arrays [3]."
    assert claim.text == "Juno is powered by solar arrays [3]."
