from rag.config import Settings
from rag.grounding import REFUSAL
from rag.qa import EVIDENCE_SCHEMA, REWRITE_SCHEMA, ask
from rag.retrieval import VISUAL_INTENT, retrieve
from rag.store import SearchHit

JUNO = "Juno arrived at Jupiter on July 5, 2016. It is powered by solar arrays."


def hit(doc_type, title, content=None, score=0.8):
    return SearchHit(content or f"content {title}", score, doc_type, title, "src", f"{title}.x", None)


class FakeStore:
    def __init__(self, hits=None):
        self.calls = []
        self.hits = hits

    def search(self, embedding, *, team, embed_model, k, doc_types=None):
        self.calls.append(doc_types)
        if self.hits is not None:
            return self.hits
        if doc_types == ["image"]:
            return [hit("image", f"img{i}") for i in range(k)]
        return [hit("text", f"t{i}") for i in range(k - 1)] + [hit("image", "img0")]


class FakeLLM:
    """Scripted LLM: JSON calls are answered by schema, chat_stream returns a fixed answer."""

    def __init__(self, ids=(), answer="", verdicts=(), rewrite=None):
        self.quotes, self.answer, self.verdicts, self.rewrite = list(ids), answer, list(verdicts), rewrite
        self.streamed = 0
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.0]

    def chat_json(self, messages, schema):
        if schema is REWRITE_SCHEMA:
            return {"question": self.rewrite}
        if schema is EVIDENCE_SCHEMA:
            return {"ids": self.quotes}
        return self.verdicts.pop(0)

    def chat_stream(self, messages, **options):
        self.streamed += 1
        yield self.answer


def test_visual_question_includes_images_first_without_duplicates():
    hits = retrieve(FakeStore(), FakeLLM(), Settings(), "Show me pictures of Saturn", 6, None)
    assert [h.title for h in hits] == ["img0", "img1", "img2", "t0", "t1", "t2"]


def test_plain_question_does_single_search():
    store = FakeStore()
    retrieve(store, FakeLLM(), Settings(), "Who flew Apollo 11?", 6, None)
    assert store.calls == [None]


def test_explicit_type_filter_skips_image_boost():
    store = FakeStore()
    retrieve(store, FakeLLM(), Settings(), "show me pdf results", 4, ["pdf"])
    assert store.calls == [["pdf"]]


def test_image_as_verb_is_not_visual_intent():
    assert not VISUAL_INTENT.search("Which black hole did the Event Horizon Telescope image first?")
    assert VISUAL_INTENT.search("show me an image of Saturn")
    assert VISUAL_INTENT.search("Any photos of the ISS?")


def test_relevance_gate_refuses_without_calling_llm():
    llm = FakeLLM()
    result = ask(FakeStore([hit("text", "Milky Way", score=0.5)]), llm, Settings(min_score=0.65), "best pizza?")
    assert result.gated and result.refused and llm.streamed == 0


def test_invented_evidence_ids_refuse_without_answering():
    llm = FakeLLM(ids=["1.7"])
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "When did Juno arrive?")
    assert result.text == REFUSAL and llm.streamed == 0
    assert len(result.rejected_quotes) == 1


def test_unsupported_claims_are_removed_in_strict_mode():
    llm = FakeLLM(ids=["1.1"],
                  answer="Juno arrived at Jupiter on July 5, 2016 [1]. It carried a nuclear reactor to Jupiter [1].",
                  verdicts=[{"verdicts": [{"claim": 1, "supported": False, "reason": "not in evidence"}]}])
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "When did Juno arrive?")
    assert result.text == "Juno arrived at Jupiter on July 5, 2016 [1]."
    assert [c.reason for c in result.removed] == ["not in evidence"]


def test_all_claims_unsupported_becomes_refusal():
    llm = FakeLLM(ids=["1.2"],
                  answer="Juno is powered by a nuclear reactor on board [1].",
                  verdicts=[{"verdicts": [{"claim": 1, "supported": False, "reason": "evidence says solar"}]}])
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "How is Juno powered?")
    assert result.text == REFUSAL


def test_fast_mode_streams_and_skips_claim_verification():
    llm = FakeLLM(ids=["1.2"], answer="Juno is powered by solar arrays [1].")
    tokens = []
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "How is Juno powered?",
                 strict=False, on_token=tokens.append)
    assert tokens == ["Juno is powered by solar arrays [1]."] and result.removed == []


def test_follow_up_is_rewritten_before_search():
    llm = FakeLLM(ids=["1.1"],
                  answer="Juno arrived at Jupiter on July 5, 2016 [1].",
                  verdicts=[{"verdicts": [{"claim": 1, "supported": True, "reason": ""}]}], rewrite="When did Juno arrive at Jupiter?")
    history = [{"role": "user", "content": "Tell me about Juno"}, {"role": "assistant", "content": "Juno is..."}]
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "When did it arrive?", history=history)
    assert llm.queries == ["When did Juno arrive at Jupiter?"]
    assert result.search_query == "When did Juno arrive at Jupiter?"


def test_stages_are_reported_in_order():
    llm = FakeLLM(ids=["1.1"], answer="Juno arrived at Jupiter on July 5, 2016 [1].",
                  verdicts=[{"verdicts": [{"claim": 1, "supported": True, "reason": ""}]}])
    stages = []
    ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "When did Juno arrive?", on_stage=stages.append)
    assert stages == ["retrieve", "evidence", "answer", "check"]

    gated = []
    ask(FakeStore([hit("text", "Milky Way", score=0.5)]), FakeLLM(), Settings(min_score=0.65), "pizza?",
        on_stage=gated.append)
    assert gated == ["retrieve"]


def test_exact_quotes_skip_the_checker_and_uncited_supported_sentences_get_cited():
    llm = FakeLLM(ids=["1.1"],
                  answer="Juno arrived at Jupiter on July 5, 2016 [1]. The Juno probe is powered by solar arrays.",
                  verdicts=[{"verdicts": [{"claim": 1, "supported": True, "reason": ""}]}])
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "Tell me about Juno")
    assert result.text == "Juno arrived at Jupiter on July 5, 2016 [1]. The Juno probe is powered by solar arrays [1]."
    exact, cited = result.grounding.claims
    assert exact.method == "exact quote" and cited.auto_cited and cited.method == "fact checker"
    assert result.removed == [] and result.cited_sources == [1]


def test_strict_mode_streams_a_draft_and_returns_the_checked_text():
    llm = FakeLLM(ids=["1.1"], answer="Juno arrived at Jupiter on July 5, 2016 [1]. It carried a nuclear reactor [1].",
                  verdicts=[{"verdicts": [{"claim": 1, "supported": False, "reason": "not in evidence"}]}])
    tokens = []
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "When did Juno arrive?", on_token=tokens.append)
    assert "reactor" in "".join(tokens) and "reactor" not in result.text


def test_standalone_questions_are_not_rewritten():
    from rag.qa import needs_rewrite
    assert needs_rewrite("When was it launched?") and needs_rewrite("Where did the rover land?")
    assert needs_rewrite("What about Kepler?") and needs_rewrite("Its commander?")
    assert not needs_rewrite("When was the Hubble Space Telescope launched?")
    llm = FakeLLM(ids=["1.1"], answer="Juno arrived at Jupiter on July 5, 2016 [1].", rewrite="SHOULD NOT BE USED")
    history = [{"role": "user", "content": "Tell me about Voyager"}, {"role": "assistant", "content": "..."}]
    result = ask(FakeStore([hit("text", "Juno", JUNO)]), llm, Settings(), "When did Juno arrive at Jupiter?", history=history)
    assert result.search_query == "When did Juno arrive at Jupiter?"
