from rag.qa import retrieve
from rag.config import Settings
from rag.store import SearchHit


def hit(doc_type, title):
    return SearchHit(f"content {title}", 0.8, doc_type, title, "src", f"{title}.x", None)


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, embedding, *, team, embed_model, k, doc_types=None):
        self.calls.append(doc_types)
        if doc_types == ["image"]:
            return [hit("image", f"img{i}") for i in range(k)]
        return [hit("text", f"t{i}") for i in range(k - 1)] + [hit("image", "img0")]


class FakeLLM:
    def embed_query(self, text):
        return [0.0]


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


class LowScoreStore(FakeStore):
    def search(self, embedding, *, team, embed_model, k, doc_types=None):
        return [SearchHit("pizza?", 0.5, "text", "Milky Way", "src", "p", None)]


class ExplodingLLM(FakeLLM):
    def chat_stream(self, messages):
        raise AssertionError("LLM must not be called when no source is relevant")


def test_relevance_gate_refuses_without_calling_llm():
    from rag.qa import ask
    result = ask(LowScoreStore(), ExplodingLLM(), Settings(min_score=0.65), "best pizza recipe?")
    assert result.gated and result.refused


def test_image_as_verb_is_not_visual_intent():
    from rag.qa import VISUAL_INTENT
    assert not VISUAL_INTENT.search("Which black hole did the Event Horizon Telescope image first?")
    assert VISUAL_INTENT.search("show me an image of Saturn")
    assert VISUAL_INTENT.search("Any photos of the ISS?")
