from rag.config import Settings
from rag.retrieval import rrf_fuse, retrieve, with_neighbors
from rag.store import SearchHit


def hit(cid, doc_type="text", title=None, content=None, doc=1, idx=0, score=0.5):
    return SearchHit(content or f"chunk {cid}", score, doc_type, title or f"T{cid}", "s", "p", None, cid, doc, idx)


def test_rrf_rewards_chunks_found_by_both_methods():
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4)]
    assert [h.chunk_id for h in rrf_fuse(vector, keyword)][:2] == [3, 1]


class NeighborStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def chunk_neighbors(self, keys):
        return {k: v for k, v in self.chunks.items()}


def test_with_neighbors_adds_context_without_repeating_overlap():
    prev = "Earlier context about the launch vehicle."
    own = "the launch vehicle.\n\nJuno arrived at Jupiter in 2016."
    nxt = "Juno arrived at Jupiter in 2016.\n\nIt studies the planet's gravity field."
    store = NeighborStore({(1, 4): prev, (1, 6): nxt})
    [h] = with_neighbors(store, [hit(5, content=own, doc=1, idx=5)])
    assert h.content == ("Earlier context about\n\nthe launch vehicle.\n\nJuno arrived at Jupiter in 2016."
                         "\n\nIt studies the planet's gravity field.")


def test_with_neighbors_leaves_images_alone():
    [h] = with_neighbors(NeighborStore({(1, 0): "x"}), [hit(1, doc_type="image", content="Image: ISS", idx=1)])
    assert h.content == "Image: ISS"


class HybridStore:
    def __init__(self):
        self.keyword_calls = 0

    def search(self, embedding, *, team, embed_model, k, doc_types=None):
        if doc_types == ["image"]:
            return [hit(100 + i, "image") for i in range(3)]
        return [hit(1, content="Juno facts"), hit(2, content="Pizza dough")]

    def keyword_search(self, query, *, team, embed_model, k, doc_types=None):
        self.keyword_calls += 1
        return [hit(3, content="Juno arrived 2016")]

    def chunk_neighbors(self, keys):
        return {}


class FakeLLM:
    def embed_query(self, text):
        return [0.0]


class KeywordReranker:
    def score(self, query, docs):
        return [0.9 if "Juno arrived" in d else 0.5 if "Juno" in d else 0.01 for d in docs]


def test_retrieve_with_reranker_uses_both_methods_and_reranker_scores():
    store = HybridStore()
    hits = retrieve(store, FakeLLM(), Settings(), "When did Juno arrive?", 2, None, reranker=KeywordReranker())
    assert store.keyword_calls == 1
    assert [(h.chunk_id, h.score) for h in hits] == [(3, 0.9), (1, 0.5)]


def test_retrieve_visual_query_puts_images_first():
    hits = retrieve(HybridStore(), FakeLLM(), Settings(), "show me pictures of Juno", 5, None,
                    reranker=KeywordReranker())
    assert [h.doc_type for h in hits][:3] == ["image"] * 3
