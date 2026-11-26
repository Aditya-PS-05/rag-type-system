"""Retrieval: hybrid (vector + keyword) candidates -> cross-encoder rerank -> neighbouring-chunk context."""

import re

from rag.config import Settings
from rag.store import SearchHit, VectorStore

IMAGES_PER_VISUAL_QUERY = 3
CANDIDATES_PER_METHOD = 30
RERANK_POOL = 24
RRF_K = 60
NEIGHBOR_CHARS = 250
VISUAL_INTENT = re.compile(r"\b(images|image of|pictures?|photos?|photographs?|show me|looks? like)\b", re.I)


def rrf_fuse(*rankings: list[SearchHit]) -> list[SearchHit]:
    """Reciprocal rank fusion: merge ranked lists, rewarding chunks ranked high by any method."""
    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1 / (RRF_K + rank)
            hits.setdefault(h.chunk_id, h)
    return [hits[c] for c in sorted(scores, key=scores.get, reverse=True)]


def _strip_overlap(before: str, after: str) -> str:
    """Chunks overlap: `after` may start with the tail of `before`. Return `after` without that repeat."""
    head, sep, rest = after.partition("\n\n")
    return rest if sep and head and before.endswith(head) else after


def with_neighbors(store: VectorStore, hits: list[SearchHit]) -> list[SearchHit]:
    """Extend text/PDF hits with up to NEIGHBOR_CHARS of the previous and next chunk of the same document.

    A neighbour that is itself one of the hits is skipped: the model already reads it as its own source.
    """
    keys = [(h.document_id, h.chunk_index) for h in hits if h.doc_type != "image" and h.document_id]
    neighbors = {k: v for k, v in store.chunk_neighbors(keys).items() if k not in set(keys)}
    for h in hits:
        if h.doc_type == "image" or not h.document_id:
            continue
        prev = neighbors.get((h.document_id, h.chunk_index - 1))
        nxt = neighbors.get((h.document_id, h.chunk_index + 1))
        content = h.content
        if prev:
            own_head = content.partition("\n\n")[0]
            if prev.endswith(own_head) and "\n\n" in content:
                prev = prev[: -len(own_head)].rstrip()
            tail = prev[-NEIGHBOR_CHARS:]
            content = f"{tail[tail.find(' ') + 1:] if len(prev) > NEIGHBOR_CHARS else tail}\n\n{content}"
        if nxt:
            nxt = _strip_overlap(h.content, nxt)
            head = nxt[:NEIGHBOR_CHARS]
            content = f"{content}\n\n{head[:head.rfind(' ')] if len(nxt) > NEIGHBOR_CHARS else head}"
        h.content = content.strip()
    return hits


def _ensure_images(ranked: list[SearchHit], k: int) -> list[SearchHit]:
    images = [h for h in ranked if h.doc_type == "image"][:IMAGES_PER_VISUAL_QUERY]
    others = [h for h in ranked if h.doc_type != "image"]
    return images + others[:max(k - len(images), 0)]


def retrieve(store: VectorStore, llm, cfg: Settings, question: str, k: int,
             doc_types: list[str] | None, reranker=None) -> list[SearchHit]:
    """Return the k best chunks. hit.score is the reranker score when a reranker is given, else cosine similarity."""
    embedding = llm.embed_query(question)
    visual = doc_types is None and bool(VISUAL_INTENT.search(question))
    search = dict(team=cfg.team, embed_model=cfg.embed_model)

    if reranker is None:
        hits = store.search(embedding, k=k, doc_types=doc_types, **search)
        if visual:
            hits = _ensure_images(store.search(embedding, k=IMAGES_PER_VISUAL_QUERY, doc_types=["image"], **search)
                                  + hits, k)
        return hits

    vector = store.search(embedding, k=CANDIDATES_PER_METHOD, doc_types=doc_types, **search)
    keyword = store.keyword_search(question, k=CANDIDATES_PER_METHOD, doc_types=doc_types, **search)
    pool = rrf_fuse(vector, keyword)[:RERANK_POOL]
    if visual:
        seen = {h.chunk_id for h in pool}
        pool += [h for h in store.search(embedding, k=10, doc_types=["image"], **search) if h.chunk_id not in seen]

    scores = reranker.score(question, [f"{h.title}\n{h.content}" for h in pool])
    for h, s in zip(pool, scores):
        h.score = s
    ranked = sorted(pool, key=lambda h: h.score, reverse=True)
    hits = _ensure_images(ranked, k) if visual else ranked[:k]
    return with_neighbors(store, hits)
