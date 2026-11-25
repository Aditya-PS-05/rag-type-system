"""Question answering pipeline: retrieve -> relevance gate -> grounded answer -> checks."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rag.config import Settings
from rag.grounding import REFUSAL, GroundingReport, check_citations, is_refusal, verify_claims
from rag.ollama import Ollama
from rag.store import SearchHit, VectorStore

SYSTEM_PROMPT = f"""You answer questions about space exploration using ONLY the numbered sources in the user's message.
Rules:
- Every sentence that states a fact must end with the citation(s) it came from, like [1] or [2][3].
- Use only facts written in the sources. Do not add facts, numbers or dates from your own knowledge, even if you are sure they are true.
- Report dates and events exactly as the sources state them; do not merge or reinterpret them.
- If the sources do not contain the answer, reply with exactly: {REFUSAL}
- When a source is an image, describe it from its description and say it can be viewed.
- Be concise."""

HISTORY_TURNS = 4
IMAGES_PER_VISUAL_QUERY = 3
VISUAL_INTENT = re.compile(r"\b(images|image of|pictures?|photos?|photographs?|show me|looks? like)\b", re.I)


@dataclass
class Answer:
    text: str
    hits: list[SearchHit]
    gated: bool = False
    grounding: GroundingReport = field(default_factory=GroundingReport)

    @property
    def refused(self) -> bool:
        return self.gated or is_refusal(self.text)


def retrieve(store: VectorStore, llm: Ollama, cfg: Settings, question: str, k: int,
             doc_types: list[str] | None) -> list[SearchHit]:
    embedding = llm.embed_query(question)
    hits = store.search(embedding, team=cfg.team, embed_model=cfg.embed_model, k=k, doc_types=doc_types)
    if doc_types is None and VISUAL_INTENT.search(question):
        images = store.search(embedding, team=cfg.team, embed_model=cfg.embed_model,
                              k=IMAGES_PER_VISUAL_QUERY, doc_types=["image"])
        others = [h for h in hits if h.doc_type != "image"]
        hits = images + others[:max(k - len(images), 0)]
    return hits


def format_sources(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f", page {h.page}" if h.page else ""
        blocks.append(f"[{i}] ({h.doc_type}) {h.title}{where}\n{h.content}")
    return "\n\n".join(blocks)


def ask(store: VectorStore, llm: Ollama, cfg: Settings, question: str, *, history: list[dict] | None = None,
        k: int = 6, doc_types: list[str] | None = None, verify: bool = False,
        on_token: Callable[[str], None] | None = None) -> Answer:
    hits = retrieve(store, llm, cfg, question, k, doc_types)
    if not hits or max(h.score for h in hits) < cfg.min_score:
        if on_token:
            on_token(REFUSAL)
        return Answer(REFUSAL, hits, gated=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or [])[-2 * HISTORY_TURNS:],
                {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"}]
    parts = []
    for token in llm.chat_stream(messages):
        parts.append(token)
        if on_token:
            on_token(token)
    text = "".join(parts).strip()

    grounding = check_citations(text, len(hits))
    if verify:
        verify_claims(llm, grounding, hits)
    return Answer(text, hits, grounding=grounding)
