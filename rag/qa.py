"""Question answering pipeline.

rewrite follow-up -> retrieve -> relevance gate -> select evidence sentences by ID (quotes are exact source text)
-> answer from verified quotes only -> check citations -> verify claims (LLM judge) -> remove unsupported
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from rag.config import Settings
from rag.grounding import (REFUSAL, Claim, GroundingReport, Quote, check_citations, is_refusal,
                           number_sentences, quotes_from_ids, remove_claims, verify_claims)
from rag.ollama import Ollama
from rag.retrieval import retrieve
from rag.store import SearchHit, VectorStore

HISTORY_TURNS = 4
MAX_EVIDENCE_SENTENCES = 10

REWRITE_PROMPT = """Rewrite the user's latest question as a standalone question that can be understood without the conversation.
Replace pronouns and vague references ("it", "they", "that mission", "the second one") with the specific names from the conversation.
If the question is already standalone, return it unchanged. Do not answer it.
Return JSON: {"question": "<standalone question>"}"""

EVIDENCE_PROMPT = f"""You find evidence. The sources below are split into sentences, each with an ID like 2.5 (source 2, sentence 5).
Select the IDs of the sentences needed to answer the question.
Rules:
- Select at most {MAX_EVIDENCE_SENTENCES} IDs, most relevant first. Include neighbouring sentences when they are needed for context.
- For a request to see pictures, images or photos, select the IDs of the matching image sources.
- If no sentence helps answer the question, return an empty list.
Return JSON: {{"ids": ["<id>", ...]}}"""

ANSWER_PROMPT = f"""You answer questions about space exploration using ONLY the evidence quotes in the user's message.
Rules:
- Every sentence that states a fact must end with the citation number(s) of the quotes it uses, like [1] or [2][3].
- Use only what the quotes say. Do not add facts, numbers or dates from your own knowledge, even if you are sure they are true.
- Report dates and events exactly as the quotes state them; do not merge or reinterpret them.
- If the quotes do not answer the question, reply with exactly: {REFUSAL}
- When a quote comes from an image, describe the image from it and say it can be viewed.
- Be concise."""

REWRITE_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
EVIDENCE_SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                   "required": ["ids"]}


@dataclass
class Answer:
    text: str
    hits: list[SearchHit]
    search_query: str
    gated: bool = False
    quotes: list[Quote] = field(default_factory=list)
    grounding: GroundingReport = field(default_factory=GroundingReport)
    removed: list[Claim] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.gated or is_refusal(self.text)

    @property
    def verified_quotes(self) -> list[Quote]:
        return [q for q in self.quotes if q.verified]

    @property
    def rejected_quotes(self) -> list[Quote]:
        """Evidence IDs the model invented (not present in the sources)."""
        return [q for q in self.quotes if not q.verified]

    @property
    def cited_sources(self) -> list[int]:
        return sorted({n for c in self.grounding.claims if c not in self.removed for n in c.citations})


def rewrite_question(llm: Ollama, history: list[dict], question: str) -> str:
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-2 * HISTORY_TURNS:])
    result = llm.chat_json([
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest question: {question}"},
    ], REWRITE_SCHEMA)
    return (result.get("question") or question).strip()


def format_numbered_sources(hits: list[SearchHit], numbered: dict[str, tuple[int, str]]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f", page {h.page}" if h.page else ""
        lines = [f"[{i}] ({h.doc_type}) {h.title}{where}"]
        lines += [f"  {sid} {text}" for sid, (src, text) in numbered.items() if src == i]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def select_evidence(llm: Ollama, question: str, hits: list[SearchHit]) -> list[Quote]:
    numbered = number_sentences(hits)
    result = llm.chat_json([
        {"role": "system", "content": EVIDENCE_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{format_numbered_sources(hits, numbered)}\n\nQuestion: {question}"},
    ], EVIDENCE_SCHEMA)
    return quotes_from_ids(result.get("ids", [])[:MAX_EVIDENCE_SENTENCES], numbered)


def format_evidence(quotes: list[Quote], hits: list[SearchHit]) -> str:
    return "\n".join(f'[{q.source}] ({hits[q.source - 1].doc_type}) {hits[q.source - 1].title}: "{q.text}"'
                     for q in quotes)


def evidence_by_source(quotes: list[Quote], hits: list[SearchHit]) -> dict[int, str]:
    evidence: dict[int, list[str]] = {}
    for q in quotes:
        evidence.setdefault(q.source, []).append(f'"{q.text}"')
    return {n: f"({hits[n - 1].doc_type}: {hits[n - 1].title}) " + " ... ".join(texts) for n, texts in evidence.items()}


def ask(store: VectorStore, llm: Ollama, cfg: Settings, question: str, *, history: list[dict] | None = None,
        k: int = 8, doc_types: list[str] | None = None, strict: bool = True, reranker=None,
        on_token: Callable[[str], None] | None = None) -> Answer:
    """Answer from the documents only.

    strict=True fact-checks every claim and removes unsupported ones before returning (nothing is streamed).
    strict=False streams the answer and only reports citation problems.
    """
    search_query = rewrite_question(llm, history or [], question)
    hits = retrieve(store, llm, cfg, search_query, k, doc_types, reranker=reranker)
    threshold = cfg.min_rerank_score if reranker else cfg.min_score
    if not hits or max(h.score for h in hits) < threshold:
        return Answer(REFUSAL, hits, search_query, gated=True)

    quotes = select_evidence(llm, search_query, hits)
    verified = [q for q in quotes if q.verified]
    if not verified:
        return Answer(REFUSAL, hits, search_query, quotes=quotes)

    messages = [{"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Evidence quotes:\n{format_evidence(verified, hits)}\n\nQuestion: {search_query}"}]
    parts = []
    for token in llm.chat_stream(messages):
        parts.append(token)
        if on_token and not strict:
            on_token(token)
    text = "".join(parts).strip()

    evidence = evidence_by_source(verified, hits)
    grounding = check_citations(text, set(evidence))
    removed: list[Claim] = []
    if strict and grounding.claims:
        verify_claims(llm, grounding, evidence)
        removed = grounding.problems
        if removed:
            text = remove_claims(text, removed)
            if len(removed) == len(grounding.claims) or not text:
                text = REFUSAL
    return Answer(text, hits, search_query, quotes=quotes, grounding=grounding, removed=removed)
