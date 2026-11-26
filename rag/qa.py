"""Question answering pipeline.

rewrite follow-up -> retrieve -> relevance gate -> select evidence sentences by ID (quotes are exact source text)
-> answer from verified quotes only -> check citations -> verify claims (exact quote or LLM judge)
-> remove unsupported, cite the uncited ones that were verified
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rag.config import Settings
from rag.grounding import (REFUSAL, Claim, GroundingReport, Quote, add_citations, check_citations, is_refusal,
                           without_refusal_sentences,
                           number_sentences, quotes_from_ids, remove_claims, verify_claims)
from rag.ollama import Ollama
from rag.retrieval import retrieve
from rag.store import SearchHit, VectorStore

HISTORY_TURNS = 4
MAX_EVIDENCE_SENTENCES = 10
MAX_ANSWER_TOKENS = 400
EVIDENCE_SOURCES = 6  # the model reads the top 6 of the retrieved chunks: prompt reading is the slowest step

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
- Answer every part of the question. Put each fact in its own short sentence, and do not add conclusions or
  comparisons that the quotes do not state themselves.
- Use only what the quotes say. Do not add facts, numbers or dates from your own knowledge, even if you are sure they are true.
- Report dates and events exactly as the quotes state them; do not merge or reinterpret them.
- If the quotes answer none of the question, reply with exactly: {REFUSAL} That sentence is only ever the whole
  reply; never combine it with other sentences.
- If the question asks for something the quotes do not contain, answer the rest and add one sentence naming what
  is missing (e.g. "The documents do not say what colour the flash was."). Do not mention missing details the
  question did not ask for.
- When a quote comes from an image, describe in your own words what the image shows, name it by its title and
  say it can be viewed. Never copy the quote list format such as [2] (image) Title: "...".
- Answer in at most 4 sentences (up to 8 if the question asks for a list or details). Use only the quotes that
  answer the question; do not restate every quote."""

REWRITE_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
EVIDENCE_SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"},
                                                            "maxItems": MAX_EVIDENCE_SENTENCES}},
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


FOLLOW_UP = re.compile(
    r"\b(it|its|it's|they|them|their|he|him|his|she|her|this|that|these|those|there|then|former|latter|same|"
    r"other|another|one|ones|else|also|too)\b"
    r"|\bthe (mission|rover|lander|telescope|spacecraft|probe|craft|satellite|station|crew|program|programme|"
    r"paper|image|picture|photo|launch|landing)\b"
    r"|^\s*(and|but|so|what about|how about)\b", re.I)


def needs_rewrite(question: str) -> bool:
    """Only questions that may refer back to the conversation are rewritten (saves an LLM call)."""
    return bool(FOLLOW_UP.search(question)) or len(question.split()) <= 2


def rewrite_question(llm: Ollama, history: list[dict], question: str) -> str:
    if not history or not needs_rewrite(question):
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
    hits = hits[:EVIDENCE_SOURCES]
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
        on_token: Callable[[str], None] | None = None,
        on_stage: Callable[[str], None] | None = None) -> Answer:
    """Answer from the documents only.

    strict=True fact-checks every claim and removes unsupported ones; the returned text is final, and
    tokens passed to on_token are a draft that may still lose sentences.
    strict=False only reports citation problems.
    on_stage is called with "rewrite", "retrieve", "evidence", "answer" and "check" as each step starts.
    """
    stage = on_stage or (lambda name: None)
    if history and needs_rewrite(question):
        stage("rewrite")
    search_query = rewrite_question(llm, history or [], question)
    stage("retrieve")
    hits = retrieve(store, llm, cfg, search_query, k, doc_types, reranker=reranker)
    threshold = cfg.min_rerank_score if reranker else cfg.min_score
    if not hits or max(h.score for h in hits) < threshold:
        return Answer(REFUSAL, hits, search_query, gated=True)

    stage("evidence")
    quotes = select_evidence(llm, search_query, hits)
    verified = [q for q in quotes if q.verified]
    if not verified:
        return Answer(REFUSAL, hits, search_query, quotes=quotes)

    messages = [{"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Evidence quotes:\n{format_evidence(verified, hits)}\n\nQuestion: {search_query}"}]
    stage("answer")
    parts = []
    for token in llm.chat_stream(messages, max_tokens=MAX_ANSWER_TOKENS):
        parts.append(token)
        if on_token:
            on_token(token)
    text = "".join(parts).strip()
    if not is_refusal(text):
        text = without_refusal_sentences(text) or REFUSAL

    evidence = evidence_by_source(verified, hits)
    grounding = check_citations(text, set(evidence))
    removed: list[Claim] = []
    if strict and grounding.claims:
        stage("check")
        verify_claims(llm, grounding, evidence)
        removed = grounding.problems
        if removed:
            text = remove_claims(text, removed)
            if len(removed) == len(grounding.claims) or not text:
                text = REFUSAL
        if text != REFUSAL:
            text = add_citations(text, [c for c in grounding.claims if c not in removed])
    return Answer(text, hits, search_query, quotes=quotes, grounding=grounding, removed=removed)
