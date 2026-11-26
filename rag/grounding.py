"""Checks that an answer is backed by the retrieved sources.

Evidence is selected by sentence ID, so every quote is exact source text by construction. A claim that
copies its cited quote word for word is supported without further checks; every other claim is judged
by an LLM against the quotes of the sources it cites. A claim with no citation is matched to the quote
it most resembles and judged against that, so a correct but uncited sentence is kept and cited.
"""

import re
from dataclasses import dataclass, field

from rag.store import SearchHit

REFUSAL = "I don't know based on the available documents."

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(*-])|\n+")
_STATES_GAP = re.compile(
    r"^\s*(the\s+)?(provided\s+|available\s+|retrieved\s+)?(documents?|sources?|quotes?|evidence)\b[^.]{0,40}?"
    r"\b(do|does|did)\s+not\s+(say|state|mention|specify|include|describe|give|cover|provide|contain)\b", re.I)
_ONLY_CITATIONS = re.compile(r"^\s*(\[\d+\]\s*)+[.]?\s*$")
_SOURCE_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SENTENCES_PER_SOURCE = 40
MAX_IMAGE_EVIDENCE_CHARS = 600


@dataclass
class Quote:
    source: int
    text: str
    verified: bool = False


@dataclass
class Claim:
    text: str
    citations: list[int]
    issue: str | None = None
    supported: bool | None = None
    reason: str = ""
    auto_cited: bool = False
    method: str = ""


@dataclass
class GroundingReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def problems(self) -> list[Claim]:
        return [c for c in self.claims if c.issue or c.supported is False]


_REFUSAL_PHRASE = "don't know based on the available documents"


def is_refusal(answer: str) -> bool:
    """True only when the whole answer is the refusal, so a refusal phrase can't hide claims from checking."""
    return _REFUSAL_PHRASE in answer.lower() and not without_refusal_sentences(answer)


def without_refusal_sentences(answer: str) -> str:
    """Drop sentences that contain the refusal phrase; the rest of a mixed answer is checked as usual."""
    pieces = re.split(r"(?<=[.!?])\s+", answer)
    return " ".join(p for p in pieces if _REFUSAL_PHRASE not in p.lower()).strip()


def _clip_to_sentence(text: str, limit: int) -> str:
    """Cut at the last sentence end within the limit, so a quote never stops mid-sentence."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > limit // 2 else cut


def number_sentences(hits: list[SearchHit]) -> dict[str, tuple[int, str]]:
    """Split each source into sentences with IDs like "2.5" (source 2, sentence 5). An image is one unit."""
    numbered = {}
    for i, h in enumerate(hits, start=1):
        if h.doc_type == "image":
            numbered[f"{i}.1"] = (i, _clip_to_sentence(" ".join(h.content.split()), MAX_IMAGE_EVIDENCE_CHARS))
            continue
        sentences = [" ".join(s.split()) for s in _SOURCE_SENTENCE.split(h.content) if s.strip()]
        for j, sentence in enumerate(sentences[:MAX_SENTENCES_PER_SOURCE], start=1):
            numbered[f"{i}.{j}"] = (i, sentence)
    return numbered


def _with_context(ids: list[str], numbered: dict[str, tuple[int, str]], context: int) -> list[str]:
    expanded = []
    for raw in ids:
        if raw not in numbered or context == 0:
            expanded.append(raw)
            continue
        source, sentence_no = raw.split(".")
        span = range(int(sentence_no) - context, int(sentence_no) + context + 1)
        expanded += [sid for n in span if (sid := f"{source}.{n}") in numbered]
    return expanded


def quotes_from_ids(ids: list[str], numbered: dict[str, tuple[int, str]], context: int = 1) -> list[Quote]:
    """Turn selected sentence IDs (plus `context` neighbouring sentences each) into quotes.

    Consecutive sentences of one source are merged. IDs that don't exist become unverified quotes,
    so made-up evidence is visible but never used.
    """
    quotes: list[Quote] = []
    last: tuple[int, int] | None = None
    ids = _with_context([str(i).strip() for i in ids], numbered, context)
    for raw in dict.fromkeys(ids):
        if raw not in numbered:
            quotes.append(Quote(0, raw, verified=False))
            continue
        source, text = numbered[raw]
        sentence_no = int(raw.split(".")[1])
        if last == (source, sentence_no - 1) and quotes and quotes[-1].verified and quotes[-1].source == source:
            quotes[-1].text = f"{quotes[-1].text} {text}"
        else:
            quotes.append(Quote(source, text, verified=True))
        last = (source, sentence_no)
    return quotes


def split_claims(answer: str) -> list[str]:
    claims = []
    for piece in _SENTENCE_BOUNDARY.split(answer):
        piece = piece.strip(" -*\t")
        if not piece:
            continue
        if _ONLY_CITATIONS.match(piece) and claims:
            claims[-1] = f"{claims[-1]} {piece.strip()}"
            continue
        claims.append(piece)
    return [c for c in claims if len(_CITATION.sub("", c).strip()) >= 25 and not c.rstrip().endswith(":")]


def check_citations(answer: str, valid_sources: set[int]) -> GroundingReport:
    report = GroundingReport()
    if is_refusal(answer):
        return report
    for text in split_claims(answer):
        cites = [int(n) for n in _CITATION.findall(text)]
        claim = Claim(text, cites)
        invalid = sorted({n for n in cites if n not in valid_sources})
        if _STATES_GAP.match(text) and not cites:
            claim.supported, claim.method = True, "states what is missing"
        elif not cites:
            claim.issue = "no citation"
        elif invalid:
            claim.issue = f"cites source(s) {invalid} with no verified quote"
        report.claims.append(claim)
    return report


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "integer"}, "supported": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["claim", "supported"]}}},
    "required": ["verdicts"],
}

VERIFY_PROMPT = """You are a fact checker. For each numbered CLAIM, decide whether the EVIDENCE from the sources it cites supports it.
A claim is SUPPORTED when its facts are stated in, or follow directly from, the cited evidence:
- Paraphrases, synonyms and alternative names for the same thing are fine (e.g. "Kepler observatory" = "Kepler space telescope"; the source title tells you what a quote is about).
- Simple inferences are fine: comparing two dates in the evidence, counting, or "arrived" for "entered orbit".
- Saying that an image is available or can be viewed is fine.
A claim is NOT SUPPORTED when it:
- states a name, number, date, place or event that is missing from the cited evidence, or
- contradicts the evidence (e.g. a different year, or treats "announced/confirmed" as "happened").
Use only the evidence; ignore what you know from elsewhere.
Return JSON with one verdict per claim: {"verdicts": [{"claim": <number>, "supported": true|false, "reason": "<only if unsupported: at most 12 words>"}]}"""


_STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is",
              "was", "were", "are", "be", "been", "it", "its", "this", "that", "which", "who", "has", "had", "have"}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[\"“”'‘’()]", " ", _CITATION.sub("", text).lower()).split()).strip(" .")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def copies_evidence(claim: str, evidence: str) -> bool:
    """True when the claim is an exact (case- and spacing-insensitive) excerpt of the evidence."""
    norm = _normalize(claim)
    return len(norm) >= 25 and norm in _normalize(evidence)


def closest_source(claim: str, evidence: dict[int, str]) -> int | None:
    words = _words(_CITATION.sub("", claim))
    scored = sorted(((len(words & _words(text)), n) for n, text in evidence.items()), reverse=True)
    return scored[0][1] if scored and scored[0][0] else None


def verify_claims(llm, report: GroundingReport, evidence: dict[int, str]) -> GroundingReport:
    """Judge every claim against its cited evidence; claims without a verdict count as unsupported.

    Uncited claims are first assigned the closest source. Exact excerpts are accepted without the LLM;
    the rest are judged in a single call.
    """
    for c in report.claims:
        if c.issue == "no citation" and (source := closest_source(c.text, evidence)):
            c.citations, c.issue, c.auto_cited = [source], None, True
    pending = []
    for c in report.claims:
        if c.issue or c.method:
            continue
        if copies_evidence(c.text, " ".join(evidence.get(n, "") for n in c.citations)):
            c.supported, c.method = True, "exact quote"
        else:
            pending.append(c)
    if not pending:
        return report
    sources = sorted({n for c in pending for n in c.citations})
    evidence_text = "\n\n".join(f"[{n}] {evidence[n]}" for n in sources)
    claims_text = "\n".join(f"{i}. {_CITATION.sub('', c.text).strip()} (cites {', '.join(f'[{n}]' for n in dict.fromkeys(c.citations))})"
                            for i, c in enumerate(pending, start=1))
    result = llm.chat_json([
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": f"EVIDENCE:\n{evidence_text}\n\nCLAIMS:\n{claims_text}"},
    ], VERIFY_SCHEMA)
    verdicts = {v.get("claim"): v for v in result.get("verdicts", []) if isinstance(v, dict)}
    for i, claim in enumerate(pending, start=1):
        verdict = verdicts.get(i)
        claim.method = "fact checker"
        claim.supported = bool(verdict and verdict.get("supported"))
        claim.reason = verdict.get("reason", "") if verdict else "no verdict returned by the checker"
    return report


def add_citations(answer: str, claims: list[Claim]) -> str:
    """Write the citation of each auto-cited claim into the answer text (before its final punctuation)."""
    for c in claims:
        if not c.auto_cited or _CITATION.search(c.text):
            continue
        marks = "".join(f"[{n}]" for n in c.citations)
        text = c.text.rstrip()
        cited = f"{text[:-1]} {marks}{text[-1]}" if text[-1:] in ".!?" else f"{text} {marks}"
        answer = answer.replace(c.text, cited, 1)
        c.text = cited
    return answer


def remove_claims(answer: str, claims: list[Claim]) -> str:
    for c in claims:
        answer = answer.replace(c.text, "")
    lines = [l.rstrip() for l in answer.splitlines()]
    lines = [l for l in lines if not l.strip() or l.strip(" -*.\t")]
    text = re.sub(r"[ \t]{2,}", " ", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
