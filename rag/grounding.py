"""Checks that an answer is backed by the retrieved sources."""

import re
from dataclasses import dataclass, field

from rag.store import SearchHit

REFUSAL = "I don't know based on the available documents."

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(*-])|\n+")
_ONLY_CITATIONS = re.compile(r"^\s*(\[\d+\]\s*)+[.]?\s*$")


@dataclass
class Claim:
    text: str
    citations: list[int]
    issue: str | None = None
    supported: bool | None = None
    reason: str = ""


@dataclass
class GroundingReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def problems(self) -> list[Claim]:
        return [c for c in self.claims if c.issue or c.supported is False]


def is_refusal(answer: str) -> bool:
    return "don't know based on the available documents" in answer.lower()


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


def check_citations(answer: str, num_sources: int) -> GroundingReport:
    report = GroundingReport()
    if is_refusal(answer):
        return report
    for text in split_claims(answer):
        cites = [int(n) for n in _CITATION.findall(text)]
        claim = Claim(text, cites)
        invalid = [n for n in cites if not 1 <= n <= num_sources]
        if not cites:
            claim.issue = "no citation"
        elif invalid:
            claim.issue = f"cites non-existent source {invalid}"
        report.claims.append(claim)
    return report


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}

VERIFY_PROMPT = """You are a strict fact checker. Decide whether the CLAIM is fully supported by the SOURCE TEXT.
Rules:
- Supported only if every fact in the claim (names, numbers, dates, causes, order of events) is stated in or directly implied by the source text.
- Use only the source text. Ignore anything you know from elsewhere.
- If the source says something different (e.g. a different year, or "announced/confirmed" vs "happened"), the claim is NOT supported.
Return JSON: {"supported": true|false, "reason": "<one short sentence>"}"""


def verify_claims(llm, report: GroundingReport, hits: list[SearchHit]) -> GroundingReport:
    for claim in report.claims:
        if claim.issue:
            continue
        sources = "\n\n".join(f"[{n}] {hits[n - 1].content}" for n in dict.fromkeys(claim.citations))
        result = llm.chat_json([
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": f"SOURCE TEXT:\n{sources}\n\nCLAIM:\n{_CITATION.sub('', claim.text).strip()}"},
        ], VERIFY_SCHEMA)
        claim.supported = bool(result.get("supported"))
        claim.reason = result.get("reason", "")
    return report
