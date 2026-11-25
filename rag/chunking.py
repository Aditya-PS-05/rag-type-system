import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, max_chars: int) -> list[str]:
    pieces, buf = [], ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Greedy paragraph packing; paragraphs longer than max_chars are split on sentences.

    Each chunk after the first starts with the tail (up to `overlap` chars) of the previous one.
    """
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        if p:
            paragraphs.extend(_split_long(p, max_chars) if len(p) > max_chars else [p])

    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + 2 + len(p) > max_chars:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            buf = f"{tail}\n\n{p}" if tail and len(tail) + 2 + len(p) <= max_chars else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks
