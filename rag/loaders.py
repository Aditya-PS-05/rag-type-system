import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

WIKI_STOP_SECTIONS = re.compile(
    r"^==+\s*(See also|References|Notes|External links|Further reading|Bibliography|Sources|Citations)\s*==+\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class Document:
    doc_type: str
    path: str
    title: str
    source: str
    license: str
    sections: list[tuple[int | None, str]]
    meta: dict = field(default_factory=dict)


def clean_wikipedia(text: str) -> str:
    m = WIKI_STOP_SECTIONS.search(text)
    if m:
        text = text[:m.start()]
    return re.sub(r"^(==+)\s*(.+?)\s*\1\s*$", r"\2", text, flags=re.MULTILINE)


def clean_pdf_page(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", text)
    return text


def load_text(root: Path, entry: dict) -> Document:
    text = (root / entry["path"]).read_text(encoding="utf-8")
    return Document("text", entry["path"], entry["title"], entry["source"], entry["license"],
                    [(None, clean_wikipedia(text))])


def load_pdf(root: Path, entry: dict) -> Document:
    reader = PdfReader(root / entry["path"])
    sections = []
    if entry.get("summary"):
        sections.append((0, f"Abstract: {entry['summary']}"))
    for i, page in enumerate(reader.pages, start=1):
        text = clean_pdf_page(page.extract_text() or "")
        if text.strip():
            sections.append((i, text))
    return Document("pdf", entry["path"], entry["title"], entry["source"], entry["license"],
                    sections, {"authors": entry.get("authors", [])})


def load_image(root: Path, entry: dict) -> Document:
    parts = [f"Image: {entry['title']}"]
    if entry.get("date"):
        parts.append(f"Date: {entry['date'][:10]}")
    if entry.get("keywords"):
        parts.append(f"Keywords: {', '.join(entry['keywords'])}")
    if entry.get("description"):
        parts.append(entry["description"])
    return Document("image", entry["path"], entry["title"], entry["source"], entry["license"],
                    [(None, "\n\n".join(parts))], {"keywords": entry.get("keywords", [])})


LOADERS = {"text": load_text, "pdf": load_pdf, "image": load_image}


def load_manifest(root: Path) -> Iterator[Document]:
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            yield LOADERS[entry["type"]](root, entry)
