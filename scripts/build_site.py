"""Build the static documentation site (deployed to rag.adityaps.work).

The pages in docs/ are written without <html>/<head> (the same files are published as claude.ai
artifacts, which add that wrapper). This wraps them into complete documents in site/, adds an index
page and a vercel.json, and copies nothing else: no code, data or configuration is uploaded.

Usage: python3 scripts/build_site.py && (cd site && vercel deploy --prod)
site/.vercel (the project link) is kept between builds.
"""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO = "https://github.com/Aditya-PS-05/rag-type-system"
PAGES = {
    "algorithms.html": ("docs/algorithms.html", "Every algorithm in the pipeline, with live demos."),
    "architecture.html": ("docs/architecture/viewer.html", "Interactive architecture diagrams, V1 to V5."),
}

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{description}">
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head>
<body>
"""

INDEX = """<title>Grounded Space Archive</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap">
<style>
:root { --bg: #f4f5f8; --surface: #fff; --fg: #171c28; --muted: #5a6273; --line: #d8dce5; --accent: #b5287a; }
@media (prefers-color-scheme: dark) { :root { --bg: #11141b; --surface: #181c25; --fg: #e6e9f0; --muted: #98a0b0; --line: #2a303c; --accent: #ec6bb5; color-scheme: dark; } }
body { background: var(--bg); color: var(--fg); font: 16px/1.6 "IBM Plex Sans", system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding-block: 12vh 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font: 700 clamp(2rem, 6vw, 3rem)/1.05 "Space Grotesk", system-ui, sans-serif; letter-spacing: -0.02em; margin: 0 0 12px; }
p.lead { color: var(--muted); max-width: 60ch; margin: 0 0 32px; }
.cards { display: grid; gap: 12px; }
a.card { display: block; text-decoration: none; color: inherit; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
a.card:hover, a.card:focus-visible { border-color: var(--accent); outline: none; }
a.card b { font: 600 1.15rem "Space Grotesk", system-ui, sans-serif; display: block; }
a.card span { color: var(--muted); font-size: 14.5px; }
.facts { margin: 32px 0 0; color: var(--muted); font-size: 14px; }
a { color: var(--accent); }
</style>
<main>
  <h1>Grounded Space Archive</h1>
  <p class="lead">A fully offline RAG chatbot over 35 Wikipedia articles, 20 arXiv papers and 100 NASA images. Every answer is built from verified quotes and every sentence is fact-checked; questions the documents can't answer are refused. These pages explain how it works.</p>
  <div class="cards">
    <a class="card" href="/algorithms"><b>Algorithm field guide</b><span>Chunking, HNSW, reciprocal rank fusion, reranking, the calibrated relevance gate, evidence by sentence ID and the verification cascade, with live demos on real evaluation data.</span></a>
    <a class="card" href="/architecture"><b>Architecture explorer</b><span>Each version from basic RAG to the current pipeline as an interactive flowchart. Click a box to trace what flows in and out.</span></a>
    <a class="card" href="{repo}"><b>Source code</b><span>Python, Postgres + pgvector, Ollama. Evaluation: 47/47 on the current question set.</span></a>
  </div>
  <p class="facts">The chatbot itself runs locally (it needs Ollama and Postgres on your machine); see the README to run it.</p>
</main>
"""


def wrap(content: str, description: str) -> str:
    title = re.search(r"<title>.*?</title>\n?", content)
    head = HEAD.format(description=description)
    if title:
        head = head.replace("</head>", title.group(0).strip() + "\n</head>")
        content = content.replace(title.group(0), "", 1)
    return head + content + "\n</body>\n</html>\n"


def main():
    SITE.mkdir(exist_ok=True)
    for old in SITE.iterdir():
        if old.name != ".vercel":
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    (SITE / "index.html").write_text(wrap(INDEX.replace("{repo}", REPO),
                                          "How an offline, fact-checked RAG chatbot works."))
    for name, (source, description) in PAGES.items():
        (SITE / name).write_text(wrap((ROOT / source).read_text(), description))
    (SITE / "vercel.json").write_text(json.dumps({"cleanUrls": True, "trailingSlash": False}, indent=2) + "\n")
    for f in sorted(SITE.iterdir()):
        print(f"site/{f.name}  {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
