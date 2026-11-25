"""Download a small multimodal space-exploration corpus (text, PDFs, images).

Sources:
  text   -> Wikipedia plain-text extracts (CC BY-SA 4.0)
  pdfs   -> arXiv papers (open access, per-paper licenses)
  images -> NASA Image and Video Library (public domain)

Usage: python3 scripts/download_dataset.py [output_dir] [text,pdf,image]
"""

import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

USER_AGENT = "local-rag-dataset-builder/0.1 (educational use)"

WIKI_TOPICS = [
    "Space exploration", "Apollo program", "Apollo 11", "Artemis program",
    "International Space Station", "Hubble Space Telescope", "James Webb Space Telescope",
    "Voyager program", "Voyager 1", "Mars", "Curiosity (rover)", "Perseverance (rover)",
    "Ingenuity (helicopter)", "Exoplanet", "Kepler space telescope",
    "Transiting Exoplanet Survey Satellite", "Cassini–Huygens", "Juno (spacecraft)",
    "New Horizons", "Pluto", "Jupiter", "Saturn", "Moon", "Black hole",
    "Event Horizon Telescope", "SpaceX Starship", "Falcon 9", "Space Shuttle program",
    "Sputnik 1", "Yuri Gagarin", "Chandrayaan-3", "Europa Clipper", "Gravitational wave",
    "Big Bang", "Milky Way",
]

ARXIV_QUERIES = [
    ("ti:exoplanet AND ti:atmosphere AND ti:JWST", 4),
    ("ti:Mars AND ti:rover", 4),
    ("ti:Europa AND ti:ocean", 3),
    ("ti:Event AND ti:Horizon AND ti:Telescope", 3),
    ("ti:gravitational AND ti:wave AND ti:detection", 3),
    ("ti:TESS AND ti:planet", 3),
]

NASA_QUERIES = [
    ("apollo 11", 12), ("international space station", 12), ("mars rover", 12),
    ("james webb", 12), ("hubble galaxy", 12), ("saturn cassini", 10),
    ("jupiter juno", 10), ("space shuttle launch", 10), ("artemis", 10),
]


def fetch(url, retries=4):
    for attempt in range(retries):
        try:
            return subprocess.run(
                ["curl", "-sSfL", "--max-time", "120", "-A", USER_AGENT, url],
                capture_output=True, check=True,
            ).stdout
        except subprocess.CalledProcessError as e:
            e = e.stderr.decode().strip() or e
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(15 * (attempt + 1))


def slug(s, n=80):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")[:n].lower()


def download_wikipedia(out, manifest):
    d = out / "text"
    d.mkdir(parents=True, exist_ok=True)
    for title in WIKI_TOPICS:
        params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts", "explaintext": 1,
            "redirects": 1, "titles": title, "format": "json",
        })
        data = json.loads(fetch(f"https://en.wikipedia.org/w/api.php?{params}"))
        page = next(iter(data["query"]["pages"].values()))
        text = page.get("extract", "")
        if not text:
            print(f"  [text] skipped {title}")
            continue
        path = d / f"{slug(page['title'])}.txt"
        path.write_text(f"# {page['title']}\n\n{text}", encoding="utf-8")
        manifest.append({
            "type": "text", "path": str(path.relative_to(out)), "title": page["title"],
            "source": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}",
            "license": "CC BY-SA 4.0",
        })
        print(f"  [text] {page['title']} ({len(text):,} chars)")
        time.sleep(2)


def download_arxiv(out, manifest):
    d = out / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    seen = set()
    for query, n in ARXIV_QUERIES:
        params = urllib.parse.urlencode({
            "search_query": query, "max_results": n,
            "sortBy": "relevance", "sortOrder": "descending",
        })
        root = ET.fromstring(fetch(f"https://export.arxiv.org/api/query?{params}"))
        time.sleep(3)
        for entry in root.findall("a:entry", ns):
            arxiv_id = entry.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            title = " ".join(entry.find("a:title", ns).text.split())
            pdf_url = next((l.get("href") for l in entry.findall("a:link", ns)
                            if l.get("title") == "pdf"), f"https://arxiv.org/pdf/{arxiv_id}")
            path = d / f"{slug(arxiv_id)}.pdf"
            try:
                path.write_bytes(fetch(pdf_url))
            except Exception as e:
                print(f"  [pdf] failed {arxiv_id}: {e}")
                continue
            manifest.append({
                "type": "pdf", "path": str(path.relative_to(out)), "title": title,
                "authors": [a.find("a:name", ns).text for a in entry.findall("a:author", ns)],
                "summary": " ".join(entry.find("a:summary", ns).text.split()),
                "source": f"https://arxiv.org/abs/{arxiv_id}", "license": "arXiv (see source)",
            })
            print(f"  [pdf] {arxiv_id}: {title[:70]} ({path.stat().st_size // 1024} KB)")
            time.sleep(3)


def download_nasa_images(out, manifest):
    d = out / "images"
    d.mkdir(parents=True, exist_ok=True)
    seen = set()
    for query, n in NASA_QUERIES:
        params = urllib.parse.urlencode({"q": query, "media_type": "image", "page_size": n * 2})
        items = json.loads(fetch(f"https://images-api.nasa.gov/search?{params}"))["collection"]["items"]
        count = 0
        for item in items:
            if count >= n:
                break
            meta, links = item["data"][0], item.get("links", [])
            nasa_id = meta["nasa_id"]
            if nasa_id in seen or not links:
                continue
            thumb = links[0]["href"]
            img = None
            for url in (thumb.replace("~thumb", "~medium"), thumb):
                try:
                    img = fetch(urllib.parse.quote(url, safe=":/~"))
                    break
                except Exception:
                    continue
            if not img:
                continue
            seen.add(nasa_id)
            path = d / f"{slug(nasa_id)}.jpg"
            path.write_bytes(img)
            manifest.append({
                "type": "image", "path": str(path.relative_to(out)), "title": meta.get("title", ""),
                "description": meta.get("description", ""), "keywords": meta.get("keywords", []),
                "date": meta.get("date_created", ""),
                "source": f"https://images.nasa.gov/details/{urllib.parse.quote(nasa_id)}",
                "license": "Public domain (NASA)",
            })
            count += 1
        print(f"  [images] '{query}': {count}")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw").resolve()
    wanted = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"text", "pdf", "image"}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    existing = []
    if manifest_path.exists():
        existing = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l]
    manifest = []
    for kind, name, fn in [("text", "Wikipedia", download_wikipedia), ("pdf", "arXiv", download_arxiv),
                           ("image", "NASA images", download_nasa_images)]:
        if kind not in wanted:
            continue
        print(f"== {name} ==")
        try:
            fn(out, manifest)
        except Exception as e:
            print(f"  {name} failed: {e}")
    manifest = [m for m in existing if m["type"] not in wanted] + manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    counts = {t: sum(m["type"] == t for m in manifest) for t in ("text", "pdf", "image")}
    print(f"\nDone: {counts} -> {out}")


if __name__ == "__main__":
    main()
