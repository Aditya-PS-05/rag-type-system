import json

from rag.loaders import clean_pdf_page, clean_wikipedia, load_manifest


def test_clean_wikipedia_drops_reference_sections_and_heading_markup():
    text = "# Mars\n\nIntro.\n\n== History ==\nOld.\n\n== See also ==\nLinks\n\n== References ==\nX"
    out = clean_wikipedia(text)
    assert "History\nOld." in out
    assert "See also" not in out and "References" not in out


def test_clean_pdf_page_joins_hyphenation_and_soft_wraps():
    assert clean_pdf_page("explo-\nration of the\nsurface.\n\nNext") == "exploration of the surface.\n\nNext"


def test_clean_pdf_page_strips_nul_and_control_chars():
    assert clean_pdf_page("ga\x00l\x07axy") == "galaxy"


def test_load_manifest_handles_text_and_image(tmp_path):
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "moon.txt").write_text("# Moon\n\nThe Moon orbits Earth.")
    entries = [
        {"type": "text", "path": "text/moon.txt", "title": "Moon", "source": "s", "license": "l"},
        {"type": "image", "path": "images/a.jpg", "title": "Apollo 11 launch", "source": "s",
         "license": "l", "description": "Saturn V lifts off.", "keywords": ["Apollo"], "date": "1969-07-16T00:00:00Z"},
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
    docs = list(load_manifest(tmp_path))
    assert [d.doc_type for d in docs] == ["text", "image"]
    assert "The Moon orbits Earth." in docs[0].sections[0][1]
    image_text = docs[1].sections[0][1]
    assert "Apollo 11 launch" in image_text and "1969-07-16" in image_text and "Saturn V" in image_text
