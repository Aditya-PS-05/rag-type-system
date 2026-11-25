from rag.chunking import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("Hello world.\n\nSecond paragraph.") == ["Hello world.\n\nSecond paragraph."]


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(30))
    chunks = chunk_text(text, max_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_carries_previous_tail():
    text = "\n\n".join(f"Para{i} " + "alpha " * 20 for i in range(8))
    chunks = chunk_text(text, max_chars=300, overlap=80)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.split("\n\n")[0] in prev


def test_long_paragraph_split_on_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, max_chars=400, overlap=0)
    assert all(len(c) <= 400 for c in chunks)
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_unbroken_text_is_hard_split():
    chunks = chunk_text("x" * 2500, max_chars=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_empty_text():
    assert chunk_text("   \n\n  ") == []
