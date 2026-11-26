"""Integration tests against a real Postgres + pgvector (docker compose up -d)."""

import os
import uuid

import psycopg
import pytest

from rag.store import Chunk, VectorStore

DB_URL = os.getenv("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
DIM = 4


@pytest.fixture
def store():
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(DB_URL, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres not available")
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(f"CREATE SCHEMA {schema}")
    s = VectorStore(f"{DB_URL}?options=-csearch_path%3D{schema},public", DIM)
    s.init_schema()
    yield s
    s.conn.close()
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def add(store, team, path, doc_type, vec, chash="h1"):
    store.upsert_document(team=team, doc_type=doc_type, path=path, title=path, source="src",
                          license="l", meta={}, chash=chash, embed_model="m",
                          chunks=[Chunk(0, None, f"content of {path}", vec)])


def test_search_ranks_by_similarity_and_filters(store):
    add(store, "a", "near.txt", "text", [1, 0, 0, 0])
    add(store, "a", "far.txt", "text", [0, 1, 0, 0])
    add(store, "a", "img.jpg", "image", [0.9, 0.1, 0, 0])
    add(store, "b", "other-team.txt", "text", [1, 0, 0, 0])

    hits = store.search([1, 0, 0, 0], team="a", embed_model="m", k=3)
    assert [h.path for h in hits] == ["near.txt", "img.jpg", "far.txt"]
    assert hits[0].score > 0.99

    only_text = store.search([1, 0, 0, 0], team="a", embed_model="m", k=3, doc_types=["text"])
    assert {h.path for h in only_text} == {"near.txt", "far.txt"}

    assert store.search([1, 0, 0, 0], team="a", embed_model="other", k=3) == []


def test_reingest_replaces_chunks(store):
    add(store, "a", "doc.txt", "text", [1, 0, 0, 0], chash="v1")
    assert store.is_current("a", "doc.txt", "v1", "m")
    add(store, "a", "doc.txt", "text", [0, 1, 0, 0], chash="v2")
    assert not store.is_current("a", "doc.txt", "v1", "m")
    assert store.stats("a") == [("text", 1, 1)]


def test_keyword_search_matches_any_term_and_title(store):
    for path, title, text in [("j.txt", "Juno", "It arrived in 2016."), ("p.txt", "Pluto", "A dwarf planet.")]:
        store.upsert_document(team="a", doc_type="text", path=path, title=title, source="s", license="l", meta={},
                              chash="h", embed_model="m", chunks=[Chunk(0, None, text, [1, 0, 0, 0])])
    hits = store.keyword_search("when did juno arrive", team="a", embed_model="m")
    assert [h.path for h in hits] == ["j.txt"]
    assert hits[0].chunk_id and hits[0].document_id
    assert store.keyword_search("the of and", team="a", embed_model="m") == []


def test_chunk_neighbors_returns_adjacent_chunks(store):
    store.upsert_document(team="a", doc_type="text", path="d.txt", title="D", source="s", license="l", meta={},
                          chash="h", embed_model="m",
                          chunks=[Chunk(i, None, f"part {i}", [1, 0, 0, 0]) for i in range(4)])
    doc_id = store.search([1, 0, 0, 0], team="a", embed_model="m", k=1)[0].document_id
    assert store.chunk_neighbors([(doc_id, 2)]) == {(doc_id, 1): "part 1", (doc_id, 3): "part 3"}


def test_corpus_fingerprint_changes_when_documents_change(store):
    add(store, "a", "one.txt", "text", [1, 0, 0, 0])
    before = store.corpus_fingerprint("a", "m")
    assert before.startswith("1 docs, 1 chunks")
    assert store.corpus_fingerprint("a", "m") == before
    add(store, "a", "one.txt", "text", [1, 0, 0, 0], chash="h2")
    assert store.corpus_fingerprint("a", "m") != before
    assert store.corpus_fingerprint("b", "m").startswith("0 docs")
