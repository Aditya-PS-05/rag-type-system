"""Postgres + pgvector storage. The only module that talks to the vector database."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


@dataclass
class Chunk:
    index: int
    page: int | None
    content: str
    embedding: list[float]


@dataclass
class SearchHit:
    content: str
    score: float
    doc_type: str
    title: str
    source: str
    path: str
    page: int | None
    chunk_id: int = 0
    document_id: int = 0
    chunk_index: int = 0


_HIT_COLUMNS = "c.content, {score} AS score, d.doc_type, d.title, d.source, d.path, c.page, c.id, d.id, c.chunk_index"


def content_hash(sections) -> str:
    return hashlib.sha256(json.dumps(sections, ensure_ascii=False).encode()).hexdigest()


class VectorStore:
    def __init__(self, database_url: str, embed_dim: int):
        self.conn = psycopg.connect(database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.embed_dim = embed_dim

    def init_schema(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id           bigserial PRIMARY KEY,
                team         text NOT NULL,
                doc_type     text NOT NULL,
                path         text NOT NULL,
                title        text NOT NULL,
                source       text,
                license      text,
                meta         jsonb NOT NULL DEFAULT '{{}}',
                content_hash text NOT NULL,
                embed_model  text NOT NULL,
                ingested_at  timestamptz NOT NULL DEFAULT now(),
                UNIQUE (team, path)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id          bigserial PRIMARY KEY,
                document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index int NOT NULL,
                page        int,
                content     text NOT NULL,
                embedding   vector({self.embed_dim}) NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);
            CREATE INDEX IF NOT EXISTS documents_team_type ON documents (team, doc_type);
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector;
            UPDATE chunks c SET tsv = to_tsvector('english', d.title || ' ' || c.content)
                FROM documents d WHERE d.id = c.document_id AND c.tsv IS NULL;
            CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
        """)

    def is_current(self, team: str, path: str, chash: str, embed_model: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE team=%s AND path=%s AND content_hash=%s AND embed_model=%s",
            (team, path, chash, embed_model)).fetchone()
        return row is not None

    def upsert_document(self, *, team, doc_type, path, title, source, license, meta,
                        chash, embed_model, chunks: list[Chunk]):
        with self.conn.transaction():
            self.conn.execute("DELETE FROM documents WHERE team=%s AND path=%s", (team, path))
            doc_id = self.conn.execute(
                """INSERT INTO documents (team, doc_type, path, title, source, license, meta,
                                          content_hash, embed_model)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (team, doc_type, path, title, source, license, json.dumps(meta), chash, embed_model),
            ).fetchone()[0]
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, page, content, embedding, tsv) "
                    "VALUES (%s,%s,%s,%s,%s, to_tsvector('english', %s || ' ' || %s))",
                    [(doc_id, c.index, c.page, c.content, np.asarray(c.embedding, dtype=np.float32), title, c.content)
                     for c in chunks])

    def search(self, embedding: list[float], *, team: str, embed_model: str, k: int = 6,
               doc_types: list[str] | None = None) -> list[SearchHit]:
        with self.conn.transaction():
            self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = self.conn.execute(
                f"""SELECT {_HIT_COLUMNS.format(score="1 - (c.embedding <=> %(q)s)")}
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE d.team = %(team)s AND d.embed_model = %(model)s
                     AND (%(types)s::text[] IS NULL OR d.doc_type = ANY(%(types)s::text[]))
                   ORDER BY c.embedding <=> %(q)s
                   LIMIT %(k)s""",
                {"q": np.asarray(embedding, dtype=np.float32),
                 "team": team, "model": embed_model, "types": doc_types, "k": k},
            ).fetchall()
        return [SearchHit(*r) for r in rows]

    def keyword_search(self, query: str, *, team: str, embed_model: str, k: int = 30,
                       doc_types: list[str] | None = None) -> list[SearchHit]:
        """Full-text search; any query term may match (OR), ranked by term density and proximity."""
        rows = self.conn.execute(
            f"""WITH q AS (SELECT NULLIF(replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | '), '')::tsquery AS q)
               SELECT {_HIT_COLUMNS.format(score="ts_rank_cd(c.tsv, q.q)")}
               FROM q, chunks c JOIN documents d ON d.id = c.document_id
               WHERE q.q IS NOT NULL AND c.tsv @@ q.q
                 AND d.team = %(team)s AND d.embed_model = %(model)s
                 AND (%(types)s::text[] IS NULL OR d.doc_type = ANY(%(types)s::text[]))
               ORDER BY score DESC
               LIMIT %(k)s""",
            {"q": query, "team": team, "model": embed_model, "types": doc_types, "k": k},
        ).fetchall()
        return [SearchHit(*r) for r in rows]

    def chunk_neighbors(self, keys: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
        """Content of the chunks immediately before and after each (document_id, chunk_index)."""
        if not keys:
            return {}
        wanted = {(d, i + off) for d, i in keys for off in (-1, 1)}
        rows = self.conn.execute(
            """SELECT document_id, chunk_index, content FROM chunks
               WHERE (document_id, chunk_index) IN (SELECT * FROM unnest(%s::bigint[], %s::int[]))""",
            ([d for d, _ in wanted], [i for _, i in wanted]),
        ).fetchall()
        return {(d, i): content for d, i, content in rows}

    def corpus_fingerprint(self, team: str, embed_model: str) -> str:
        """Changes whenever a document is added, removed, edited or re-chunked."""
        rows = self.conn.execute(
            """SELECT d.path, d.content_hash, count(c.id) FROM documents d
               LEFT JOIN chunks c ON c.document_id = d.id
               WHERE d.team=%s AND d.embed_model=%s GROUP BY d.id ORDER BY d.path""",
            (team, embed_model)).fetchall()
        digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]
        return f"{len(rows)} docs, {sum(r[2] for r in rows)} chunks, {digest}"

    def list_documents(self, team: str) -> list[tuple[str, str]]:
        return self.conn.execute(
            "SELECT doc_type, title FROM documents WHERE team=%s ORDER BY doc_type, title", (team,)).fetchall()

    def stats(self, team: str) -> list[tuple]:
        return self.conn.execute(
            """SELECT d.doc_type, count(DISTINCT d.id), count(c.id)
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               WHERE d.team=%s GROUP BY d.doc_type ORDER BY 1""", (team,)).fetchall()

