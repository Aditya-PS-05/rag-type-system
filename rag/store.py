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
                    "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    [(doc_id, c.index, c.page, c.content, np.asarray(c.embedding, dtype=np.float32))
                     for c in chunks])

    def search(self, embedding: list[float], *, team: str, embed_model: str, k: int = 6,
               doc_types: list[str] | None = None) -> list[SearchHit]:
        with self.conn.transaction():
            self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = self.conn.execute(
                """SELECT c.content, 1 - (c.embedding <=> %(q)s) AS score, d.doc_type, d.title,
                          d.source, d.path, c.page
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE d.team = %(team)s AND d.embed_model = %(model)s
                     AND (%(types)s::text[] IS NULL OR d.doc_type = ANY(%(types)s::text[]))
                   ORDER BY c.embedding <=> %(q)s
                   LIMIT %(k)s""",
                {"q": np.asarray(embedding, dtype=np.float32),
                 "team": team, "model": embed_model, "types": doc_types, "k": k},
            ).fetchall()
        return [SearchHit(*r) for r in rows]

    def list_documents(self, team: str) -> list[tuple[str, str]]:
        return self.conn.execute(
            "SELECT doc_type, title FROM documents WHERE team=%s ORDER BY doc_type, title", (team,)).fetchall()

    def stats(self, team: str) -> list[tuple]:
        return self.conn.execute(
            """SELECT d.doc_type, count(DISTINCT d.id), count(c.id)
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               WHERE d.team=%s GROUP BY d.doc_type ORDER BY 1""", (team,)).fetchall()

