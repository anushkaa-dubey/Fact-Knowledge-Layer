"""
SQLite storage layer.

Deliberately plain sqlite3 (no ORM) so the schema is easy to read and the
whole project stays inspectable in one sitting. Swap for Postgres later by
replacing this module only -- nothing else in the app talks to the DB file
directly.
"""
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("FACTLAYER_DB", "factlayer.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    page_count INTEGER,
    status TEXT NOT NULL DEFAULT 'processing'  -- processing | done | failed
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    page INTEGER,
    subject TEXT,
    metric TEXT,
    value TEXT,          -- kept as text; may be numeric or descriptive
    unit TEXT,
    period TEXT,
    scope TEXT,
    quote TEXT NOT NULL,
    confidence REAL,
    -- normalized fields, filled in by the normalization pass
    canonical_subject TEXT,
    canonical_metric TEXT,
    canonical_period TEXT,
    normalized_value REAL,
    normalized_unit TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a INTEGER NOT NULL REFERENCES facts(id),
    fact_id_b INTEGER NOT NULL REFERENCES facts(id),
    type TEXT NOT NULL,       -- corroborates | contradicts | reconciled
    explanation TEXT,
    confidence REAL
);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER REFERENCES documents(id),
    stage TEXT,               -- extraction | extraction_filtered | normalization | reasoning | pipeline
    detail TEXT,
    raw_snippet TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_document(filename: str, uploaded_at: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, uploaded_at, status) VALUES (?, ?, 'processing')",
            (filename, uploaded_at),
        )
        return cur.lastrowid


def update_document_status(doc_id: int, status: str, page_count: Optional[int] = None):
    with get_conn() as conn:
        if page_count is not None:
            conn.execute(
                "UPDATE documents SET status = ?, page_count = ? WHERE id = ?",
                (status, page_count, doc_id),
            )
        else:
            conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))


def insert_fact(doc_id: int, fact: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO facts
               (doc_id, page, subject, metric, value, unit, period, scope, quote, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                fact.get("page"),
                fact.get("subject"),
                fact.get("metric"),
                str(fact.get("value")),
                fact.get("unit"),
                fact.get("period"),
                fact.get("scope"),
                fact.get("quote"),
                fact.get("confidence"),
            ),
        )
        return cur.lastrowid


def update_fact_normalization(fact_id: int, canonical_subject: str, canonical_metric: str,
                               canonical_period: str, normalized_value: Optional[float],
                               normalized_unit: Optional[str]):
    with get_conn() as conn:
        conn.execute(
            """UPDATE facts SET canonical_subject=?, canonical_metric=?, canonical_period=?,
               normalized_value=?, normalized_unit=? WHERE id=?""",
            (canonical_subject, canonical_metric, canonical_period, normalized_value,
             normalized_unit, fact_id),
        )


def insert_relationship(fact_id_a: int, fact_id_b: int, rel_type: str,
                         explanation: str, confidence: float) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO relationships (fact_id_a, fact_id_b, type, explanation, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (fact_id_a, fact_id_b, rel_type, explanation, confidence),
        )
        return cur.lastrowid


def insert_failure(doc_id: Optional[int], stage: str, detail: str, raw_snippet: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO failures (doc_id, stage, detail, raw_snippet) VALUES (?, ?, ?, ?)",
            (doc_id, stage, detail, raw_snippet),
        )


def get_document(doc_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None


def list_documents() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def get_facts_for_doc(doc_id: int) -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM facts WHERE doc_id=? ORDER BY page", (doc_id,)).fetchall()
        return [dict(r) for r in rows]


def get_all_facts() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM facts").fetchall()
        return [dict(r) for r in rows]


def get_fact(fact_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        return dict(row) if row else None


def get_relationships(rel_type: Optional[str] = None) -> list:
    with get_conn() as conn:
        if rel_type:
            rows = conn.execute("SELECT * FROM relationships WHERE type=?", (rel_type,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM relationships").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["fact_a"] = get_fact(d["fact_id_a"])
            d["fact_b"] = get_fact(d["fact_id_b"])
            result.append(d)
        return result


def get_failures(stage: Optional[str] = None) -> list:
    with get_conn() as conn:
        if stage:
            rows = conn.execute("SELECT * FROM failures WHERE stage=?", (stage,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM failures").fetchall()
        return [dict(r) for r in rows]
