"""SQLite-backed short + long-term memory store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


class MemoryStore:
    """Persistent memory using SQLite full-text search."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(query, response, content='memories', content_rowid='id')
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, query, response)
                    VALUES (new.id, new.query, new.response);
                END
            """)
            conn.commit()

    def store(self, query: str, response: str, tags: list[str] | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories (query, response, tags, created_at) VALUES (?, ?, ?, ?)",
                (query, response, json.dumps(tags or []), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT m.query, m.response FROM memories_fts
                    JOIN memories m ON memories_fts.rowid = m.id
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, top_k),
                ).fetchall()
            return [f"Q: {r[0][:100]}\nA: {r[1][:200]}" for r in rows]
        except Exception:
            return []

    def recent(self, n: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT query, response, created_at FROM memories ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"query": r[0], "response": r[1], "created_at": r[2]} for r in rows]

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories")
            conn.commit()
