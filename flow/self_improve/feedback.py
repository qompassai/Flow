"""Session feedback collection and storage."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class FeedbackStore:
    """Store user feedback on agent sessions for self-improvement."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    prompt_used TEXT,
                    outcome TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def record(
        self,
        session_id: str,
        rating: int,
        comment: str = "",
        prompt_used: str = "",
        outcome: str = "",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO feedback
                (session_id, rating, comment, prompt_used, outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, rating, comment, prompt_used, outcome, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_low_rated(self, threshold: int = 3, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE rating <= ? ORDER BY created_at DESC LIMIT ?",
                (threshold, limit),
            ).fetchall()
        cols = ["id", "session_id", "rating", "comment", "prompt_used", "outcome", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def average_rating(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()
        return row[0] or 0.0
