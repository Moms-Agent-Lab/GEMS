"""
ExperienceDB — Cross-session persistent experience database.

Stores workflow topologies, verifier scores, and compressed lessons in a
SQLite database so that knowledge from previous sessions can warm-start
new runs.

Key features:
- Workflow topology fingerprinting for retrieval-by-similarity.
- Best-topology warm-start: for a new prompt, retrieve the highest-scoring
  topology from a semantically similar past prompt.
- Skill usage tracking: which skills contributed to high scores.

Usage::

    db = ExperienceDB("experiences.db")
    db.record(prompt="a red fox", score=0.85, topology={...}, ...)
    warm_start = db.get_warm_start("a golden retriever")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """One experience record from the database."""

    id: int
    prompt: str
    score: float
    topology_hash: str
    topology_json: str
    skills_used: list[str]
    node_count: int
    experience_text: str
    created_at: float
    session_id: str


class ExperienceDB:
    """SQLite-backed cross-session experience database.

    Parameters
    ----------
    db_path   : Path to the SQLite database file.
    session_id : Unique identifier for the current session (auto-generated
                 if not provided).
    """

    def __init__(
        self,
        db_path: str | Path = "comfyclaw_experiences.db",
        session_id: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.session_id = session_id or f"session_{int(time.time())}"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                score REAL NOT NULL,
                topology_hash TEXT NOT NULL,
                topology_json TEXT NOT NULL,
                skills_used TEXT DEFAULT '[]',
                node_count INTEGER DEFAULT 0,
                experience_text TEXT DEFAULT '',
                passed_requirements TEXT DEFAULT '[]',
                failed_requirements TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                session_id TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_score ON experiences(score DESC);
            CREATE INDEX IF NOT EXISTS idx_prompt ON experiences(prompt);
            CREATE INDEX IF NOT EXISTS idx_topology ON experiences(topology_hash);
            CREATE INDEX IF NOT EXISTS idx_session ON experiences(session_id);

            CREATE TABLE IF NOT EXISTS skill_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                score REAL NOT NULL,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_skill_name
                ON skill_performance(skill_name);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        prompt: str,
        score: float,
        topology: dict,
        skills_used: list[str] | None = None,
        experience_text: str = "",
        passed: list[str] | None = None,
        failed: list[str] | None = None,
    ) -> int:
        """Record an experience entry. Returns the row ID."""
        skills_used = skills_used or []
        passed = passed or []
        failed = failed or []
        topo_json = json.dumps(topology, sort_keys=True)
        topo_hash = self._topology_hash(topology)
        node_count = len(topology)

        cursor = self._conn.execute(
            """INSERT INTO experiences
               (prompt, score, topology_hash, topology_json, skills_used,
                node_count, experience_text, passed_requirements,
                failed_requirements, created_at, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prompt, score, topo_hash, topo_json,
                json.dumps(skills_used), node_count, experience_text,
                json.dumps(passed), json.dumps(failed),
                time.time(), self.session_id,
            ),
        )
        self._conn.commit()

        for skill in skills_used:
            self._conn.execute(
                """INSERT INTO skill_performance
                   (skill_name, prompt, score, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (skill, prompt, score, self.session_id, time.time()),
            )
        self._conn.commit()

        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Topology retrieval
    # ------------------------------------------------------------------

    def get_warm_start(
        self,
        prompt: str,
        min_score: float = 0.5,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve high-scoring topologies similar to the given prompt.

        Uses keyword overlap as a lightweight similarity proxy (no
        embedding model required). Returns topology dicts sorted by score.
        """
        keywords = set(prompt.lower().split())
        if not keywords:
            return []

        rows = self._conn.execute(
            """SELECT DISTINCT topology_hash, prompt, score, topology_json, skills_used
               FROM experiences
               WHERE score >= ?
               ORDER BY score DESC
               LIMIT 100""",
            (min_score,),
        ).fetchall()

        candidates: list[tuple[float, dict]] = []
        seen_hashes: set[str] = set()
        for row in rows:
            if row["topology_hash"] in seen_hashes:
                continue
            seen_hashes.add(row["topology_hash"])

            row_keywords = set(row["prompt"].lower().split())
            overlap = len(keywords & row_keywords) / max(len(keywords | row_keywords), 1)
            relevance = 0.5 * row["score"] + 0.5 * overlap

            candidates.append((
                relevance,
                {
                    "prompt": row["prompt"],
                    "score": row["score"],
                    "topology": json.loads(row["topology_json"]),
                    "skills_used": json.loads(row["skills_used"]),
                    "relevance": relevance,
                },
            ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:top_k]]

    def get_best_topology(self, prompt: str) -> dict | None:
        """Retrieve the single best-scoring topology for exactly this prompt."""
        row = self._conn.execute(
            """SELECT topology_json FROM experiences
               WHERE prompt = ?
               ORDER BY score DESC LIMIT 1""",
            (prompt,),
        ).fetchone()
        if row:
            return json.loads(row["topology_json"])
        return None

    # ------------------------------------------------------------------
    # Skill performance analytics
    # ------------------------------------------------------------------

    def get_skill_stats(self) -> dict[str, dict[str, float]]:
        """Return {skill_name: {mean_score, usage_count, best_score}}."""
        rows = self._conn.execute(
            """SELECT skill_name,
                      AVG(score) as mean_score,
                      COUNT(*) as usage_count,
                      MAX(score) as best_score
               FROM skill_performance
               GROUP BY skill_name
               ORDER BY mean_score DESC"""
        ).fetchall()
        return {
            row["skill_name"]: {
                "mean_score": row["mean_score"],
                "usage_count": row["usage_count"],
                "best_score": row["best_score"],
            }
            for row in rows
        }

    def get_session_summary(self, session_id: str | None = None) -> dict:
        """Summary stats for a session (defaults to current)."""
        sid = session_id or self.session_id
        rows = self._conn.execute(
            """SELECT COUNT(*) as count,
                      AVG(score) as mean_score,
                      MAX(score) as best_score,
                      MIN(score) as worst_score
               FROM experiences WHERE session_id = ?""",
            (sid,),
        ).fetchone()
        return dict(rows) if rows else {}

    def get_all_sessions(self) -> list[dict]:
        """List all sessions with summary stats."""
        rows = self._conn.execute(
            """SELECT session_id,
                      COUNT(*) as count,
                      AVG(score) as mean_score,
                      MAX(score) as best_score,
                      MIN(created_at) as started_at
               FROM experiences
               GROUP BY session_id
               ORDER BY started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _topology_hash(topology: dict) -> str:
        """Deterministic hash of the workflow graph structure (ignoring scalar params)."""
        structure: list[tuple[str, list[str]]] = []
        for nid in sorted(topology.keys()):
            node = topology[nid]
            ct = node.get("class_type", "?")
            connections = []
            for inp_name, val in sorted(node.get("inputs", {}).items()):
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                    src_ct = topology.get(str(val[0]), {}).get("class_type", "?")
                    connections.append(f"{inp_name}->{src_ct}[{val[1]}]")
            structure.append((ct, connections))
        sig = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExperienceDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
