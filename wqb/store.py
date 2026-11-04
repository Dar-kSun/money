"""Persistence for the automation loop.

SQLite holds every attempt so a re-run skips expressions already simulated
(Brain charges simulation quota, so repeats are pure waste), and winners are
additionally appended to a human-readable JSONL success file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable

from .config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    expression   TEXT PRIMARY KEY,
    template     TEXT,
    field        TEXT,
    window       INTEGER,
    status       TEXT,
    alpha_id     TEXT,
    sharpe       REAL,
    fitness      REAL,
    turnover     REAL,
    returns      REAL,
    drawdown     REAL,
    margin       REAL,
    passed       INTEGER DEFAULT 0,
    reason       TEXT,
    submitted    INTEGER DEFAULT 0,
    metrics_json TEXT,
    created_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_attempts_passed ON attempts(passed);
CREATE INDEX IF NOT EXISTS idx_attempts_sharpe ON attempts(sharpe DESC);
"""


class ResultStore:
    def __init__(self, path: str | Path | None = None) -> None:
        base = Path(path) if path else DATA_DIR / "alphas.db"
        base.parent.mkdir(parents=True, exist_ok=True)
        self.path = base
        self.success_file = base.parent / "successful_alphas.jsonl"
        # The automation loop writes from worker threads, so the connection must
        # not be thread-bound; a lock serialises access instead.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(base, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock, closing(self.conn.cursor()) as cur:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def _cursor(self, commit: bool = False):
        with self._lock:
            with closing(self.conn.cursor()) as cur:
                yield cur
            if commit:
                self.conn.commit()

    # ------------------------------------------------------------------ reads
    def seen(self) -> set[str]:
        """Expressions already simulated to a terminal state."""
        with self._cursor() as cur:
            cur.execute("SELECT expression FROM attempts WHERE status IS NOT NULL")
            return {row[0] for row in cur.fetchall()}

    def winners(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM attempts WHERE passed = 1 ORDER BY sharpe DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._cursor() as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]

    def top(self, limit: int = 20) -> list[dict[str, Any]]:
        """Sharpe leaderboard. Rows with no Sharpe (errored or metric-less
        simulations) are excluded; use :meth:`counts` or :meth:`seen` for those."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM attempts WHERE sharpe IS NOT NULL "
                "ORDER BY sharpe DESC LIMIT ?", (int(limit),)
            )
            return [dict(r) for r in cur.fetchall()]

    def counts(self) -> dict[str, int]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) total, "
                "COALESCE(SUM(passed), 0) passed, "
                "COALESCE(SUM(submitted), 0) submitted FROM attempts"
            )
            row = cur.fetchone()
            return {"total": row[0], "passed": row[1], "submitted": row[2]}

    # ----------------------------------------------------------------- writes
    def record(
        self,
        *,
        expression: str,
        template: str = "",
        field: str = "",
        window: int = 0,
        status: str,
        alpha_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        passed: bool = False,
        reason: str = "",
    ) -> None:
        metrics = metrics or {}
        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO attempts (expression, template, field, window, status, alpha_id,
                    sharpe, fitness, turnover, returns, drawdown, margin,
                    passed, reason, metrics_json, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(expression) DO UPDATE SET
                    status=excluded.status, alpha_id=excluded.alpha_id,
                    sharpe=excluded.sharpe, fitness=excluded.fitness,
                    turnover=excluded.turnover, returns=excluded.returns,
                    drawdown=excluded.drawdown, margin=excluded.margin,
                    passed=excluded.passed, reason=excluded.reason,
                    metrics_json=excluded.metrics_json
                """,
                (
                    expression, template, field, window, status, alpha_id,
                    _num(metrics.get("sharpe")), _num(metrics.get("fitness")),
                    _num(metrics.get("turnover")), _num(metrics.get("returns")),
                    _num(metrics.get("drawdown")), _num(metrics.get("margin")),
                    int(passed), reason, json.dumps(metrics, default=str), time.time(),
                ),
            )

        if passed:
            self._append_success(
                {
                    "expression": expression,
                    "template": template,
                    "field": field,
                    "window": window,
                    "alpha_id": alpha_id,
                    "metrics": metrics,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )

    def mark_submitted(self, expression: str, submitted: bool = True) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE attempts SET submitted = ? WHERE expression = ?",
                (int(submitted), expression),
            )
            self.conn.commit()

    def pending_submission(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM attempts WHERE passed = 1 AND submitted = 0 "
                "AND alpha_id IS NOT NULL ORDER BY sharpe DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def _append_success(self, record: dict[str, Any]) -> None:
        with self._lock, self.success_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
