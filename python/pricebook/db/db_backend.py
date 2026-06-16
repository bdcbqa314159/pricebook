"""SQLite storage backend for PricebookDB.

    from pricebook.db.db_backend import SQLiteBackend
    backend = SQLiteBackend("my_book.db")
"""

from __future__ import annotations

import re
import sqlite3


_VALID_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _safe_name(name: str) -> str:
    """Validate and return a safe SQL identifier. Raises on injection attempts."""
    if not _VALID_NAME.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


class SQLiteBackend:
    """SQLite backend — zero dependencies, file-based or in-memory."""

    def __init__(self, path: str = ":memory:"):
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        cursor = self._conn.execute(sql, params)
        if cursor.description is None:
            return []
        return [dict(row) for row in cursor.fetchall()]

    def execute_many(self, sql: str, rows: list[tuple]) -> None:
        self._conn.executemany(sql, rows)

    def table_exists(self, name: str) -> bool:
        rows = self.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return len(rows) > 0

    _VALID_SQL_TYPES = {"TEXT", "REAL", "INTEGER", "BLOB", "NUMERIC"}

    def create_table(self, name: str, columns: dict[str, str]) -> None:
        safe = _safe_name(name)
        parts = []
        for k, v in columns.items():
            if v.upper() not in self._VALID_SQL_TYPES:
                raise ValueError(f"Invalid SQL type: {v!r}")
            parts.append(f"{_safe_name(k)} {v}")
        self.execute(f"CREATE TABLE IF NOT EXISTS {safe} ({', '.join(parts)})")

    def drop_table(self, name: str) -> None:
        self.execute(f"DROP TABLE IF EXISTS {_safe_name(name)}")

    def list_tables(self) -> list[str]:
        rows = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r["name"] for r in rows]

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn
