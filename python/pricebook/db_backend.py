"""Storage backend abstraction for PricebookDB.

Default: SQLiteBackend (zero dependencies).
Future: DuckDBBackend, PostgresBackend — same interface, swap one line.

    from pricebook.db_backend import SQLiteBackend
    backend = SQLiteBackend("my_book.db")
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract storage backend. All backends implement this interface."""

    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute SQL, return rows as list of dicts."""

    @abstractmethod
    def execute_many(self, sql: str, rows: list[tuple]) -> None:
        """Execute SQL for multiple parameter sets."""

    @abstractmethod
    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""

    @abstractmethod
    def create_table(self, name: str, columns: dict[str, str]) -> None:
        """Create a table with {column_name: sql_type} mapping."""

    @abstractmethod
    def drop_table(self, name: str) -> None:
        """Drop a table if it exists."""

    @abstractmethod
    def list_tables(self) -> list[str]:
        """List all user tables."""

    @abstractmethod
    def commit(self) -> None:
        """Commit current transaction."""

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""


class SQLiteBackend(StorageBackend):
    """SQLite backend — zero dependencies, file-based or in-memory."""

    def __init__(self, path: str = ":memory:"):
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
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

    def create_table(self, name: str, columns: dict[str, str]) -> None:
        cols = ", ".join(f"{k} {v}" for k, v in columns.items())
        self.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")

    def drop_table(self, name: str) -> None:
        self.execute(f"DROP TABLE IF EXISTS {name}")

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
