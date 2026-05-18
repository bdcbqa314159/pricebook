"""Tests for db.db_backend: SQLiteBackend."""
import pytest
from pricebook.db.db_backend import SQLiteBackend

class TestSQLiteBackend:
    def test_execute(self):
        db = SQLiteBackend(":memory:")
        db.execute("CREATE TABLE t (x INTEGER)")
        db.execute("INSERT INTO t VALUES (42)")
        rows = db.execute("SELECT x FROM t")
        assert rows[0]["x"] == 42

    def test_table_exists(self):
        db = SQLiteBackend(":memory:")
        assert not db.table_exists("t")
        db.execute("CREATE TABLE t (x INTEGER)")
        assert db.table_exists("t")
