"""Tests for PricebookDB: all 7 tables + custom tables + CSV."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from pricebook.db import PricebookDB


@pytest.fixture
def db():
    d = PricebookDB(":memory:")
    yield d
    d.close()


# ── Entities ──

class TestEntities:

    def test_save_load_entity(self, db):
        db.save_entity("JPM", legal_name="JPMorgan Chase & Co.",
                       entity_type="counterparty", sector="financials", country="US")
        e = db.load_entity("JPM")
        assert e["short_name"] == "JPM"
        assert e["legal_name"] == "JPMorgan Chase & Co."
        assert e["entity_type"] == "counterparty"
        assert e["sector"] == "financials"
        assert e["country"] == "US"

    def test_entity_upsert(self, db):
        db.save_entity("JPM", entity_type="counterparty")
        db.save_entity("JPM", legal_name="JPMorgan Updated", entity_type="counterparty")
        e = db.load_entity("JPM")
        assert e["legal_name"] == "JPMorgan Updated"

    def test_entity_attributes(self, db):
        db.save_entity("AAPL", entity_type="issuer",
                       attributes={"ticker": "AAPL", "gics": "45"})
        e = db.load_entity("AAPL")
        assert e["attributes"]["ticker"] == "AAPL"

    def test_list_entities_filtered(self, db):
        db.save_entity("JPM", entity_type="counterparty", sector="financials")
        db.save_entity("AAPL", entity_type="issuer", sector="tech")
        db.save_entity("GS", entity_type="counterparty", sector="financials")
        result = db.list_entities(entity_type="counterparty")
        assert len(result) == 2
        result = db.list_entities(sector="tech")
        assert len(result) == 1

    def test_load_missing_entity(self, db):
        assert db.load_entity("NONEXIST") is None


# ── Ratings ──

class TestRatings:

    def test_save_and_latest_rating(self, db):
        db.save_entity("JPM", entity_type="counterparty")
        db.save_rating("JPM", "SP", "A", date(2025, 1, 1))
        db.save_rating("JPM", "SP", "A+", date(2026, 3, 1), outlook="stable")
        assert db.latest_rating("JPM", "SP") == "A+"

    def test_rating_history(self, db):
        db.save_entity("JPM", entity_type="counterparty")
        db.save_rating("JPM", "SP", "A-", date(2024, 1, 1))
        db.save_rating("JPM", "SP", "A", date(2025, 1, 1))
        db.save_rating("JPM", "Moodys", "Aa3", date(2025, 6, 1))
        history = db.rating_history("JPM")
        assert len(history) == 3

    def test_missing_rating(self, db):
        assert db.latest_rating("NONE", "SP") is None


# ── Trades ──

class TestTrades:

    def test_save_load_trade_dict(self, db):
        inst = {"type": "irs", "params": {"notional": 10e6, "rate": 0.05}}
        db.save_trade("irs_001", inst, book="rates", direction="pay",
                      notional=10e6, currency="USD")
        raw = db.load_trade_raw("irs_001")
        assert raw["book"] == "rates"
        assert raw["direction"] == "pay"
        assert raw["notional"] == 10e6
        assert raw["instrument"]["type"] == "irs"

    def test_save_trade_with_to_dict(self, db):
        from pricebook.deposit import Deposit
        dep = Deposit(date(2026, 1, 1), date(2026, 7, 1), 0.05, 1e6)
        db.save_trade("dep_001", dep, book="mm")
        raw = db.load_trade_raw("dep_001")
        assert raw["instrument_type"] == "deposit"

    def test_list_trades_filtered(self, db):
        db.save_trade("t1", {"type": "irs"}, book="rates", desk="flow", currency="USD")
        db.save_trade("t2", {"type": "cds"}, book="credit", desk="flow", currency="EUR")
        db.save_trade("t3", {"type": "irs"}, book="rates", desk="prop", currency="USD")
        assert len(db.list_trades(book="rates")) == 2
        assert len(db.list_trades(currency="EUR")) == 1
        assert len(db.list_trades(desk="flow")) == 2

    def test_update_trade_status(self, db):
        db.save_trade("t1", {"type": "irs"})
        db.update_trade_status("t1", "terminated")
        raw = db.load_trade_raw("t1")
        assert raw["status"] == "terminated"

    def test_trade_with_counterparty(self, db):
        db.save_entity("JPM", entity_type="counterparty")
        db.save_trade("t1", {"type": "irs"}, counterparty_id="JPM")
        trades = db.list_trades(counterparty_id="JPM")
        assert len(trades) == 1

    def test_trade_tags(self, db):
        db.save_trade("t1", {"type": "irs"}, tags={"strategy": "hedge", "hedge_of": "bond_1"})
        raw = db.load_trade_raw("t1")
        assert raw["tags"]["strategy"] == "hedge"


# ── Market Snapshots ──

class TestMarketSnapshots:

    def test_save_load_snapshot(self, db):
        data = {"type": "flat_curve", "rate": 0.05}
        db.save_snapshot(date(2026, 5, 11), "discount_curve", "OIS_USD", data)
        loaded = db.load_snapshot_raw(date(2026, 5, 11), "discount_curve", "OIS_USD")
        assert loaded["data"]["rate"] == 0.05

    def test_list_snapshots(self, db):
        db.save_snapshot(date(2026, 5, 10), "discount_curve", "OIS_USD", {"r": 0.05})
        db.save_snapshot(date(2026, 5, 11), "discount_curve", "OIS_USD", {"r": 0.051})
        db.save_snapshot(date(2026, 5, 11), "vol_surface", "SPX", {"v": 0.20})
        snaps = db.list_snapshots(date(2026, 5, 10), date(2026, 5, 11))
        assert len(snaps) == 3


# ── Pricing Results ──

class TestPricingResults:

    def test_save_and_history(self, db):
        db.save_trade("t1", {"type": "irs"})
        db.save_result("t1", date(2026, 5, 10), {"price": 100.5, "dv01": 45})
        db.save_result("t1", date(2026, 5, 11), {"price": 101.2, "dv01": 44})
        history = db.result_history("t1")
        assert len(history) == 2
        assert history[0]["result"]["price"] == 100.5
        assert history[1]["result"]["price"] == 101.2


# ── P&L History ──

class TestPnLHistory:

    def test_save_and_series(self, db):
        db.save_pnl("t1", date(2026, 5, 10), pv=1000, pnl_1d=10, dv01=45)
        db.save_pnl("t1", date(2026, 5, 11), pv=1012, pnl_1d=12, dv01=44)
        series = db.pnl_series("t1")
        assert len(series) == 2
        assert series[0]["pv"] == 1000
        assert series[1]["pnl_1d"] == 12

    def test_pnl_upsert(self, db):
        db.save_pnl("t1", date(2026, 5, 10), pv=1000)
        db.save_pnl("t1", date(2026, 5, 10), pv=1001)
        series = db.pnl_series("t1")
        assert len(series) == 1
        assert series[0]["pv"] == 1001


# ── Key-Value Store ──

class TestKVStore:

    def test_put_get(self, db):
        db.put("config", "default_recovery", 0.4)
        assert db.get("config", "default_recovery") == 0.4

    def test_get_missing(self, db):
        assert db.get("config", "nope") is None
        assert db.get("config", "nope", default=42) == 42

    def test_put_complex(self, db):
        db.put("notes", "irs_001", {"text": "Hedging bond portfolio", "reviewed": True})
        val = db.get("notes", "irs_001")
        assert val["text"] == "Hedging bond portfolio"

    def test_list_keys(self, db):
        db.put("config", "a", 1)
        db.put("config", "b", 2)
        db.put("other", "c", 3)
        assert db.list_keys("config") == ["a", "b"]


# ── Custom Tables ──

class TestCustomTables:

    def test_save_and_load_2_columns(self, db):
        db.save_table("spreads", [
            {"issuer": "JPM", "spread": 0.0045},
            {"issuer": "AAPL", "spread": 0.0032},
        ])
        rows = db.load_table("spreads")
        assert len(rows) == 2
        assert rows[0]["issuer"] == "AAPL" or rows[1]["issuer"] == "AAPL"

    def test_save_and_load_6_columns(self, db):
        db.save_table("snapshot", [
            {"date": "2026-05-11", "issuer": "JPM", "rating": "A+",
             "spread": 45, "notional": 10e6, "sector": "fin"},
            {"date": "2026-05-11", "issuer": "GS", "rating": "A",
             "spread": 50, "notional": 5e6, "sector": "fin"},
        ])
        rows = db.load_table("snapshot")
        assert len(rows) == 2
        assert all(r["sector"] == "fin" for r in rows)

    def test_append_rows(self, db):
        db.save_table("data", [{"x": 1, "y": "a"}])
        db.append_rows("data", [{"x": 2, "y": "b"}])
        assert len(db.load_table("data")) == 2

    def test_query_table_filtered(self, db):
        db.save_table("portfolio", [
            {"name": "A", "sector": "tech", "weight": 0.3},
            {"name": "B", "sector": "fin", "weight": 0.5},
            {"name": "C", "sector": "tech", "weight": 0.2},
        ])
        tech = db.query_table("portfolio", sector="tech")
        assert len(tech) == 2

    def test_load_table_df(self, db):
        db.save_table("data", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        df = db.load_table_df("data")
        assert len(df) == 2
        assert list(df.columns) == ["x", "y"]

    def test_list_custom_tables(self, db):
        db.save_table("my_data", [{"a": 1}])
        db.save_table("other", [{"b": 2}])
        custom = db.list_custom_tables()
        assert "my_data" in custom
        assert "other" in custom
        assert "trades" not in custom

    def test_drop_custom_table(self, db):
        db.save_table("temp", [{"x": 1}])
        db.drop_table("temp")
        assert "temp" not in db.list_custom_tables()

    def test_cannot_drop_system_table(self, db):
        with pytest.raises(ValueError, match="system table"):
            db.drop_table("trades")

    def test_cannot_overwrite_system_table(self, db):
        with pytest.raises(ValueError, match="system table"):
            db.save_table("trades", [{"x": 1}])

    def test_replace_table(self, db):
        db.save_table("data", [{"x": 1}])
        db.save_table("data", [{"x": 2}, {"x": 3}], replace=True)
        assert len(db.load_table("data")) == 2


# ── CSV Import/Export ──

class TestCSV:

    def test_export_import_roundtrip(self, db):
        db.save_table("spreads", [
            {"issuer": "JPM", "spread": 0.0045},
            {"issuer": "AAPL", "spread": 0.0032},
        ])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            db.export_csv("spreads", path)
            # Verify CSV content
            with open(path) as f:
                content = f.read()
            assert "JPM" in content
            assert "0.0045" in content

            # Import into new table
            count = db.import_csv("spreads_copy", path)
            assert count == 2
            rows = db.load_table("spreads_copy")
            assert len(rows) == 2
        finally:
            os.unlink(path)

    def test_export_system_table(self, db):
        db.save_entity("JPM", entity_type="counterparty")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            db.export_csv("entities", path)
            with open(path) as f:
                content = f.read()
            assert "JPM" in content
        finally:
            os.unlink(path)


# ── Context Manager ──

class TestContextManager:

    def test_with_statement(self):
        with PricebookDB(":memory:") as db:
            db.save_entity("TEST", entity_type="issuer")
            assert db.load_entity("TEST") is not None


# ── File Persistence ──

# ── SQL Injection Defense ──

class TestSQLInjection:

    def test_malicious_table_name_rejected(self, db):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            db.save_table("trades; DROP TABLE entities", [{"x": 1}])

    def test_malicious_column_name_rejected(self, db):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            db.save_table("safe", [{"valid": 1, "1=1; --": 2}])

    def test_malicious_filter_key_rejected(self, db):
        db.save_table("data", [{"x": 1}])
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            db.load_table("data", **{"1=1; --": "x"})


# ── Delete Rows ──

class TestDeleteRows:

    def test_delete_rows(self, db):
        db.save_table("data", [
            {"name": "A", "val": 1},
            {"name": "B", "val": 2},
            {"name": "C", "val": 1},
        ])
        db.delete_rows("data", val=1)
        remaining = db.load_table("data")
        assert len(remaining) == 1
        assert remaining[0]["name"] == "B"

    def test_delete_requires_filter(self, db):
        db.save_table("data", [{"x": 1}])
        with pytest.raises(ValueError, match="requires at least one filter"):
            db.delete_rows("data")


# ── Trade Reconstruction ──

class TestTradeReconstruction:

    def test_load_trade_reconstructs_instrument(self, db):
        from pricebook.deposit import Deposit
        dep = Deposit(date(2026, 1, 1), date(2026, 7, 1), 0.05, 1e6)
        db.save_trade("dep_001", dep)
        loaded = db.load_trade("dep_001")
        assert loaded is not None
        assert hasattr(loaded, "rate")

    def test_load_missing_trade(self, db):
        assert db.load_trade("nonexist") is None


# ── File Persistence ──

class TestFilePersistence:

    def test_data_survives_reopen(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            with PricebookDB(path) as db:
                db.save_entity("JPM", entity_type="counterparty")
                db.save_table("test", [{"x": 42}])

            with PricebookDB(path) as db:
                assert db.load_entity("JPM") is not None
                assert db.load_table("test")[0]["x"] == 42
        finally:
            os.unlink(path)
