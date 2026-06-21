"""PricebookDB: persistent storage for trades, entities, market data, results.

SQLite-backed with backend abstraction for future DuckDB/PostgreSQL upgrade.
Stores pricebook objects via to_dict/from_dict, plus custom reference data
(issuers, ratings, counterparties) and arbitrary user tables.

    from pricebook.db.db import PricebookDB

    db = PricebookDB("my_book.db")
    db.save_entity("JPM", legal_name="JPMorgan Chase & Co.",
                   entity_type="counterparty", sector="financials")
    db.save_trade("irs_001", my_swap, book="rates", counterparty_id="JPM")
    db.export_csv("issuer_spreads", "spreads.csv")
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pricebook.db.db_backend import SQLiteBackend, _safe_name

if TYPE_CHECKING:
    from uuid import UUID

    from pricebook.calibration import CalibrationResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _infer_sql_type(value) -> str:
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _serialise_value(v) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


class PricebookDB:
    """Main database interface for pricebook persistence."""

    # Fixed table names — protected from drop_table
    _SYSTEM_TABLES = {
        "entities", "ratings", "trades", "market_snapshots",
        "pricing_results", "pnl_history", "kv_store", "calibration_results",
    }

    def __init__(self, path: str = ":memory:", backend: SQLiteBackend | None = None):
        self._backend = backend or SQLiteBackend(path)
        self._init_schema()

    def _init_schema(self) -> None:
        b = self._backend
        b.execute("""CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            short_name TEXT NOT NULL,
            legal_name TEXT,
            entity_type TEXT NOT NULL,
            sector TEXT,
            country TEXT,
            lei TEXT,
            attributes TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        b.execute("""CREATE TABLE IF NOT EXISTS ratings (
            entity_id TEXT NOT NULL,
            agency TEXT NOT NULL,
            rating TEXT NOT NULL,
            outlook TEXT,
            effective_date TEXT NOT NULL,
            PRIMARY KEY (entity_id, agency, effective_date)
        )""")
        b.execute("""CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            instrument_type TEXT NOT NULL,
            instrument_json TEXT NOT NULL,
            book TEXT DEFAULT 'default',
            desk TEXT,
            counterparty_id TEXT,
            issuer_id TEXT,
            direction TEXT,
            notional REAL,
            currency TEXT,
            maturity_date TEXT,
            trader TEXT,
            status TEXT DEFAULT 'live',
            tags TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        b.execute("""CREATE TABLE IF NOT EXISTS market_snapshots (
            snapshot_date TEXT NOT NULL,
            snapshot_type TEXT NOT NULL,
            curve_name TEXT NOT NULL,
            data_json TEXT NOT NULL,
            source TEXT DEFAULT 'manual',
            created_at TEXT NOT NULL,
            PRIMARY KEY (snapshot_date, snapshot_type, curve_name)
        )""")
        b.execute("""CREATE TABLE IF NOT EXISTS pricing_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            result_json TEXT NOT NULL,
            method TEXT,
            computed_at TEXT NOT NULL
        )""")
        b.execute("""CREATE TABLE IF NOT EXISTS pnl_history (
            trade_id TEXT NOT NULL,
            valuation_date TEXT NOT NULL,
            pv REAL NOT NULL,
            pnl_1d REAL,
            delta REAL, gamma REAL, vega REAL, dv01 REAL, cs01 REAL,
            PRIMARY KEY (trade_id, valuation_date)
        )""")
        b.execute("""CREATE TABLE IF NOT EXISTS kv_store (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (namespace, key)
        )""")
        # Calibration results — the canonical CalibrationResult artefact.
        # Identity columns (model_class, timestamp, objective, converged,
        # rms/max residual, market_snapshot_id) are denormalised out of the
        # JSON blob so the audit chain is queryable without reconstructing.
        b.execute("""CREATE TABLE IF NOT EXISTS calibration_results (
            calibration_id TEXT PRIMARY KEY,
            model_class TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            code_version TEXT,
            objective TEXT,
            converged INTEGER,
            iterations INTEGER,
            rms_residual REAL,
            max_residual REAL,
            market_snapshot_id TEXT,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        # Indices for common queries
        b.execute("CREATE INDEX IF NOT EXISTS idx_trades_book ON trades(book, status)")
        b.execute("CREATE INDEX IF NOT EXISTS idx_trades_cpty ON trades(counterparty_id)")
        b.execute("CREATE INDEX IF NOT EXISTS idx_trades_issuer ON trades(issuer_id)")
        b.execute("CREATE INDEX IF NOT EXISTS idx_result_trade ON pricing_results(trade_id, snapshot_date)")
        b.execute("CREATE INDEX IF NOT EXISTS idx_snap_date ON market_snapshots(snapshot_date)")
        b.execute("CREATE INDEX IF NOT EXISTS idx_calib_model ON calibration_results(model_class)")
        b.execute("CREATE INDEX IF NOT EXISTS idx_calib_snapshot ON calibration_results(market_snapshot_id)")
        b.commit()

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def save_entity(
        self,
        entity_id: str,
        short_name: str | None = None,
        legal_name: str | None = None,
        entity_type: str = "issuer",
        sector: str | None = None,
        country: str | None = None,
        lei: str | None = None,
        attributes: dict | None = None,
    ) -> None:
        now = _now()
        self._backend.execute(
            """INSERT INTO entities
               (entity_id, short_name, legal_name, entity_type, sector, country, lei, attributes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(entity_id) DO UPDATE SET
               short_name=excluded.short_name, legal_name=excluded.legal_name,
               entity_type=excluded.entity_type, sector=excluded.sector,
               country=excluded.country, lei=excluded.lei,
               attributes=excluded.attributes, updated_at=excluded.updated_at""",
            (entity_id, short_name or entity_id, legal_name, entity_type,
             sector, country, lei, json.dumps(attributes or {}), now, now),
        )
        self._backend.commit()

    def load_entity(self, entity_id: str) -> dict | None:
        rows = self._backend.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,))
        if not rows:
            return None
        row = rows[0]
        row["attributes"] = json.loads(row["attributes"])
        return row

    def list_entities(self, **filters) -> list[dict]:
        where, params = self._build_where(filters)
        rows = self._backend.execute(
            f"SELECT * FROM entities{where} ORDER BY entity_id", params)
        for r in rows:
            r["attributes"] = json.loads(r["attributes"])
        return rows

    # ------------------------------------------------------------------
    # Ratings
    # ------------------------------------------------------------------

    def save_rating(
        self,
        entity_id: str,
        agency: str,
        rating: str,
        effective_date: date | str,
        outlook: str | None = None,
    ) -> None:
        if isinstance(effective_date, date):
            effective_date = effective_date.isoformat()
        self._backend.execute(
            """INSERT INTO ratings (entity_id, agency, rating, outlook, effective_date)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(entity_id, agency, effective_date) DO UPDATE SET
               rating=excluded.rating, outlook=excluded.outlook""",
            (entity_id, agency, rating, outlook, effective_date),
        )
        self._backend.commit()

    def latest_rating(self, entity_id: str, agency: str) -> str | None:
        rows = self._backend.execute(
            """SELECT rating FROM ratings
               WHERE entity_id = ? AND agency = ?
               ORDER BY effective_date DESC LIMIT 1""",
            (entity_id, agency),
        )
        return rows[0]["rating"] if rows else None

    def rating_history(self, entity_id: str) -> list[dict]:
        return self._backend.execute(
            """SELECT * FROM ratings WHERE entity_id = ?
               ORDER BY agency, effective_date""",
            (entity_id,),
        )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(
        self,
        trade_id: str,
        instrument,
        book: str = "default",
        desk: str | None = None,
        counterparty_id: str | None = None,
        issuer_id: str | None = None,
        direction: str | None = None,
        notional: float | None = None,
        currency: str | None = None,
        maturity_date: date | str | None = None,
        trader: str | None = None,
        status: str = "live",
        tags: dict | None = None,
    ) -> None:
        if hasattr(instrument, "to_dict"):
            inst_dict = instrument.to_dict()
            inst_type = inst_dict.get("type", type(instrument).__name__)
            inst_json = json.dumps(inst_dict)
        elif isinstance(instrument, dict):
            inst_type = instrument.get("type", "unknown")
            inst_json = json.dumps(instrument)
        else:
            raise TypeError(f"instrument must have to_dict() or be a dict, got {type(instrument)}")

        if isinstance(maturity_date, date):
            maturity_date = maturity_date.isoformat()

        now = _now()
        self._backend.execute(
            """INSERT INTO trades
               (trade_id, instrument_type, instrument_json, book, desk,
                counterparty_id, issuer_id, direction, notional, currency,
                maturity_date, trader, status, tags, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(trade_id) DO UPDATE SET
               instrument_type=excluded.instrument_type,
               instrument_json=excluded.instrument_json,
               book=excluded.book, desk=excluded.desk,
               counterparty_id=excluded.counterparty_id,
               issuer_id=excluded.issuer_id, direction=excluded.direction,
               notional=excluded.notional, currency=excluded.currency,
               maturity_date=excluded.maturity_date, trader=excluded.trader,
               status=excluded.status, tags=excluded.tags,
               updated_at=excluded.updated_at""",
            (trade_id, inst_type, inst_json, book, desk,
             counterparty_id, issuer_id, direction, notional, currency,
             maturity_date, trader, status, json.dumps(tags or {}), now, now),
        )
        self._backend.commit()

    def load_trade(self, trade_id: str):
        """Load and reconstruct a pricebook instrument via from_dict."""
        rows = self._backend.execute(
            "SELECT instrument_json FROM trades WHERE trade_id = ?", (trade_id,))
        if not rows:
            return None
        from pricebook.core.serialisable import from_dict
        return from_dict(json.loads(rows[0]["instrument_json"]))

    def load_trade_raw(self, trade_id: str) -> dict | None:
        """Load trade metadata + instrument JSON without reconstruction."""
        rows = self._backend.execute(
            "SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        if not rows:
            return None
        row = rows[0]
        row["tags"] = json.loads(row["tags"])
        row["instrument"] = json.loads(row["instrument_json"])
        return row

    def list_trades(self, **filters) -> list[dict]:
        where, params = self._build_where(filters)
        rows = self._backend.execute(
            f"SELECT trade_id, instrument_type, book, desk, direction, notional, "
            f"currency, maturity_date, trader, status, counterparty_id, issuer_id "
            f"FROM trades{where} ORDER BY trade_id", params)
        return rows

    def update_trade_status(self, trade_id: str, status: str) -> None:
        self._backend.execute(
            "UPDATE trades SET status = ?, updated_at = ? WHERE trade_id = ?",
            (status, _now(), trade_id))
        self._backend.commit()

    # ------------------------------------------------------------------
    # Market snapshots
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        snapshot_date: date | str,
        snapshot_type: str,
        curve_name: str,
        data,
        source: str = "manual",
    ) -> None:
        if isinstance(snapshot_date, date):
            snapshot_date = snapshot_date.isoformat()
        if hasattr(data, "to_dict"):
            data_json = json.dumps(data.to_dict())
        elif isinstance(data, dict):
            data_json = json.dumps(data)
        else:
            raise TypeError(f"data must have to_dict() or be a dict, got {type(data)}")

        self._backend.execute(
            """INSERT INTO market_snapshots
               (snapshot_date, snapshot_type, curve_name, data_json, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(snapshot_date, snapshot_type, curve_name) DO UPDATE SET
               data_json=excluded.data_json, source=excluded.source""",
            (snapshot_date, snapshot_type, curve_name, data_json, source, _now()),
        )
        self._backend.commit()

    def load_snapshot(self, snapshot_date: date | str, snapshot_type: str, curve_name: str):
        """Load and reconstruct a curve/surface via from_dict."""
        if isinstance(snapshot_date, date):
            snapshot_date = snapshot_date.isoformat()
        rows = self._backend.execute(
            """SELECT data_json FROM market_snapshots
               WHERE snapshot_date = ? AND snapshot_type = ? AND curve_name = ?""",
            (snapshot_date, snapshot_type, curve_name),
        )
        if not rows:
            return None
        from pricebook.core.serialisable import from_dict
        return from_dict(json.loads(rows[0]["data_json"]))

    def load_snapshot_raw(self, snapshot_date: date | str, snapshot_type: str, curve_name: str) -> dict | None:
        if isinstance(snapshot_date, date):
            snapshot_date = snapshot_date.isoformat()
        rows = self._backend.execute(
            """SELECT * FROM market_snapshots
               WHERE snapshot_date = ? AND snapshot_type = ? AND curve_name = ?""",
            (snapshot_date, snapshot_type, curve_name),
        )
        if not rows:
            return None
        row = rows[0]
        row["data"] = json.loads(row["data_json"])
        return row

    def list_snapshots(self, date_from: date | str | None = None,
                       date_to: date | str | None = None) -> list[dict]:
        if isinstance(date_from, date):
            date_from = date_from.isoformat()
        if isinstance(date_to, date):
            date_to = date_to.isoformat()
        base = "SELECT snapshot_date, snapshot_type, curve_name, source FROM market_snapshots"
        if date_from and date_to:
            return self._backend.execute(
                f"{base} WHERE snapshot_date BETWEEN ? AND ? ORDER BY snapshot_date",
                (date_from, date_to))
        if date_from:
            return self._backend.execute(
                f"{base} WHERE snapshot_date >= ? ORDER BY snapshot_date", (date_from,))
        if date_to:
            return self._backend.execute(
                f"{base} WHERE snapshot_date <= ? ORDER BY snapshot_date", (date_to,))
        return self._backend.execute(f"{base} ORDER BY snapshot_date")

    # ------------------------------------------------------------------
    # Calibration results
    # ------------------------------------------------------------------

    def save_calibration(self, result) -> str:
        """Persist a calibration artefact; return its id as a string.

        Accepts either a canonical `CalibrationResult` or any family result
        that exposes `to_calibration_result()` (`HWCalibrationResult`,
        `JumpCalibrationResult`, `G2PPCalibrationResult`, …) — the latter is
        converted via that accessor. This makes the persistence path the
        canonical *consumer* of every calibrator's record, closing the
        build → store → read loop.

        Idempotent on the calibration id — re-saving the same result updates
        the row. The full record is stored as JSON (`result_json`) and the
        identity/quality fields are denormalised into columns so the audit
        chain can be queried (e.g. `list_calibrations(model_class="HullWhite")`
        or by `market_snapshot_id`) without reconstructing every blob.
        """
        if hasattr(result, "to_calibration_result"):
            result = result.to_calibration_result()
        cid = str(result.id)
        msid = str(result.market_snapshot_id) if result.market_snapshot_id else None
        self._backend.execute(
            """INSERT INTO calibration_results
               (calibration_id, model_class, timestamp, code_version, objective,
                converged, iterations, rms_residual, max_residual,
                market_snapshot_id, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(calibration_id) DO UPDATE SET
               model_class=excluded.model_class, timestamp=excluded.timestamp,
               code_version=excluded.code_version, objective=excluded.objective,
               converged=excluded.converged, iterations=excluded.iterations,
               rms_residual=excluded.rms_residual, max_residual=excluded.max_residual,
               market_snapshot_id=excluded.market_snapshot_id,
               result_json=excluded.result_json""",
            (cid, result.model_class, result.timestamp.isoformat(),
             result.code_version, result.objective.value,
             int(result.converged), result.iterations,
             result.rms_residual, result.max_residual,
             msid, json.dumps(result.to_dict()), _now()),
        )
        self._backend.commit()
        return cid

    def load_calibration(self, calibration_id: str | UUID) -> "CalibrationResult | None":
        """Load and reconstruct a `CalibrationResult` by id."""
        rows = self._backend.execute(
            "SELECT result_json FROM calibration_results WHERE calibration_id = ?",
            (str(calibration_id),))
        if not rows:
            return None
        # Direct from_dict: the convention payload is flat (no "type" key), so
        # the generic registry dispatch can't be used — but we know the type.
        from pricebook.calibration import CalibrationResult
        return CalibrationResult.from_dict(json.loads(rows[0]["result_json"]))

    def load_calibration_raw(self, calibration_id: str | UUID) -> dict | None:
        """Load calibration row + parsed JSON without reconstruction."""
        rows = self._backend.execute(
            "SELECT * FROM calibration_results WHERE calibration_id = ?",
            (str(calibration_id),))
        if not rows:
            return None
        row = rows[0]
        row["result"] = json.loads(row["result_json"])
        return row

    def list_calibrations(self, **filters) -> list[dict]:
        """List calibration metadata (no reconstruction). Filter by any column,
        e.g. `model_class`, `converged`, `market_snapshot_id`."""
        where, params = self._build_where(filters)
        return self._backend.execute(
            f"SELECT calibration_id, model_class, timestamp, code_version, objective, "
            f"converged, iterations, rms_residual, max_residual, market_snapshot_id "
            f"FROM calibration_results{where} ORDER BY timestamp", params)

    # ------------------------------------------------------------------
    # Pricing results
    # ------------------------------------------------------------------

    def save_result(self, trade_id: str, snapshot_date: date | str,
                    result, method: str | None = None) -> None:
        if isinstance(snapshot_date, date):
            snapshot_date = snapshot_date.isoformat()
        if hasattr(result, "to_dict"):
            result_json = json.dumps(result.to_dict())
        elif isinstance(result, dict):
            result_json = json.dumps(result)
        else:
            result_json = json.dumps({"value": float(result)})

        self._backend.execute(
            """INSERT INTO pricing_results (trade_id, snapshot_date, result_json, method, computed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (trade_id, snapshot_date, result_json, method, _now()),
        )
        self._backend.commit()

    def result_history(self, trade_id: str) -> list[dict]:
        rows = self._backend.execute(
            """SELECT snapshot_date, result_json, method, computed_at
               FROM pricing_results WHERE trade_id = ?
               ORDER BY snapshot_date""", (trade_id,))
        for r in rows:
            r["result"] = json.loads(r["result_json"])
        return rows

    # ------------------------------------------------------------------
    # P&L history
    # ------------------------------------------------------------------

    def save_pnl(self, trade_id: str, valuation_date: date | str,
                 pv: float, pnl_1d: float | None = None,
                 delta: float | None = None, gamma: float | None = None,
                 vega: float | None = None, dv01: float | None = None,
                 cs01: float | None = None) -> None:
        if isinstance(valuation_date, date):
            valuation_date = valuation_date.isoformat()
        self._backend.execute(
            """INSERT INTO pnl_history
               (trade_id, valuation_date, pv, pnl_1d, delta, gamma, vega, dv01, cs01)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(trade_id, valuation_date) DO UPDATE SET
               pv=excluded.pv, pnl_1d=excluded.pnl_1d,
               delta=excluded.delta, gamma=excluded.gamma, vega=excluded.vega,
               dv01=excluded.dv01, cs01=excluded.cs01""",
            (trade_id, valuation_date, pv, pnl_1d, delta, gamma, vega, dv01, cs01),
        )
        self._backend.commit()

    def pnl_series(self, trade_id: str) -> list[dict]:
        return self._backend.execute(
            """SELECT * FROM pnl_history WHERE trade_id = ?
               ORDER BY valuation_date""", (trade_id,))

    def pnl_series_by_book(self, book: str) -> list[dict]:
        """Aggregate P&L history across all trades in a book."""
        return self._backend.execute(
            """SELECT h.valuation_date,
                      SUM(h.pv) as pv, SUM(h.pnl_1d) as pnl_1d,
                      SUM(h.delta) as delta, SUM(h.gamma) as gamma,
                      SUM(h.vega) as vega, SUM(h.dv01) as dv01, SUM(h.cs01) as cs01
               FROM pnl_history h JOIN trades t ON h.trade_id = t.trade_id
               WHERE t.book = ?
               GROUP BY h.valuation_date ORDER BY h.valuation_date""",
            (book,))

    def pnl_series_by_desk(self, desk: str) -> list[dict]:
        """Aggregate P&L history across all trades in a desk."""
        return self._backend.execute(
            """SELECT h.valuation_date,
                      SUM(h.pv) as pv, SUM(h.pnl_1d) as pnl_1d,
                      SUM(h.delta) as delta, SUM(h.gamma) as gamma,
                      SUM(h.vega) as vega, SUM(h.dv01) as dv01, SUM(h.cs01) as cs01
               FROM pnl_history h JOIN trades t ON h.trade_id = t.trade_id
               WHERE t.desk = ?
               GROUP BY h.valuation_date ORDER BY h.valuation_date""",
            (desk,))

    # ------------------------------------------------------------------
    # Key-value store
    # ------------------------------------------------------------------

    def put(self, namespace: str, key: str, value) -> None:
        self._backend.execute(
            """INSERT INTO kv_store (namespace, key, value_json, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(namespace, key) DO UPDATE SET
               value_json=excluded.value_json, updated_at=excluded.updated_at""",
            (namespace, key, json.dumps(value), _now()),
        )
        self._backend.commit()

    def get(self, namespace: str, key: str, default=None):
        rows = self._backend.execute(
            "SELECT value_json FROM kv_store WHERE namespace = ? AND key = ?",
            (namespace, key))
        if not rows:
            return default
        return json.loads(rows[0]["value_json"])

    def list_keys(self, namespace: str) -> list[str]:
        rows = self._backend.execute(
            "SELECT key FROM kv_store WHERE namespace = ? ORDER BY key",
            (namespace,))
        return [r["key"] for r in rows]

    # ------------------------------------------------------------------
    # Custom tables (any columns, DataFrame-native)
    # ------------------------------------------------------------------

    def save_table(self, name: str, rows: list[dict], replace: bool = True) -> None:
        """Save a list of dicts as a custom table. Columns inferred from first row."""
        if not rows:
            return
        if name in self._SYSTEM_TABLES:
            raise ValueError(f"Cannot overwrite system table '{name}'")

        columns = {k: _infer_sql_type(v) for k, v in rows[0].items()}

        if replace and self._backend.table_exists(name):
            self._backend.drop_table(name)

        if not self._backend.table_exists(name):
            self._backend.create_table(name, columns)

        cols = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        safe_table = _safe_name(name)
        safe_cols = ", ".join(_safe_name(c) for c in cols)
        sql = f"INSERT INTO {safe_table} ({safe_cols}) VALUES ({placeholders})"
        self._backend.execute_many(
            sql, [tuple(_serialise_value(row.get(c)) for c in cols) for row in rows])
        self._backend.commit()

    def append_rows(self, name: str, rows: list[dict]) -> None:
        """Append rows to an existing custom table."""
        if not rows:
            return
        if name in self._SYSTEM_TABLES:
            raise ValueError(f"Cannot append to system table '{name}'")
        if not self._backend.table_exists(name):
            return self.save_table(name, rows, replace=False)
        cols = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        safe_table = _safe_name(name)
        safe_cols = ", ".join(_safe_name(c) for c in cols)
        sql = f"INSERT INTO {safe_table} ({safe_cols}) VALUES ({placeholders})"
        self._backend.execute_many(
            sql, [tuple(_serialise_value(row.get(c)) for c in cols) for row in rows])
        self._backend.commit()

    def load_table(self, name: str, **filters) -> list[dict]:
        """Load all rows from a custom table, optionally filtered."""
        safe = _safe_name(name)
        if not self._backend.table_exists(name):
            raise ValueError(f"Table '{name}' does not exist")
        where, params = self._build_where(filters)
        return self._backend.execute(f"SELECT * FROM {safe}{where}", params)

    def load_table_df(self, name: str, **filters):
        """Load a custom table as a pandas DataFrame."""
        import pandas as pd
        rows = self.load_table(name, **filters)
        return pd.DataFrame(rows)

    def delete_rows(self, name: str, **filters) -> None:
        """Delete rows matching filters."""
        if not self._backend.table_exists(name):
            raise ValueError(f"Table '{name}' does not exist")
        where, params = self._build_where(filters)
        if not where:
            raise ValueError("delete_rows requires at least one filter")
        self._backend.execute(f"DELETE FROM {_safe_name(name)}{where}", params)
        self._backend.commit()

    _INTERNAL_TABLES = {"sqlite_sequence"}

    def list_custom_tables(self) -> list[str]:
        """List user-created tables (excludes system and SQLite internal tables)."""
        all_tables = self._backend.list_tables()
        exclude = self._SYSTEM_TABLES | self._INTERNAL_TABLES
        return [t for t in all_tables if t not in exclude]

    def drop_table(self, name: str) -> None:
        """Drop a custom table. Cannot drop system tables."""
        if name in self._SYSTEM_TABLES:
            raise ValueError(f"Cannot drop system table '{name}'")
        self._backend.drop_table(name)
        self._backend.commit()

    # ------------------------------------------------------------------
    # CSV import/export
    # ------------------------------------------------------------------

    def export_csv(self, name: str, filepath: str | Path) -> None:
        """Export a table (custom or system) to CSV."""
        safe = _safe_name(name)
        if not self._backend.table_exists(name):
            raise ValueError(f"Table '{name}' does not exist")
        rows = self._backend.execute(f"SELECT * FROM {safe}")
        if not rows:
            return
        filepath = Path(filepath)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def import_csv(self, name: str, filepath: str | Path, replace: bool = True) -> int:
        """Import a CSV file into a custom table. Returns row count."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return 0
        # Convert numeric strings
        converted = []
        for row in rows:
            r = {}
            for k, v in row.items():
                try:
                    r[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        r[k] = float(v)
                    except (ValueError, TypeError):
                        r[k] = v
            converted.append(r)
        self.save_table(name, converted, replace=replace)
        return len(converted)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_where(self, filters: dict) -> tuple[str, tuple]:
        if not filters:
            return "", ()
        clauses = []
        params = []
        for k, v in filters.items():
            clauses.append(f"{_safe_name(k)} = ?")
            params.append(v)
        return " WHERE " + " AND ".join(clauses), tuple(params)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
