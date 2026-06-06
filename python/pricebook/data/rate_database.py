"""Rate database: store and query historical fixings and calibrated curves.

SQLite storage for any RateSource. Stores raw fixings and optionally
Nelson-Siegel parameters for each calibrated curve.

    from pricebook.data.rate_database import RateDatabase

    db = RateDatabase()                         # default path
    db.ingest_year(EuriborSource(), 2024)       # fetch + store
    db.ingest_history(EuriborSource())          # full history

    fixings = db.query("EUR", "euriborrates.com", date(2024,1,1), date(2024,12,31))
    curve = db.curve("EUR", "euriborrates.com", date(2024,6,3))

DATA SOURCES: attributed per-record in the database.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pricebook.data.rate_source import RateSource, RateFixing, RateType


_DEFAULT_DB_PATH = Path(__file__).parent / "rates.db"


class RateDatabase:
    """SQLite database for rate fixings and calibrated curves.

    Schema:
        fixings: date, currency, source, tenor, rate, rate_type
        curves:  date, currency, source, method, params (JSON)
    """

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path or _DEFAULT_DB_PATH)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    @property
    def path(self) -> Path:
        return self._path

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS fixings (
                date        TEXT NOT NULL,
                currency    TEXT NOT NULL,
                source      TEXT NOT NULL,
                tenor       TEXT NOT NULL,
                rate        REAL NOT NULL,
                rate_type   TEXT NOT NULL DEFAULT 'deposit',
                ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (date, currency, source, tenor)
            );

            CREATE TABLE IF NOT EXISTS curves (
                date        TEXT NOT NULL,
                currency    TEXT NOT NULL,
                source      TEXT NOT NULL,
                method      TEXT NOT NULL DEFAULT 'bootstrap',
                params      TEXT,
                ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (date, currency, source, method)
            );

            CREATE INDEX IF NOT EXISTS idx_fixings_date ON fixings(date);
            CREATE INDEX IF NOT EXISTS idx_fixings_ccy ON fixings(currency, source);
            CREATE INDEX IF NOT EXISTS idx_curves_date ON curves(date);
        """)
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    # Ingestion
    # ═══════════════════════════════════════════════════════════════

    def store_fixing(self, fixing: RateFixing):
        """Store a single day's fixings."""
        for tenor, rate in fixing.rates.items():
            self._conn.execute(
                """INSERT OR REPLACE INTO fixings (date, currency, source, tenor, rate, rate_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (fixing.date.isoformat(), fixing.currency, fixing.source,
                 tenor, rate, fixing.rate_type.value),
            )
        self._conn.commit()

    def store_fixings(self, fixings: list[RateFixing]):
        """Store multiple fixings (batch)."""
        rows = []
        for f in fixings:
            for tenor, rate in f.rates.items():
                rows.append((f.date.isoformat(), f.currency, f.source,
                             tenor, rate, f.rate_type.value))
        self._conn.executemany(
            """INSERT OR REPLACE INTO fixings (date, currency, source, tenor, rate, rate_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def store_curve_params(self, d: date, currency: str, source: str,
                           method: str, params: dict):
        """Store calibrated curve parameters (e.g. Nelson-Siegel)."""
        self._conn.execute(
            """INSERT OR REPLACE INTO curves (date, currency, source, method, params)
               VALUES (?, ?, ?, ?, ?)""",
            (d.isoformat(), currency, source, method, json.dumps(params)),
        )
        self._conn.commit()

    def ingest_year(self, source: RateSource, year: int, calibrate: bool = True,
                    progress: bool = True):
        """Fetch a full year from a RateSource and store in DB.

        Optionally calibrates Nelson-Siegel for each day.

        Args:
            source: any RateSource implementation.
            year: calendar year.
            calibrate: if True, also store NS parameters.
            progress: print progress.
        """
        if progress:
            print(f"Fetching {source.currency} {year} from {source.source_name}...")

        fixings = source.fetch_year(year)

        if progress:
            print(f"  {len(fixings)} business days fetched. Storing...")

        self.store_fixings(fixings)

        if calibrate:
            from pricebook.data.market_curve import MarketCurve
            for f in fixings:
                try:
                    curve = MarketCurve._build(f, source.tenors, source.source_name, source.attribution)
                    ns = curve.ns_fit()
                    self.store_curve_params(f.date, source.currency, source.source_name,
                                           "nelson_siegel", ns)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Calibration failed for {f.date}: {e}")

        if progress:
            print(f"  Done. {len(fixings)} fixings stored.")

    def ingest_history(self, source: RateSource, start_year: int = 1999,
                       end_year: int | None = None, calibrate: bool = True,
                       delay_between_years: float = 2.0):
        """Fetch complete history and store in DB.

        Args:
            source: any RateSource.
            start_year: first year (default 1999 for Euribor).
            end_year: last year (default: current year).
            delay_between_years: polite delay between year requests.
        """
        import time

        if end_year is None:
            end_year = date.today().year

        total = 0
        for year in range(start_year, end_year + 1):
            self.ingest_year(source, year, calibrate=calibrate)
            total += self.count_fixings(source.currency, source.source_name, year)
            if year < end_year:
                time.sleep(delay_between_years)

        print(f"\nComplete: {total} total fixings for {source.currency} "
              f"({start_year}–{end_year}) from {source.source_name}")

    def ingest_today(self, source: RateSource, calibrate: bool = True) -> bool:
        """Fetch today's fixing and store. Returns True if new data stored.

        Designed for cron job.
        """
        today = date.today()

        # Check if already stored
        existing = self.query_date(today, source.currency, source.source_name)
        if existing:
            return False

        fixing = source.fetch(today)
        if fixing is None:
            return False

        self.store_fixing(fixing)

        if calibrate:
            try:
                from pricebook.data.market_curve import MarketCurve
                curve = MarketCurve._build(fixing, source.tenors, source.source_name,
                                           source.attribution)
                ns = curve.ns_fit()
                self.store_curve_params(today, source.currency, source.source_name,
                                       "nelson_siegel", ns)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Calibration failed for {today}: {e}")

        return True

    # ═══════════════════════════════════════════════════════════════
    # Query
    # ═══════════════════════════════════════════════════════════════

    def query_date(self, d: date, currency: str, source: str) -> dict[str, float] | None:
        """Get fixings for a specific date. Returns tenor→rate dict or None."""
        rows = self._conn.execute(
            "SELECT tenor, rate FROM fixings WHERE date=? AND currency=? AND source=?",
            (d.isoformat(), currency, source),
        ).fetchall()
        if not rows:
            return None
        return {r["tenor"]: r["rate"] for r in rows}

    def query_range(self, currency: str, source: str,
                    start: date, end: date) -> list[RateFixing]:
        """Get all fixings in a date range."""
        rows = self._conn.execute(
            """SELECT date, tenor, rate, rate_type FROM fixings
               WHERE currency=? AND source=? AND date BETWEEN ? AND ?
               ORDER BY date, tenor""",
            (currency, source, start.isoformat(), end.isoformat()),
        ).fetchall()

        # Group by date
        by_date: dict[str, dict[str, float]] = {}
        rate_type = RateType.DEPOSIT
        for r in rows:
            d = r["date"]
            if d not in by_date:
                by_date[d] = {}
            by_date[d][r["tenor"]] = r["rate"]
            rate_type = RateType(r["rate_type"])

        return [
            RateFixing(date=date.fromisoformat(d), rates=rates,
                       rate_type=rate_type, source=source, currency=currency)
            for d, rates in sorted(by_date.items())
        ]

    def query_curve_params(self, d: date, currency: str, source: str,
                           method: str = "nelson_siegel") -> dict | None:
        """Get calibrated curve parameters for a date."""
        row = self._conn.execute(
            "SELECT params FROM curves WHERE date=? AND currency=? AND source=? AND method=?",
            (d.isoformat(), currency, source, method),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["params"])

    def curve(self, d: date, currency: str, source: str) -> Any:
        """Build a MarketCurve from stored fixings for a date.

        No network request — uses only the local database.
        """
        rates = self.query_date(d, currency, source)
        if rates is None:
            raise ValueError(f"No fixings in DB for {currency} {d} from {source}")

        from pricebook.data.rate_source import DEPOSIT_TENORS
        from pricebook.data.market_curve import MarketCurve

        fixing = RateFixing(date=d, rates=rates, rate_type=RateType.DEPOSIT,
                            source=source, currency=currency)
        return MarketCurve._build(fixing, DEPOSIT_TENORS, source,
                                  f"Data from {source}")

    # ═══════════════════════════════════════════════════════════════
    # Info
    # ═══════════════════════════════════════════════════════════════

    def count_fixings(self, currency: str | None = None,
                      source: str | None = None,
                      year: int | None = None) -> int:
        """Count fixing records."""
        sql = "SELECT COUNT(DISTINCT date) FROM fixings WHERE 1=1"
        params: list = []
        if currency:
            sql += " AND currency=?"
            params.append(currency)
        if source:
            sql += " AND source=?"
            params.append(source)
        if year:
            sql += " AND date LIKE ?"
            params.append(f"{year}-%")
        return self._conn.execute(sql, params).fetchone()[0]

    def date_range(self, currency: str = "EUR",
                   source: str = "euriborrates.com") -> tuple[date, date] | None:
        # Note: defaults kept for backward compat but caller should specify
        """Get earliest and latest dates in DB."""
        row = self._conn.execute(
            "SELECT MIN(date) as min_d, MAX(date) as max_d FROM fixings WHERE currency=? AND source=?",
            (currency, source),
        ).fetchone()
        if row["min_d"] is None:
            return None
        return date.fromisoformat(row["min_d"]), date.fromisoformat(row["max_d"])

    def summary(self) -> dict:
        """Database summary."""
        n_fixings = self._conn.execute("SELECT COUNT(*) FROM fixings").fetchone()[0]
        n_curves = self._conn.execute("SELECT COUNT(*) FROM curves").fetchone()[0]
        sources = self._conn.execute(
            "SELECT DISTINCT currency, source FROM fixings"
        ).fetchall()
        return {
            "db_path": str(self._path),
            "total_fixing_records": n_fixings,
            "total_curve_params": n_curves,
            "sources": [{"currency": r["currency"], "source": r["source"]} for r in sources],
        }

    def close(self):
        self._conn.close()

    def __repr__(self) -> str:
        n = self.count_fixings()
        return f"RateDatabase({self._path.name}, {n} fixing-dates)"
