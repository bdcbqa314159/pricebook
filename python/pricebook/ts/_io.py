"""I/O: load TimeSeries from PricebookDB, CSV, or raw data."""

from __future__ import annotations

import csv
from datetime import date

import numpy as np

from pricebook.ts._core import TimeSeries


def from_db(db, trade_id: str, field: str = "pnl_1d") -> TimeSeries:
    """Load a TimeSeries from PricebookDB pnl_history.

    Args:
        db: PricebookDB instance.
        trade_id: trade identifier.
        field: column to extract — 'pv', 'pnl_1d', 'delta', 'gamma',
               'vega', 'dv01', 'cs01'.
    """
    rows = db.pnl_series(trade_id)
    if not rows:
        return TimeSeries.empty(f"{trade_id}_{field}")
    dates = [r["valuation_date"] for r in rows]
    values = [float(r.get(field, 0) or 0) for r in rows]
    return TimeSeries.from_lists(dates, values, f"{trade_id}_{field}")


def from_db_book(db, book: str, field: str = "pnl_1d") -> TimeSeries:
    """Load aggregated TimeSeries for all trades in a book."""
    rows = db.pnl_series_by_book(book)
    if not rows:
        return TimeSeries.empty(f"book_{book}_{field}")
    dates = [r["valuation_date"] for r in rows]
    values = [float(r.get(field, 0) or 0) for r in rows]
    return TimeSeries.from_lists(dates, values, f"book_{book}_{field}")


def from_db_desk(db, desk: str, field: str = "pnl_1d") -> TimeSeries:
    """Load aggregated TimeSeries for all trades in a desk."""
    rows = db.pnl_series_by_desk(desk)
    if not rows:
        return TimeSeries.empty(f"desk_{desk}_{field}")
    dates = [r["valuation_date"] for r in rows]
    values = [float(r.get(field, 0) or 0) for r in rows]
    return TimeSeries.from_lists(dates, values, f"desk_{desk}_{field}")


def greeks_from_db(db, trade_id: str) -> dict[str, TimeSeries]:
    """Load all greek time series for a trade.

    Returns dict with keys: 'delta', 'gamma', 'vega', 'dv01', 'cs01'.
    """
    rows = db.pnl_series(trade_id)
    if not rows:
        return {k: TimeSeries.empty(f"{trade_id}_{k}")
                for k in ("delta", "gamma", "vega", "dv01", "cs01")}
    dates = [r["valuation_date"] for r in rows]
    result = {}
    for field in ("delta", "gamma", "vega", "dv01", "cs01"):
        values = [float(r.get(field, 0) or 0) for r in rows]
        result[field] = TimeSeries.from_lists(dates, values, f"{trade_id}_{field}")
    return result


def from_csv(
    filepath: str,
    date_col: str = "date",
    value_col: str = "value",
    name: str = "",
) -> TimeSeries:
    """Load a TimeSeries from a CSV file (no pandas dependency).

    Args:
        filepath: path to CSV.
        date_col: column name for dates (ISO format).
        value_col: column name for values.
        name: series name.
    """
    dates = []
    values = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row[date_col])
            values.append(float(row[value_col]))
    return TimeSeries.from_lists(dates, values, name or value_col)
