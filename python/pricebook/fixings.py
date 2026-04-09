"""Fixings manager: store and retrieve daily rate fixings.

File-based storage for SOFR, EURIBOR, CPI, and other daily fixings.
Used for retroactive floating leg valuation and historical analysis.

    from pricebook.fixings import FixingsStore

    store = FixingsStore()
    store.set("SOFR", date(2024, 1, 15), 0.043)
    rate = store.get("SOFR", date(2024, 1, 15))
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any


class FixingsStore:
    """File-backed store for daily rate fixings.

    Args:
        path: directory for storage files. If None, in-memory only.
    """

    def __init__(self, path: str | None = None):
        self._path = path
        self._data: dict[str, dict[date, float]] = {}
        if path and os.path.isdir(path):
            self._load_all()

    def set(self, rate_name: str, d: date, value: float) -> None:
        """Store a fixing."""
        if rate_name not in self._data:
            self._data[rate_name] = {}
        self._data[rate_name][d] = value

    def get(self, rate_name: str, d: date) -> float | None:
        """Retrieve a fixing. Returns None if not found."""
        return self._data.get(rate_name, {}).get(d)

    def get_or_raise(self, rate_name: str, d: date) -> float:
        """Retrieve a fixing, raising KeyError if not found."""
        val = self.get(rate_name, d)
        if val is None:
            raise KeyError(f"No fixing for {rate_name} on {d}")
        return val

    def has(self, rate_name: str, d: date) -> bool:
        return self.get(rate_name, d) is not None

    def rate_names(self) -> list[str]:
        return sorted(self._data.keys())

    def dates_for(self, rate_name: str) -> list[date]:
        """All dates with fixings for a rate, sorted."""
        return sorted(self._data.get(rate_name, {}).keys())

    def series(self, rate_name: str, start: date | None = None, end: date | None = None) -> list[tuple[date, float]]:
        """Get a time series of fixings."""
        data = self._data.get(rate_name, {})
        result = sorted(data.items())
        if start:
            result = [(d, v) for d, v in result if d >= start]
        if end:
            result = [(d, v) for d, v in result if d <= end]
        return result

    def bulk_set(self, rate_name: str, fixings: list[tuple[date, float]]) -> None:
        """Store multiple fixings at once."""
        if rate_name not in self._data:
            self._data[rate_name] = {}
        for d, v in fixings:
            self._data[rate_name][d] = v

    # ---- Persistence ----

    def save(self, path: str | None = None) -> None:
        """Save all fixings to disk as JSON."""
        p = path or self._path
        if p is None:
            raise ValueError("No path specified for saving")
        os.makedirs(p, exist_ok=True)
        for rate_name, data in self._data.items():
            filepath = os.path.join(p, f"{rate_name}.json")
            serialised = {d.isoformat(): v for d, v in sorted(data.items())}
            with open(filepath, "w") as f:
                json.dump(serialised, f, indent=2)

    def _load_all(self) -> None:
        """Load all JSON files from the storage directory."""
        if self._path is None:
            return
        for filename in os.listdir(self._path):
            if filename.endswith(".json"):
                rate_name = filename[:-5]
                filepath = os.path.join(self._path, filename)
                with open(filepath) as f:
                    raw = json.load(f)
                self._data[rate_name] = {
                    date.fromisoformat(k): v for k, v in raw.items()
                }

    def load_csv(self, rate_name: str, filepath: str, date_col: str = "date", value_col: str = "value") -> int:
        """Load fixings from a CSV file.

        Returns number of fixings loaded.
        """
        count = 0
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = date.fromisoformat(row[date_col])
                v = float(row[value_col])
                self.set(rate_name, d, v)
                count += 1
        return count


# ---- Sample fixings ----

def create_sample_fixings(reference_date: date, history_days: int = 252) -> FixingsStore:
    """Create a FixingsStore with sample data for testing.

    Generates synthetic SOFR, ESTR, and CPI fixings.
    """
    import random
    from datetime import timedelta

    store = FixingsStore()
    rng = random.Random(42)

    rates = {"SOFR": 0.043, "ESTR": 0.035, "FED_FUNDS": 0.043}
    for name, base in rates.items():
        rate = base
        for i in range(history_days):
            d = reference_date - timedelta(days=history_days - i)
            if d.weekday() < 5:
                rate = max(rate + rng.gauss(0, 0.0005), 0.001)
                store.set(name, d, round(rate, 6))

    # CPI: monthly, increasing
    cpi = 310.0
    for m in range(24):
        d = date(reference_date.year - 2 + m // 12, 1 + m % 12, 1)
        cpi *= 1 + rng.uniform(0.001, 0.004)
        store.set("CPI", d, round(cpi, 2))

    return store
