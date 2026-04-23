"""Market data tools: quote parsing, synthetic data, snapshots.

Extends existing market_data.py with practical tools for daily use.

    from pricebook.market_data_tools import (
        synthetic_market, parse_json_quotes, MarketSnapshot,
    )
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np


# ============================================================================
# Quote parsing
# ============================================================================

@dataclass
class MarketQuote:
    """A single market quote."""
    instrument: str
    value: float
    quote_date: date
    source: str = ""
    quote_type: str = "mid"


def parse_csv_quotes(filepath: str, date_col: str = "date",
                     instrument_col: str = "instrument",
                     value_col: str = "value") -> list[MarketQuote]:
    """Parse market quotes from a CSV file."""
    quotes = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                quotes.append(MarketQuote(
                    instrument=row[instrument_col],
                    value=float(row[value_col]),
                    quote_date=date.fromisoformat(row[date_col]),
                ))
            except (ValueError, KeyError):
                continue
    return quotes


def parse_json_quotes(data: dict | list) -> list[MarketQuote]:
    """Parse market quotes from a JSON structure.

        quotes = parse_json_quotes({"USD_OIS_5Y": 0.04, "EUR_OIS_5Y": 0.03})
    """
    if isinstance(data, dict):
        return [MarketQuote(k, v, date.today()) for k, v in data.items()
                if isinstance(v, (int, float))]
    return []


# ============================================================================
# Synthetic data
# ============================================================================

@dataclass
class SyntheticMarket:
    """Synthetic market data for testing."""
    reference_date: date
    deposits: dict[str, float]
    swaps: dict[str, float]
    fx_spots: dict[str, float]
    equity_spots: dict[str, float]
    commodity_spots: dict[str, float]


def synthetic_market(
    reference_date: date | None = None,
    base_rate: float = 0.04,
    seed: int = 42,
) -> SyntheticMarket:
    """Generate synthetic market data for testing."""
    rng = np.random.default_rng(seed)
    ref = reference_date or date.today()

    deposits = {"1M": round(base_rate - 0.002 + rng.normal(0, 0.001), 5),
                "3M": round(base_rate + rng.normal(0, 0.001), 5),
                "6M": round(base_rate + 0.001 + rng.normal(0, 0.001), 5)}
    swaps = {f"{t}Y": round(base_rate + 0.002 * t + rng.normal(0, 0.001), 5)
             for t in [1, 2, 3, 5, 7, 10, 15, 20, 30]}
    fx = {"EUR/USD": round(1.10 + rng.normal(0, 0.01), 4),
          "GBP/USD": round(1.27 + rng.normal(0, 0.01), 4),
          "USD/JPY": round(150.0 + rng.normal(0, 1.0), 2)}
    equities = {"SPX": round(5500 + rng.normal(0, 50), 0),
                "NDX": round(19000 + rng.normal(0, 200), 0)}
    commodities = {"CL": round(72 + rng.normal(0, 2), 2),
                   "GC": round(2350 + rng.normal(0, 20), 0)}

    return SyntheticMarket(ref, deposits, swaps, fx, equities, commodities)


# ============================================================================
# Snapshots
# ============================================================================

@dataclass
class MarketSnapshot:
    """Point-in-time market data snapshot."""
    snapshot_date: date
    quotes: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_snapshot(snapshot: MarketSnapshot, filepath: str) -> None:
    """Save snapshot to JSON."""
    with open(filepath, "w") as f:
        json.dump({"date": snapshot.snapshot_date.isoformat(),
                    "quotes": snapshot.quotes, "metadata": snapshot.metadata}, f, indent=2)


def load_snapshot(filepath: str) -> MarketSnapshot:
    """Load snapshot from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return MarketSnapshot(date.fromisoformat(data["date"]),
                          data.get("quotes", {}), data.get("metadata", {}))
