"""Position management: Book, Desk, and position limits.

A Book is a named container of trades (e.g. "USD_Swaps").
A Desk is a collection of books (e.g. "Rates").
Positions aggregate net exposure by instrument type and tenor bucket.
Limits enforce DV01, notional, and tenor-bucket constraints with breach detection.

    book = Book("USD_Swaps")
    book.add(trade1)
    book.add(trade2)

    desk = Desk("Rates")
    desk.add_book(book)

    positions = book.positions(ctx)
    breaches = book.check_limits(ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade


# ---- Tenor bucketing ----

TENOR_BUCKET_BOUNDARIES = [
    0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30,
]

# Labels use ≤ convention: "≤3M" means years_to_maturity <= 0.25
TENOR_BUCKET_LABELS = [
    "≤3M", "3M-6M", "6M-1Y", "1Y-2Y", "2Y-3Y", "3Y-5Y",
    "5Y-7Y", "7Y-10Y", "10Y-15Y", "15Y-20Y", "20Y-30Y", "30Y+",
]


def tenor_bucket(ref: date, end: date) -> str:
    """Assign a tenor bucket based on time from ref to end.

    Boundaries are inclusive: a 5Y tenor goes into '3Y-5Y', not '5Y-7Y'.
    """
    years = (end - ref).days / 365.25
    for boundary, label in zip(TENOR_BUCKET_BOUNDARIES, TENOR_BUCKET_LABELS):
        if years <= boundary:
            return label
    return TENOR_BUCKET_LABELS[-1]


def _instrument_type(instrument: object) -> str:
    return type(instrument).__name__


def _instrument_notional(instrument: object) -> float:
    if hasattr(instrument, "notional"):
        return instrument.notional
    return 0.0


def _instrument_end(instrument: object) -> date | None:
    for attr in ("end", "swap_end", "maturity", "expiry"):
        val = getattr(instrument, attr, None)
        if val is not None:
            return val
    return None


# ---- Position ----

@dataclass
class Position:
    """Net exposure in a single bucket."""
    instrument_type: str
    tenor_bucket: str
    net_notional: float
    trade_count: int


# ---- Limit framework ----

@dataclass
class LimitBreach:
    """A limit violation."""
    limit_type: str
    limit_name: str
    limit_value: float
    actual_value: float
    details: str = ""


@dataclass
class BookLimits:
    """Position limits for a book.

    Attributes:
        max_dv01: maximum absolute DV01 for the book.
        max_notional_per_counterparty: per-counterparty notional cap.
        tenor_dv01_limits: DV01 limit per tenor bucket.
    """
    max_dv01: float | None = None
    max_notional_per_counterparty: dict[str, float] = field(default_factory=dict)
    tenor_dv01_limits: dict[str, float] = field(default_factory=dict)


# ---- Book ----

class Book:
    """A named container of trades.

    Args:
        name: book name (e.g. "USD_Swaps", "EUR_Vol").
        limits: optional position limits.
    """

    def __init__(self, name: str, limits: BookLimits | None = None):
        self.name = name
        self.limits = limits or BookLimits()
        self._trades: list[Trade] = []

    def add(self, trade: Trade) -> None:
        self._trades.append(trade)

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def __len__(self) -> int:
        return len(self._trades)

    def pv(self, ctx: PricingContext) -> float:
        return sum(t.pv(ctx) for t in self._trades)

    def pv_by_trade(self, ctx: PricingContext) -> list[tuple[str, float]]:
        return [
            (t.trade_id or f"trade_{i}", t.pv(ctx))
            for i, t in enumerate(self._trades)
        ]

    def dv01(self, ctx: PricingContext, shift: float = 0.0001) -> float:
        if ctx.discount_curve is None:
            return 0.0
        pv_base = self.pv(ctx)
        bumped = ctx.replace(discount_curve=ctx.discount_curve.bumped(shift))
        return self.pv(bumped) - pv_base

    def positions(self, ctx: PricingContext) -> list[Position]:
        """Aggregate positions by instrument type and tenor bucket."""
        buckets: dict[tuple[str, str], tuple[float, int]] = {}
        val = ctx.valuation_date
        for trade in self._trades:
            inst_type = _instrument_type(trade.instrument)
            end = _instrument_end(trade.instrument)
            bucket = tenor_bucket(val, end) if end else "unknown"
            key = (inst_type, bucket)
            notional = trade.direction * trade.notional_scale * _instrument_notional(trade.instrument)
            prev_not, prev_cnt = buckets.get(key, (0.0, 0))
            buckets[key] = (prev_not + notional, prev_cnt + 1)
        return [
            Position(t, b, n, c)
            for (t, b), (n, c) in sorted(buckets.items())
        ]

    def notional_by_counterparty(self) -> dict[str, float]:
        """Aggregate absolute notional by counterparty."""
        result: dict[str, float] = {}
        for trade in self._trades:
            cp = trade.counterparty or "unknown"
            notional = abs(trade.notional_scale * _instrument_notional(trade.instrument))
            result[cp] = result.get(cp, 0.0) + notional
        return result

    def tenor_dv01(self, ctx: PricingContext, shift: float = 0.0001) -> dict[str, float]:
        """DV01 broken down by tenor bucket."""
        if ctx.discount_curve is None:
            return {}
        bumped_ctx = ctx.replace(discount_curve=ctx.discount_curve.bumped(shift))
        val = ctx.valuation_date
        result: dict[str, float] = {}
        for trade in self._trades:
            end = _instrument_end(trade.instrument)
            bucket = tenor_bucket(val, end) if end else "unknown"
            dv01 = trade.pv(bumped_ctx) - trade.pv(ctx)
            result[bucket] = result.get(bucket, 0.0) + dv01
        return result

    def check_limits(self, ctx: PricingContext) -> list[LimitBreach]:
        """Check all limits and return any breaches."""
        breaches: list[LimitBreach] = []

        if self.limits.max_dv01 is not None:
            actual = abs(self.dv01(ctx))
            if actual > self.limits.max_dv01:
                breaches.append(LimitBreach(
                    "dv01", f"book:{self.name}",
                    self.limits.max_dv01, actual,
                    f"Book DV01 {actual:.0f} exceeds limit {self.limits.max_dv01:.0f}",
                ))

        if self.limits.max_notional_per_counterparty:
            cp_notionals = self.notional_by_counterparty()
            for cp, limit in self.limits.max_notional_per_counterparty.items():
                actual = cp_notionals.get(cp, 0.0)
                if actual > limit:
                    breaches.append(LimitBreach(
                        "counterparty_notional", cp,
                        limit, actual,
                        f"Counterparty {cp} notional {actual:,.0f} exceeds {limit:,.0f}",
                    ))

        if self.limits.tenor_dv01_limits:
            bucket_dv01 = self.tenor_dv01(ctx)
            for bucket, limit in self.limits.tenor_dv01_limits.items():
                actual = abs(bucket_dv01.get(bucket, 0.0))
                if actual > limit:
                    breaches.append(LimitBreach(
                        "tenor_dv01", bucket,
                        limit, actual,
                        f"Tenor {bucket} DV01 {actual:.0f} exceeds {limit:.0f}",
                    ))

        return breaches


# ---- Desk ----

class Desk:
    """A collection of books.

    Args:
        name: desk name (e.g. "Rates", "Credit").
        max_dv01: optional DV01 limit for the entire desk.
    """

    def __init__(self, name: str, max_dv01: float | None = None):
        self.name = name
        self.max_dv01 = max_dv01
        self._books: dict[str, Book] = {}

    def add_book(self, book: Book) -> None:
        if book.name in self._books:
            raise ValueError(f"Book '{book.name}' already in desk {self.name}")
        self._books[book.name] = book

    def get_book(self, name: str) -> Book:
        if name not in self._books:
            raise KeyError(f"Book '{name}' not in desk {self.name}")
        return self._books[name]

    @property
    def books(self) -> dict[str, Book]:
        return dict(self._books)

    def pv(self, ctx: PricingContext) -> float:
        return sum(b.pv(ctx) for b in self._books.values())

    def pv_by_book(self, ctx: PricingContext) -> dict[str, float]:
        return {name: b.pv(ctx) for name, b in self._books.items()}

    def dv01(self, ctx: PricingContext, shift: float = 0.0001) -> float:
        if ctx.discount_curve is None:
            return 0.0
        pv_base = self.pv(ctx)
        bumped = ctx.replace(discount_curve=ctx.discount_curve.bumped(shift))
        return self.pv(bumped) - pv_base

    def positions(self, ctx: PricingContext) -> list[Position]:
        """Aggregate positions across all books."""
        merged: dict[tuple[str, str], tuple[float, int]] = {}
        for book in self._books.values():
            for pos in book.positions(ctx):
                key = (pos.instrument_type, pos.tenor_bucket)
                prev_n, prev_c = merged.get(key, (0.0, 0))
                merged[key] = (prev_n + pos.net_notional, prev_c + pos.trade_count)
        return [
            Position(t, b, n, c)
            for (t, b), (n, c) in sorted(merged.items())
        ]

    def check_limits(self, ctx: PricingContext) -> list[LimitBreach]:
        """Check desk-level and book-level limits."""
        breaches: list[LimitBreach] = []

        if self.max_dv01 is not None:
            actual = abs(self.dv01(ctx))
            if actual > self.max_dv01:
                breaches.append(LimitBreach(
                    "dv01", f"desk:{self.name}",
                    self.max_dv01, actual,
                    f"Desk DV01 {actual:.0f} exceeds limit {self.max_dv01:.0f}",
                ))

        for book in self._books.values():
            breaches.extend(book.check_limits(ctx))

        return breaches

    def risk_summary(self, ctx: PricingContext) -> dict[str, Any]:
        """Desk-level risk summary."""
        return {
            "desk": self.name,
            "n_books": len(self._books),
            "total_pv": self.pv(ctx),
            "total_dv01": self.dv01(ctx),
            "pv_by_book": self.pv_by_book(ctx),
            "positions": [
                {
                    "type": p.instrument_type,
                    "bucket": p.tenor_bucket,
                    "net_notional": p.net_notional,
                    "trade_count": p.trade_count,
                }
                for p in self.positions(ctx)
            ],
            "breaches": [
                {
                    "limit_type": b.limit_type,
                    "limit_name": b.limit_name,
                    "limit": b.limit_value,
                    "actual": b.actual_value,
                    "details": b.details,
                }
                for b in self.check_limits(ctx)
            ],
        }
