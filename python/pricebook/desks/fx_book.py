"""FX position management: FXBook, per-pair and per-currency aggregation, limits.

Mirrors the equity/commodity/bond desk pattern for FX. Positions are
tracked by currency pair, with net exposure computed per individual
currency and P&L translated to the reporting currency.

    book = FXBook("G10_Spot", reporting_currency="USD")
    book.add(trade, pair="EUR/USD", direction=1, notional=10_000_000,
             spot_rate=1.0850, forward_points=0.0025)
    positions = book.positions_by_pair()
    ccy_exposures = book.net_currency_exposure()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.trade import Trade


# ---- Position dataclasses ----

@dataclass
class FXPairPosition:
    """Net position in a single currency pair."""
    pair: str
    base_ccy: str
    quote_ccy: str
    net_notional: float       # signed, in base currency
    long_notional: float
    short_notional: float
    net_pv_reporting: float   # PV translated to reporting ccy
    trade_count: int


@dataclass
class CurrencyExposure:
    """Net exposure to a single currency (across all pairs)."""
    currency: str
    net_exposure: float       # positive = long, negative = short


@dataclass
class FXLimitBreach:
    """Limit violation for an FX book."""
    limit_type: str
    limit_name: str
    limit_value: float
    actual_value: float
    details: str = ""


@dataclass
class FXLimits:
    """Position limits for an FX book.

    Attributes:
        max_notional_per_pair: per-pair notional cap.
        max_exposure_per_ccy: per-currency net exposure cap.
        max_gross_notional: gross notional across all pairs.
    """
    max_notional_per_pair: dict[str, float] = field(default_factory=dict)
    max_exposure_per_ccy: dict[str, float] = field(default_factory=dict)
    max_gross_notional: float | None = None


# ---- Trade entry ----

@dataclass
class FXTradeEntry:
    """An FX trade with desk metadata.

    ``pair`` uses the "BASE/QUOTE" convention (e.g. "EUR/USD").
    ``notional`` is always in the base currency. A long EUR/USD trade
    means buying EUR and selling USD.

    ``spot_rate`` is the FX spot at trade inception.
    ``forward_points`` are the pip-adjusted forward premium.
    """
    trade: Trade
    pair: str
    notional: float = 0.0
    spot_rate: float = 0.0
    forward_points: float = 0.0
    reporting_rate: float = 1.0  # pair_ccy → reporting_ccy FX rate

    @property
    def base_ccy(self) -> str:
        return self.pair.split("/")[0]

    @property
    def quote_ccy(self) -> str:
        return self.pair.split("/")[1]


# ---- FX Book ----

class FXBook:
    """A named container of FX trades with per-pair and per-currency aggregation.

    Args:
        name: book name.
        reporting_currency: base currency for P&L aggregation.
        limits: optional :class:`FXLimits`.
    """

    def __init__(
        self,
        name: str,
        reporting_currency: str = "USD",
        limits: FXLimits | None = None,
    ):
        self.name = name
        self.reporting_currency = reporting_currency
        self.limits = limits or FXLimits()
        self._entries: list[FXTradeEntry] = []

    def add(
        self,
        trade: Trade,
        pair: str,
        notional: float = 0.0,
        spot_rate: float = 0.0,
        forward_points: float = 0.0,
        reporting_rate: float = 1.0,
    ) -> None:
        """Add an FX trade with metadata."""
        self._entries.append(FXTradeEntry(
            trade=trade, pair=pair, notional=notional,
            spot_rate=spot_rate, forward_points=forward_points,
            reporting_rate=reporting_rate,
        ))

    @property
    def entries(self) -> list[FXTradeEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def n_pairs(self) -> int:
        return len({e.pair for e in self._entries})

    # ---- Aggregations ----

    def positions_by_pair(self) -> list[FXPairPosition]:
        """Aggregate net positions per currency pair."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            sign = e.trade.direction * e.trade.notional_scale
            signed_notional = sign * e.notional

            if e.pair not in agg:
                agg[e.pair] = {
                    "pair": e.pair, "base": e.base_ccy, "quote": e.quote_ccy,
                    "net": 0.0, "long": 0.0, "short": 0.0,
                    "pv_rpt": 0.0, "count": 0,
                }
            d = agg[e.pair]
            d["net"] += signed_notional
            d["pv_rpt"] += signed_notional * e.reporting_rate
            if signed_notional > 0:
                d["long"] += signed_notional
            elif signed_notional < 0:
                d["short"] += -signed_notional
            d["count"] += 1

        return [
            FXPairPosition(
                pair=d["pair"], base_ccy=d["base"], quote_ccy=d["quote"],
                net_notional=d["net"], long_notional=d["long"],
                short_notional=d["short"], net_pv_reporting=d["pv_rpt"],
                trade_count=d["count"],
            )
            for d in sorted(agg.values(), key=lambda x: x["pair"])
        ]

    def net_currency_exposure(self) -> list[CurrencyExposure]:
        """Net exposure per individual currency across all pairs.

        A long EUR/USD trade creates +EUR exposure and −USD exposure
        (notional × spot_rate in USD terms).
        """
        ccy_agg: dict[str, float] = {}
        for e in self._entries:
            sign = e.trade.direction * e.trade.notional_scale
            signed = sign * e.notional

            # Long base = +base, −quote (in quote terms: notional × spot)
            base = e.base_ccy
            quote = e.quote_ccy
            ccy_agg[base] = ccy_agg.get(base, 0.0) + signed
            ccy_agg[quote] = ccy_agg.get(quote, 0.0) - signed * e.spot_rate

        return [
            CurrencyExposure(ccy, exp)
            for ccy, exp in sorted(ccy_agg.items())
        ]

    def gross_notional(self) -> float:
        """Sum of absolute notionals across all trades."""
        return sum(
            abs(e.trade.direction * e.trade.notional_scale * e.notional)
            for e in self._entries
        )

    # ---- Limits ----

    def check_limits(self) -> list[FXLimitBreach]:
        """Check all configured limits and return any breaches."""
        breaches: list[FXLimitBreach] = []

        if self.limits.max_notional_per_pair:
            pair_notionals = {
                p.pair: abs(p.net_notional) for p in self.positions_by_pair()
            }
            for pair, lim in self.limits.max_notional_per_pair.items():
                actual = pair_notionals.get(pair, 0.0)
                if actual > lim:
                    breaches.append(FXLimitBreach(
                        "per_pair", pair, lim, actual,
                        f"{pair} notional {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_exposure_per_ccy:
            ccy_exp = {c.currency: abs(c.net_exposure) for c in self.net_currency_exposure()}
            for ccy, lim in self.limits.max_exposure_per_ccy.items():
                actual = ccy_exp.get(ccy, 0.0)
                if actual > lim:
                    breaches.append(FXLimitBreach(
                        "per_ccy", ccy, lim, actual,
                        f"{ccy} exposure {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_gross_notional is not None:
            actual = self.gross_notional()
            if actual > self.limits.max_gross_notional:
                breaches.append(FXLimitBreach(
                    "gross_notional", f"book:{self.name}",
                    self.limits.max_gross_notional, actual,
                    f"gross notional {actual:,.0f} exceeds {self.limits.max_gross_notional:,.0f}",
                ))

        return breaches
