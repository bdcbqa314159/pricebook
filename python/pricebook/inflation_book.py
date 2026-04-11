"""Inflation position management: InflationBook, IE01, limits.

* :class:`InflationBook` — linker positions, ZC/YoY swaps, caps/floors.
* :class:`InflationPosition` — per-issuer aggregation with IE01.
* :class:`InflationLimits` — per-issuer, IE01, real rate limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.trade import Trade


@dataclass
class InflationTradeEntry:
    """An inflation trade with desk metadata."""
    trade: Trade
    issuer: str
    product_type: str = "linker"  # linker, zc_swap, yoy_swap, cap, floor
    currency: str = "USD"
    notional: float = 0.0
    maturity: date | None = None
    real_yield: float = 0.0
    breakeven: float = 0.0
    ie01: float = 0.0           # breakeven DV01 per 1M notional
    real_dv01: float = 0.0      # real rate DV01 per 1M notional


@dataclass
class InflationPosition:
    """Aggregated inflation position per issuer."""
    issuer: str
    product_type: str
    net_notional: float
    long_notional: float
    short_notional: float
    net_ie01: float
    net_real_dv01: float
    trade_count: int


@dataclass
class InflationLimitBreach:
    limit_type: str
    limit_name: str
    limit_value: float
    actual_value: float
    details: str = ""


@dataclass
class InflationLimits:
    max_notional_per_issuer: dict[str, float] = field(default_factory=dict)
    max_ie01: float | None = None
    max_real_dv01: float | None = None


class InflationBook:
    """Inflation position book with per-issuer/type aggregation and limits."""

    def __init__(self, name: str, limits: InflationLimits | None = None, currency: str = "USD"):
        self.name = name
        self.limits = limits or InflationLimits()
        self.currency = currency
        self._entries: list[InflationTradeEntry] = []

    def add(self, trade: Trade, issuer: str, product_type: str = "linker",
            currency: str | None = None, notional: float = 0.0,
            maturity: date | None = None, real_yield: float = 0.0,
            breakeven: float = 0.0, ie01: float = 0.0, real_dv01: float = 0.0) -> None:
        self._entries.append(InflationTradeEntry(
            trade=trade, issuer=issuer, product_type=product_type,
            currency=currency or self.currency, notional=notional,
            maturity=maturity, real_yield=real_yield, breakeven=breakeven,
            ie01=ie01, real_dv01=real_dv01,
        ))

    @property
    def entries(self): return list(self._entries)
    def __len__(self): return len(self._entries)
    @property
    def n_issuers(self): return len({e.issuer for e in self._entries})

    def positions_by_issuer(self) -> list[InflationPosition]:
        agg: dict[str, dict] = {}
        for e in self._entries:
            sign = e.trade.direction * e.trade.notional_scale
            signed = sign * e.notional
            if e.issuer not in agg:
                agg[e.issuer] = {"type": e.product_type, "net": 0.0, "long": 0.0,
                                  "short": 0.0, "ie01": 0.0, "rdv01": 0.0, "count": 0}
            d = agg[e.issuer]
            d["net"] += signed
            d["ie01"] += sign * e.notional * e.ie01 / 1_000_000
            d["rdv01"] += sign * e.notional * e.real_dv01 / 1_000_000
            if signed > 0: d["long"] += signed
            elif signed < 0: d["short"] += -signed
            d["count"] += 1
        return [
            InflationPosition(iss, d["type"], d["net"], d["long"], d["short"],
                              d["ie01"], d["rdv01"], d["count"])
            for iss, d in sorted(agg.items())
        ]

    def net_ie01(self) -> float:
        return sum(
            e.trade.direction * e.trade.notional_scale * e.notional * e.ie01 / 1_000_000
            for e in self._entries
        )

    def net_real_dv01(self) -> float:
        return sum(
            e.trade.direction * e.trade.notional_scale * e.notional * e.real_dv01 / 1_000_000
            for e in self._entries
        )

    def check_limits(self) -> list[InflationLimitBreach]:
        breaches: list[InflationLimitBreach] = []
        if self.limits.max_notional_per_issuer:
            issuer_n = {p.issuer: abs(p.net_notional) for p in self.positions_by_issuer()}
            for iss, lim in self.limits.max_notional_per_issuer.items():
                actual = issuer_n.get(iss, 0.0)
                if actual > lim:
                    breaches.append(InflationLimitBreach("per_issuer", iss, lim, actual))
        if self.limits.max_ie01 is not None:
            actual = abs(self.net_ie01())
            if actual > self.limits.max_ie01:
                breaches.append(InflationLimitBreach("ie01", self.name, self.limits.max_ie01, actual))
        if self.limits.max_real_dv01 is not None:
            actual = abs(self.net_real_dv01())
            if actual > self.limits.max_real_dv01:
                breaches.append(InflationLimitBreach("real_dv01", self.name, self.limits.max_real_dv01, actual))
        return breaches
