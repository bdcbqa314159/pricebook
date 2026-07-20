"""Settlement — how a flow settles (L0).

Finance-free: it describes the *shape* of settlement, not its value. `Cashflow(date,
Money)` (S4) covers cash in the contract currency; this completes the vocabulary — a
physical `Delivery(date, Quantity)` alongside it, a settlement currency that may differ
from the contract currency (quanto / NDF), and settlement-date arithmetic (trade + lag
business days under a calendar).

`AUCTION` is a **marker** only: the CDS credit-event / recovery / auction-price
mechanics belong to the credit topic (higher layer), never L0. Collateral/CSA
discounting is a *numeraire* — an L3 model choice — and is deliberately absent here.

Provenance:
  quarry: python/pricebook/core/settlement.py (zero-fan-in orphan — content mined, structure ignored)
  source: ISDA settlement definitions; ACI Model Code (FX T+2)
  oracle: settlement date = trade + lag under a calendar; cash vs physical flow types;
          settlement currency ≠ contract currency
  slice:  settlement (Topic 0 S4b)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from pricebook_ng.foundation.calendars import CalendarProtocol
from pricebook_ng.foundation.market_calendars import calendar_for_currency
from pricebook_ng.foundation.money import Currency, CurrencyPair, Quantity


class SettlementType(Enum):
    CASH = "cash"
    PHYSICAL = "physical"
    AUCTION = (
        "auction"  # CDS credit event (marker only — mechanics live in the credit topic)
    )


@dataclass(frozen=True)
class Delivery:
    """A physical settlement flow: a `Quantity` delivered on a date — the `Cashflow`
    of the physical world (barrels, MWh, tonnes).

    Scope: this delivers a commodity `Quantity` only. Delivering a **security** — a
    deliverable obligation on a CDS credit event ('settlement on default'), a repo, or a
    bond-future — needs a *security identity*, which is a higher-layer (product/index)
    concept and does not exist yet. Forward-link: *on the credit/product topic, allow a
    security as a delivered obligation.* The credit-event MECHANICS (auction recovery,
    the `(1-R)·N` protection payout) are the credit topic, never L0 — `AUCTION` here is a
    type marker only."""

    date: date
    quantity: Quantity

    def to_dict(self) -> dict:
        return {
            "kind": "delivery",
            "date": self.date.isoformat(),
            "quantity": self.quantity.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Delivery:
        return cls(date.fromisoformat(d["date"]), Quantity.from_dict(d["quantity"]))


@dataclass(frozen=True)
class SettlementTerms:
    """How a contract settles: the type, the settlement currency (which may differ from
    the contract currency for quanto/NDF; `None` for physical delivery), and the lag
    (T+`lag` business days)."""

    settlement_type: SettlementType
    currency: Currency | None
    lag: int = 2

    def __post_init__(self) -> None:
        if self.settlement_type is SettlementType.PHYSICAL:
            if self.currency is not None:
                raise ValueError(
                    "physical settlement delivers a Quantity, not a currency"
                )
        elif self.currency is None:
            raise ValueError(
                f"{self.settlement_type.value} settlement requires a currency"
            )


def settlement_date(
    trade_date: date, terms: SettlementTerms, calendar: CalendarProtocol
) -> date:
    """The settlement date: `trade_date` + `terms.lag` business days on `calendar`."""
    return calendar.add_business_days(trade_date, terms.lag)


# ── FX spot (audit 3.6) ───────────────────────────────────────────────────────────
# Pair-conventions registry: the spot lag by pair name (default T+2). Kept OUT of
# `CurrencyPair` identity so a lag never fragments dict keys. T+1 pairs vs USD:
_SPOT_LAGS: dict[str, int] = {"USDCAD": 1, "USDTRY": 1, "USDPHP": 1}


def spot_lag(pair: CurrencyPair) -> int:
    """The FX spot settlement lag for `pair` (market default T+2; USD/CAD etc. are T+1)."""
    return _SPOT_LAGS.get(pair.name, 2)


def fx_spot_date(pair: CurrencyPair, trade_date: date) -> date:
    """The FX spot value date: `trade_date` + `spot_lag` good business days, where each
    counted day is a business day in BOTH currency centres. For a **cross** (neither leg USD)
    the spot date must additionally be a US business day — a USD holiday cannot be the value
    date, though it does not reduce the count (ACI market practice).

    SCOPE (B3, AC-3.6b): intermediate days are counted **jointly** — a day pauses the count
    if EITHER centre is closed. The stricter asymmetric ACI convention (for a USD pair, a USD
    holiday on an *intermediate* day does not pause the count; only the LatAm-style pairs
    pause on it) is **not** implemented: it would change some USD-pair spot dates by a day, and
    the green-oracle gate forbids coding a market convention without a citable source pinned to
    a verifiable worked example. Revisit when the FX market-data topic supplies one."""
    base_cal = calendar_for_currency(pair.base.code)
    quote_cal = calendar_for_currency(pair.quote.code)
    usd_cal = calendar_for_currency("USD")
    is_cross = "USD" not in (pair.base.code, pair.quote.code)

    d, counted = trade_date, 0
    while counted < spot_lag(pair):
        d += timedelta(days=1)
        if base_cal.is_business_day(d) and quote_cal.is_business_day(d):
            counted += 1
    while not (
        base_cal.is_business_day(d)
        and quote_cal.is_business_day(d)
        and (not is_cross or usd_cal.is_business_day(d))
    ):
        d += timedelta(days=1)
    return d
