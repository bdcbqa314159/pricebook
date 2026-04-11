"""FX daily P&L: spot move, carry, basis, attribution.

Official P&L decomposes into:
* **spot** — Δspot × position,
* **carry** — forward points (interest rate differential) earned,
* **basis** — cross-currency basis change,
* **new trades** — day-one P&L,
* **amendments** — caller-supplied adjustments.

Attribution splits the total by pair and by currency.

    pnl = compute_fx_daily_pnl(book, prior_spots, current_spots, ...)
    attrib = attribute_fx_pnl(book, prior_spots, current_spots, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.fx_book import FXBook


# ---- Official P&L ----

@dataclass
class FXDailyPnL:
    """Official daily P&L breakdown for an FX book."""
    book_name: str
    prior_date: date
    current_date: date
    spot_pnl: float
    carry_pnl: float
    basis_pnl: float
    new_trade_pnl: float
    amendment_pnl: float
    total_pnl: float


def compute_fx_daily_pnl(
    book: FXBook,
    prior_spots: dict[str, float],
    current_spots: dict[str, float],
    prior_date: date,
    current_date: date,
    carry_points: dict[str, float] | None = None,
    basis_changes: dict[str, float] | None = None,
    new_trade_pnls: dict[str, float] | None = None,
    amendments: dict[str, float] | None = None,
) -> FXDailyPnL:
    """Compute official daily P&L for an FX book.

    Args:
        book: yesterday's FX book (EOD positions).
        prior_spots: {pair → spot} at prior date.
        current_spots: {pair → spot} at current date.
        carry_points: {pair → daily carry in quote ccy per unit base}.
            Typically forward_points / days_to_maturity.
        basis_changes: {pair → basis change in quote ccy per unit base}.
        new_trade_pnls: {trade_id → P&L} for trades booked today.
        amendments: {trade_id → P&L adjustment}.
    """
    carry_points = carry_points or {}
    basis_changes = basis_changes or {}

    spot_pnl = 0.0
    carry_pnl = 0.0
    basis_pnl = 0.0

    for entry in book.entries:
        sign = entry.trade.direction * entry.trade.notional_scale
        signed = sign * entry.notional
        pair = entry.pair

        prior_s = prior_spots.get(pair, entry.spot_rate)
        current_s = current_spots.get(pair, entry.spot_rate)

        spot_pnl += signed * (current_s - prior_s)
        carry_pnl += signed * carry_points.get(pair, 0.0)
        basis_pnl += signed * basis_changes.get(pair, 0.0)

    new_trade_pnl = sum(new_trade_pnls.values()) if new_trade_pnls else 0.0
    amendment_pnl = sum(amendments.values()) if amendments else 0.0

    total = spot_pnl + carry_pnl + basis_pnl + new_trade_pnl + amendment_pnl

    return FXDailyPnL(
        book_name=book.name,
        prior_date=prior_date,
        current_date=current_date,
        spot_pnl=spot_pnl,
        carry_pnl=carry_pnl,
        basis_pnl=basis_pnl,
        new_trade_pnl=new_trade_pnl,
        amendment_pnl=amendment_pnl,
        total_pnl=total,
    )


# ---- Attribution ----

@dataclass
class FXPairAttribution:
    """P&L attribution for a single currency pair."""
    pair: str
    spot_pnl: float
    carry_pnl: float
    basis_pnl: float
    total_pnl: float


@dataclass
class FXCurrencyAttribution:
    """P&L attributed to a single currency (across all pairs)."""
    currency: str
    pnl: float


@dataclass
class FXBookAttribution:
    """Aggregate FX attribution."""
    book_name: str
    total_pnl: float
    spot_pnl: float
    carry_pnl: float
    basis_pnl: float
    by_pair: list[FXPairAttribution]
    by_currency: list[FXCurrencyAttribution]


def attribute_fx_pnl(
    book: FXBook,
    prior_spots: dict[str, float],
    current_spots: dict[str, float],
    prior_date: date,
    current_date: date,
    carry_points: dict[str, float] | None = None,
    basis_changes: dict[str, float] | None = None,
) -> FXBookAttribution:
    """Decompose FX P&L by pair and by currency.

    Per-currency attribution allocates each pair's total P&L to
    both its base and quote currencies (50/50 split for simplicity;
    a more sophisticated approach would weight by the size of each
    leg's FX move).
    """
    carry_points = carry_points or {}
    basis_changes = basis_changes or {}

    pair_agg: dict[str, dict[str, float]] = {}
    ccy_agg: dict[str, float] = {}

    for entry in book.entries:
        sign = entry.trade.direction * entry.trade.notional_scale
        signed = sign * entry.notional
        pair = entry.pair

        prior_s = prior_spots.get(pair, entry.spot_rate)
        current_s = current_spots.get(pair, entry.spot_rate)

        e_spot = signed * (current_s - prior_s)
        e_carry = signed * carry_points.get(pair, 0.0)
        e_basis = signed * basis_changes.get(pair, 0.0)
        e_total = e_spot + e_carry + e_basis

        if pair not in pair_agg:
            pair_agg[pair] = {"spot": 0.0, "carry": 0.0, "basis": 0.0, "total": 0.0}
        pair_agg[pair]["spot"] += e_spot
        pair_agg[pair]["carry"] += e_carry
        pair_agg[pair]["basis"] += e_basis
        pair_agg[pair]["total"] += e_total

        # Attribute to base and quote currencies
        base = entry.base_ccy
        quote = entry.quote_ccy
        ccy_agg[base] = ccy_agg.get(base, 0.0) + e_total * 0.5
        ccy_agg[quote] = ccy_agg.get(quote, 0.0) + e_total * 0.5

    by_pair = [
        FXPairAttribution(pair, d["spot"], d["carry"], d["basis"], d["total"])
        for pair, d in sorted(pair_agg.items())
    ]
    by_currency = [
        FXCurrencyAttribution(ccy, pnl)
        for ccy, pnl in sorted(ccy_agg.items())
    ]

    total = sum(d["total"] for d in pair_agg.values())
    spot = sum(d["spot"] for d in pair_agg.values())
    carry = sum(d["carry"] for d in pair_agg.values())
    basis = sum(d["basis"] for d in pair_agg.values())

    return FXBookAttribution(
        book_name=book.name,
        total_pnl=total,
        spot_pnl=spot,
        carry_pnl=carry,
        basis_pnl=basis,
        by_pair=by_pair,
        by_currency=by_currency,
    )
