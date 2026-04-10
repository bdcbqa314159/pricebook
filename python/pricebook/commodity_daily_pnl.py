"""Commodity daily P&L: mark-to-market, carry, roll, attribution.

Mirrors the IR / equity desk pattern (:mod:`pricebook.daily_pnl`,
:mod:`pricebook.equity_daily_pnl`) for the commodity world. The official
P&L decomposes into:

* **spot**       — forward curve moves at constant delivery dates,
* **carry**      — convenience yield earned minus storage cost accrued,
* **roll**       — gain/loss when a position is rolled to a new delivery,
* **new trades** — value of trades booked today,
* **amendments** — caller-supplied per-trade adjustments.

The attribution helper additionally splits the spot move into a
parallel-shift component and a curve-shape residual, and produces
per-commodity / per-tenor breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.commodity_book import (
    CommodityBook,
    CommodityTradeEntry,
    commodity_tenor_bucket,
)


# ---- Official P&L ----

@dataclass
class CommodityDailyPnL:
    """Official daily P&L breakdown for a commodity book."""
    book_name: str
    prior_date: date
    current_date: date
    spot_pnl: float
    carry_pnl: float
    roll_pnl: float
    new_trade_pnl: float
    amendment_pnl: float
    total_pnl: float

    @property
    def market_move_pnl(self) -> float:
        return self.spot_pnl + self.carry_pnl + self.roll_pnl


def _signed_qty(entry: CommodityTradeEntry) -> float:
    return entry.trade.direction * entry.trade.notional_scale * entry.quantity


def _curve_lookup(
    curves: dict[str, dict[date, float]],
    commodity: str,
    delivery: date,
) -> float:
    return curves.get(commodity, {}).get(delivery, 0.0)


def compute_commodity_daily_pnl(
    book: CommodityBook,
    prior_curves: dict[str, dict[date, float]],
    current_curves: dict[str, dict[date, float]],
    prior_date: date,
    current_date: date,
    storage_rates: dict[str, float] | None = None,
    convenience_yields: dict[str, float] | None = None,
    rolls: list[tuple[str, date, date]] | None = None,
    new_trades: list[CommodityTradeEntry] | None = None,
    amendments: dict[str, float] | None = None,
) -> CommodityDailyPnL:
    """Compute the official daily P&L for a commodity book.

    Args:
        book: yesterday's commodity book (positions as of EOD prior).
        prior_curves: {commodity → {delivery_date → forward}} at prior date.
        current_curves: same shape, at current date.
        prior_date: as-of date of ``prior_curves``.
        current_date: as-of date of ``current_curves``.
        storage_rates: per-commodity annualised storage cost.
        convenience_yields: per-commodity annualised convenience yield.
        rolls: list of (commodity, old_delivery, new_delivery) for any
            positions rolled today; the roll P&L is the curve spread
            ``new_fwd − old_fwd`` at the current valuation.
        new_trades: trades entered today, each priced as
            ``qty × (curr_fwd(delivery) − reference_price)``.
        amendments: trade_id → P&L impact of amendments.

    Returns:
        :class:`CommodityDailyPnL` with full decomposition.
    """
    storage_rates = storage_rates or {}
    convenience_yields = convenience_yields or {}
    rolls_by_old: dict[tuple[str, date], date] = {
        (c, old): new for c, old, new in (rolls or [])
    }

    dt_days = (current_date - prior_date).days
    dt_year = dt_days / 365.0

    spot_pnl = 0.0
    carry_pnl = 0.0
    roll_pnl = 0.0

    for entry in book.entries:
        if entry.delivery_date is None:
            continue
        commodity = entry.commodity
        if commodity not in prior_curves or commodity not in current_curves:
            continue

        sq = _signed_qty(entry)
        delivery = entry.delivery_date

        prior_fwd = _curve_lookup(prior_curves, commodity, delivery)
        current_fwd = _curve_lookup(current_curves, commodity, delivery)
        spot_pnl += sq * (current_fwd - prior_fwd)

        cy = convenience_yields.get(commodity, 0.0)
        sc = storage_rates.get(commodity, 0.0)
        carry_pnl += sq * entry.reference_price * (cy - sc) * dt_year

        roll_key = (commodity, delivery)
        if roll_key in rolls_by_old:
            new_delivery = rolls_by_old[roll_key]
            new_fwd = _curve_lookup(current_curves, commodity, new_delivery)
            roll_pnl += sq * (new_fwd - current_fwd)

    new_trade_pnl = 0.0
    if new_trades:
        for entry in new_trades:
            if entry.delivery_date is None:
                continue
            sq = _signed_qty(entry)
            current_fwd = _curve_lookup(
                current_curves, entry.commodity, entry.delivery_date,
            )
            new_trade_pnl += sq * (current_fwd - entry.reference_price)

    amendment_pnl = sum(amendments.values()) if amendments else 0.0

    total = spot_pnl + carry_pnl + roll_pnl + new_trade_pnl + amendment_pnl

    return CommodityDailyPnL(
        book_name=book.name,
        prior_date=prior_date,
        current_date=current_date,
        spot_pnl=spot_pnl,
        carry_pnl=carry_pnl,
        roll_pnl=roll_pnl,
        new_trade_pnl=new_trade_pnl,
        amendment_pnl=amendment_pnl,
        total_pnl=total,
    )


# ---- Attribution ----

@dataclass
class CommodityAttribution:
    """Per-commodity / per-tenor / parallel-vs-shape attribution."""
    book_name: str
    total_pnl: float
    spot_pnl: float
    carry_pnl: float
    roll_pnl: float
    parallel_pnl: float          # Σ qty × ⟨Δfwd⟩_per_commodity
    shape_pnl: float             # spot_pnl − parallel_pnl
    by_commodity: dict[str, dict[str, float]]
    by_tenor: dict[str, dict[str, float]]


def attribute_commodity_pnl(
    book: CommodityBook,
    prior_curves: dict[str, dict[date, float]],
    current_curves: dict[str, dict[date, float]],
    prior_date: date,
    current_date: date,
    storage_rates: dict[str, float] | None = None,
    convenience_yields: dict[str, float] | None = None,
    rolls: list[tuple[str, date, date]] | None = None,
) -> CommodityAttribution:
    """Decompose P&L by commodity, tenor bucket, and parallel-vs-shape.

    The parallel component is computed per commodity as the average forward
    move across the deliveries actually held in that commodity:

        parallel_pnl_c = (Σ qty_c) × mean_d(Δfwd_c(d))

    The shape residual is the rest of the spot P&L. Carry and roll are
    not split into parallel/shape — they appear unchanged in the
    per-commodity / per-tenor buckets.
    """
    storage_rates = storage_rates or {}
    convenience_yields = convenience_yields or {}
    rolls_by_old: dict[tuple[str, date], date] = {
        (c, old): new for c, old, new in (rolls or [])
    }

    dt_days = (current_date - prior_date).days
    dt_year = dt_days / 365.0

    by_commodity: dict[str, dict[str, float]] = {}
    by_tenor: dict[str, dict[str, float]] = {}

    spot_pnl = 0.0
    carry_pnl = 0.0
    roll_pnl = 0.0
    parallel_pnl = 0.0

    # Per-commodity sums needed for parallel decomposition
    per_c_signed_qty: dict[str, float] = {}
    per_c_delta_fwds: dict[str, list[float]] = {}

    for entry in book.entries:
        if entry.delivery_date is None:
            continue
        commodity = entry.commodity
        if commodity not in prior_curves or commodity not in current_curves:
            continue

        sq = _signed_qty(entry)
        delivery = entry.delivery_date
        bucket = commodity_tenor_bucket(prior_date, delivery)

        prior_fwd = _curve_lookup(prior_curves, commodity, delivery)
        current_fwd = _curve_lookup(current_curves, commodity, delivery)
        d_fwd = current_fwd - prior_fwd
        e_spot = sq * d_fwd

        cy = convenience_yields.get(commodity, 0.0)
        sc = storage_rates.get(commodity, 0.0)
        e_carry = sq * entry.reference_price * (cy - sc) * dt_year

        e_roll = 0.0
        roll_key = (commodity, delivery)
        if roll_key in rolls_by_old:
            new_delivery = rolls_by_old[roll_key]
            new_fwd = _curve_lookup(current_curves, commodity, new_delivery)
            e_roll = sq * (new_fwd - current_fwd)

        spot_pnl += e_spot
        carry_pnl += e_carry
        roll_pnl += e_roll

        per_c_signed_qty[commodity] = per_c_signed_qty.get(commodity, 0.0) + sq
        per_c_delta_fwds.setdefault(commodity, []).append(d_fwd)

        c_bucket = by_commodity.setdefault(commodity, {
            "spot_pnl": 0.0, "carry_pnl": 0.0, "roll_pnl": 0.0, "total_pnl": 0.0,
        })
        c_bucket["spot_pnl"] += e_spot
        c_bucket["carry_pnl"] += e_carry
        c_bucket["roll_pnl"] += e_roll
        c_bucket["total_pnl"] += e_spot + e_carry + e_roll

        t_bucket = by_tenor.setdefault(bucket, {
            "spot_pnl": 0.0, "carry_pnl": 0.0, "roll_pnl": 0.0, "total_pnl": 0.0,
        })
        t_bucket["spot_pnl"] += e_spot
        t_bucket["carry_pnl"] += e_carry
        t_bucket["roll_pnl"] += e_roll
        t_bucket["total_pnl"] += e_spot + e_carry + e_roll

    # Parallel attribution: per-commodity mean Δfwd × signed qty
    for commodity, deltas in per_c_delta_fwds.items():
        if not deltas:
            continue
        mean_d = sum(deltas) / len(deltas)
        parallel_pnl += per_c_signed_qty[commodity] * mean_d

    shape_pnl = spot_pnl - parallel_pnl
    total_pnl = spot_pnl + carry_pnl + roll_pnl

    return CommodityAttribution(
        book_name=book.name,
        total_pnl=total_pnl,
        spot_pnl=spot_pnl,
        carry_pnl=carry_pnl,
        roll_pnl=roll_pnl,
        parallel_pnl=parallel_pnl,
        shape_pnl=shape_pnl,
        by_commodity=by_commodity,
        by_tenor=by_tenor,
    )
