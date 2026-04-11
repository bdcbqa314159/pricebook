"""Bond daily P&L: mark-to-market, coupon accrual, attribution.

Official P&L decomposes into:
* **MTM** — dirty price change × face amount,
* **coupon accrual** — interest earned over the day,
* **new trades** — day-one P&L of trades booked today,
* **amendments** — caller-supplied adjustments.

Attribution further decomposes the MTM into:
* **carry** — coupon income net of financing cost,
* **roll-down** — aging effect along the yield curve,
* **curve move** — parallel DV01 × parallel shift,
* **spread move** — credit spread DV01 × spread change,
* **unexplained** — residual.

    pnl = compute_bond_daily_pnl(book, prior_prices, current_prices, ...)
    attrib = attribute_bond_pnl(book, prior_prices, current_prices, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.bond_book import BondBook, BondTradeEntry, bond_tenor_bucket


# ---- Official P&L ----

@dataclass
class BondDailyPnL:
    """Official daily P&L breakdown for a bond book."""
    book_name: str
    prior_date: date
    current_date: date
    mtm_pnl: float
    accrual_pnl: float
    new_trade_pnl: float
    amendment_pnl: float
    total_pnl: float

    @property
    def market_move_pnl(self) -> float:
        return self.mtm_pnl + self.accrual_pnl


def _signed(entry: BondTradeEntry) -> float:
    return entry.trade.direction * entry.trade.notional_scale


def compute_bond_daily_pnl(
    book: BondBook,
    prior_prices: dict[str, float],
    current_prices: dict[str, float],
    prior_date: date,
    current_date: date,
    accrual_rates: dict[str, float] | None = None,
    new_trades: list[BondTradeEntry] | None = None,
    amendments: dict[str, float] | None = None,
) -> BondDailyPnL:
    """Compute official daily P&L for a bond book.

    Args:
        book: yesterday's bond book (EOD positions).
        prior_prices: {issuer → dirty_price} at prior date.
        current_prices: {issuer → dirty_price} at current date.
        prior_date: as-of date of prior_prices.
        current_date: as-of date of current_prices.
        accrual_rates: {issuer → annual_coupon_rate} for daily accrual.
            If None, uses the coupon_rate from each BondTradeEntry.
        new_trades: trades entered today — P&L = face × (current - ref) / 100.
        amendments: trade_id → P&L impact.

    Returns:
        :class:`BondDailyPnL` with full decomposition.
    """
    accrual_rates = accrual_rates or {}
    dt_days = (current_date - prior_date).days

    mtm_pnl = 0.0
    accrual_pnl = 0.0

    for entry in book.entries:
        sign = _signed(entry)
        issuer = entry.issuer

        prior_dp = prior_prices.get(issuer, entry.dirty_price)
        current_dp = current_prices.get(issuer, entry.dirty_price)

        # MTM: price change × face
        mtm_pnl += sign * entry.face_amount * (current_dp - prior_dp) / 100.0

        # Daily coupon accrual
        coupon = accrual_rates.get(issuer, entry.coupon_rate)
        accrual_pnl += sign * entry.face_amount * coupon * dt_days / 365.0

    new_trade_pnl = 0.0
    if new_trades:
        for entry in new_trades:
            sign = _signed(entry)
            current_dp = current_prices.get(entry.issuer, entry.dirty_price)
            new_trade_pnl += sign * entry.face_amount * (current_dp - entry.dirty_price) / 100.0

    amendment_pnl = sum(amendments.values()) if amendments else 0.0

    total = mtm_pnl + accrual_pnl + new_trade_pnl + amendment_pnl

    return BondDailyPnL(
        book_name=book.name,
        prior_date=prior_date,
        current_date=current_date,
        mtm_pnl=mtm_pnl,
        accrual_pnl=accrual_pnl,
        new_trade_pnl=new_trade_pnl,
        amendment_pnl=amendment_pnl,
        total_pnl=total,
    )


# ---- Attribution ----

@dataclass
class BondTradeAttribution:
    """P&L attribution for a single bond position."""
    issuer: str
    tenor_bucket: str
    total_pnl: float
    carry_pnl: float
    rolldown_pnl: float
    curve_pnl: float
    spread_pnl: float
    unexplained: float

    @property
    def explained(self) -> float:
        return (
            self.carry_pnl + self.rolldown_pnl
            + self.curve_pnl + self.spread_pnl
        )


@dataclass
class BondBookAttribution:
    """Aggregate attribution for a bond book."""
    book_name: str
    total_pnl: float
    carry_pnl: float
    rolldown_pnl: float
    curve_pnl: float
    spread_pnl: float
    unexplained: float
    by_trade: list[BondTradeAttribution]
    by_issuer: dict[str, dict[str, float]]
    by_tenor: dict[str, dict[str, float]]


def attribute_bond_pnl(
    book: BondBook,
    prior_prices: dict[str, float],
    current_prices: dict[str, float],
    prior_date: date,
    current_date: date,
    parallel_shift: float = 0.0,
    spread_changes: dict[str, float] | None = None,
    rolldown_prices: dict[str, float] | None = None,
    financing_rates: dict[str, float] | None = None,
) -> BondBookAttribution:
    """Decompose bond P&L into carry, roll-down, curve, spread, unexplained.

    Args:
        book: bond book (yesterday's positions).
        prior_prices: {issuer → dirty_price} at prior date.
        current_prices: {issuer → dirty_price} at current date.
        prior_date / current_date: valuation dates.
        parallel_shift: observed parallel yield shift (in bps, e.g. 5.0 = 5bp).
        spread_changes: {issuer → spread change in bps}.
        rolldown_prices: {issuer → dirty_price from aging the prior curve by 1 day
            without moving yields}. If None, roll-down is zero.
        financing_rates: {issuer → repo/financing rate} for carry.
            If None, financing cost is zero (carry = coupon only).

    Returns:
        :class:`BondBookAttribution`.
    """
    spread_changes = spread_changes or {}
    rolldown_prices = rolldown_prices or {}
    financing_rates = financing_rates or {}

    dt_days = (current_date - prior_date).days
    dt_year = dt_days / 365.0

    by_trade: list[BondTradeAttribution] = []

    for entry in book.entries:
        sign = _signed(entry)
        issuer = entry.issuer
        bucket = bond_tenor_bucket(prior_date, entry.maturity)

        prior_dp = prior_prices.get(issuer, entry.dirty_price)
        current_dp = current_prices.get(issuer, entry.dirty_price)
        total = sign * entry.face_amount * (current_dp - prior_dp) / 100.0

        # Carry = (coupon income - financing cost) per day
        coupon_income = sign * entry.face_amount * entry.coupon_rate * dt_year
        fin_rate = financing_rates.get(issuer, 0.0)
        financing_cost = sign * entry.face_amount * (prior_dp / 100.0) * fin_rate * dt_year
        carry = coupon_income - financing_cost

        # Roll-down: price change from aging without yield moves
        rd_price = rolldown_prices.get(issuer)
        if rd_price is not None:
            rolldown = sign * entry.face_amount * (rd_price - prior_dp) / 100.0
        else:
            rolldown = 0.0

        # Curve move: DV01 × parallel shift (shift in bps)
        dv01 = sign * entry.face_amount * entry.dv01_per_million / 1_000_000.0
        curve = dv01 * parallel_shift

        # Spread move: spread DV01 × spread change (in bps)
        ds = spread_changes.get(issuer, 0.0)
        spread = dv01 * ds

        unexplained = total - (carry + rolldown + curve + spread)

        by_trade.append(BondTradeAttribution(
            issuer=issuer, tenor_bucket=bucket,
            total_pnl=total, carry_pnl=carry, rolldown_pnl=rolldown,
            curve_pnl=curve, spread_pnl=spread, unexplained=unexplained,
        ))

    # Aggregate
    total_pnl = sum(a.total_pnl for a in by_trade)
    carry_pnl = sum(a.carry_pnl for a in by_trade)
    rolldown_pnl = sum(a.rolldown_pnl for a in by_trade)
    curve_pnl = sum(a.curve_pnl for a in by_trade)
    spread_pnl = sum(a.spread_pnl for a in by_trade)
    unexplained = sum(a.unexplained for a in by_trade)

    by_issuer: dict[str, dict[str, float]] = {}
    by_tenor: dict[str, dict[str, float]] = {}

    for a in by_trade:
        for key, bucket_dict in [(a.issuer, by_issuer), (a.tenor_bucket, by_tenor)]:
            if key not in bucket_dict:
                bucket_dict[key] = {
                    "total_pnl": 0.0, "carry_pnl": 0.0, "rolldown_pnl": 0.0,
                    "curve_pnl": 0.0, "spread_pnl": 0.0, "unexplained": 0.0,
                }
            d = bucket_dict[key]
            d["total_pnl"] += a.total_pnl
            d["carry_pnl"] += a.carry_pnl
            d["rolldown_pnl"] += a.rolldown_pnl
            d["curve_pnl"] += a.curve_pnl
            d["spread_pnl"] += a.spread_pnl
            d["unexplained"] += a.unexplained

    return BondBookAttribution(
        book_name=book.name,
        total_pnl=total_pnl,
        carry_pnl=carry_pnl,
        rolldown_pnl=rolldown_pnl,
        curve_pnl=curve_pnl,
        spread_pnl=spread_pnl,
        unexplained=unexplained,
        by_trade=by_trade,
        by_issuer=by_issuer,
        by_tenor=by_tenor,
    )
