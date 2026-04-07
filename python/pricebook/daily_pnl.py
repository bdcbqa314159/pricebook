"""Daily P&L workflow: official P&L and attribution at book/desk level.

Official P&L decomposes into market move, new trades, and amendments.
Attribution breaks market-move P&L into rate, vol, credit, theta,
and unexplained via sequential bump-and-reprice.

    result = compute_daily_pnl(book, prior_ctx, current_ctx, new_trades=[t3])
    attrib = attribute_pnl(book, prior_ctx, current_ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pricebook.book import Book, Desk, tenor_bucket, _instrument_end
from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade


# ---- Official P&L ----

@dataclass
class DailyPnL:
    """Official daily P&L breakdown."""
    book_name: str
    prior_date: date
    current_date: date
    prior_pv: float
    current_pv: float
    market_move_pnl: float
    new_trade_pnl: float
    amendment_pnl: float
    total_pnl: float


def compute_daily_pnl(
    book: Book,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
    new_trades: list[Trade] | None = None,
    amendments: dict[str, float] | None = None,
) -> DailyPnL:
    """Compute official daily P&L for a book.

    Args:
        book: the trading book (yesterday's portfolio).
        prior_ctx: yesterday's market data.
        current_ctx: today's market data.
        new_trades: trades entered today (priced on today's curves).
        amendments: trade_id -> P&L impact of amendments.

    Returns:
        DailyPnL with full decomposition.
    """
    prior_pv = book.pv(prior_ctx)
    market_move_pv = book.pv(current_ctx)
    market_move_pnl = market_move_pv - prior_pv

    new_trade_pnl = 0.0
    if new_trades:
        new_trade_pnl = sum(t.pv(current_ctx) for t in new_trades)

    amendment_pnl = 0.0
    if amendments:
        amendment_pnl = sum(amendments.values())

    total_pnl = market_move_pnl + new_trade_pnl + amendment_pnl

    return DailyPnL(
        book_name=book.name,
        prior_date=prior_ctx.valuation_date,
        current_date=current_ctx.valuation_date,
        prior_pv=prior_pv,
        current_pv=market_move_pv,
        market_move_pnl=market_move_pnl,
        new_trade_pnl=new_trade_pnl,
        amendment_pnl=amendment_pnl,
        total_pnl=total_pnl,
    )


def compute_desk_pnl(
    desk: Desk,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
) -> list[DailyPnL]:
    """Compute daily P&L for each book in a desk."""
    return [
        compute_daily_pnl(book, prior_ctx, current_ctx)
        for book in desk.books.values()
    ]


# ---- P&L Attribution ----

@dataclass
class TradeAttribution:
    """P&L attribution for a single trade."""
    trade_id: str
    total_pnl: float
    rate_pnl: float
    vol_pnl: float
    theta_pnl: float
    unexplained: float

    @property
    def explained(self) -> float:
        return self.rate_pnl + self.vol_pnl + self.theta_pnl


@dataclass
class BookAttribution:
    """P&L attribution for a book."""
    book_name: str
    total_pnl: float
    rate_pnl: float
    vol_pnl: float
    theta_pnl: float
    unexplained: float
    by_trade: list[TradeAttribution]
    by_bucket: dict[str, dict[str, float]]


def _attribute_trade(
    trade: Trade,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
) -> TradeAttribution:
    """Sequential bump-and-reprice attribution for one trade.

    Order: rates → vol → theta → unexplained.
    """
    trade_id = trade.trade_id or "unknown"
    pv_prior = trade.pv(prior_ctx)
    pv_current = trade.pv(current_ctx)
    total = pv_current - pv_prior

    # Step 1: rate move — swap in today's discount curve, keep prior vol and date
    ctx_rates = prior_ctx.replace(
        discount_curve=current_ctx.discount_curve,
        projection_curves=current_ctx.projection_curves,
    )
    pv_after_rates = trade.pv(ctx_rates)
    rate_pnl = pv_after_rates - pv_prior

    # Step 2: vol move — also swap in today's vol surfaces
    ctx_rates_vol = ctx_rates.replace(vol_surfaces=current_ctx.vol_surfaces)
    pv_after_vol = trade.pv(ctx_rates_vol)
    vol_pnl = pv_after_vol - pv_after_rates

    # Step 3: theta — advance date (this gives us current_ctx)
    # ctx_rates_vol with current_date = current_ctx
    ctx_full = ctx_rates_vol.replace(
        valuation_date=current_ctx.valuation_date,
        credit_curves=current_ctx.credit_curves,
        fx_spots=current_ctx.fx_spots,
    )
    pv_after_theta = trade.pv(ctx_full)
    theta_pnl = pv_after_theta - pv_after_vol

    # Unexplained = total - explained
    unexplained = total - (rate_pnl + vol_pnl + theta_pnl)

    return TradeAttribution(
        trade_id=trade_id,
        total_pnl=total,
        rate_pnl=rate_pnl,
        vol_pnl=vol_pnl,
        theta_pnl=theta_pnl,
        unexplained=unexplained,
    )


def attribute_pnl(
    book: Book,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
) -> BookAttribution:
    """Full P&L attribution for a book.

    Returns per-trade and per-bucket breakdown.
    """
    by_trade = [
        _attribute_trade(trade, prior_ctx, current_ctx)
        for trade in book.trades
    ]

    total_pnl = sum(a.total_pnl for a in by_trade)
    rate_pnl = sum(a.rate_pnl for a in by_trade)
    vol_pnl = sum(a.vol_pnl for a in by_trade)
    theta_pnl = sum(a.theta_pnl for a in by_trade)
    unexplained = sum(a.unexplained for a in by_trade)

    # Bucket attribution
    by_bucket: dict[str, dict[str, float]] = {}
    val = prior_ctx.valuation_date
    for trade, attrib in zip(book.trades, by_trade):
        end = _instrument_end(trade.instrument)
        bucket = tenor_bucket(val, end) if end else "unknown"
        if bucket not in by_bucket:
            by_bucket[bucket] = {
                "total_pnl": 0.0, "rate_pnl": 0.0,
                "vol_pnl": 0.0, "theta_pnl": 0.0, "unexplained": 0.0,
            }
        by_bucket[bucket]["total_pnl"] += attrib.total_pnl
        by_bucket[bucket]["rate_pnl"] += attrib.rate_pnl
        by_bucket[bucket]["vol_pnl"] += attrib.vol_pnl
        by_bucket[bucket]["theta_pnl"] += attrib.theta_pnl
        by_bucket[bucket]["unexplained"] += attrib.unexplained

    return BookAttribution(
        book_name=book.name,
        total_pnl=total_pnl,
        rate_pnl=rate_pnl,
        vol_pnl=vol_pnl,
        theta_pnl=theta_pnl,
        unexplained=unexplained,
        by_trade=by_trade,
        by_bucket=by_bucket,
    )
