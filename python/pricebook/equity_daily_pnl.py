"""Daily P&L workflow for equity books: official P&L and Greek attribution.

Mirrors the IR `daily_pnl` module but for `EquityBook`. Official P&L
decomposes into market move, new trades, and amendments. Greek attribution
breaks the per-trade move into delta, gamma, vega, theta, and rho:

    ΔPV ≈ Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + θ·Δt + ρ·Δr

    pnl = compute_equity_daily_pnl(book, prior_ctx, current_ctx)
    attrib = attribute_equity_pnl(
        book, prior_ctx, current_ctx,
        spot_changes={"AAPL": 2.5}, vol_changes={"AAPL": 0.005},
        greeks={"t1": TradeGreeks(delta=0.55, gamma=0.02, vega=120.0)},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.equity_book import EquityBook, EquityTradeEntry
from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade


# ---- Official P&L ----

@dataclass
class EquityDailyPnL:
    """Official daily P&L breakdown for an equity book."""
    book_name: str
    prior_date: date
    current_date: date
    prior_pv: float
    current_pv: float
    market_move_pnl: float
    new_trade_pnl: float
    amendment_pnl: float
    total_pnl: float


def compute_equity_daily_pnl(
    book: EquityBook,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
    new_trades: list[Trade] | None = None,
    amendments: dict[str, float] | None = None,
) -> EquityDailyPnL:
    """Compute official daily P&L for an equity book.

    Args:
        book: yesterday's equity book (EOD positions).
        prior_ctx: yesterday's market data.
        current_ctx: today's market data.
        new_trades: trades entered today (priced on today's context).
        amendments: trade_id -> P&L impact of amendments.

    Returns:
        EquityDailyPnL with full decomposition.
    """
    prior_pv = book.pv(prior_ctx)
    current_pv = book.pv(current_ctx)
    market_move_pnl = current_pv - prior_pv

    new_trade_pnl = 0.0
    if new_trades:
        new_trade_pnl = sum(t.pv(current_ctx) for t in new_trades)

    amendment_pnl = 0.0
    if amendments:
        amendment_pnl = sum(amendments.values())

    total_pnl = market_move_pnl + new_trade_pnl + amendment_pnl

    return EquityDailyPnL(
        book_name=book.name,
        prior_date=prior_ctx.valuation_date,
        current_date=current_ctx.valuation_date,
        prior_pv=prior_pv,
        current_pv=current_pv,
        market_move_pnl=market_move_pnl,
        new_trade_pnl=new_trade_pnl,
        amendment_pnl=amendment_pnl,
        total_pnl=total_pnl,
    )


# ---- Greek-based attribution ----

@dataclass
class TradeGreeks:
    """Greeks for a single equity trade.

    All Greeks are at unit-direction; the attribution function applies
    the trade's signed scale (direction × notional_scale).

    Attributes:
        delta: ∂PV / ∂S — change in PV per unit spot move.
        gamma: ∂²PV / ∂S² — second-order spot sensitivity.
        vega: ∂PV / ∂σ — change in PV per unit absolute vol move
            (e.g. 0.01 = 1 vol point).
        theta: ∂PV / ∂t — daily time decay (per calendar day).
        rho: ∂PV / ∂r — change in PV per unit rate move.
    """
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0


@dataclass
class GreekAttribution:
    """Greek-based P&L attribution for a single trade."""
    trade_id: str
    ticker: str
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_pnl: float
    unexplained: float

    @property
    def explained(self) -> float:
        return (
            self.delta_pnl + self.gamma_pnl + self.vega_pnl
            + self.theta_pnl + self.rho_pnl
        )


@dataclass
class EquityBookAttribution:
    """Aggregate Greek attribution for an equity book."""
    book_name: str
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_pnl: float
    unexplained: float
    by_trade: list[GreekAttribution]
    by_ticker: dict[str, dict[str, float]]

    @property
    def explained(self) -> float:
        return (
            self.delta_pnl + self.gamma_pnl + self.vega_pnl
            + self.theta_pnl + self.rho_pnl
        )


def _trade_total_pnl(
    trade: Trade,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
) -> float:
    """Total P&L for a trade if it can be priced; else 0.

    EquityBook may carry instruments without ``pv_ctx`` (e.g. raw equity
    positions). For those, the official market move is captured separately
    via spot moves; the attribution leans on the supplied Greeks.
    """
    inst = trade.instrument
    if not hasattr(inst, "pv_ctx"):
        return 0.0
    return trade.pv(current_ctx) - trade.pv(prior_ctx)


def attribute_equity_pnl(
    book: EquityBook,
    prior_ctx: PricingContext,
    current_ctx: PricingContext,
    spot_changes: dict[str, float],
    vol_changes: dict[str, float] | None = None,
    rate_change: float = 0.0,
    greeks: dict[str, TradeGreeks] | None = None,
) -> EquityBookAttribution:
    """Greek-based P&L attribution for an equity book.

    For each trade we apply the supplied Greeks to the corresponding
    market changes:

        delta_pnl = sign · Δ · ΔS
        gamma_pnl = sign · ½ · Γ · ΔS²
        vega_pnl  = sign · ν · Δσ
        theta_pnl = sign · θ · Δt(days)
        rho_pnl   = sign · ρ · Δr

    where ``sign = direction × notional_scale``. The unexplained residual
    is computed against the actual re-priced total (when the instrument
    supports ``pv_ctx``).

    Args:
        book: equity book (yesterday's positions).
        prior_ctx: yesterday's market data.
        current_ctx: today's market data.
        spot_changes: ticker -> ΔS.
        vol_changes: ticker -> Δσ (default empty).
        rate_change: parallel rate move Δr (default 0).
        greeks: trade_id -> TradeGreeks (missing => zero Greeks).

    Returns:
        EquityBookAttribution with per-trade and per-ticker breakdown.
    """
    vol_changes = vol_changes or {}
    greeks = greeks or {}
    dt_days = float((current_ctx.valuation_date - prior_ctx.valuation_date).days)

    by_trade: list[GreekAttribution] = []

    for entry in book.entries:
        trade = entry.trade
        trade_id = trade.trade_id or f"{entry.ticker}:unknown"
        ticker = entry.ticker
        sign = trade.direction * trade.notional_scale

        g = greeks.get(trade_id, TradeGreeks())
        ds = spot_changes.get(ticker, 0.0)
        dv = vol_changes.get(ticker, 0.0)

        delta_pnl = sign * g.delta * ds
        gamma_pnl = sign * 0.5 * g.gamma * ds * ds
        vega_pnl = sign * g.vega * dv
        theta_pnl = sign * g.theta * dt_days
        rho_pnl = sign * g.rho * rate_change

        total = _trade_total_pnl(trade, prior_ctx, current_ctx)
        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
        unexplained = total - explained

        by_trade.append(GreekAttribution(
            trade_id=trade_id, ticker=ticker, total_pnl=total,
            delta_pnl=delta_pnl, gamma_pnl=gamma_pnl, vega_pnl=vega_pnl,
            theta_pnl=theta_pnl, rho_pnl=rho_pnl, unexplained=unexplained,
        ))

    total_pnl = sum(a.total_pnl for a in by_trade)
    delta_pnl = sum(a.delta_pnl for a in by_trade)
    gamma_pnl = sum(a.gamma_pnl for a in by_trade)
    vega_pnl = sum(a.vega_pnl for a in by_trade)
    theta_pnl = sum(a.theta_pnl for a in by_trade)
    rho_pnl = sum(a.rho_pnl for a in by_trade)
    unexplained = sum(a.unexplained for a in by_trade)

    by_ticker: dict[str, dict[str, float]] = {}
    for a in by_trade:
        if a.ticker not in by_ticker:
            by_ticker[a.ticker] = {
                "total_pnl": 0.0, "delta_pnl": 0.0, "gamma_pnl": 0.0,
                "vega_pnl": 0.0, "theta_pnl": 0.0, "rho_pnl": 0.0,
                "unexplained": 0.0,
            }
        d = by_ticker[a.ticker]
        d["total_pnl"] += a.total_pnl
        d["delta_pnl"] += a.delta_pnl
        d["gamma_pnl"] += a.gamma_pnl
        d["vega_pnl"] += a.vega_pnl
        d["theta_pnl"] += a.theta_pnl
        d["rho_pnl"] += a.rho_pnl
        d["unexplained"] += a.unexplained

    return EquityBookAttribution(
        book_name=book.name,
        total_pnl=total_pnl,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        rho_pnl=rho_pnl,
        unexplained=unexplained,
        by_trade=by_trade,
        by_ticker=by_ticker,
    )
