"""End-of-day process: MTM, P&L, risk report, limit monitoring.

    from pricebook.eod import eod_mtm, eod_pnl, eod_risk_report, check_limits

    mtm = eod_mtm(trades, curve_today)
    pnl = eod_pnl(mtm_today, mtm_yesterday)
    report = eod_risk_report(mtm, curve, sensitivities)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np


# ============================================================================
# Mark-to-market
# ============================================================================

@dataclass
class MTMResult:
    """Mark-to-market result."""
    trade_id: str
    pv: float
    currency: str = "USD"


def eod_mtm(
    trade_pvs: dict[str, float],
    currency: str = "USD",
) -> list[MTMResult]:
    """Mark-to-market all trades.

        mtm = eod_mtm({"swap_5y": 15000, "bond_10y": -3000})
    """
    return [MTMResult(tid, pv, currency) for tid, pv in trade_pvs.items()]


# ============================================================================
# P&L
# ============================================================================

@dataclass
class PnLResult:
    """Daily P&L result."""
    total_pnl: float
    market_move_pnl: float
    new_trade_pnl: float
    trade_pnls: dict[str, float]
    eod_date: date


def eod_pnl(
    mtm_today: dict[str, float],
    mtm_yesterday: dict[str, float],
    new_trades: dict[str, float] | None = None,
    eod_date: date | None = None,
) -> PnLResult:
    """Compute daily P&L.

    P&L = (MTM_today - MTM_yesterday) for existing trades + new trades.

        pnl = eod_pnl(today_pvs, yesterday_pvs)
    """
    trade_pnls = {}
    market_move = 0.0

    # Existing trades
    for tid in set(mtm_today) | set(mtm_yesterday):
        today = mtm_today.get(tid, 0.0)
        yesterday = mtm_yesterday.get(tid, 0.0)
        pl = today - yesterday
        trade_pnls[tid] = pl
        if tid in mtm_yesterday:
            market_move += pl

    # New trades
    new_pnl = 0.0
    if new_trades:
        for tid, pv in new_trades.items():
            trade_pnls[tid] = pv
            new_pnl += pv

    total = market_move + new_pnl
    return PnLResult(total, market_move, new_pnl, trade_pnls,
                     eod_date or date.today())


@dataclass
class PnLAttribution:
    """P&L attribution by risk factor."""
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    unexplained: float


def attribute_pnl(
    total_pnl: float,
    delta: float, spot_move: float,
    gamma: float = 0.0,
    vega: float = 0.0, vol_move: float = 0.0,
    theta: float = 0.0, dt: float = 1.0 / 365,
) -> PnLAttribution:
    """Greek-based P&L attribution.

        attrib = attribute_pnl(pnl, delta=0.5, spot_move=2.0, gamma=0.01)
    """
    d_pnl = delta * spot_move
    g_pnl = 0.5 * gamma * spot_move ** 2
    v_pnl = vega * vol_move
    t_pnl = theta * dt
    explained = d_pnl + g_pnl + v_pnl + t_pnl
    unexplained = total_pnl - explained

    return PnLAttribution(total_pnl, d_pnl, g_pnl, v_pnl, t_pnl, unexplained)


# ============================================================================
# Risk report
# ============================================================================

@dataclass
class RiskReport:
    """Daily risk report."""
    eod_date: date
    total_pv: float
    total_dv01: float
    total_vega: float
    var_95: float
    var_99: float
    n_trades: int
    limit_breaches: list[str]
    concentrations: dict[str, float]


def eod_risk_report(
    trade_pvs: dict[str, float],
    sensitivities: dict[str, dict[str, float]] | None = None,
    var_95: float = 0.0,
    var_99: float = 0.0,
    eod_date: date | None = None,
) -> RiskReport:
    """Generate daily risk report.

        report = eod_risk_report(pvs, sensitivities, var_95=50000, var_99=80000)
    """
    total_pv = sum(trade_pvs.values())
    n_trades = len(trade_pvs)

    total_dv01 = 0.0
    total_vega = 0.0
    if sensitivities:
        for tid, sens in sensitivities.items():
            total_dv01 += sens.get("dv01", 0.0)
            total_vega += sens.get("vega", 0.0)

    # Concentration by absolute PV
    total_abs = sum(abs(v) for v in trade_pvs.values())
    concentrations = {}
    if total_abs > 0:
        for tid, pv in sorted(trade_pvs.items(), key=lambda x: -abs(x[1]))[:5]:
            concentrations[tid] = abs(pv) / total_abs

    return RiskReport(
        eod_date=eod_date or date.today(),
        total_pv=total_pv,
        total_dv01=total_dv01,
        total_vega=total_vega,
        var_95=var_95,
        var_99=var_99,
        n_trades=n_trades,
        limit_breaches=[],
        concentrations=concentrations,
    )


# ============================================================================
# Limit monitoring
# ============================================================================

@dataclass
class LimitBreach:
    """A limit breach."""
    limit_type: str
    limit_value: float
    actual_value: float
    breach_amount: float


def check_limits(
    total_dv01: float = 0.0,
    total_vega: float = 0.0,
    total_pv: float = 0.0,
    var_99: float = 0.0,
    dv01_limit: float | None = None,
    vega_limit: float | None = None,
    pv_limit: float | None = None,
    var_limit: float | None = None,
) -> list[LimitBreach]:
    """Check risk limits and return breaches.

        breaches = check_limits(total_dv01=150_000, dv01_limit=100_000)
    """
    breaches = []
    if dv01_limit is not None and abs(total_dv01) > dv01_limit:
        breaches.append(LimitBreach("DV01", dv01_limit, abs(total_dv01),
                                     abs(total_dv01) - dv01_limit))
    if vega_limit is not None and abs(total_vega) > vega_limit:
        breaches.append(LimitBreach("Vega", vega_limit, abs(total_vega),
                                     abs(total_vega) - vega_limit))
    if pv_limit is not None and abs(total_pv) > pv_limit:
        breaches.append(LimitBreach("PV", pv_limit, abs(total_pv),
                                     abs(total_pv) - pv_limit))
    if var_limit is not None and var_99 > var_limit:
        breaches.append(LimitBreach("VaR", var_limit, var_99,
                                     var_99 - var_limit))
    return breaches
