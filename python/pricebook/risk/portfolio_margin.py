"""Portfolio margin and SPAN-style cross-margining.

Implements SPAN-like scenario-based margining, cross-margining offsets,
strategy-level margin rules (Reg-T / exchange), and VaR-based initial margin
in the spirit of ISDA SIMM.

    from pricebook.risk.portfolio_margin import (
        span_margin, cross_margin_offset, strategy_margin,
        var_based_margin, margin_call,
    )

    positions = [
        Position("equity_option", quantity=10, delta=0.5, gamma=0.02, vega=0.1, notional=100_000),
        Position("equity_option", quantity=-10, delta=-0.3, gamma=0.01, vega=0.08, notional=80_000),
    ]
    result = span_margin(positions, scenarios=None, price_scan_range=0.15, vol_scan_range=0.30)
    print(result.initial_margin, result.offset_benefit)

References:
    CME Group, *SPAN Margining Overview*, 2023.
    OCC, *Margin Policy and Methodology*, 2024.
    ISDA, *SIMM Methodology*, v2.6, 2024.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---- Data classes -----------------------------------------------------------

@dataclass
class Position:
    """A single position with its risk sensitivities."""
    instrument_type: str   # e.g. "equity_option", "future", "bond"
    quantity: float        # signed (positive = long)
    delta: float           # dollar delta per unit of notional move
    gamma: float           # dollar gamma
    vega: float            # dollar vega per unit of vol move
    notional: float        # notional value


@dataclass
class OptionLeg:
    """One leg of an option strategy."""
    option_type: str    # "call" or "put"
    strike: float
    quantity: float     # signed
    premium: float      # per-unit premium (positive = paid)


@dataclass
class MarginResult:
    """Result from a margin computation."""
    initial_margin: float
    maintenance_margin: float
    margin_components: dict[str, float]
    worst_scenario: dict[str, float]
    offset_benefit: float


# ---- SPAN scenario grid -----------------------------------------------------

def _build_scenarios(price_scan_range: float, vol_scan_range: float) -> list[tuple[float, float]]:
    """Build a 14-scenario SPAN-style grid.

    Returns list of (price_move_fraction, vol_move_fraction) tuples.
    The 14 scenarios cover ±1/3, ±2/3, ±1 price moves at each of +vol, -vol,
    plus two extreme price moves at a fraction of the delta loss.
    """
    price_fracs = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    vol_signs = [1.0, -1.0]
    scenarios: list[tuple[float, float]] = []
    for pf in price_fracs:
        for vs in vol_signs:
            scenarios.append((pf * price_scan_range, vs * vol_scan_range))
    # Two extreme scenarios at 2× PSR, capped at 35% of loss (SPAN convention)
    scenarios.append((2.0 * price_scan_range, 0.0))
    scenarios.append((-2.0 * price_scan_range, 0.0))
    return scenarios


def _scenario_pnl(pos: Position, price_move: float, vol_move: float) -> float:
    """Approximate P&L of a position under a price and vol scenario.

    Uses a second-order Taylor expansion:
        PnL ≈ delta*dS + 0.5*gamma*dS^2 + vega*dVol
    where dS = price_move * notional.
    """
    dS = price_move * pos.notional
    return pos.quantity * (
        pos.delta * dS
        + 0.5 * pos.gamma * dS ** 2
        + pos.vega * vol_move
    )


# ---- SPAN margin ------------------------------------------------------------

def span_margin(
    positions: list[Position],
    scenarios: list[tuple[float, float]] | None,
    price_scan_range: float,
    vol_scan_range: float,
) -> MarginResult:
    """SPAN-style scenario-based portfolio margin.

    Computes the worst-case portfolio loss across a grid of price and vol
    scenarios.  The initial margin equals that worst-case loss (floored at
    zero).  The maintenance margin is set at 75% of initial margin following
    typical exchange conventions.

    Args:
        positions: List of Position objects.
        scenarios: Optional pre-built list of (price_move, vol_move) tuples.
            If None, a standard 16-scenario SPAN grid is generated from
            price_scan_range and vol_scan_range.
        price_scan_range: Maximum price move as a fraction of notional (e.g.
            0.15 for 15%).  Used only when scenarios is None.
        vol_scan_range: Maximum vol move (absolute, e.g. 0.30 for 30 vol pts).
            Used only when scenarios is None.

    Returns:
        MarginResult with initial and maintenance margin.
    """
    if scenarios is None:
        scenarios = _build_scenarios(price_scan_range, vol_scan_range)

    worst_loss = 0.0
    worst_scen: dict[str, float] = {}
    all_losses: list[float] = []

    n_scenarios = len(scenarios)
    for idx, (price_move, vol_move) in enumerate(scenarios):
        portfolio_pnl = sum(_scenario_pnl(p, price_move, vol_move) for p in positions)
        loss = -portfolio_pnl  # margin is a positive number
        # SPAN: extreme scenarios (last 2) contribute only 35% of their loss
        if idx >= n_scenarios - 2 and scenarios is not None:
            loss *= 0.35
        all_losses.append(loss)
        if loss > worst_loss:
            worst_loss = loss
            worst_scen = {"price_move": price_move, "vol_move": vol_move, "loss": loss}

    initial_margin = max(worst_loss, 0.0)
    maintenance_margin = 0.75 * initial_margin

    # Delta margin component (flat worst-case delta move)
    net_delta = sum(p.quantity * p.delta * p.notional for p in positions)
    delta_margin = abs(net_delta) * price_scan_range

    # Vega margin component
    net_vega = sum(p.quantity * p.vega for p in positions)
    vega_margin = abs(net_vega) * vol_scan_range

    components = {
        "scanning_risk": float(np.max(all_losses)) if all_losses else 0.0,
        "delta_component": delta_margin,
        "vega_component": vega_margin,
    }

    return MarginResult(
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        margin_components=components,
        worst_scenario=worst_scen,
        offset_benefit=0.0,
    )


# ---- Cross-margin offset ----------------------------------------------------

def cross_margin_offset(
    margin_standalone: list[float],
    margin_portfolio: float,
    asset_classes: list[str],
) -> dict[str, float]:
    """Compute the diversification benefit from cross-margining.

    Standalone margins (one per asset class) are compared with the combined
    portfolio margin to quantify the benefit of netting.

    Args:
        margin_standalone: List of standalone initial margins, one per asset class.
        margin_portfolio: Portfolio-level initial margin (after cross-margining).
        asset_classes: Asset class labels matching margin_standalone.

    Returns:
        Dict with keys: sum_standalone, portfolio_margin, offset_benefit,
        offset_ratio, and per-asset-class standalone values.
    """
    total_standalone = sum(margin_standalone)
    offset_amount = max(total_standalone - margin_portfolio, 0.0)
    offset_ratio = offset_amount / total_standalone if total_standalone > 0.0 else 0.0

    result: dict[str, float] = {
        "sum_standalone": total_standalone,
        "portfolio_margin": margin_portfolio,
        "offset_benefit": offset_amount,
        "offset_ratio": offset_ratio,
    }
    for ac, m in zip(asset_classes, margin_standalone):
        result[f"standalone_{ac}"] = m
    return result


# ---- Strategy margin (Reg-T and exchange rules) -----------------------------

_REG_T_RATE = 0.50   # 50% initial margin requirement under Reg-T


def _net_premium(legs: list[OptionLeg]) -> float:
    return sum(leg.quantity * leg.premium for leg in legs)


def _calls(legs: list[OptionLeg]) -> list[OptionLeg]:
    return [l for l in legs if l.option_type.lower() == "call"]


def _puts(legs: list[OptionLeg]) -> list[OptionLeg]:
    return [l for l in legs if l.option_type.lower() == "put"]


def strategy_margin(
    legs: list[OptionLeg],
    margin_type: str = "reg_t",
) -> dict[str, float]:
    """Margin requirement for common option strategies.

    Recognises: covered_call, vertical_spread, straddle, iron_condor.
    Falls back to Reg-T naked option margin for unrecognised strategies.

    Args:
        legs: List of OptionLeg objects defining the strategy.
        margin_type: "reg_t" (default) or "exchange".  Exchange margin uses
            the maximum loss of the spread as the margin requirement.

    Returns:
        Dict with keys: strategy, initial_margin, maintenance_margin,
        net_premium, max_loss.
    """
    calls = _calls(legs)
    puts = _puts(legs)
    n_legs = len(legs)
    net_prem = _net_premium(legs)

    strategy = "unknown"
    initial_margin = 0.0
    max_loss = float("inf")

    # ---- Covered call: long stock + short call (simplified: legs has 1 leg) -
    if n_legs == 1 and calls and calls[0].quantity < 0:
        strategy = "covered_call"
        notional = calls[0].strike * abs(calls[0].quantity) * 100
        initial_margin = _REG_T_RATE * notional - abs(net_prem)
        max_loss = calls[0].strike * abs(calls[0].quantity) * 100 - abs(net_prem)

    # ---- Vertical spread: 2 legs, same type, different strikes ---------------
    elif n_legs == 2 and (
        (len(calls) == 2 and calls[0].quantity * calls[1].quantity < 0) or
        (len(puts) == 2 and puts[0].quantity * puts[1].quantity < 0)
    ):
        strategy = "vertical_spread"
        same_type = calls if len(calls) == 2 else puts
        strikes = sorted(l.strike for l in same_type)
        spread_width = (strikes[1] - strikes[0]) * abs(same_type[0].quantity) * 100
        if margin_type == "exchange":
            initial_margin = max(spread_width - abs(net_prem), 0.0)
        else:
            initial_margin = spread_width
        max_loss = spread_width - abs(net_prem)

    # ---- Straddle: 1 call + 1 put, same strike, same sign -------------------
    elif n_legs == 2 and len(calls) == 1 and len(puts) == 1 and \
            calls[0].quantity == puts[0].quantity and \
            math.isclose(calls[0].strike, puts[0].strike, rel_tol=1e-6):
        strategy = "straddle"
        # Reg-T: margin is the larger of call or put naked margin
        notional = calls[0].strike * abs(calls[0].quantity) * 100
        put_notional = puts[0].strike * abs(puts[0].quantity) * 100
        initial_margin = max(
            _REG_T_RATE * notional,
            _REG_T_RATE * put_notional,
        )
        max_loss = abs(net_prem)  # premium paid

    # ---- Iron condor: 4 legs, 2 calls + 2 puts with opposing signs ----------
    elif n_legs == 4 and len(calls) == 2 and len(puts) == 2:
        strategy = "iron_condor"
        call_strikes = sorted(l.strike for l in calls)
        put_strikes = sorted(l.strike for l in puts)
        call_spread = (call_strikes[1] - call_strikes[0]) * 100
        put_spread = (put_strikes[1] - put_strikes[0]) * 100
        # Margin = wider spread width (max possible loss side)
        spread_width = max(call_spread, put_spread)
        if margin_type == "exchange":
            initial_margin = max(spread_width - abs(net_prem), 0.0)
        else:
            initial_margin = spread_width
        max_loss = spread_width - abs(net_prem)

    # ---- Fallback: naked Reg-T margin ---------------------------------------
    else:
        strategy = "naked"
        total_notional = sum(abs(l.quantity) * l.strike * 100 for l in legs)
        initial_margin = _REG_T_RATE * total_notional
        max_loss = float("inf")

    maintenance_margin = 0.75 * initial_margin

    return {
        "strategy": strategy,
        "initial_margin": max(initial_margin, 0.0),
        "maintenance_margin": max(maintenance_margin, 0.0),
        "net_premium": net_prem,
        "max_loss": max_loss,
    }


# ---- VaR-based initial margin (ISDA SIMM style) ----------------------------

def var_based_margin(
    portfolio_value: float,
    var_99: float,
    expected_shortfall: float,
    multiplier: float = 1.0,
) -> dict[str, float]:
    """VaR-based initial margin in the spirit of ISDA SIMM.

    ISDA SIMM uses a sensitivity-based approach; this function provides a
    simplified portfolio-level analog using VaR and Expected Shortfall (ES).
    The initial margin is:

        IM = multiplier × max(VaR_99, 0.5 × ES)

    Args:
        portfolio_value: Current mark-to-market portfolio value.
        var_99: 99th-percentile 10-day VaR (positive number = potential loss).
        expected_shortfall: Expected Shortfall (CVaR) at 97.5% confidence
            (positive number = expected loss beyond VaR threshold).
        multiplier: Scaling factor for regulatory add-ons (default 1.0).
            Set to >1 for stressed VaR or procyclicality buffers.

    Returns:
        Dict with keys: initial_margin, var_99, expected_shortfall,
        binding_measure, im_to_nav_ratio.
    """
    floor_es = 0.5 * expected_shortfall
    binding = max(var_99, floor_es)
    im = multiplier * binding
    binding_measure = "var_99" if var_99 >= floor_es else "0.5_es"
    nav_ratio = im / abs(portfolio_value) if portfolio_value != 0.0 else float("nan")

    return {
        "initial_margin": im,
        "var_99": var_99,
        "expected_shortfall": expected_shortfall,
        "binding_measure": binding_measure,
        "im_to_nav_ratio": nav_ratio,
    }


# ---- Margin call ------------------------------------------------------------

def margin_call(
    equity: float,
    initial_margin: float,
    maintenance_margin: float,
    market_value: float,
) -> dict[str, float]:
    """Compute margin call amount if equity falls below maintenance margin.

    A margin call is triggered when:
        equity < maintenance_margin

    The call amount restores equity to the initial margin level.

    Args:
        equity: Current account equity (market_value + cash - liabilities).
        initial_margin: Required initial margin (variation margin floor).
        maintenance_margin: Maintenance margin threshold; breach triggers call.
        market_value: Current mark-to-market of the portfolio.

    Returns:
        Dict with keys: margin_call_triggered (bool), call_amount,
        equity, shortfall, equity_after_call.
    """
    shortfall = max(maintenance_margin - equity, 0.0)
    triggered = equity < maintenance_margin
    call_amount = max(initial_margin - equity, 0.0) if triggered else 0.0
    equity_after = equity + call_amount

    return {
        "margin_call_triggered": triggered,
        "call_amount": call_amount,
        "equity": equity,
        "shortfall": shortfall,
        "equity_after_call": equity_after,
    }
