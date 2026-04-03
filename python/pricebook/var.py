"""
Value at Risk and stress testing.

Historical VaR: full revaluation under historical scenarios.
Parametric VaR: delta-normal approximation.
Stress testing: predefined or custom risk factor shocks.

    from pricebook.var import historical_var, parametric_var, stress_test

    var_95 = historical_var(pnl_series, confidence=0.95)
    var_dn = parametric_var(deltas, cov_matrix, confidence=0.99)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def historical_var(
    pnl_series: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Historical Value at Risk.

    VaR = negative of the (1-confidence) percentile of P&L.
    Positive VaR = potential loss.

    Args:
        pnl_series: historical P&L observations.
        confidence: e.g. 0.95 for 95% VaR.
    """
    pnl = np.asarray(pnl_series)
    percentile = (1 - confidence) * 100
    return -float(np.percentile(pnl, percentile))


def historical_cvar(
    pnl_series: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Conditional VaR (Expected Shortfall): average of losses beyond VaR.

    CVaR = -E[PnL | PnL < -VaR]
    """
    pnl = np.asarray(pnl_series)
    var = historical_var(pnl, confidence)
    tail = pnl[pnl <= -var]
    if len(tail) == 0:
        return var
    return -float(tail.mean())


def parametric_var(
    deltas: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Parametric (delta-normal) VaR.

    VaR = z_alpha * sqrt(delta' @ Sigma @ delta)

    Assumes linear risk and normal returns.

    Args:
        deltas: vector of portfolio sensitivities to risk factors.
        cov_matrix: covariance matrix of risk factor returns.
        confidence: e.g. 0.99.
    """
    deltas = np.asarray(deltas)
    cov_matrix = np.asarray(cov_matrix)
    portfolio_var = float(deltas @ cov_matrix @ deltas)
    portfolio_std = math.sqrt(max(portfolio_var, 0.0))
    z = norm.ppf(confidence)
    return z * portfolio_std


def stress_test(
    pricer,
    base_ctx,
    scenarios: list[dict[str, float]],
    scenario_names: list[str] | None = None,
) -> list[dict[str, float]]:
    """Run stress tests: reprice under each scenario.

    Args:
        pricer: callable(ctx) → float, returns portfolio PV.
        base_ctx: base PricingContext.
        scenarios: list of dicts, each mapping risk factor to shock value.
            E.g. [{"rate_shift": 0.01}, {"rate_shift": -0.01, "vol_shift": 0.05}]
        scenario_names: optional names for each scenario.

    Returns:
        List of dicts with: name, base_pv, scenario_pv, pnl.
    """
    base_pv = pricer(base_ctx)

    if scenario_names is None:
        scenario_names = [f"scenario_{i}" for i in range(len(scenarios))]

    results = []
    for name, shocks in zip(scenario_names, scenarios):
        # Apply shocks to context
        ctx = base_ctx
        if "rate_shift" in shocks:
            if ctx.discount_curve is not None:
                from pricebook.pricing_context import PricingContext
                new_disc = ctx.discount_curve.bumped(shocks["rate_shift"])
                ctx = PricingContext(
                    valuation_date=ctx.valuation_date,
                    discount_curve=new_disc,
                    projection_curves=ctx.projection_curves,
                    vol_surfaces=ctx.vol_surfaces,
                    credit_curves=ctx.credit_curves,
                    fx_spots=ctx.fx_spots,
                )

        scenario_pv = pricer(ctx)
        results.append({
            "name": name,
            "base_pv": base_pv,
            "scenario_pv": scenario_pv,
            "pnl": scenario_pv - base_pv,
        })

    return results


# Standard stress scenarios
STANDARD_STRESSES = [
    ("parallel_up_100bp", {"rate_shift": 0.01}),
    ("parallel_down_100bp", {"rate_shift": -0.01}),
    ("parallel_up_25bp", {"rate_shift": 0.0025}),
    ("parallel_down_25bp", {"rate_shift": -0.0025}),
]
