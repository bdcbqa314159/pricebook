"""
Value at Risk and stress testing.

Historical VaR: full revaluation under historical scenarios.
Parametric VaR: delta-normal approximation.
Stress testing: predefined or custom risk factor shocks.

    from pricebook.risk.var import historical_var, parametric_var, stress_test

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


_SUPPORTED_SHOCKS = frozenset({"rate_shift", "credit_shift"})


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
            Supported shock keys:
              - ``rate_shift``    parallel shift of every discount curve
                                  (singular ``discount_curve`` and every
                                  entry of the plural ``discount_curves``
                                  and ``projection_curves`` dicts).
              - ``credit_shift``  parallel shift of every entry of
                                  ``credit_curves`` (hazard rates).
            Unknown shock keys raise ``ValueError`` — pre-fix used to
            silently no-op on ``vol_shift``, ``fx_shift``, etc., giving
            unchanged P&L that looked successful.
        scenario_names: optional names for each scenario.

    Returns:
        List of dicts with: name, base_pv, scenario_pv, pnl.

    Fix T4-RISK1: pre-fix only inspected ``rate_shift`` and silently
    dropped every other shock key (including ``vol_shift`` shown in
    the original docstring example).  Also, the bumped context was
    constructed from scratch with a 6-field subset, silently dropping
    the plural ``discount_curves``, ``inflation_curves``, ``repo_curves``,
    ``reporting_currency``, ``stochastic_credit_models``,
    ``credit_vol_surfaces``, ``credit_correlations``, and
    ``numerical_config``.  Now uses ``dataclasses.replace`` to preserve
    every untouched field and raises on unknown shock keys.
    """
    from dataclasses import replace

    base_pv = pricer(base_ctx)

    if scenario_names is None:
        scenario_names = [f"scenario_{i}" for i in range(len(scenarios))]

    results = []
    for name, shocks in zip(scenario_names, scenarios):
        unknown = set(shocks) - _SUPPORTED_SHOCKS
        if unknown:
            raise ValueError(
                f"stress_test scenario {name!r} contains unsupported shock "
                f"keys {sorted(unknown)}.  Supported: {sorted(_SUPPORTED_SHOCKS)}."
            )

        ctx = base_ctx
        rate_shift = shocks.get("rate_shift")
        if rate_shift is not None:
            updates: dict = {}
            if ctx.discount_curve is not None:
                updates["discount_curve"] = ctx.discount_curve.bumped(rate_shift)
            if ctx.discount_curves:
                updates["discount_curves"] = {
                    k: v.bumped(rate_shift) for k, v in ctx.discount_curves.items()
                }
            if ctx.projection_curves:
                updates["projection_curves"] = {
                    k: v.bumped(rate_shift) for k, v in ctx.projection_curves.items()
                }
            if updates:
                ctx = replace(ctx, **updates)

        credit_shift = shocks.get("credit_shift")
        if credit_shift is not None and ctx.credit_curves:
            ctx = replace(
                ctx,
                credit_curves={
                    k: v.bumped(credit_shift) for k, v in ctx.credit_curves.items()
                },
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
