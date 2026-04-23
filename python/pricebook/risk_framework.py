"""Risk framework: VaR, CVaR, stress testing, drawdown management.

    from pricebook.risk_framework import historical_var, parametric_var, stress_test

    var_95 = historical_var(returns, confidence=0.95)
    pvar = parametric_var(portfolio_dv01, portfolio_vol, confidence=0.99)
    stress = stress_test(portfolio_pv_func, scenarios)

References:
    McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton, 2005.
    Jorion, *Value at Risk*, McGraw-Hill, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ============================================================================
# Value at Risk
# ============================================================================

@dataclass
class VaRResult:
    """Value at Risk result."""
    var: float              # VaR (positive = loss)
    cvar: float             # CVaR / Expected Shortfall
    confidence: float
    n_observations: int
    method: str


def historical_var(
    returns: np.ndarray | list[float],
    confidence: float = 0.95,
    holding_period: int = 1,
) -> VaRResult:
    """Historical VaR and CVaR from P&L or return series.

    VaR = -quantile(returns, 1 - confidence)
    CVaR = -mean(returns below -VaR)

        var = historical_var(daily_returns, confidence=0.99)
        print(var.var, var.cvar)
    """
    ret = np.asarray(returns, dtype=float)
    if len(ret) < 2:
        return VaRResult(0.0, 0.0, confidence, len(ret), "historical")

    # Scale for holding period (sqrt-T for returns)
    if holding_period > 1:
        ret = ret * math.sqrt(holding_period)

    cutoff = np.percentile(ret, (1 - confidence) * 100)
    var = -cutoff
    tail = ret[ret <= cutoff]
    cvar = -float(tail.mean()) if len(tail) > 0 else var

    return VaRResult(var, cvar, confidence, len(ret), "historical")


def parametric_var(
    portfolio_value: float,
    portfolio_vol: float,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> VaRResult:
    """Parametric (delta-normal) VaR.

    VaR = portfolio_value × vol × z_α × √T

        pvar = parametric_var(10_000_000, 0.01, confidence=0.99)
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)
    vol_scaled = portfolio_vol * math.sqrt(holding_period)
    var = portfolio_value * vol_scaled * z
    # CVaR for normal: E[X | X < -VaR] = μ + σ × φ(z)/Φ(-z)
    cvar = portfolio_value * vol_scaled * norm.pdf(z) / (1 - confidence)

    return VaRResult(var, cvar, confidence, 0, "parametric")


def delta_gamma_var(
    delta: float,
    gamma: float,
    underlying_vol: float,
    underlying_value: float,
    confidence: float = 0.95,
) -> VaRResult:
    """Delta-gamma VaR for nonlinear positions.

    Uses the Cornish-Fisher expansion for the portfolio distribution.

    VaR ≈ -δ×σ×z + 0.5×γ×σ²×(z²-1)
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)
    sigma = underlying_value * underlying_vol
    var_delta = -delta * sigma * z
    var_gamma = 0.5 * gamma * sigma**2 * (z**2 - 1)
    var = var_delta + var_gamma
    return VaRResult(abs(var), abs(var) * 1.1, confidence, 0, "delta_gamma")


# ============================================================================
# Component VaR
# ============================================================================

@dataclass
class ComponentVaRResult:
    """Component VaR decomposition."""
    total_var: float
    component_vars: dict[str, float]
    marginal_vars: dict[str, float]
    pct_contributions: dict[str, float]


def component_var(
    positions: dict[str, float],
    covariance: np.ndarray,
    names: list[str],
    confidence: float = 0.95,
) -> ComponentVaRResult:
    """Component VaR: decompose total VaR by position.

    Component VaR_i = w_i × (Σw)_i / σ_p × VaR_total

        result = component_var({"USD_10Y": 1e6, "EUR_5Y": 5e5}, cov, names)
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)
    w = np.array([positions[n] for n in names])
    port_var = float(w @ covariance @ w)
    port_vol = math.sqrt(max(port_var, 0))
    total_var = port_vol * z

    # Marginal VaR: ∂VaR/∂w_i = z × (Σw)_i / σ_p
    sigma_w = covariance @ w
    marginal = {}
    component = {}
    pct = {}
    for i, n in enumerate(names):
        if port_vol > 1e-10:
            m = z * sigma_w[i] / port_vol
        else:
            m = 0.0
        marginal[n] = m
        component[n] = m * w[i]
        pct[n] = component[n] / total_var if total_var > 1e-10 else 0.0

    return ComponentVaRResult(total_var, component, marginal, pct)


# ============================================================================
# Stress testing
# ============================================================================

@dataclass
class StressScenario:
    """A named stress scenario."""
    name: str
    shifts: dict[str, float]   # {risk_factor: shift_amount}
    description: str = ""


@dataclass
class StressResult:
    """Result of a stress test."""
    scenario: str
    base_pv: float
    stressed_pv: float
    pnl: float
    pnl_pct: float


# Pre-defined scenarios
GFC_2008 = StressScenario(
    "GFC 2008",
    {"rates": -0.02, "spreads": 0.03, "equity": -0.40, "fx": 0.10, "vol": 0.20},
    "Global Financial Crisis: rates down, spreads wide, equity crash, vol spike",
)

COVID_2020 = StressScenario(
    "COVID 2020",
    {"rates": -0.015, "spreads": 0.02, "equity": -0.30, "fx": 0.05, "vol": 0.30},
    "COVID crash: rates down, spreads wide, equity drop, massive vol spike",
)

RATES_UP_200 = StressScenario(
    "Rates +200bp",
    {"rates": 0.02},
    "Parallel rates shock +200bp",
)

RATES_DOWN_100 = StressScenario(
    "Rates -100bp",
    {"rates": -0.01},
    "Parallel rates shock -100bp",
)

EQUITY_CRASH = StressScenario(
    "Equity -20%",
    {"equity": -0.20, "vol": 0.15},
    "Equity correction with vol spike",
)

STANDARD_SCENARIOS = [GFC_2008, COVID_2020, RATES_UP_200, RATES_DOWN_100, EQUITY_CRASH]


def stress_test(
    base_pv: float,
    sensitivities: dict[str, float],
    scenarios: list[StressScenario] | None = None,
) -> list[StressResult]:
    """Run stress scenarios using first-order sensitivities.

    P&L ≈ Σ sensitivity_i × shift_i

        results = stress_test(1_000_000, {"rates": -50_000, "equity": 200_000})
    """
    if scenarios is None:
        scenarios = STANDARD_SCENARIOS

    results = []
    for sc in scenarios:
        pnl = sum(
            sensitivities.get(rf, 0.0) * shift
            for rf, shift in sc.shifts.items()
        )
        stressed = base_pv + pnl
        pnl_pct = pnl / abs(base_pv) if abs(base_pv) > 1e-10 else 0.0
        results.append(StressResult(sc.name, base_pv, stressed, pnl, pnl_pct))

    return results


# ============================================================================
# Drawdown management
# ============================================================================

@dataclass
class DrawdownResult:
    """Drawdown analysis."""
    current_drawdown: float
    max_drawdown: float
    max_drawdown_duration: int
    in_drawdown: bool
    peak: float
    trough: float


def analyse_drawdown(equity_curve: np.ndarray | list[float]) -> DrawdownResult:
    """Analyse drawdown from an equity curve.

        dd = analyse_drawdown(equity_curve)
        if dd.current_drawdown > 0.10:
            print("WARNING: 10% drawdown")
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return DrawdownResult(0.0, 0.0, 0, False, eq[0] if len(eq) > 0 else 0.0, eq[0] if len(eq) > 0 else 0.0)

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(dd.max())
    current_dd = float(dd[-1])

    # Duration
    in_dd = dd > 0
    dur = 0
    max_dur = 0
    for d in in_dd:
        if d:
            dur += 1
            max_dur = max(max_dur, dur)
        else:
            dur = 0

    trough_idx = int(np.argmax(dd))

    return DrawdownResult(
        current_drawdown=current_dd,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dur,
        in_drawdown=current_dd > 0.001,
        peak=float(peak.max()),
        trough=float(eq[trough_idx]),
    )


# ============================================================================
# Concentration limits
# ============================================================================

@dataclass
class ConcentrationResult:
    """Concentration analysis."""
    herfindahl: float           # HHI (sum of squared weights)
    effective_n: float          # 1/HHI — effective number of positions
    top_1_pct: float
    top_5_pct: float
    is_concentrated: bool       # HHI > threshold


def concentration_check(
    positions: dict[str, float],
    hhi_threshold: float = 0.15,
) -> ConcentrationResult:
    """Check portfolio concentration.

        conc = concentration_check({"AAPL": 0.30, "MSFT": 0.25, ...})
    """
    if not positions:
        return ConcentrationResult(0.0, 0.0, 0.0, 0.0, False)

    total = sum(abs(v) for v in positions.values())
    if total < 1e-10:
        return ConcentrationResult(0.0, 0.0, 0.0, 0.0, False)

    weights = sorted([abs(v) / total for v in positions.values()], reverse=True)
    hhi = sum(w**2 for w in weights)
    eff_n = 1.0 / hhi if hhi > 1e-10 else len(weights)
    top_1 = weights[0] if weights else 0.0
    top_5 = sum(weights[:5])

    return ConcentrationResult(
        herfindahl=hhi,
        effective_n=eff_n,
        top_1_pct=top_1,
        top_5_pct=top_5,
        is_concentrated=hhi > hhi_threshold,
    )
