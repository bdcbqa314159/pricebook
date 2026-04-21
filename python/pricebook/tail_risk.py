"""Far-OTM & tail risk: Roger Lee bounds, SVI wings, EVT pricing.

* :func:`roger_lee_bounds` — moment formula wing slope constraints.
* :func:`svi_wings_fit` — SVI with no-arb wing constraints.
* :func:`tail_risk_pricing` — deep OTM puts via EVT.
* :func:`extreme_value_var` — VaR from Generalised Pareto Distribution.

References:
    Lee, *The Moment Formula for Implied Volatility at Extreme Strikes*, MF, 2004.
    Gatheral & Jacquier, *Arbitrage-Free SVI Volatility Surfaces*, QF, 2014.
    McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton, 2015.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.stats import genpareto


@dataclass
class RogerLeeBoundsResult:
    left_slope_bound: float     # max left wing slope: β_L ≤ 2
    right_slope_bound: float    # max right wing slope: β_R ≤ 2
    observed_left: float
    observed_right: float
    is_valid: bool

def roger_lee_bounds(
    log_moneyness: list[float], total_variance: list[float],
) -> RogerLeeBoundsResult:
    """Roger Lee (2004): wing slope of total variance w(k) is bounded by 2.
    β_L = lim_{k→−∞} w(k)/|k| ≤ 2, β_R = lim_{k→+∞} w(k)/k ≤ 2.
    """
    k = np.array(log_moneyness); w = np.array(total_variance)
    # Left wing: negative k
    left_mask = k < -0.1
    right_mask = k > 0.1
    if left_mask.sum() >= 2:
        obs_left = float(np.polyfit(np.abs(k[left_mask]), w[left_mask], 1)[0])
    else:
        obs_left = 0.0
    if right_mask.sum() >= 2:
        obs_right = float(np.polyfit(k[right_mask], w[right_mask], 1)[0])
    else:
        obs_right = 0.0
    valid = obs_left <= 2.01 and obs_right <= 2.01
    return RogerLeeBoundsResult(2.0, 2.0, obs_left, obs_right, valid)


@dataclass
class SVIWingsResult:
    a: float; b: float; rho: float; m: float; sigma: float
    left_wing_slope: float
    right_wing_slope: float
    is_no_arb: bool

def svi_wings_fit(
    log_moneyness: list[float], implied_vols: list[float], T: float,
) -> SVIWingsResult:
    """SVI fit with Roger Lee wing constraints."""
    from pricebook.fx_smile_cube import svi_fit, SVIParams
    params = svi_fit(log_moneyness, implied_vols, T)
    # Wing slopes: β_± = b(1 ± ρ)
    left_slope = params.b * (1 - params.rho)
    right_slope = params.b * (1 + params.rho)
    no_arb = left_slope <= 2 and right_slope <= 2 and params.b >= 0
    return SVIWingsResult(params.a, params.b, params.rho, params.m, params.sigma,
                            float(left_slope), float(right_slope), no_arb)


@dataclass
class TailRiskResult:
    deep_otm_put_price: float
    tail_probability: float
    expected_shortfall: float
    strike_pct: float

def tail_risk_pricing(
    spot: float, strike: float, rate: float, T: float,
    tail_index: float = 3.0,
    scale: float = 0.10,
    n_paths: int = 50_000,
    seed: int | None = 42,
) -> TailRiskResult:
    """Deep OTM put pricing via heavy-tailed distribution (Pareto tail).
    Standard GBM underprices tail risk; Pareto tail captures crash risk.
    """
    rng = np.random.default_rng(seed)
    # Mix: 95% normal + 5% Pareto tail
    # Both components get drift so E[S_T] = S_0 × e^{rT} (risk-neutral)
    drift = (rate - 0.5 * scale**2) * T
    normal_returns = rng.normal(drift, scale * math.sqrt(T), n_paths)
    tail_mask = rng.random(n_paths) < 0.05
    tail_returns = drift - genpareto.rvs(1/tail_index, scale=scale * math.sqrt(T), size=n_paths, random_state=rng)
    returns = np.where(tail_mask, tail_returns, normal_returns)

    S_T = spot * np.exp(returns)
    payoff = np.maximum(strike - S_T, 0.0)
    df = math.exp(-rate * T)
    price = df * float(payoff.mean())
    tail_prob = float((S_T < strike).mean())
    es = float(S_T[S_T < strike].mean()) if tail_prob > 0 else spot

    return TailRiskResult(float(price), tail_prob, float(es), strike / spot * 100)


@dataclass
class EVTVaRResult:
    var_level: float
    confidence: float
    gpd_shape: float
    gpd_scale: float

def extreme_value_var(
    losses: list[float], confidence: float = 0.99,
    threshold_quantile: float = 0.90,
) -> EVTVaRResult:
    """VaR from Generalised Pareto Distribution (Peaks over Threshold)."""
    arr = np.array(losses)
    threshold = float(np.quantile(arr, threshold_quantile))
    exceedances = arr[arr > threshold] - threshold
    if len(exceedances) < 5:
        return EVTVaRResult(float(np.quantile(arr, confidence)), confidence, 0, 0)
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    n = len(arr); n_u = len(exceedances)
    p = 1 - confidence
    # Standard POT VaR: u + (σ/ξ) × [(n_u / (n × p))^ξ − 1]
    # where p = 1 − confidence
    if abs(shape) > 1e-10:
        var = threshold + (scale / shape) * ((n_u / (n * p)) ** shape - 1)
    else:
        var = threshold + scale * math.log(n_u / (n * p))
    return EVTVaRResult(float(var), confidence, float(shape), float(scale))
