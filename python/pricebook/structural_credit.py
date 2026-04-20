"""Structural credit: Merton, KMV, first-passage (Black-Cox).

* :func:`merton_equity_credit` — equity price → credit spread via Merton.
* :func:`kmv_distance_to_default` — KMV DD from equity vol + leverage.
* :func:`black_cox_first_passage` — first-passage default with barrier.
* :func:`implied_credit_from_equity` — back out spread from equity vol.

References:
    Merton, *On the Pricing of Corporate Debt*, JF, 1974.
    Black & Cox, *Valuing Corporate Securities: Some Effects of Bond Indenture
    Provisions*, JF, 1976.
    Crosbie & Bohn, *Modeling Default Risk*, KMV, 2003.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class MertonResult:
    equity_value: float
    debt_value: float
    asset_value: float
    asset_vol: float
    default_probability: float
    credit_spread_bps: float

def merton_equity_credit(
    asset_value: float, debt_face: float, asset_vol: float,
    rate: float, T: float,
) -> MertonResult:
    """Merton (1974): equity = call on assets; debt = assets − equity.
    E = V × N(d₁) − D × e^{−rT} × N(d₂).
    Default prob = N(−d₂).
    Credit spread = −ln(D_value / (D_face × e^{−rT})) / T − r + r.
    """
    d1 = (math.log(asset_value / debt_face) + (rate + 0.5 * asset_vol**2) * T) / (asset_vol * math.sqrt(T))
    d2 = d1 - asset_vol * math.sqrt(T)

    equity = asset_value * norm.cdf(d1) - debt_face * math.exp(-rate * T) * norm.cdf(d2)
    debt = asset_value - equity
    default_prob = norm.cdf(-d2)

    risk_free_debt = debt_face * math.exp(-rate * T)
    if debt > 0 and risk_free_debt > 0:
        spread = -math.log(debt / risk_free_debt) / max(T, 1e-10) * 10000
    else:
        spread = 0.0

    return MertonResult(float(equity), float(debt), asset_value, asset_vol,
                          float(default_prob), float(max(spread, 0)))


@dataclass
class KMVResult:
    distance_to_default: float
    default_probability: float
    asset_value: float
    default_point: float

def kmv_distance_to_default(
    equity_value: float, equity_vol: float,
    short_term_debt: float, long_term_debt: float,
    rate: float, T: float = 1.0,
) -> KMVResult:
    """KMV distance-to-default.
    Default point = STD + 0.5 × LTD.
    DD = (V − DP) / (V × σ_V).
    Asset value V ≈ equity + debt (simplification).
    Asset vol σ_V ≈ equity_vol × equity / (equity + debt).
    """
    dp = short_term_debt + 0.5 * long_term_debt
    total_debt = short_term_debt + long_term_debt
    V = equity_value + total_debt
    sigma_V = equity_vol * equity_value / max(V, 1e-10)

    dd = (V - dp) / max(V * sigma_V, 1e-10)
    pd = norm.cdf(-dd)

    return KMVResult(float(dd), float(pd), float(V), float(dp))


@dataclass
class BlackCoxResult:
    survival_probability: float
    default_probability: float
    credit_spread_bps: float
    barrier: float

def black_cox_first_passage(
    asset_value: float, barrier: float, asset_vol: float,
    rate: float, T: float,
) -> BlackCoxResult:
    """Black-Cox: default occurs first time V hits barrier H (H < V₀).
    P(τ ≤ T) = N(−a) + (H/V)^{2μ/σ²} × N(−b)
    where a = [ln(V/H) + μT] / (σ√T), b = [ln(V/H) − μT] / (σ√T).
    """
    if asset_value <= barrier:
        return BlackCoxResult(0.0, 1.0, 10000.0, barrier)

    mu = rate - 0.5 * asset_vol**2
    sigma_sqrt_T = asset_vol * math.sqrt(T)
    log_VH = math.log(asset_value / barrier)

    a = (log_VH + mu * T) / sigma_sqrt_T
    b = (log_VH - mu * T) / sigma_sqrt_T

    exponent = 2 * mu / (asset_vol**2)
    pd = norm.cdf(-a) + (barrier / asset_value)**exponent * norm.cdf(-b)
    pd = min(max(pd, 0), 1)
    surv = 1 - pd

    if surv > 1e-10:
        spread = -math.log(surv) / max(T, 1e-10) * 10000
    else:
        spread = 10000.0

    return BlackCoxResult(float(surv), float(pd), float(spread), barrier)


@dataclass
class ImpliedCreditResult:
    implied_spread_bps: float
    implied_default_prob: float
    equity_vol: float
    leverage: float

def implied_credit_from_equity(
    equity_value: float, equity_vol: float,
    total_debt: float, rate: float, T: float = 5.0,
) -> ImpliedCreditResult:
    """Back out credit spread from equity vol and leverage."""
    V = equity_value + total_debt
    leverage = total_debt / max(V, 1e-10)
    sigma_V = equity_vol * equity_value / max(V, 1e-10)

    merton = merton_equity_credit(V, total_debt, sigma_V, rate, T)

    return ImpliedCreditResult(
        float(merton.credit_spread_bps), float(merton.default_probability),
        equity_vol, float(leverage))
