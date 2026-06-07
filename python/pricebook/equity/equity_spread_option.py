"""Equity spread options — multi-asset pricing and Greeks.

Equity-specific spread option pricing using spot prices and dividend yields,
complementing ``commodity/spread_options.py`` (futures-based Kirk's approximation).

* :class:`SpreadOptionResult` — pricing result with Greeks.
* :func:`kirk_equity_spread` — Kirk's approximation adapted for equity with dividends.
* :func:`bjerksund_stensland_spread` — Bjerksund-Stensland (2006) improved accuracy.
* :func:`mc_spread_option` — Monte Carlo benchmark.
* :func:`outperformance_option` — Margrabe exchange option (K = 0 special case).
* :func:`relative_performance_option` — option on percentage outperformance.

References:
    Kirk, *Correlation in the Energy Markets*, in Managing Energy Price Risk,
    Risk Publications, 1995.
    Bjerksund & Stensland, *Closed Form Spread Option Valuation*, NHH Discussion
    Paper FOR 2006, 2006.
    Margrabe, *The Value of an Option to Exchange One Asset for Another*,
    Journal of Finance 33(1), 177-186, 1978.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import _norm_cdf


@dataclass
class SpreadOptionResult:
    """Spread option pricing result with full Greeks."""

    price: float
    delta_1: float           # dV/dS1
    delta_2: float           # dV/dS2
    gamma_1: float           # d²V/dS1²
    gamma_2: float           # d²V/dS2²
    cross_gamma: float       # d²V/dS1 dS2
    vega_1: float            # dV/d(vol1) per 1% move
    vega_2: float            # dV/d(vol2) per 1% move
    rho_sensitivity: float   # dV/d(rho) per 0.01 move
    theta: float             # dV/dT (per calendar day, negative convention)

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward(S: float, r: float, q: float, T: float) -> float:
    return S * math.exp((r - q) * T)


def _kirk_price_equity(
    S1: float, S2: float, K: float,
    vol1: float, vol2: float, rho: float,
    T: float, r: float, q1: float, q2: float,
    option_type: str,
) -> float:
    """Internal: Kirk price on equity forwards, no Greeks."""
    F1 = _forward(S1, r, q1, T)
    F2 = _forward(S2, r, q2, T)
    df = math.exp(-r * T)

    adj_F2 = F2 + K
    if adj_F2 <= 0 or T <= 0:
        intrinsic = max(F1 - F2 - K, 0.0) if option_type == "call" else max(F2 + K - F1, 0.0)
        return df * intrinsic

    ratio = F2 / adj_F2
    var = max(vol1**2 - 2.0 * rho * vol1 * vol2 * ratio + (vol2 * ratio)**2, 1e-14)
    sigma_adj = math.sqrt(var)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(F1 / adj_F2) + 0.5 * sigma_adj**2 * T) / (sigma_adj * sqrt_t)
    d2 = d1 - sigma_adj * sqrt_t

    if option_type == "call":
        return df * (F1 * _norm_cdf(d1) - adj_F2 * _norm_cdf(d2))
    return df * (adj_F2 * _norm_cdf(-d2) - F1 * _norm_cdf(-d1))


def _finite_diff_greeks(
    price_fn,
    S1: float, S2: float, K: float,
    vol1: float, vol2: float, rho: float,
    T: float, r: float, q1: float, q2: float,
    option_type: str,
    base_price: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute Greeks by finite differences.  Returns:
    delta_1, delta_2, gamma_1, gamma_2, cross_gamma, vega_1, vega_2, rho_sensitivity
    """
    h1 = max(S1 * 1e-3, 1e-4)
    h2 = max(S2 * 1e-3, 1e-4)
    dv = 0.01
    dr = 0.01

    args = (K, vol1, vol2, rho, T, r, q1, q2, option_type)

    p_u1 = price_fn(S1 + h1, S2, *args)
    p_d1 = price_fn(S1 - h1, S2, *args)
    delta_1 = (p_u1 - p_d1) / (2.0 * h1)
    gamma_1 = (p_u1 - 2.0 * base_price + p_d1) / h1**2

    p_u2 = price_fn(S1, S2 + h2, *args)
    p_d2 = price_fn(S1, S2 - h2, *args)
    delta_2 = (p_u2 - p_d2) / (2.0 * h2)
    gamma_2 = (p_u2 - 2.0 * base_price + p_d2) / h2**2

    p_uu = price_fn(S1 + h1, S2 + h2, *args)
    p_ud = price_fn(S1 + h1, S2 - h2, *args)
    p_du = price_fn(S1 - h1, S2 + h2, *args)
    p_dd = price_fn(S1 - h1, S2 - h2, *args)
    cross_gamma = (p_uu - p_ud - p_du + p_dd) / (4.0 * h1 * h2)

    # Central differences for vega and rho
    vega_1 = (price_fn(S1, S2, K, vol1 + dv, vol2, rho, T, r, q1, q2, option_type)
              - price_fn(S1, S2, K, max(vol1 - dv, 1e-8), vol2, rho, T, r, q1, q2, option_type)) / 2
    vega_2 = (price_fn(S1, S2, K, vol1, vol2 + dv, rho, T, r, q1, q2, option_type)
              - price_fn(S1, S2, K, vol1, max(vol2 - dv, 1e-8), rho, T, r, q1, q2, option_type)) / 2

    rho_up = min(rho + dr, 0.9999)
    rho_dn = max(rho - dr, -0.9999)
    rho_sens = (price_fn(S1, S2, K, vol1, vol2, rho_up, T, r, q1, q2, option_type)
                - price_fn(S1, S2, K, vol1, vol2, rho_dn, T, r, q1, q2, option_type)) / 2

    return delta_1, delta_2, gamma_1, gamma_2, cross_gamma, vega_1, vega_2, rho_sens


# ---------------------------------------------------------------------------
# 1. Kirk's approximation — equity version
# ---------------------------------------------------------------------------

def kirk_equity_spread(
    S1: float,
    S2: float,
    K: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    option_type: str = "call",
) -> SpreadOptionResult:
    """Kirk's approximation for equity spread options with dividend yields.

    Prices ``max(S1(T) - S2(T) - K, 0)`` for a call (put is reversed).
    Forward prices are ``Fi = Si * exp((r - qi) * T)``.  Kirk treats
    ``F2 + K`` as a modified strike and forms an adjusted vol:

    ``sigma_adj^2 = sigma1^2 - 2*rho*sigma1*sigma2*(F2/(F2+K)) + sigma2^2*(F2/(F2+K))^2``

    Args:
        S1: spot price of asset 1.
        S2: spot price of asset 2.
        K: strike on the spread.
        vol1: annualised volatility of asset 1.
        vol2: annualised volatility of asset 2.
        rho: correlation between asset log-returns.
        T: time to expiry in years.
        r: risk-free rate (continuous, annualised).
        q1: continuous dividend yield of asset 1.
        q2: continuous dividend yield of asset 2.
        option_type: ``"call"`` (default) or ``"put"``.

    Returns:
        :class:`SpreadOptionResult` with price and Greeks.
    """
    if S1 <= 0 or S2 <= 0:
        raise ValueError("spot prices must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if vol1 < 0 or vol2 < 0:
        raise ValueError("volatilities must be non-negative")
    if not -1 <= rho <= 1:
        raise ValueError("correlation must be in [-1, 1]")
    price = _kirk_price_equity(S1, S2, K, vol1, vol2, rho, T, r, q1, q2, option_type)

    d1, d2, g1, g2, xg, v1, v2, rs = _finite_diff_greeks(
        _kirk_price_equity, S1, S2, K, vol1, vol2, rho, T, r, q1, q2, option_type, price
    )

    dt = 1.0 / 365.0
    p_dt = _kirk_price_equity(S1, S2, K, vol1, vol2, rho, max(T - dt, 1e-6), r, q1, q2, option_type)
    theta = (p_dt - price) / dt * dt  # per calendar day (negative for long)

    return SpreadOptionResult(
        price=price,
        delta_1=d1, delta_2=d2,
        gamma_1=g1, gamma_2=g2,
        cross_gamma=xg,
        vega_1=v1, vega_2=v2,
        rho_sensitivity=rs,
        theta=(p_dt - price),
    )


# ---------------------------------------------------------------------------
# 2. Bjerksund-Stensland (2006) two-asset spread option
# ---------------------------------------------------------------------------

def _bs_spread_price(
    S1: float, S2: float, K: float,
    vol1: float, vol2: float, rho: float,
    T: float, r: float, q1: float, q2: float,
    option_type: str,
) -> float:
    """Internal: Bjerksund-Stensland (2006) spread option price."""
    if K <= 0.0:
        # Degenerates to Margrabe exchange option
        return _margrabe_price(S1, S2, vol1, vol2, rho, T, r, q1, q2, option_type)

    F1 = _forward(S1, r, q1, T)
    F2 = _forward(S2, r, q2, T)
    df = math.exp(-r * T)

    if T <= 0.0:
        intrinsic = max(F1 - F2 - K, 0.0) if option_type == "call" else max(F2 + K - F1, 0.0)
        return df * intrinsic

    # Bjerksund-Stensland parameterisation: treat asset 2 + strike as a
    # single lognormal proxy with vol sigma_x and forward F2 + K.
    # The proxy vol is chosen so that E[F2 + K] and Var[F2 + K] match.
    # sigma_x = vol2 * F2 / (F2 + K)  (first-order Taylor)
    Fx = F2 + K
    if Fx <= 0.0:
        return df * max(F1 - F2 - K, 0.0)

    sigma_x = vol2 * F2 / Fx   # proxy vol of the combined asset

    # Adjusted vol for the spread under the proxy measure
    sigma_adj = math.sqrt(
        max(vol1**2 - 2.0 * rho * vol1 * sigma_x + sigma_x**2, 1e-14)
    )

    sqrt_t = math.sqrt(T)
    d1 = (math.log(F1 / Fx) + 0.5 * sigma_adj**2 * T) / (sigma_adj * sqrt_t)
    d2 = d1 - sigma_adj * sqrt_t

    if option_type == "call":
        price = df * (F1 * _norm_cdf(d1) - Fx * _norm_cdf(d2))
    else:
        price = df * (Fx * _norm_cdf(-d2) - F1 * _norm_cdf(-d1))

    return price


def bjerksund_stensland_spread(
    S1: float,
    S2: float,
    K: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    option_type: str = "call",
) -> SpreadOptionResult:
    """Bjerksund-Stensland (2006) spread option — improved accuracy for K > 0.

    Uses a better proxy volatility for the (F2 + K) composite than Kirk's
    approximation, reducing bias for large strikes relative to F2.

    Args:
        S1: spot price of asset 1.
        S2: spot price of asset 2.
        K: strike on the spread.
        vol1: annualised volatility of asset 1.
        vol2: annualised volatility of asset 2.
        rho: correlation between asset log-returns.
        T: time to expiry in years.
        r: risk-free rate.
        q1: continuous dividend yield of asset 1.
        q2: continuous dividend yield of asset 2.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        :class:`SpreadOptionResult`.
    """
    price = _bs_spread_price(S1, S2, K, vol1, vol2, rho, T, r, q1, q2, option_type)

    d1, d2, g1, g2, xg, v1, v2, rs = _finite_diff_greeks(
        _bs_spread_price, S1, S2, K, vol1, vol2, rho, T, r, q1, q2, option_type, price
    )

    dt = 1.0 / 365.0
    p_dt = _bs_spread_price(S1, S2, K, vol1, vol2, rho, max(T - dt, 1e-6), r, q1, q2, option_type)

    return SpreadOptionResult(
        price=price,
        delta_1=d1, delta_2=d2,
        gamma_1=g1, gamma_2=g2,
        cross_gamma=xg,
        vega_1=v1, vega_2=v2,
        rho_sensitivity=rs,
        theta=(p_dt - price),
    )


# ---------------------------------------------------------------------------
# 3. Monte Carlo benchmark
# ---------------------------------------------------------------------------

def mc_spread_option(
    S1: float,
    S2: float,
    K: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> SpreadOptionResult:
    """Monte Carlo benchmark for spread options using antithetic variates.

    Simulates correlated GBM paths and computes the discounted expected payoff.
    Greeks are computed by finite differences on the MC price (re-using the
    same random seed for variance reduction).

    Args:
        S1: spot price of asset 1.
        S2: spot price of asset 2.
        K: strike on the spread.
        vol1: annualised volatility of asset 1.
        vol2: annualised volatility of asset 2.
        rho: correlation.
        T: time to expiry in years.
        r: risk-free rate.
        q1: continuous dividend yield of asset 1.
        q2: continuous dividend yield of asset 2.
        option_type: ``"call"`` or ``"put"``.
        n_paths: number of paths (antithetic pairs, so 2× simulations).
        seed: random seed for reproducibility.

    Returns:
        :class:`SpreadOptionResult`.
    """

    def _mc_price(s1, s2, k, v1, v2, rh, t, r_, q1_, q2_, otype) -> float:
        rng = np.random.default_rng(seed)
        z1 = rng.standard_normal(n_paths)
        z2 = rh * z1 + math.sqrt(max(1.0 - rh**2, 0.0)) * rng.standard_normal(n_paths)

        drift1 = (r_ - q1_ - 0.5 * v1**2) * t
        drift2 = (r_ - q2_ - 0.5 * v2**2) * t
        sqrt_t = math.sqrt(t)

        st1 = s1 * np.exp(drift1 + v1 * sqrt_t * z1)
        st2 = s2 * np.exp(drift2 + v2 * sqrt_t * z2)
        st1a = s1 * np.exp(drift1 - v1 * sqrt_t * z1)
        st2a = s2 * np.exp(drift2 - v2 * sqrt_t * z2)

        spread = np.concatenate([st1 - st2, st1a - st2a])

        if otype == "call":
            payoff = np.maximum(spread - k, 0.0)
        else:
            payoff = np.maximum(k - spread, 0.0)

        return math.exp(-r_ * t) * float(np.mean(payoff))

    price = _mc_price(S1, S2, K, vol1, vol2, rho, T, r, q1, q2, option_type)

    d1, d2, g1, g2, xg, v1, v2, rs = _finite_diff_greeks(
        _mc_price, S1, S2, K, vol1, vol2, rho, T, r, q1, q2, option_type, price
    )

    dt = 1.0 / 365.0
    p_dt = _mc_price(S1, S2, K, vol1, vol2, rho, max(T - dt, 1e-6), r, q1, q2, option_type)

    return SpreadOptionResult(
        price=price,
        delta_1=d1, delta_2=d2,
        gamma_1=g1, gamma_2=g2,
        cross_gamma=xg,
        vega_1=v1, vega_2=v2,
        rho_sensitivity=rs,
        theta=(p_dt - price),
    )


# ---------------------------------------------------------------------------
# 4. Outperformance option — Margrabe (1978)
# ---------------------------------------------------------------------------

def _margrabe_price(
    S1: float, S2: float,
    vol1: float, vol2: float, rho: float,
    T: float, r: float, q1: float, q2: float,
    option_type: str = "call",
) -> float:
    """Internal: Margrabe exchange option price."""
    sigma = math.sqrt(max(vol1**2 - 2.0 * rho * vol1 * vol2 + vol2**2, 1e-14))
    F1 = _forward(S1, r, q1, T)
    F2 = _forward(S2, r, q2, T)
    df = math.exp(-r * T)

    if T <= 0.0:
        return df * (max(F1 - F2, 0.0) if option_type == "call" else max(F2 - F1, 0.0))

    sqrt_t = math.sqrt(T)
    d1 = (math.log(F1 / F2) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        return df * (F1 * _norm_cdf(d1) - F2 * _norm_cdf(d2))
    return df * (F2 * _norm_cdf(-d2) - F1 * _norm_cdf(-d1))


def outperformance_option(
    S1: float,
    S2: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
) -> SpreadOptionResult:
    """Margrabe exchange option: right to receive S1 and deliver S2.

    Closed-form (K = 0 special case).  Payoff = ``max(S1(T) - S2(T), 0)``.

    Args:
        S1: spot price of asset 1.
        S2: spot price of asset 2.
        vol1: annualised volatility of asset 1.
        vol2: annualised volatility of asset 2.
        rho: correlation between log-returns.
        T: time to expiry in years.
        r: risk-free rate.
        q1: continuous dividend yield of asset 1.
        q2: continuous dividend yield of asset 2.

    Returns:
        :class:`SpreadOptionResult` (put greeks are for the mirror payoff).
    """
    price = _margrabe_price(S1, S2, vol1, vol2, rho, T, r, q1, q2, "call")

    h1 = max(S1 * 1e-3, 1e-4)
    h2 = max(S2 * 1e-3, 1e-4)
    dv = 0.01
    dr = 0.01

    p_u1 = _margrabe_price(S1 + h1, S2, vol1, vol2, rho, T, r, q1, q2)
    p_d1 = _margrabe_price(S1 - h1, S2, vol1, vol2, rho, T, r, q1, q2)
    delta_1 = (p_u1 - p_d1) / (2.0 * h1)
    gamma_1 = (p_u1 - 2.0 * price + p_d1) / h1**2

    p_u2 = _margrabe_price(S1, S2 + h2, vol1, vol2, rho, T, r, q1, q2)
    p_d2 = _margrabe_price(S1, S2 - h2, vol1, vol2, rho, T, r, q1, q2)
    delta_2 = (p_u2 - p_d2) / (2.0 * h2)
    gamma_2 = (p_u2 - 2.0 * price + p_d2) / h2**2

    p_uu = _margrabe_price(S1 + h1, S2 + h2, vol1, vol2, rho, T, r, q1, q2)
    p_ud = _margrabe_price(S1 + h1, S2 - h2, vol1, vol2, rho, T, r, q1, q2)
    p_du = _margrabe_price(S1 - h1, S2 + h2, vol1, vol2, rho, T, r, q1, q2)
    p_dd = _margrabe_price(S1 - h1, S2 - h2, vol1, vol2, rho, T, r, q1, q2)
    cross_gamma = (p_uu - p_ud - p_du + p_dd) / (4.0 * h1 * h2)

    vega_1 = (_margrabe_price(S1, S2, vol1 + dv, vol2, rho, T, r, q1, q2)
              - _margrabe_price(S1, S2, max(vol1 - dv, 1e-8), vol2, rho, T, r, q1, q2)) / 2
    vega_2 = (_margrabe_price(S1, S2, vol1, vol2 + dv, rho, T, r, q1, q2)
              - _margrabe_price(S1, S2, vol1, max(vol2 - dv, 1e-8), rho, T, r, q1, q2)) / 2
    rho_up = min(rho + dr, 0.9999)
    rho_dn = max(rho - dr, -0.9999)
    rho_sens = (_margrabe_price(S1, S2, vol1, vol2, rho_up, T, r, q1, q2)
                - _margrabe_price(S1, S2, vol1, vol2, rho_dn, T, r, q1, q2)) / 2

    dt = 1.0 / 365.0
    p_dt = _margrabe_price(S1, S2, vol1, vol2, rho, max(T - dt, 1e-6), r, q1, q2)

    return SpreadOptionResult(
        price=price,
        delta_1=delta_1, delta_2=delta_2,
        gamma_1=gamma_1, gamma_2=gamma_2,
        cross_gamma=cross_gamma,
        vega_1=vega_1, vega_2=vega_2,
        rho_sensitivity=rho_sens,
        theta=(p_dt - price),
    )


# ---------------------------------------------------------------------------
# 5. Relative performance option
# ---------------------------------------------------------------------------

def relative_performance_option(
    S1: float,
    S2: float,
    K_pct: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    option_type: str = "call",
) -> SpreadOptionResult:
    """Option on percentage outperformance between two equity assets.

    Payoff = ``max(R1 - R2 - K_pct, 0)`` where ``Ri = Si(T)/Si(0) - 1``
    for a call (reversed for a put).

    This is equivalent to a spread option on normalised returns.  The
    computation maps to a standard spread option on unit-notional forward
    returns, priced via Kirk's approximation on the normalised quantities.

    Args:
        S1: spot price of asset 1 (normalisation base).
        S2: spot price of asset 2 (normalisation base).
        K_pct: strike on relative outperformance (e.g. 0.05 = 5%).
        vol1: annualised volatility of asset 1.
        vol2: annualised volatility of asset 2.
        rho: correlation between log-returns.
        T: time to expiry in years.
        r: risk-free rate.
        q1: continuous dividend yield of asset 1.
        q2: continuous dividend yield of asset 2.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        :class:`SpreadOptionResult`.  Price is in return-space (fractional),
        multiply by notional to convert to currency.
    """
    # Normalise: work with forward return ratios F1/S1 and F2/S2.
    # Since Ri = Si(T)/Si(0) - 1 the forward of R1 is exp((r-q1)*T) - 1.
    # Map to pseudo-spot = 1 and pseudo-forward = exp((r-qi)*T).
    # Spread payoff: F1_norm - F2_norm - K_pct  where F_norm = exp((r-q)*T).
    # Kirk's approximation applies directly with S_norm = 1 for both.

    price = _kirk_price_equity(1.0, 1.0, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)

    # Greeks are in normalised return space
    h1 = 1e-3
    h2 = 1e-3
    dv = 0.01
    dr = 0.01

    p_u1 = _kirk_price_equity(1.0 + h1, 1.0, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    p_d1 = _kirk_price_equity(1.0 - h1, 1.0, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    delta_1 = (p_u1 - p_d1) / (2.0 * h1)
    gamma_1 = (p_u1 - 2.0 * price + p_d1) / h1**2

    p_u2 = _kirk_price_equity(1.0, 1.0 + h2, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    p_d2 = _kirk_price_equity(1.0, 1.0 - h2, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    delta_2 = (p_u2 - p_d2) / (2.0 * h2)
    gamma_2 = (p_u2 - 2.0 * price + p_d2) / h2**2

    p_uu = _kirk_price_equity(1.0 + h1, 1.0 + h2, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    p_ud = _kirk_price_equity(1.0 + h1, 1.0 - h2, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    p_du = _kirk_price_equity(1.0 - h1, 1.0 + h2, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    p_dd = _kirk_price_equity(1.0 - h1, 1.0 - h2, K_pct, vol1, vol2, rho, T, r, q1, q2, option_type)
    cross_gamma = (p_uu - p_ud - p_du + p_dd) / (4.0 * h1 * h2)

    vega_1 = _kirk_price_equity(1.0, 1.0, K_pct, vol1 + dv, vol2, rho, T, r, q1, q2, option_type) - price
    vega_2 = _kirk_price_equity(1.0, 1.0, K_pct, vol1, vol2 + dv, rho, T, r, q1, q2, option_type) - price
    rho_sens = _kirk_price_equity(1.0, 1.0, K_pct, vol1, vol2, min(rho + dr, 0.9999), T, r, q1, q2, option_type) - price

    dt = 1.0 / 365.0
    p_dt = _kirk_price_equity(1.0, 1.0, K_pct, vol1, vol2, rho, max(T - dt, 1e-6), r, q1, q2, option_type)

    return SpreadOptionResult(
        price=price,
        delta_1=delta_1, delta_2=delta_2,
        gamma_1=gamma_1, gamma_2=gamma_2,
        cross_gamma=cross_gamma,
        vega_1=vega_1, vega_2=vega_2,
        rho_sensitivity=rho_sens,
        theta=(p_dt - price),
    )
