"""Convexity adjustments for interest rate derivatives.

Covers the key corrections needed when payment timing or index
differs from natural measure:

* :func:`cms_convexity_adjustment` — CMS rate vs forward swap rate.
* :func:`cms_rate_replication` — CMS rate via swaption replication.
* :func:`arrears_adjustment` — LIBOR-in-arrears vs LIBOR-in-advance.
* :func:`timing_adjustment` — payment delay / timing mismatch.
* :func:`quanto_ir_adjustment` — cross-currency rate adjustment.

References:
    Pelsser, *Mathematical Foundation of Convexity Correction*, QF, 2003.
    Hagan, *Convexity Conundrums*, Wilmott, 2003.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- CMS convexity adjustment ----

@dataclass
class CMSConvexityResult:
    """CMS convexity adjustment result."""
    forward_swap_rate: float
    cms_rate: float
    convexity_adjustment: float
    method: str


def cms_convexity_adjustment(
    forward_swap_rate: float,
    swap_annuity: float,
    swap_duration: float,
    vol: float,
    expiry: float,
    mean_reversion: float = 0.0,
) -> CMSConvexityResult:
    """CMS convexity adjustment (Hagan linear swap rate model).

    The CMS rate E^A[S(T)] differs from the forward swap rate S(0) by:
        CMS = S(0) + S(0)² × G''(S(0))/G'(S(0)) × σ² × T

    where G(S) = annuity as function of swap rate.

    For a flat curve: G''(S)/G'(S) ≈ duration.

    With mean reversion a:
        effective_T = (1 − e^{−aT}) / a if a > 0 else T.

    Args:
        forward_swap_rate: forward par swap rate.
        swap_annuity: PV01 of the swap.
        swap_duration: modified duration of the annuity.
        vol: swaption implied vol (lognormal).
        expiry: time to CMS fixing.
        mean_reversion: mean-reversion speed (0 = no adjustment).
    """
    if mean_reversion > 1e-10:
        effective_T = (1 - math.exp(-mean_reversion * expiry)) / mean_reversion
    else:
        effective_T = expiry

    # Convexity correction
    adj = forward_swap_rate**2 * swap_duration * vol**2 * effective_T

    cms_rate = forward_swap_rate + adj

    return CMSConvexityResult(forward_swap_rate, cms_rate, adj, "hagan_linear")


def cms_rate_replication(
    forward_swap_rate: float,
    vol: float,
    expiry: float,
    swap_tenor: float,
    n_strikes: int = 50,
    strike_range: float = 3.0,
) -> CMSConvexityResult:
    """CMS rate via static replication with payer/receiver swaptions.

    E^P[S(T)] = S(0) + ∫₀^∞ call(K) w(K) dK + ∫₋∞^0 put(K) w(K) dK

    where w(K) = −G''(K)/G'(0) (replication weights).

    Simplified: uses lognormal swaption prices with trapezoidal integration.
    """
    from pricebook.black76 import black76_price, OptionType

    S0 = forward_swap_rate
    # Generate strikes around forward
    K_min = S0 * math.exp(-strike_range * vol * math.sqrt(expiry))
    K_max = S0 * math.exp(strike_range * vol * math.sqrt(expiry))
    K_min = max(K_min, 1e-6)

    strikes = np.linspace(K_min, K_max, n_strikes)
    dK = strikes[1] - strikes[0]

    # Approximate annuity function G(K) = Σ 1/(1+K)^i for i=1..n
    n_periods = max(1, int(swap_tenor * 2))  # semi-annual

    def annuity_func(K):
        """PV of unit annuity at rate K."""
        if K <= 0:
            return n_periods * 0.5
        return sum(1.0 / (1 + 0.5 * K)**i for i in range(1, n_periods + 1)) * 0.5

    def annuity_deriv2(K):
        """Second derivative of annuity w.r.t. K (numerical)."""
        h = 1e-4
        return (annuity_func(K + h) - 2 * annuity_func(K) + annuity_func(K - h)) / h**2

    G0_prime = (annuity_func(S0 + 1e-4) - annuity_func(S0 - 1e-4)) / (2e-4)

    # Replication integral
    cms_adj = 0.0
    for K in strikes:
        w = -annuity_deriv2(K) / G0_prime if abs(G0_prime) > 1e-12 else 0.0
        if K >= S0:
            # Payer swaption (call on rate)
            price = black76_price(S0, K, expiry, vol, 1.0, OptionType.CALL)
        else:
            # Receiver swaption (put on rate)
            price = black76_price(S0, K, expiry, vol, 1.0, OptionType.PUT)
        cms_adj += w * price * dK

    cms_rate = S0 + cms_adj
    adj = cms_rate - S0

    return CMSConvexityResult(S0, cms_rate, adj, "replication")


# ---- Arrears adjustment ----

@dataclass
class ArrearsResult:
    """LIBOR-in-arrears adjustment result."""
    forward_rate: float
    arrears_rate: float
    adjustment: float


def arrears_adjustment(
    forward_rate: float,
    vol: float,
    start: float,
    tenor: float,
) -> ArrearsResult:
    """LIBOR-in-arrears convexity adjustment.

    When LIBOR L(T, T+τ) is paid at time T instead of T+τ:
        E^T[L(T)] = L(0) × (1 + L(0) τ σ² T) / (1 + L(0) τ) + correction

    Simplified: E^T[L] ≈ L(0) + L(0)² × τ × σ² × T / (1 + L(0)τ)

    Args:
        forward_rate: forward LIBOR rate.
        vol: LIBOR caplet vol.
        start: start time of the LIBOR period.
        tenor: LIBOR tenor (e.g. 0.25 for 3M).
    """
    L = forward_rate
    adj = L**2 * tenor * vol**2 * start / (1 + L * tenor)
    arrears_rate = L + adj

    return ArrearsResult(L, arrears_rate, adj)


# ---- Timing adjustment ----

@dataclass
class TimingResult:
    """Payment timing adjustment result."""
    unadjusted_rate: float
    adjusted_rate: float
    adjustment: float
    payment_delay: float


def timing_adjustment(
    forward_rate: float,
    vol: float,
    fixing_time: float,
    natural_payment_time: float,
    actual_payment_time: float,
    discount_rate: float = 0.05,
) -> TimingResult:
    """Timing/payment delay adjustment.

    When payment occurs at time T_p instead of natural time T_n:
        adj = −L × σ² × (T_p − T_n) × T_fix × ρ

    where ρ is correlation between rate and discount factor (≈ −1 for
    rates, simplified to use discount rate for approximation).

    Args:
        forward_rate: forward rate being paid.
        vol: rate volatility.
        fixing_time: time of rate fixing.
        natural_payment_time: when payment would naturally occur.
        actual_payment_time: when payment actually occurs.
        discount_rate: rate for discounting (for correlation proxy).
    """
    delay = actual_payment_time - natural_payment_time
    # Correlation proxy: rate and ZCB are negatively correlated
    # Adjustment ≈ L × σ² × T_fix × delay × r / (1 + r × delay)
    adj = forward_rate * vol**2 * fixing_time * delay * discount_rate / (1 + discount_rate * abs(delay))

    return TimingResult(forward_rate, forward_rate + adj, adj, delay)


# ---- Quanto IR adjustment ----

@dataclass
class QuantoIRResult:
    """Cross-currency rate adjustment result."""
    domestic_rate: float
    quanto_rate: float
    adjustment: float


def quanto_ir_adjustment(
    foreign_forward_rate: float,
    rate_vol: float,
    fx_vol: float,
    correlation: float,
    expiry: float,
) -> QuantoIRResult:
    """Quanto adjustment for IR rate paid in different currency.

    When a foreign rate L^f is paid in domestic currency:
        E^d[L^f(T)] = L^f(0) − σ_L × σ_FX × ρ × T × L^f(0)

    The adjustment is negative when correlation > 0 (rate up → FX up
    means domestic value of foreign rate is lower).

    Args:
        foreign_forward_rate: foreign forward rate.
        rate_vol: volatility of the foreign rate.
        fx_vol: volatility of the FX rate.
        correlation: correlation between rate and FX.
        expiry: time to fixing.
    """
    L = foreign_forward_rate
    adj = -L * rate_vol * fx_vol * correlation * expiry
    quanto_rate = L + adj

    return QuantoIRResult(L, quanto_rate, adj)
