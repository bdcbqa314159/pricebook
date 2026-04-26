"""Yield-based bond pricing: price from yield, IRR, risk factor, derivatives.

Pure-math functions on raw floats and arrays — no dates, no curves.
Used by Treasury Lock (Pucci 2019), CMT (Pucci 2014), and FixedRateBond.

    from pricebook.bond_yield import bond_price_from_yield, bond_irr

References:
    Pucci, M. (2019). Hedging the Treasury Lock. SSRN 3386521, Eq 2-5.
    Pucci, M. (2014). CMT Convexity Correction. IJTAF 17(8), Eq 4.
"""

from __future__ import annotations

import math


def bond_price_from_yield(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """Simply-compounded bond price from yield (Pucci Eq 2).

    P(y) = prod_i 1/(1+alpha_i*y) + c * sum_i alpha_i * prod_{j<=i} 1/(1+alpha_j*y)
    """
    n = len(accrual_factors)
    if n == 0:
        return 1.0
    cum_df = 1.0
    coupon_pv = 0.0
    for i in range(n):
        cum_df /= (1 + accrual_factors[i] * y)
        coupon_pv += coupon_rate * accrual_factors[i] * cum_df
    return cum_df + coupon_pv


def bond_price_from_yield_stub(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
    stub_fraction: float,
) -> float:
    """Simply-compounded bond price within a coupon period (Pucci Eq 3).

    Fractional first-period discount, then remaining bond + first coupon.
    """
    if len(accrual_factors) < 1:
        return 1.0
    base = 1 + accrual_factors[0] * y
    if base <= 0:
        raise ValueError(
            f"Negative discount base in stub: 1 + {accrual_factors[0]}*{y} = {base}")
    first_df = (1 / base) ** stub_fraction
    remaining = accrual_factors[1:]
    bond_from_2 = bond_price_from_yield(coupon_rate, remaining, y) if remaining else 1.0
    value_at_t1 = coupon_rate * accrual_factors[0] + bond_from_2
    return first_df * value_at_t1


def bond_price_continuous(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> float:
    """Continuously-compounded bond price, Hull-form (Pucci Eq 4).

    P(y) = e^{-T*y} + c * sum alpha_i * e^{-(t_i-t)*y}
    """
    redemption = math.exp(-time_to_maturity * y)
    annuity = sum(
        coupon_rate * alpha * math.exp(-tau * y)
        for alpha, tau in zip(accrual_factors, times_to_coupon)
    )
    return redemption + annuity


def bond_yield_derivatives(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> tuple[float, float, float]:
    """First, second, third yield derivatives of continuous bond price (Pucci Eq 5).

    D_y^k[P] = (-T)^k * e^{-Ty} + c * sum alpha_i * (-tau_i)^k * e^{-tau_i*y}

    Returns (D1, D2, D3). Signs alternate: D1 < 0, D2 > 0, D3 < 0.
    """
    T = time_to_maturity
    exp_T = math.exp(-T * y)
    D1 = (-T) * exp_T
    D2 = T**2 * exp_T
    D3 = (-T)**3 * exp_T
    for alpha, tau in zip(accrual_factors, times_to_coupon):
        q = -tau
        exp_tau = math.exp(-tau * y)
        D1 += coupon_rate * alpha * q * exp_tau
        D2 += coupon_rate * alpha * q**2 * exp_tau
        D3 += coupon_rate * alpha * q**3 * exp_tau
    return D1, D2, D3


def bond_irr(
    market_price: float,
    coupon_rate: float,
    accrual_factors: list[float],
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Internal rate of return: solve P(y) = market_price (simply-compounded).

    Newton from y0 = coupon_rate, bisect fallback.
    """
    def price_and_deriv(y):
        p = bond_price_from_yield(coupon_rate, accrual_factors, y)
        h = 1e-6
        dp = (bond_price_from_yield(coupon_rate, accrual_factors, y + h)
              - bond_price_from_yield(coupon_rate, accrual_factors, y - h)) / (2 * h)
        return p, dp

    y = coupon_rate if coupon_rate > 0 else 0.05
    for _ in range(max_iter):
        p, dp = price_and_deriv(y)
        if abs(p - market_price) < tol:
            return y
        if abs(dp) < 1e-15:
            break
        y -= (p - market_price) / dp
        y = max(-0.5, min(y, 2.0))

    # Bisect: expand bounds until root is bracketed
    lo, hi = -0.05, 0.5
    p_lo = bond_price_from_yield(coupon_rate, accrual_factors, lo)
    p_hi = bond_price_from_yield(coupon_rate, accrual_factors, hi)
    for _ in range(20):
        if p_lo >= market_price >= p_hi:
            break
        if p_lo < market_price:
            lo -= 0.1
            p_lo = bond_price_from_yield(coupon_rate, accrual_factors, lo)
        if p_hi > market_price:
            hi += 0.5
            p_hi = bond_price_from_yield(coupon_rate, accrual_factors, hi)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = bond_price_from_yield(coupon_rate, accrual_factors, mid)
        if p > market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def bond_risk_factor(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """RiskFactor = -dP/dy (Pucci notation). Positive for positive yields."""
    h = 1e-6
    p_up = bond_price_from_yield(coupon_rate, accrual_factors, y + h)
    p_dn = bond_price_from_yield(coupon_rate, accrual_factors, y - h)
    return -(p_up - p_dn) / (2 * h)


def ytm_cmt_bridge(
    R_cmt: float,
    K: float,
    B: float,
    n: int,
) -> float:
    """YTM-CMT Taylor bridge (Pucci 2014, Eq 4).

    R^ytm ≈ R^cmt + (K - R^cmt) - R^cmt / (1 - (1+R^cmt)^{-n}) * (B - 1)

    Maps a CMT rate to an approximate YTM given coupon K and bond price B.
    Exact when B = 1 and K = R^cmt.
    """
    if abs(R_cmt) < 1e-15:
        return K - (B - 1) / max(n, 1)

    discount_factor_n = (1 + R_cmt) ** (-n)
    annuity_factor = (1 - discount_factor_n) / R_cmt

    if abs(annuity_factor) < 1e-15:
        return R_cmt + (K - R_cmt)

    return R_cmt + (K - R_cmt) - R_cmt / annuity_factor * (B - 1)


def bond_dv01_from_yield(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """Dollar value of a basis point from yield."""
    h = 0.00005
    return (bond_price_from_yield(coupon_rate, accrual_factors, y - h)
            - bond_price_from_yield(coupon_rate, accrual_factors, y + h))
