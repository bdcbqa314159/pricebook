"""Treasury Lock (T-Lock) pricing, hedging, and roll P&L.

Implements Pucci (2019), "Hedging the Treasury Lock", Banca IMI.
SSRN working paper no. 3386521.

* :func:`bond_price_simply_compounded` — Eq (2): full T-Lock prescription.
* :func:`bond_price_stub` — Eq (3): within a coupon period.
* :func:`bond_price_continuous` — Eq (4): Hull-form for greek analysis.
* :func:`bond_yield_derivatives` — Eq (5): D_y^k[P] for k=1,2,3.
* :func:`irr` — internal rate of return (Newton + bisect fallback).
* :func:`risk_factor` — RiskFactor = -D_y[P] = -10000 * DV01.
* :func:`tlock_payoff` — Eq (1): a * N * RiskFactor * (IRR - L).
* :func:`forward_price_repo` — Eq (21): zero-haircut forward.
* :func:`forward_price_haircut` — Eq (24): haircut + funding blend.
* :func:`tlock_booking_value` — v = a * D * (K - ForwardPrice).
* :func:`tlock_delta` — Eq (14): D_P[g].
* :func:`tlock_gamma` — Eq (16-17): D^2_P[g].
* :func:`gamma_sign_threshold` — Eq (18): yield above which gamma flips.
* :func:`overhedge_bound` — Eq (10-11): |R1(y)| <= 0.5*M*(y-L)^2.
* :func:`roll_pnl` — Eq (31): full closed-form roll P&L.
* :func:`roll_pnl_first_order` — Eq (33): first-order approximation.

References:
    Pucci, M. (2019). Hedging the Treasury Lock. SSRN 3386521.
    Hull, J. Options, Futures, and Other Derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---- Bond price from yield ----

def bond_price_simply_compounded(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """Bond price using simply-compounded discounting (Pucci Eq 2).

    P(y) = prod_i 1/(1+alpha_i*y) + c * sum_i alpha_i * prod_{j<=i} 1/(1+alpha_j*y)

    Args:
        coupon_rate: annual coupon rate (e.g. 0.03 for 3%).
        accrual_factors: alpha_1, ..., alpha_n (year fractions per period).
        y: yield to maturity.
    """
    n = len(accrual_factors)
    if n == 0:
        return 1.0

    # Cumulative discount factors
    cum_df = 1.0
    redemption = 1.0
    coupon_pv = 0.0

    for i in range(n):
        cum_df /= (1 + accrual_factors[i] * y)
        coupon_pv += coupon_rate * accrual_factors[i] * cum_df

    redemption = cum_df
    return redemption + coupon_pv


def bond_price_stub(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
    stub_fraction: float,
) -> float:
    """Bond price within a coupon period (Pucci Eq 3).

    Discounts the first period by a fractional power:
    P(y) = (1/(1+alpha_1*y))^stub * [rest of bond from period 2 onward]

    Args:
        stub_fraction: (t1 - t) / (t1 - t0), the fraction of the first
            period remaining.
    """
    if len(accrual_factors) < 1:
        return 1.0

    # First period stub discount
    first_df = (1 / (1 + accrual_factors[0] * y)) ** stub_fraction

    # At t1 (ex-stub), bondholder receives: first coupon + remaining bond
    remaining_alphas = accrual_factors[1:]
    if remaining_alphas:
        bond_from_2 = bond_price_simply_compounded(coupon_rate, remaining_alphas, y)
    else:
        bond_from_2 = 1.0

    # Include first coupon at t1
    value_at_t1 = coupon_rate * accrual_factors[0] + bond_from_2

    return first_df * value_at_t1


def bond_price_continuous(
    coupon_rate: float,
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> float:
    """Bond price with continuous compounding (Pucci Eq 4, Hull-form).

    P(y) = exp(-T*y) + c * sum alpha_i * exp(-(t_i - t)*y)

    Used for analytical greek derivations.

    Args:
        times_to_coupon: (t_i - t) for each remaining coupon.
        time_to_maturity: (t_n - t).
        y: continuously-compounded yield.
    """
    redemption = math.exp(-time_to_maturity * y)
    annuity = sum(
        coupon_rate * alpha * math.exp(-tau * y)
        for alpha, tau in zip([0.5] * len(times_to_coupon), times_to_coupon)
    )
    # Actually use proper accrual factors — assume semi-annual (0.5) by default
    # Re-derive: P(y) = e^{-T*y} + c * sum_i alpha_i * e^{-(t_i-t)*y}
    # For semi-annual, alpha_i = 0.5 and t_i - t = times_to_coupon[i]
    return redemption + annuity


def bond_price_continuous_general(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> float:
    """Bond price continuous compounding, general schedule (Pucci Eq 4).

    P(y) = e^{-(tn-t)*y} + c * sum_i alpha_i * e^{-(ti-t)*y}
    """
    redemption = math.exp(-time_to_maturity * y)
    annuity = sum(
        coupon_rate * alpha * math.exp(-tau * y)
        for alpha, tau in zip(accrual_factors, times_to_coupon)
    )
    return redemption + annuity


# ---- Yield derivatives (Pucci Eq 5) ----

def bond_yield_derivatives(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> tuple[float, float, float]:
    """First, second, and third yield derivatives of bond price (Pucci Eq 5).

    D_y^k[P] = (t-tn)^k * e^{-(tn-t)*y} + c * sum alpha_i * (t-ti)^k * e^{-(ti-t)*y}

    Returns (D1, D2, D3) where D1 = D_y[P], D2 = D_y^2[P], D3 = D_y^3[P].
    Note: D1 < 0, D2 > 0, D3 < 0 (alternating signs).
    """
    T = time_to_maturity
    exp_T = math.exp(-T * y)

    D1 = (-T) * exp_T
    D2 = T**2 * exp_T
    D3 = (-T)**3 * exp_T

    for alpha, tau in zip(accrual_factors, times_to_coupon):
        q = -tau  # q_i = t - t_i <= 0
        exp_tau = math.exp(-tau * y)
        D1 += coupon_rate * alpha * q * exp_tau
        D2 += coupon_rate * alpha * q**2 * exp_tau
        D3 += coupon_rate * alpha * q**3 * exp_tau

    return D1, D2, D3


# ---- IRR solver ----

def irr(
    market_price: float,
    coupon_rate: float,
    accrual_factors: list[float],
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Internal rate of return: solve P(y) = market_price.

    Newton from y0 = coupon_rate, bisect fallback when |D_y[P]| is small.
    """
    n = len(accrual_factors)

    def price_and_deriv(y):
        p = bond_price_simply_compounded(coupon_rate, accrual_factors, y)
        # Numerical derivative
        h = 1e-6
        dp = (bond_price_simply_compounded(coupon_rate, accrual_factors, y + h)
              - bond_price_simply_compounded(coupon_rate, accrual_factors, y - h)) / (2 * h)
        return p, dp

    # Newton
    y = coupon_rate if coupon_rate > 0 else 0.05
    for _ in range(max_iter):
        p, dp = price_and_deriv(y)
        if abs(p - market_price) < tol:
            return y
        if abs(dp) < 1e-15:
            break
        y -= (p - market_price) / dp
        y = max(-0.5, min(y, 2.0))

    # Bisect fallback
    lo, hi = -0.1, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = bond_price_simply_compounded(coupon_rate, accrual_factors, mid)
        if p > market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---- Risk factor and DV01 ----

def risk_factor(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """RiskFactor = -dP/dy evaluated at y (Pucci notation).

    Equals -10000 * DV01 when DV01 is per bp.
    """
    h = 1e-6
    p_up = bond_price_simply_compounded(coupon_rate, accrual_factors, y + h)
    p_dn = bond_price_simply_compounded(coupon_rate, accrual_factors, y - h)
    return -(p_up - p_dn) / (2 * h)


def dv01(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """Dollar value of a basis point: (P(y-0.5bp) - P(y+0.5bp)) / 1."""
    h = 0.00005  # half a bp
    p_up = bond_price_simply_compounded(coupon_rate, accrual_factors, y + h)
    p_dn = bond_price_simply_compounded(coupon_rate, accrual_factors, y - h)
    return p_dn - p_up


# ---- T-Lock payoff (Pucci Eq 1) ----

def tlock_payoff(
    irr_te: float,
    locked_yield: float,
    coupon_rate: float,
    accrual_factors: list[float],
    notional: float = 1.0,
    direction: int = 1,
) -> float:
    """T-Lock payoff at expiry (Pucci Eq 1).

    Payoff = a * N * RiskFactor(IRR_te) * (IRR_te - L)

    Args:
        irr_te: IRR at expiry.
        locked_yield: L, the locked yield (strike).
        direction: +1 (long), -1 (short).
    """
    rf = risk_factor(coupon_rate, accrual_factors, irr_te)
    return direction * notional * rf * (irr_te - locked_yield)


# ---- Forward price under repo (Pucci Eq 21, 24) ----

def forward_price_repo(
    market_price: float,
    repo_rate: float,
    time_to_expiry: float,
    coupon_rate: float,
    coupon_accruals: list[float],
    coupon_times_to_expiry: list[float],
    repo_rates_coupon: list[float] | None = None,
) -> float:
    """Forward price under repo (Pucci Eq 21, zero haircut).

    ForwardPrice = P^mkt * (1 + r_repo * tau) - c * sum alpha_i * (1 + r_repo_i * tau_i)

    Args:
        market_price: current bond price.
        repo_rate: repo rate from t to t_e.
        time_to_expiry: t_e - t.
        coupon_accruals: alpha_i for coupons in (t, t_e].
        coupon_times_to_expiry: t_e - t_i for each coupon in (t, t_e].
        repo_rates_coupon: repo rate from t_i to t_e for each coupon.
            If None, uses repo_rate for all.
    """
    fwd = market_price * (1 + repo_rate * time_to_expiry)

    if repo_rates_coupon is None:
        repo_rates_coupon = [repo_rate] * len(coupon_accruals)

    for alpha, tau, r in zip(coupon_accruals, coupon_times_to_expiry, repo_rates_coupon):
        fwd -= coupon_rate * alpha * (1 + r * tau)

    return fwd


def forward_price_haircut(
    market_price: float,
    repo_rate: float,
    funding_rate: float,
    haircut: float,
    time_to_expiry: float,
    coupon_rate: float,
    coupon_amounts: list[float],
    coupon_times_to_expiry: list[float],
) -> float:
    """Forward price with haircut and unsecured funding blend (Pucci Eq 24).

    ForwardPrice = X * [(1-h)*exp(r_repo*tau) + h*exp(r_fun*tau)]
                   - sum c_i * exp((h*r_fun + (1-h)*r_repo) * tau_i)

    Args:
        haircut: h^cut in [0, 1].
        funding_rate: unsecured funding rate.
    """
    T = time_to_expiry
    blend_rate = haircut * funding_rate + (1 - haircut) * repo_rate

    fwd = market_price * (
        (1 - haircut) * math.exp(repo_rate * T)
        + haircut * math.exp(funding_rate * T)
    )

    for ci, tau_i in zip(coupon_amounts, coupon_times_to_expiry):
        fwd -= ci * math.exp(blend_rate * tau_i)

    return fwd


# ---- T-Lock booking value ----

@dataclass
class TLockResult:
    """T-Lock booking result."""
    value: float
    forward_price: float
    strike_price: float       # K = P_{te}(L)
    risk_factor: float
    locked_yield: float
    direction: int


def tlock_booking_value(
    locked_yield: float,
    forward_price: float,
    coupon_rate: float,
    accrual_factors_from_expiry: list[float],
    discount_factor: float,
    notional: float = 1.0,
    direction: int = 1,
) -> TLockResult:
    """T-Lock booking value as forward contract (Pucci Eq 8, Section 3.1).

    v = a * D_{t,te} * (K - ForwardPrice)
    K = P_{te}(L)

    Args:
        accrual_factors_from_expiry: schedule from expiry onward.
        discount_factor: D_{t,te} = discount factor to expiry.
    """
    K = bond_price_simply_compounded(coupon_rate, accrual_factors_from_expiry, locked_yield)
    rf = risk_factor(coupon_rate, accrual_factors_from_expiry, locked_yield)
    value = direction * discount_factor * notional * (K - forward_price)

    return TLockResult(value, forward_price, K, rf, locked_yield, direction)


# ---- Greeks (Pucci Section 4) ----

def tlock_delta(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
    locked_yield: float,
    direction: int = 1,
) -> float:
    """T-Lock delta in price (Pucci Eq 14).

    Delta = D_P[g] = D_y[g] / D_y[P]

    For a long T-Lock, delta < 0 near L (short the bond's price).
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity, y)

    # D_y[g] = -a * [D2 * (y - L) + D1]  (Eq 12)
    Dy_g = -direction * (D2 * (y - locked_yield) + D1)

    if abs(D1) < 1e-30:
        return 0.0

    return Dy_g / D1


def tlock_gamma(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
    locked_yield: float,
    direction: int = 1,
) -> float:
    """T-Lock gamma in price (Pucci Eq 16-17).

    At y = L: Gamma = -a * D2 / D1^2  (Eq 17).
    Negative for long T-Lock.
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity, y)

    if abs(D1) < 1e-30:
        return 0.0

    # D_y[g] = -a * [D2 * (y - L) + D1]
    Dy_g = -direction * (D2 * (y - locked_yield) + D1)
    # D_y^2[g] = -a * [D3 * (y - L) + 2*D2]
    D2y_g = -direction * (D3 * (y - locked_yield) + 2 * D2)

    # D^2_P[g] = D2y_g / D1^2 - Dy_g * D2 / D1^3  (Eq 16)
    gamma = D2y_g / D1**2 - Dy_g * D2 / D1**3
    return gamma


def gamma_sign_threshold(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    locked_yield: float,
) -> float:
    """Yield threshold above which gamma flips sign (Pucci Eq 18).

    Gamma < 0 iff y - L < D1*D3 / ((D2)^2 - D1*D3).

    When the denominator is non-positive, gamma stays negative for all
    yields — returns inf (no sign flip).

    For 10Y benchmarks, the paper reports thresholds ~11.4%.
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity,
        locked_yield)

    denom = D2**2 - D1 * D3
    if denom <= 0:
        # Gamma never flips sign — always negative for long T-Lock
        return float('inf')

    threshold_diff = D1 * D3 / denom
    if threshold_diff <= 0:
        return float('inf')

    return locked_yield + threshold_diff


# ---- Overhedge bound (Pucci Eq 10-11) ----

def overhedge_bound(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    yield_change: float,
) -> float:
    """Upper bound on overhedge error |R1(y)| (Pucci Eq 10-11).

    |R1| <= 0.5 * M * (y - L)^2
    M <= (tn-t)^2 + c * sum alpha_i * (ti-t)^2
    """
    T = time_to_maturity
    M = T**2
    for alpha, tau in zip(accrual_factors, times_to_coupon):
        M += coupon_rate * alpha * tau**2

    return 0.5 * M * yield_change**2


# ---- Roll P&L (Pucci Eq 31, 33) ----

def roll_pnl(
    coupon_old: float,
    coupon_new: float,
    irr_old: float,
    irr_new: float,
    locked_yield: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
) -> float:
    """Full closed-form roll P&L (Pucci Eq 31).

    (pi_hat - pi) / D = (c_hat - c) * [A(L) - A(R_hat)] + [P(R) - P(R_hat)]

    where A(y) = sum alpha_i * exp(-(t_i - t_e) * y).
    """
    def annuity_cont(y):
        return sum(alpha * math.exp(-tau * y)
                   for alpha, tau in zip(accrual_factors, times_to_coupon))

    def price_cont(y):
        return bond_price_continuous_general(
            0.0, accrual_factors, times_to_coupon, time_to_maturity, y)

    dc = coupon_new - coupon_old
    A_L = annuity_cont(locked_yield)
    A_Rhat = annuity_cont(irr_new)
    P_R = bond_price_continuous_general(
        coupon_old, accrual_factors, times_to_coupon, time_to_maturity, irr_old)
    P_Rhat = bond_price_continuous_general(
        coupon_old, accrual_factors, times_to_coupon, time_to_maturity, irr_new)

    return dc * (A_L - A_Rhat) + (P_R - P_Rhat)


def roll_pnl_first_order(
    coupon_old: float,
    coupon_new: float,
    irr_old: float,
    irr_new: float,
    locked_yield: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
) -> float:
    """First-order roll P&L approximation (Pucci Eq 33).

    (pi_hat - pi) / D ≈ (c_hat - c)(R_hat - L) * sum alpha_i * tau_i
                        + (R_hat - R) * [T + c * sum alpha_i * tau_i]
    """
    dc = coupon_new - coupon_old
    dR = irr_new - irr_old
    T = time_to_maturity

    weighted_tau = sum(alpha * tau for alpha, tau in zip(accrual_factors, times_to_coupon))

    term1 = dc * (irr_new - locked_yield) * weighted_tau
    term2 = dR * (T + coupon_old * weighted_tau)

    return term1 + term2
