"""Treasury Lock (T-Lock) pricing, hedging, and roll P&L.

Thin layer over :mod:`pricebook.bond` (yield-based pricing) and
:mod:`pricebook.bond_forward` (repo-funded forward). T-Lock-specific
payoff, booking, greeks, overhedge bound, and roll P&L.

References:
    Pucci, M. (2019). Hedging the Treasury Lock. SSRN 3386521.

Functions:
    :func:`tlock_payoff` — Eq (1): a * N * RiskFactor * (IRR - L).
    :func:`tlock_booking_value` — forward-contract proxy.
    :func:`tlock_delta` — Eq (14): D_P[g].
    :func:`tlock_gamma` — Eq (16-17): D^2_P[g].
    :func:`gamma_sign_threshold` — Eq (18).
    :func:`overhedge_bound` — Eq (10-11).
    :func:`roll_pnl` — Eq (31): full closed-form.
    :func:`roll_pnl_first_order` — Eq (33).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.bond import (
    bond_price_from_yield,
    bond_price_from_yield_stub,
    bond_price_continuous,
    bond_yield_derivatives,
    bond_irr,
    bond_risk_factor,
    bond_dv01_from_yield,
)
from pricebook.bond_forward import forward_price_repo, forward_price_haircut


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
    """
    rf = bond_risk_factor(coupon_rate, accrual_factors, irr_te)
    return direction * notional * rf * (irr_te - locked_yield)


# ---- T-Lock booking value ----

@dataclass
class TLockResult:
    """T-Lock booking result."""
    value: float
    forward_price: float
    strike_price: float
    risk_factor: float
    locked_yield: float
    direction: int

    @property
    def price(self) -> float:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.value, "forward_price": self.forward_price,
            "strike_price": self.strike_price, "risk_factor": self.risk_factor,
        }


def tlock_booking_value(
    locked_yield: float,
    forward_price: float,
    coupon_rate: float,
    accrual_factors_from_expiry: list[float],
    discount_factor: float,
    notional: float = 1.0,
    direction: int = 1,
) -> TLockResult:
    """T-Lock booking value as forward contract (Pucci Eq 8).

    v = a * D_{t,te} * (K - ForwardPrice), K = P_{te}(L).
    """
    K = bond_price_from_yield(coupon_rate, accrual_factors_from_expiry, locked_yield)
    rf = bond_risk_factor(coupon_rate, accrual_factors_from_expiry, locked_yield)
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

    Delta = D_P[g] = D_y[g] / D_y[P]. Negative for long T-Lock near L.
    """
    D1, D2, _ = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity, y)
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

    At y = L: Gamma = -a * D2 / D1^2. Negative for long T-Lock.
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity, y)
    if abs(D1) < 1e-30:
        return 0.0
    Dy_g = -direction * (D2 * (y - locked_yield) + D1)
    D2y_g = -direction * (D3 * (y - locked_yield) + 2 * D2)
    return D2y_g / D1**2 - Dy_g * D2 / D1**3


def gamma_sign_threshold(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    locked_yield: float,
) -> float:
    """Yield threshold above which gamma flips sign (Pucci Eq 18).

    Returns inf if gamma never flips (common case — always negative).
    """
    D1, D2, D3 = bond_yield_derivatives(
        coupon_rate, accrual_factors, times_to_coupon, time_to_maturity,
        locked_yield)
    denom = D2**2 - D1 * D3
    if denom <= 0:
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

    |R1| <= 0.5 * M * (y - L)^2, M <= T^2 + c * sum alpha_i * tau_i^2.
    """
    M = time_to_maturity**2
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
    """
    def annuity_cont(y):
        return sum(alpha * math.exp(-tau * y)
                   for alpha, tau in zip(accrual_factors, times_to_coupon))

    dc = coupon_new - coupon_old
    A_L = annuity_cont(locked_yield)
    A_Rhat = annuity_cont(irr_new)
    P_R = bond_price_continuous(
        coupon_old, accrual_factors, times_to_coupon, time_to_maturity, irr_old)
    P_Rhat = bond_price_continuous(
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
    """First-order roll P&L approximation (Pucci Eq 33)."""
    dc = coupon_new - coupon_old
    dR = irr_new - irr_old
    T = time_to_maturity
    weighted_tau = sum(alpha * tau for alpha, tau in zip(accrual_factors, times_to_coupon))
    return dc * (irr_new - locked_yield) * weighted_tau + dR * (T + coupon_old * weighted_tau)
