"""Constant Maturity Asset Swap (CMASW) convexity correction.

An ASW-let pays the prevailing ASW spread at a future date. The convexity
correction arises from the cross-gamma between the swap rate and the ASW
spread when hedging via the forward asset-swap package.

* :func:`cmasw_convexity_correction` — Pucci (2012a) Eq (9): general CC.
* :func:`cmasw_cc_lognormal` — Pucci (2012a) Eq (14): lognormal limit.
* :func:`cmasw_aswlet_value` — normalised ASW-let value D_{0,Tp}(R^asw + CC).

References:
    Pucci, M. (2012a). Constant Maturity Asset Swap Convexity Correction.
    SSRN 1961545. Risk Magazine, April 2012.
    Hagan, P.S. (2003). Convexity Conundrums. Wilmott Magazine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.cms import (
    linear_swap_rate_calibrate,
    displaced_lognormal_cross_moment,
)


@dataclass
class CMASWResult:
    """CMASW convexity correction result."""
    convexity_correction: float
    aswlet_value: float         # D_{0,Tp} * (R^asw_0 + CC)
    prefactor: float            # 1 - A_0 * alpha / D_{0,Tp}
    cross_moment: float         # E^A[R^swp * R^asw]
    R_asw_0: float
    R_swp_0: float


def cmasw_convexity_correction(
    R_asw_0: float,
    R_swp_0: float,
    annuity: float,
    payment_df: float,
    year_fractions: list[float],
    discount_factors: list[float],
    sigma_swp: float,
    sigma_asw: float,
    rho: float,
    T0: float,
    a_swp: float = 0.0,
    a_asw: float = 0.0,
) -> CMASWResult:
    """CMASW convexity correction (Pucci 2012a, Proposition 2, Eq 9).

    CC = (1 - A_0 * alpha / D_{0,Tp}) * (E[R^asw R^swp] / R^swp_0 - R^asw_0)

    Args:
        R_asw_0: forward ASW spread at t=0.
        R_swp_0: forward swap rate at t=0.
        annuity: A_0 = sum(y_i * D_{0,T_i}).
        payment_df: D_{0,Tp} discount factor to payment date.
        year_fractions: y_i for each coupon period.
        discount_factors: D_{0,T_i} for each T_i.
        sigma_swp: lognormal vol of swap rate.
        sigma_asw: lognormal vol of ASW spread.
        rho: instantaneous correlation.
        T0: time to fixing (observation date).
        a_swp: displacement for swap rate (0 = lognormal).
        a_asw: displacement for ASW spread (0 = lognormal).
    """
    # Calibrate linear model
    alpha, _ = linear_swap_rate_calibrate(
        year_fractions, discount_factors, annuity, R_swp_0)

    # Prefactor
    prefactor = 1.0 - annuity * alpha / payment_df

    # Cross-moment
    cross_moment = displaced_lognormal_cross_moment(
        R_swp_0, R_asw_0, a_swp, a_asw,
        sigma_swp, sigma_asw, rho, T0)

    # CC (Eq 9)
    if abs(R_swp_0) < 1e-15:
        cc = 0.0
    else:
        cc = prefactor * (cross_moment / R_swp_0 - R_asw_0)

    # ASW-let value
    aswlet = payment_df * (R_asw_0 + cc)

    return CMASWResult(cc, aswlet, prefactor, cross_moment, R_asw_0, R_swp_0)


def cmasw_cc_lognormal(
    R_asw_0: float,
    annuity: float,
    payment_df: float,
    year_fractions: list[float],
    sigma_swp: float,
    sigma_asw: float,
    rho: float,
    T0: float,
) -> float:
    """Lognormal CMASW convexity correction (Pucci Eq 14).

    CC = R^asw_0 * (1 - A_0 * alpha / D_{0,Tp}) * (exp(sigma_swp * sigma_asw * rho * T0) - 1)

    Special case of Eq (9) with a_swp = a_asw = 0.
    """
    sum_yi = sum(year_fractions)
    alpha = 1.0 / sum_yi if sum_yi > 0 else 0.0
    prefactor = 1.0 - annuity * alpha / payment_df

    return R_asw_0 * prefactor * (math.exp(sigma_swp * sigma_asw * rho * T0) - 1)
