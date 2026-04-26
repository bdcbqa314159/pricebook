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

    @property
    def price(self) -> float:
        return self.aswlet_value

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.aswlet_value, "convexity_correction": self.convexity_correction,
            "prefactor": self.prefactor, "R_asw_0": self.R_asw_0, "R_swp_0": self.R_swp_0,
        }


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


class CMASWInstrument:
    """Constant Maturity Asset Swap instrument for Trade/Portfolio.

        cmasw = CMASWInstrument(fixing_date, payment_date, swap_tenor=5,
                                bond_price=0.95, sigma_swp=0.30, sigma_asw=0.25, rho=0.5)
        result = cmasw.price(discount_curve)
    """

    def __init__(
        self,
        fixing_date,
        payment_date,
        swap_tenor: int = 5,
        bond_price: float = 1.0,
        notional: float = 1_000_000.0,
        sigma_swp: float = 0.30,
        sigma_asw: float = 0.30,
        rho: float = 0.5,
        a_swp: float = 0.0,
        a_asw: float = 0.0,
        frequency: int = 2,
    ):
        self.fixing_date = fixing_date
        self.payment_date = payment_date
        self.swap_tenor = swap_tenor
        self.bond_price = bond_price
        self.notional = notional
        self.sigma_swp = sigma_swp
        self.sigma_asw = sigma_asw
        self.rho = rho
        self.a_swp = a_swp
        self.a_asw = a_asw
        self.frequency = frequency

    def price(self, curve) -> CMASWResult:
        """Price the CMASW-let using a discount curve."""
        import math
        from pricebook.day_count import year_fraction, DayCountConvention
        from pricebook.par_asset_swap import forward_asw_spread

        T0 = year_fraction(curve.reference_date, self.fixing_date,
                           DayCountConvention.ACT_365_FIXED)
        n = self.swap_tenor * self.frequency
        dt_period = 1.0 / self.frequency
        yfs = [dt_period] * n

        # Build schedule dates and DFs from fixing date forward
        from datetime import timedelta
        dates = [self.fixing_date + timedelta(days=int(dt_period * 365 * (i + 1)))
                 for i in range(n)]
        dfs = [curve.df(d) for d in dates]
        annuity = sum(y * d for y, d in zip(yfs, dfs))

        # Forward swap rate
        R_swp = (curve.df(self.fixing_date) - dfs[-1]) / annuity if annuity > 0 else 0.0

        # Forward ASW spread
        B_rf = sum(R_swp * y * d for y, d in zip(yfs, dfs)) + dfs[-1]
        R_asw = forward_asw_spread(B_rf, self.bond_price, annuity)

        payment_df = curve.df(self.payment_date)

        return cmasw_convexity_correction(
            R_asw, R_swp, annuity, payment_df, yfs, dfs,
            self.sigma_swp, self.sigma_asw, self.rho, T0,
            self.a_swp, self.a_asw)

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv()."""
        result = self.price(ctx.discount_curve)
        return self.notional * result.aswlet_value


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
