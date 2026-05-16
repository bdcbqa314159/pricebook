"""Credit-risk-adjusted building blocks and convexity model utilities.

Shared across CMASW (Pucci 2012a), CMT (Pucci 2014), and future
credit-aware constant maturity products.

    from pricebook.credit_adjustment import (
        cra_discount, risky_annuity, risky_swap_rate,
        linear_swap_rate_calibrate, displaced_lognormal_cross_moment,
    )

References:
    Pucci, M. (2014). CMT Convexity Correction. IJTAF 17(8), Eq 7-12, 21.
    Pucci, M. (2012a). CMASW Convexity Correction. SSRN 1961545, Eq 7, 13.
    Hagan, P.S. (2003). Convexity Conundrums. Wilmott Magazine.
"""

from __future__ import annotations

import math


# ---- CRA discount and risky annuity (Pucci 2014, Eq 7-10) ----

def cra_discount(risk_free_df: float, gamma_t: float, gamma_T: float) -> float:
    """Credit-risk-adjusted discount D̂_tT = D_tT × e^{Γ_t - Γ_T} (Pucci Eq 7).

    Under default independence (DI), the CRA discount is the risk-free
    discount scaled by the survival ratio.
    """
    return risk_free_df * math.exp(gamma_t - gamma_T)


def risky_annuity(
    year_fractions: list[float],
    cra_discount_factors: list[float],
) -> float:
    """CRA risky annuity Â = Σ y_i D̂_{0,T_i} (Pucci Eq 10)."""
    return sum(y * d for y, d in zip(year_fractions, cra_discount_factors))


def risky_swap_rate(
    cra_df_start: float,
    cra_df_end: float,
    risky_ann: float,
) -> float:
    """Recoveryless risky swap rate R̂^swp = (D̂_Ts - D̂_Tn) / Â (Pucci Eq 11)."""
    if abs(risky_ann) < 1e-15:
        return 0.0
    return (cra_df_start - cra_df_end) / risky_ann


# ---- Linear swap-rate model (Hagan 2003 / Pucci 2012a, 2014) ----

def linear_swap_rate_calibrate(
    year_fractions: list[float],
    discount_factors: list[float],
    annuity: float,
    forward_swap_rate: float,
    chi: float = 1.0,
) -> tuple[float, list[float]]:
    """Calibrate the linear swap-rate model G_U(x) = alpha + beta_U * x.

    Risk-free version (Pucci 2012a, Eq 7): chi = 1.
    Risky version (Pucci 2014, Eq 21): chi = e^{Γ_Ts - Γ_0}.

    alpha = 1 / sum(y_i)
    beta_U = (chi * D_0U / A_0 - alpha) / R_0

    Args:
        chi: e^{Γ_Ts - Γ_0}, = 1 for risk-free, > 1 for risky.
    """
    sum_yi = sum(year_fractions)
    alpha = 1.0 / sum_yi if sum_yi > 0 else 0.0

    betas = []
    for df_i in discount_factors:
        if abs(forward_swap_rate) < 1e-15:
            betas.append(0.0)
        else:
            betas.append((chi * df_i / annuity - alpha) / forward_swap_rate)

    return alpha, betas


# ---- Displaced lognormal cross-moment (Pucci 2012a, Eq 13) ----

def displaced_lognormal_cross_moment(
    R_swp_0: float,
    R_asw_0: float,
    a_swp: float,
    a_asw: float,
    sigma_swp: float,
    sigma_asw: float,
    rho: float,
    T: float,
) -> float:
    """Cross-moment E^A[R^swp * R^asw] under displaced lognormal (Pucci Eq 13).

    E[R^swp R^asw] = (R^swp_0 + a_swp)(R^asw_0 + a_asw) exp(sigma_swp sigma_asw rho T)
                     - a_swp(R^asw_0 + a_asw) - a_asw(R^swp_0 + a_swp) + a_swp a_asw

    With a_swp = a_asw = 0 (lognormal):
        E[R^swp R^asw] = R^swp_0 R^asw_0 exp(sigma_swp sigma_asw rho T)
    """
    X0 = R_swp_0 + a_swp
    Y0 = R_asw_0 + a_asw
    exp_term = math.exp(sigma_swp * sigma_asw * rho * T)

    return (X0 * Y0 * exp_term
            - a_swp * Y0
            - a_asw * X0
            + a_swp * a_asw)
