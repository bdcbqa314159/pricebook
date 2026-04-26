"""Constant Maturity Treasury (CMT) convexity correction.

Credit-aware CMT pricing with three payoff variants (A, B, C) under
the linear risky swap-rate model with lognormal CMT dynamics.

* :func:`cmt_cc_ab` — Eq (34): CC for variants A and B.
* :func:`cmt_cc_c` — Eq (35): CC for variant C (fix-or-pre-default).
* :func:`cmt_cc_no_default` — Eq (37): Pelsser/Hagan limit (gamma=0).

References:
    Pucci, M. (2014). Constant Maturity Treasury Convexity Correction.
    IJTAF 17(8). SSRN 3387961.
    Pelsser, A. (2001). Mathematical Foundation of Convexity Correction.
    Hagan, P.S. (2003). Convexity Conundrums. Wilmott Magazine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.cms import (
    cra_discount,
    risky_annuity,
    risky_swap_rate,
    linear_swap_rate_calibrate,
)


@dataclass
class CMTConvexityResult:
    """CMT convexity correction result."""
    cc_A: float       # variant A (survival to payment)
    cc_B: float       # variant B (survival to fix)
    cc_C: float       # variant C (fix-or-pre-default)
    R_cmt_0: float
    prefactor: float   # 1 - alpha * Â_0 / (chi_0 * D̂_{0,Tp})

    @property
    def price(self) -> float:
        return self.cc_A

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.cc_A, "cc_A": self.cc_A, "cc_B": self.cc_B,
            "cc_C": self.cc_C, "R_cmt_0": self.R_cmt_0, "prefactor": self.prefactor,
        }


def cmt_cc_ab(
    sigma: float,
    Ts: float,
    alpha: float,
    risky_ann_0: float,
    chi_0: float,
    cra_df_Tp: float,
) -> float:
    """CMT convexity correction for variants A and B (Pucci Eq 34).

    CC^(A) = CC^(B) = (exp(sigma^2 Ts) - 1) * (1 - alpha Â_0 / (chi_0 D̂_{0,Tp}))
    """
    prefactor = 1.0 - alpha * risky_ann_0 / (chi_0 * cra_df_Tp)
    return (math.exp(sigma**2 * Ts) - 1) * prefactor


def cmt_cc_c(
    sigma: float,
    gamma: float,
    Ts: float,
    alpha: float,
    risky_ann_0: float,
    chi_0: float,
    cra_df_Tp: float,
    cra_df_Ts: float,
    rf_df_Ts: float,
) -> float:
    """CMT convexity correction for variant C (Pucci Eq 35).

    CC^(C) = (1 + CC^(B)) * D̂_{0,Ts}/D_{0,Ts} - 1
             + alpha*Â_0/(chi_0*D̂_{0,Tp}) * (1 - e^{-gamma Ts})
             + (1 - alpha*Â_0/(chi_0*D̂_{0,Tp})) * (e^{(sigma^2-gamma)Ts}-1)/(sigma^2-gamma) * gamma

    Handles the sigma^2 ≈ gamma singularity via Taylor expansion.
    """
    cc_b = cmt_cc_ab(sigma, Ts, alpha, risky_ann_0, chi_0, cra_df_Tp)
    prefactor = 1.0 - alpha * risky_ann_0 / (chi_0 * cra_df_Tp)
    ratio = cra_df_Ts / rf_df_Ts if abs(rf_df_Ts) > 1e-15 else 1.0

    term1 = (1 + cc_b) * ratio - 1
    term2 = (alpha * risky_ann_0 / (chi_0 * cra_df_Tp)) * (1 - math.exp(-gamma * Ts))

    # Handle sigma^2 ≈ gamma singularity
    diff = sigma**2 - gamma
    if abs(diff) < 1e-10:
        # Taylor: (e^{diff*Ts} - 1) / diff ≈ Ts + 0.5*diff*Ts^2
        integral = Ts + 0.5 * diff * Ts**2
    else:
        integral = (math.exp(diff * Ts) - 1) / diff

    term3 = prefactor * integral * gamma

    return term1 + term2 + term3


def cmt_cc_no_default(
    sigma: float,
    Ts: float,
    alpha: float,
    annuity_0: float,
    df_Tp: float,
) -> float:
    """No-default limit: Pelsser/Hagan CMS formula (Pucci Eq 37).

    CC = (exp(sigma^2 Ts) - 1) * (1 - alpha * A_0 / D_{0,Tp})

    This is the cleanest regression test: gamma=0 => all three CCs = this.
    """
    prefactor = 1.0 - alpha * annuity_0 / df_Tp
    return (math.exp(sigma**2 * Ts) - 1) * prefactor


def cmt_convexity_corrections(
    R_cmt_0: float,
    sigma: float,
    gamma: float,
    Ts: float,
    year_fractions: list[float],
    rf_discount_factors: list[float],
    rf_df_Ts: float,
    rf_df_Tp: float,
) -> CMTConvexityResult:
    """Compute all three CMT convexity corrections (Pucci 2014).

    Builds CRA quantities from risk-free curve + flat hazard rate.

    Args:
        R_cmt_0: forward CMT rate.
        sigma: lognormal CMT vol.
        gamma: constant hazard rate (intensity).
        Ts: fixing date (years).
        year_fractions: y_i for each coupon period.
        rf_discount_factors: D_{0,T_i} for each T_i.
        rf_df_Ts: D_{0,Ts}.
        rf_df_Tp: D_{0,Tp}.
    """
    n = len(year_fractions)

    # CRA discount factors: D̂ = D * e^{-gamma*T_i} (under DI, Γ_0=0, Γ_T=gamma*T)
    # Actually: D̂_{0,T} = D_{0,T} * e^{Γ_0 - Γ_T} = D_{0,T} * e^{-gamma*T}
    # But we need the times T_i. Infer from DFs: T_i such that D = e^{-r*T_i}
    # Simpler: use the fact that CRA df = risk-free df * survival(T_i)
    # Under flat hazard: survival(T) = e^{-gamma*T}
    # We need T_i. Since we have year_fractions (period lengths) and Ts (fixing),
    # T_i = Ts + cumulative year_fractions
    times = []
    t_cum = Ts
    for yi in year_fractions:
        t_cum += yi
        times.append(t_cum)

    cra_dfs = [d * math.exp(-gamma * t) for d, t in zip(rf_discount_factors, times)]
    cra_df_Ts = rf_df_Ts * math.exp(-gamma * Ts)
    Tp_time = Ts + year_fractions[0]  # typically T_p = T_1 = T_s + y_1
    cra_df_Tp = rf_df_Tp * math.exp(-gamma * Tp_time)

    # Risky annuity
    risky_ann = sum(y * d for y, d in zip(year_fractions, cra_dfs))

    # chi_0 = e^{Γ_Ts - Γ_0} = e^{gamma * Ts}
    chi_0 = math.exp(gamma * Ts)

    # alpha
    alpha = 1.0 / sum(year_fractions) if sum(year_fractions) > 0 else 0.0

    # CCs
    cc_ab = cmt_cc_ab(sigma, Ts, alpha, risky_ann, chi_0, cra_df_Tp)
    cc_c = cmt_cc_c(sigma, gamma, Ts, alpha, risky_ann, chi_0,
                     cra_df_Tp, cra_df_Ts, rf_df_Ts)
    prefactor = 1.0 - alpha * risky_ann / (chi_0 * cra_df_Tp)

    return CMTConvexityResult(
        cc_A=cc_ab,
        cc_B=cc_ab,
        cc_C=cc_c,
        R_cmt_0=R_cmt_0,
        prefactor=prefactor,
    )


# ---- Instrument class ----

class CMTInstrument:
    """CMT-let instrument for Trade/Portfolio integration.

        cmt = CMTInstrument(fixing_date, payment_date, bond_tenor=10,
                            sigma=0.20, credit_curve_name="UST")
        result = cmt.price(discount_curve, hazard_rate=0.01)
        portfolio.add(Trade(cmt))
    """

    def __init__(
        self,
        fixing_date,
        payment_date,
        bond_tenor: int = 10,
        notional: float = 1_000_000.0,
        sigma: float = 0.20,
        hazard_rate: float = 0.0,
        credit_curve_name: str = "default",
        frequency: int = 1,
    ):
        self.fixing_date = fixing_date
        self.payment_date = payment_date
        self.bond_tenor = bond_tenor
        self.notional = notional
        self.sigma = sigma
        self.hazard_rate = hazard_rate
        self.credit_curve_name = credit_curve_name
        self.frequency = frequency

    def price(self, curve, hazard_rate: float | None = None) -> CMTConvexityResult:
        """Price the CMT-let. Uses stored hazard_rate if not overridden."""
        from pricebook.day_count import year_fraction, DayCountConvention
        from datetime import timedelta

        gamma = hazard_rate if hazard_rate is not None else self.hazard_rate
        Ts = year_fraction(curve.reference_date, self.fixing_date,
                           DayCountConvention.ACT_365_FIXED)

        n = self.bond_tenor * self.frequency
        dt_period = 1.0 / self.frequency
        yfs = [dt_period] * n

        # Schedule from fixing date forward
        dates = [self.fixing_date + timedelta(days=int(dt_period * 365 * (i + 1)))
                 for i in range(n)]
        rf_dfs = [curve.df(d) for d in dates]
        rf_df_Ts = curve.df(self.fixing_date)
        rf_df_Tp = curve.df(self.payment_date)

        # CMT rate
        risky_ann_val = risky_annuity(yfs,
            [d * math.exp(-gamma * (Ts + dt_period * (i + 1))) for i, d in enumerate(rf_dfs)])
        cra_Ts = rf_df_Ts * math.exp(-gamma * Ts)
        cra_Tn = rf_dfs[-1] * math.exp(-gamma * (Ts + self.bond_tenor))
        R_cmt = risky_swap_rate(cra_Ts, cra_Tn, risky_ann_val)

        return cmt_convexity_corrections(
            R_cmt, self.sigma, gamma, Ts, yfs, rf_dfs, rf_df_Ts, rf_df_Tp)

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv()."""
        curve = ctx.discount_curve
        gamma = self.hazard_rate
        try:
            surv = ctx.get_credit_curve(self.credit_curve_name)
            # Extract flat hazard from survival curve at fixing date
            gamma = surv.hazard_rate(self.fixing_date)
        except KeyError:
            pass

        result = self.price(curve, hazard_rate=gamma)
        df_tp = curve.df(self.payment_date)
        return self.notional * df_tp * result.R_cmt_0 * (1 + result.cc_A)
