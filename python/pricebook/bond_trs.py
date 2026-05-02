"""Bond Total Return Swap — Burgess (2024) closed-form pricing.

PV(TRS) = φ × [coupon_leg + performance_leg - funding_leg - lgd_leg]

where:
  coupon_leg    = Σ N_B × r × τ_i × Q̃P(t_i)           (Eq 5)
  performance   = Σ N_B × (B(t_{i-1}) - B(t_i)) × Q̃P(t_i)  (Eq 6)
  funding_leg   = Σ N_C × (F_j + s) × τ_j × Q̃P(t_j)  (Eq 10)
  lgd_leg       = Σ N_B × (1-RR) × ΔPD_i × P(t_i)     (Eq 12)

Q̃P(t) = Q(t) × P(t) is the risky discount factor.
P(t) is the riskfree (OIS) discount factor.
Q(t) is the survival probability.
ΔPD_i = Q(t_{i-1}) - Q(t_i) is the incremental default probability.

    from pricebook.bond_trs import bond_trs_pv, BondTRSResult, par_funding_spread

References:
    Burgess, N. (2024). Bond Total Return Swaps: Theory, Pricing & Practice.
    SSRN 5024091.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.bond_forward import forward_price_repo
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, generate_schedule
from pricebook.survival_curve import SurvivalCurve


@dataclass
class BondTRSResult:
    """Burgess Eq 13 decomposition."""
    pv: float
    coupon_pv: float         # Eq 5
    performance_pv: float    # Eq 6
    funding_pv: float        # Eq 10
    lgd_pv: float            # Eq 12
    par_spread: float        # Eq 14
    risky_annuity: float     # Eq 15

    def to_dict(self) -> dict:
        return {
            "pv": self.pv,
            "coupon_pv": self.coupon_pv,
            "performance_pv": self.performance_pv,
            "funding_pv": self.funding_pv,
            "lgd_pv": self.lgd_pv,
            "par_spread": self.par_spread,
            "risky_annuity": self.risky_annuity,
        }


def _risky_df(
    d: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve | None,
) -> float:
    """Q̃P(t) = Q(t) × P(t). If no survival curve, Q(t) = 1."""
    p = discount_curve.df(d)
    q = survival_curve.survival(d) if survival_curve else 1.0
    return q * p


def bond_trs_pv(
    bond: FixedRateBond,
    trs_start: date,
    trs_end: date,
    funding_spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve | None = None,
    recovery: float = 0.4,
    bond_notional: float | None = None,
    cash_notional: float | None = None,
    repo_rate: float | None = None,
    repo_curve: DiscountCurve | None = None,
    funding_frequency: Frequency = Frequency.QUARTERLY,
    phi: float = 1.0,
    notional_mode: str = "constant_notional",
) -> BondTRSResult:
    """Bond TRS PV via Burgess (2024) Eq 13.

    PV(TRS) = φ × [coupon + performance - funding - LGD]

    Args:
        bond: reference bond.
        trs_start: TRS effective date.
        trs_end: TRS maturity.
        funding_spread: s in the funding leg (F_j + s).
        discount_curve: OIS discount curve P(t).
        survival_curve: issuer survival curve Q(t). None = no default risk.
        recovery: recovery rate RR.
        bond_notional: N_B = Units × FaceValue. Default = bond.face_value.
        cash_notional: N_C = funding notional. Default = N_B × B(t₀).
        repo_rate: flat bond repo rate for forward pricing. None = OIS forward.
        repo_curve: repo discount curve (overrides repo_rate if provided).
            Forward repo rates extracted per period.
        funding_frequency: floating leg frequency.
        phi: +1 if receiving bond return, -1 if paying.
        notional_mode: "constant_notional" (fixed N_C, units float) or
            "constant_units" (fixed units, N_C adjusts to bond price).
            Both give same PV at inception (Burgess Remark 1).
    """
    if not 0.0 <= recovery <= 1.0:
        raise ValueError(f"recovery must be in [0, 1], got {recovery}")

    N_B = bond_notional or bond.face_value
    B_0 = bond.dirty_price(discount_curve) / 100.0  # per unit face
    N_C = cash_notional or (N_B * B_0)

    # Schedules — coupon dates within TRS period
    bond_schedule = [cf.payment_date for cf in bond.coupon_leg.cashflows
                     if trs_start < cf.payment_date <= trs_end]
    # Add trs_end if not already there
    if not bond_schedule or bond_schedule[-1] != trs_end:
        bond_schedule.append(trs_end)

    funding_schedule = generate_schedule(trs_start, trs_end, funding_frequency)

    # Repo rate for forward bond pricing
    def _repo_forward(d1: date, d2: date) -> float:
        """Forward repo rate for period [d1, d2]."""
        if repo_curve is not None:
            return repo_curve.forward_rate(d1, d2)
        if repo_rate is not None:
            return repo_rate
        T = year_fraction(trs_start, trs_end, DayCountConvention.ACT_365_FIXED)
        return -math.log(discount_curve.df(trs_end)) / T if T > 0 else 0.0

    # ---- (i) Coupon leg: Eq 5 ----
    coupon_pv = 0.0
    for cf in bond.coupon_leg.cashflows:
        if trs_start < cf.payment_date <= trs_end:
            rpv = _risky_df(cf.payment_date, discount_curve, survival_curve)
            coupon_pv += N_B / bond.face_value * cf.amount * rpv

    # ---- (ii) Performance leg: Eq 6 ----
    # Project forward bond prices via repo (Eq 8-9)
    performance_pv = 0.0
    perf_dates = [trs_start] + bond_schedule
    B_prev = B_0

    for i in range(1, len(perf_dates)):
        d_prev = perf_dates[i - 1]
        d_i = perf_dates[i]
        tau = year_fraction(d_prev, d_i, DayCountConvention.ACT_365_FIXED)

        # Forward repo rate for this period (from repo curve or flat)
        r_repo_period = _repo_forward(d_prev, d_i)

        # Forward price via repo (Burgess Eq 8-9):
        # B(t_i) = B(t_{i-1}) × (1 + r_repo × τ) - coupons × (1 + r_repo × τ₂)
        coupon_income = 0.0
        for cf in bond.coupon_leg.cashflows:
            if d_prev < cf.payment_date <= d_i:
                # Reinvest coupon at repo from coupon date to period end
                tau2 = year_fraction(cf.payment_date, d_i, DayCountConvention.ACT_365_FIXED)
                coupon_income += cf.amount / bond.face_value * (1 + r_repo_period * tau2)

        B_i = B_prev * (1 + r_repo_period * tau) - coupon_income

        rpv = _risky_df(d_i, discount_curve, survival_curve)
        performance_pv += N_B * (B_prev - B_i) * rpv
        B_prev = B_i

    # ---- (iii) Funding leg: Eq 10 ----
    # Constant-units: N_C adjusts each period to Units × FaceValue × B(t)
    # Constant-notional: N_C is fixed throughout
    funding_pv = 0.0
    risky_annuity = 0.0

    # Build forward bond price at each funding date for constant-units
    fwd_bond_prices: dict[date, float] = {trs_start: B_0}
    if notional_mode == "constant_units":
        b_cur = B_0
        for i in range(1, len(perf_dates)):
            d_prev = perf_dates[i - 1]
            d_i = perf_dates[i]
            tau_p = year_fraction(d_prev, d_i, DayCountConvention.ACT_365_FIXED)
            r_p = _repo_forward(d_prev, d_i)
            cpn = 0.0
            for cf in bond.coupon_leg.cashflows:
                if d_prev < cf.payment_date <= d_i:
                    tau2 = year_fraction(cf.payment_date, d_i, DayCountConvention.ACT_365_FIXED)
                    cpn += cf.amount / bond.face_value * (1 + r_p * tau2)
            b_cur = b_cur * (1 + r_p * tau_p) - cpn
            fwd_bond_prices[d_i] = b_cur

    units = N_B / bond.face_value  # fixed units count

    for i in range(1, len(funding_schedule)):
        d_prev = funding_schedule[i - 1]
        d_i = funding_schedule[i]
        tau_j = year_fraction(d_prev, d_i, DayCountConvention.ACT_360)
        rpv = _risky_df(d_i, discount_curve, survival_curve)

        # Funding notional for this period
        if notional_mode == "constant_units":
            # N_C = Units × FaceValue × B(t_{j-1})
            # Find closest forward price at or before d_prev
            b_at_start = B_0
            for d_key in sorted(fwd_bond_prices):
                if d_key <= d_prev:
                    b_at_start = fwd_bond_prices[d_key]
            nc_period = units * bond.face_value * b_at_start
        else:
            nc_period = N_C

        fwd = discount_curve.forward_rate(d_prev, d_i)
        funding_pv += nc_period * (fwd + funding_spread) * tau_j * rpv
        risky_annuity += nc_period * tau_j * rpv

    # ---- (iv) LGD leg: Eq 12 ----
    lgd_pv = 0.0
    if survival_curve is not None:
        lgd_dates = [trs_start] + bond_schedule
        for i in range(1, len(lgd_dates)):
            d_prev = lgd_dates[i - 1]
            d_i = lgd_dates[i]
            q_prev = survival_curve.survival(d_prev)
            q_i = survival_curve.survival(d_i)
            default_prob = q_prev - q_i
            # Note: P(t_i), NOT Q̃P(t_i) — cash flow conditional on default
            p_i = discount_curve.df(d_i)
            lgd_pv += N_B * (1 - recovery) * default_prob * p_i

    # ---- Assembly: Eq 13 ----
    pv = phi * (coupon_pv + performance_pv - funding_pv - lgd_pv)

    # ---- Par spread: Eq 14 ----
    if risky_annuity > 1e-10:
        # PV at s=0: recompute funding without spread (same notional logic)
        funding_pv_no_spread = funding_pv - funding_spread * risky_annuity
        pv_no_spread = phi * (coupon_pv + performance_pv - funding_pv_no_spread - lgd_pv)
        par_spread = pv_no_spread / risky_annuity
    else:
        par_spread = 0.0

    return BondTRSResult(
        pv=pv, coupon_pv=coupon_pv, performance_pv=performance_pv,
        funding_pv=funding_pv, lgd_pv=lgd_pv,
        par_spread=par_spread, risky_annuity=risky_annuity,
    )


def par_funding_spread(
    bond: FixedRateBond,
    trs_start: date,
    trs_end: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve | None = None,
    recovery: float = 0.4,
    bond_notional: float | None = None,
    cash_notional: float | None = None,
    repo_rate: float | None = None,
    repo_curve: DiscountCurve | None = None,
    funding_frequency: Frequency = Frequency.QUARTERLY,
    phi: float = 1.0,
) -> float:
    """Par funding spread: s such that PV(TRS) = 0 at inception (Burgess Eq 14)."""
    result = bond_trs_pv(
        bond, trs_start, trs_end,
        funding_spread=0.0,
        discount_curve=discount_curve,
        survival_curve=survival_curve,
        recovery=recovery,
        bond_notional=bond_notional,
        cash_notional=cash_notional,
        repo_rate=repo_rate,
        repo_curve=repo_curve,
        funding_frequency=funding_frequency,
        phi=phi,
    )
    return result.par_spread
