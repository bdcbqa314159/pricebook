"""Total Return Swap pricing with repo financing and FVA.

Implements Lou (2018), "Pricing Total Return Swap", SSRN 3217420.

Equity TRS:
* :func:`trs_equity_full_csa` — Eq (7): full-CSA closed form with repo drift.
* :func:`trs_fva` — Eq (8): hedge financing cost (repo vs OIS spread).
* :func:`trs_precrisis` — Eq (2): pre-crisis valuation (rs = r).
* :func:`trs_repo_style_symmetric` — Eq (11): repo-style margining, symmetric funding.

Bond TRS:
* :func:`trs_bond_full_csa` — Eq (25): bond TRS with default risk.
* :func:`trs_bond_forward` — Eq (27-28): bond forward under haircut.

Multi-period:
* :func:`trs_multi_period` — Eq (31): multi-period with resets.

References:
    Lou, W. (2018). Pricing Total Return Swap. SSRN 3217420.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.bond_forward import repo_financing_factor, blended_repo_rate


# ---- Result dataclasses ----

@dataclass
class TRSResult:
    """Equity TRS pricing result (Lou framework)."""
    value: float            # V(t) — NPV to the payer (bank)
    funding_leg: float      # M0 rf T D(t,T) — funding leg PV
    asset_leg: float        # St × repo factor — asset leg PV
    fva: float              # hedge financing cost adjustment
    repo_factor: float      # exp(∫(rs-r)du) — repo-vs-OIS factor

    @property
    def price(self) -> float:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.value, "funding_leg": self.funding_leg,
            "asset_leg": self.asset_leg, "fva": self.fva,
            "repo_factor": self.repo_factor,
        }


@dataclass
class TRSBondResult:
    """Bond TRS pricing result."""
    value: float
    funding_leg: float
    asset_leg: float
    fva: float
    bond_price: float
    dpv: float              # default PV component

    @property
    def price(self) -> float:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.value, "funding_leg": self.funding_leg,
            "asset_leg": self.asset_leg, "fva": self.fva,
            "bond_price": self.bond_price,
        }


# ---- Equity TRS, full CSA (Section 2) ----

def trs_precrisis(
    S_t: float,
    S_0: float,
    r_f: float,
    T: float,
    t_0: float,
    D_tT: float,
    M_0: float | None = None,
) -> float:
    """Pre-crisis TRS valuation, Eq (2). Assumes rs = r.

    V(t) = (M0 rf (T - t0) + S0) D(t,T) - St

    Args:
        S_t: current spot price.
        S_0: initial spot (= M0 if not specified).
        r_f: funding rate (Libor + spread).
        T: maturity.
        t_0: trade date.
        D_tT: risk-free discount factor D(t, T).
        M_0: funding notional (defaults to S_0).
    """
    if M_0 is None:
        M_0 = S_0
    return (M_0 * r_f * (T - t_0) + S_0) * D_tT - S_t


def trs_equity_full_csa(
    S_t: float,
    S_0: float,
    r_f: float,
    T: float,
    t_0: float,
    D_tT: float,
    rs_minus_r: float = 0.0,
    t: float = 0.0,
    M_0: float | None = None,
) -> TRSResult:
    """Full-CSA equity TRS with repo financing, Eq (7).

    V(t) = (M0 rf (T-t0) + S0) D(t,T) - St exp(∫(rs-r)du)

    The asset leg is repo-adjusted: the hedge is financed at rs, not r.

    Args:
        rs_minus_r: repo spread over OIS (rs - r). 0 recovers pre-crisis.
        t: current time (for computing repo accrual from t to T).
    """
    if M_0 is None:
        M_0 = S_0

    repo_factor = repo_financing_factor(rs_minus_r, T, t)
    funding_leg = (M_0 * r_f * (T - t_0) + S_0) * D_tT
    asset_leg = S_t * repo_factor
    value = funding_leg - asset_leg
    fva = (repo_factor - 1) * S_t

    return TRSResult(value, funding_leg, asset_leg, fva, repo_factor)


def trs_fva(
    S_t: float,
    rs_minus_r: float,
    T: float,
    t: float = 0.0,
) -> float:
    """Hedge financing cost adjustment, Eq (8).

    fva = (exp(∫(rs-r)du) - 1) × St ≥ 0

    This is the PV of repo-vs-OIS spread on the hedge stock position.
    """
    return (repo_financing_factor(rs_minus_r, T, t) - 1) * S_t


# ---- Repo-style margining (Section 3) ----

def trs_repo_style_symmetric(
    S_t: float,
    S_0: float,
    r_f: float,
    T: float,
    t_0: float,
    r: float,
    r_b: float,
    rs_minus_r: float,
    t: float = 0.0,
    M_0: float | None = None,
) -> float:
    """Repo-style margined TRS, symmetric funding (rb = rc), Eq (11).

    Closed-form for one-period TRS under repo-style variation margin.

    Args:
        r: OIS rate (continuously compounded).
        r_b: bank's unsecured funding rate.
        rs_minus_r: repo spread.
    """
    if M_0 is None:
        M_0 = S_0

    rs = r + rs_minus_r
    tau = T - t

    # Discount factors
    D_rb = math.exp(-r_b * tau)
    D_rb_rs = math.exp(-(r_b - rs) * tau)

    # Eq (11) terms
    term1 = -S_t * (r_b - r - (rs - r) * D_rb_rs) / (r_b - rs) if abs(r_b - rs) > 1e-15 else \
            -S_t * (r_b - r) * tau  # limit when rb ≈ rs

    term2 = (S_0 - t_0 * M_0 * r_f) * (1 - r * (1 - D_rb) / r_b) if abs(r_b) > 1e-15 else \
            (S_0 - t_0 * M_0 * r_f)

    if abs(r_b) > 1e-15:
        term3 = M_0 * r_f * (T * D_rb * r / r_b + (t + (1 - D_rb) / r_b) * (r_b - r) / r_b)
    else:
        term3 = M_0 * r_f * T

    return term1 + term2 + term3


# ---- Bond TRS (Section 6) ----

def trs_bond_full_csa(
    B_t: float,
    B_0: float,
    r_f: float,
    T: float,
    t_0: float,
    D_tT: float,
    q_tT: float,
    coupons: list[tuple[float, float]],
    rs_bar_minus_r: float = 0.0,
    lambda_avg: float = 0.0,
    recovery: float = 0.4,
    t: float = 0.0,
    M_0: float | None = None,
) -> TRSBondResult:
    """Bond TRS under full CSA, Eq (25).

    V(t) = M0 rf T D̄(t,T) + B0 (D̄(t,T) + dpv) - Bt - fva

    Args:
        B_t: current bond dirty price.
        B_0: initial bond price (at trade date).
        D_tT: risk-free discount factor.
        q_tT: survival probability Q(t,T).
        coupons: list of (time_to_coupon, coupon_amount) for coupons in (t, T].
        rs_bar_minus_r: blended repo-vs-OIS spread (Eq 19: r̄s - r).
        lambda_avg: average default intensity.
        recovery: recovery rate R.
        M_0: funding notional (defaults to B_0).
    """
    if M_0 is None:
        M_0 = B_0

    tau = T - t
    D_bar = D_tT * q_tT  # D̄(t,T) = D(t,T) × q(t,T)

    # Default PV: dpv ≈ ∫ λ D̄(t,τ) dτ ≈ λ_avg × (D(t,T) - D̄(t,T)) / (rs_bar + λ)
    # Simplified: dpv = 1 - D̄/D - (1-q) for small λ
    # More precisely: dpv = D_tT × (1 - q_tT) for deterministic λ (integral of hazard × risky df)
    dpv = D_tT * (1 - q_tT)

    # Funding leg
    funding_leg = M_0 * r_f * tau * D_bar

    # Asset leg components
    gamma = math.expm1(rs_bar_minus_r * tau)  # repo funding factor - 1 (expm1 for precision)

    # Coupon fva
    coupon_fva = sum(
        c * D_tT * math.exp(-lambda_avg * t_c) * gamma
        for t_c, c in coupons
    )

    # Recovery fva (simplified for deterministic λ)
    recovery_fva = recovery * D_tT * (1 - q_tT) * gamma

    fva = gamma * B_t - coupon_fva - recovery_fva

    # Total value
    value = funding_leg + B_0 * (D_bar + dpv) - B_t - fva

    return TRSBondResult(value, funding_leg, B_t + fva, fva, B_t, dpv)


# ---- Bond forward (Section 6) ----

def trs_bond_forward(
    B_t: float,
    rs_bar: float,
    T: float,
    coupons: list[tuple[float, float]],
    lambda_val: float = 0.0,
    t: float = 0.0,
) -> float:
    """Bond forward under haircut-blended repo, Eq (27-28).

    For Treasuries (λ=0): F = Bt exp(r̄s T) - Σ ci exp(r̄s (T-Ti))

    Args:
        B_t: current bond price.
        rs_bar: all-in financing rate (Eq 19).
        coupons: list of (time_to_coupon, coupon_amount).
        lambda_val: default intensity (0 for Treasuries).
    """
    tau = T - t

    if lambda_val < 1e-15:
        # Treasury limit (Eq 28)
        fwd = B_t * math.exp(rs_bar * tau)
        for t_c, c in coupons:
            fwd -= c * math.exp(rs_bar * (tau - t_c))
        return fwd

    # Risky case (Eq 27) — simplified for constant λ and rs_bar
    D_tilde = math.exp(-(rs_bar + lambda_val) * tau)
    fwd = B_t / D_tilde

    # Subtract default leg contribution
    # ∫ λ exp(∫(rs_bar + λ)du) dτ ≈ λ/(rs_bar+λ) × (1/D_tilde - 1)
    if abs(rs_bar + lambda_val) > 1e-15:
        fwd -= lambda_val / (rs_bar + lambda_val) * (1 / D_tilde - 1)

    # Subtract coupons
    for t_c, c in coupons:
        D_c = math.exp(-(rs_bar + lambda_val) * t_c)
        fwd -= c * D_c / D_tilde

    return fwd


# ---- Multi-period (Section 6) ----

def trs_multi_period(
    forwards: list[float],
    funding_rates: list[float],
    funding_notionals: list[float],
    periods: list[float],
    discount_factors: list[float],
    coupon_pvs: list[float] | None = None,
    dpv_increments: list[float] | None = None,
    recovery: float = 0.4,
) -> float:
    """Multi-period bond TRS, Eq (31).

    V = Σ [(Mj rf,j Δtj + Fj - Fj+1) D(t, tj+1) - ci D(t, Ti)
           + (Fj - R)(dpv(t,tj+1) - dpv(t,tj))]

    Args:
        forwards: Fj at each reset date (j = 0, ..., K).
        funding_rates: rf,j for each period.
        funding_notionals: Mj for each period (M0 for fixed-loan, Fj for MTM).
        periods: Δtj = tj+1 - tj.
        discount_factors: D(t, tj+1) for each period end.
        coupon_pvs: PV of coupons in each period (optional).
        dpv_increments: dpv(t, tj+1) - dpv(t, tj) for each period (optional, for defaultable bonds).
    """
    K = len(periods)
    if coupon_pvs is None:
        coupon_pvs = [0.0] * K
    if dpv_increments is None:
        dpv_increments = [0.0] * K

    value = 0.0
    for j in range(K):
        F_j = forwards[j]
        F_next = forwards[j + 1] if j + 1 < len(forwards) else forwards[-1]

        # Period cashflow: funding + price return
        period_cf = funding_notionals[j] * funding_rates[j] * periods[j] + F_j - F_next

        # Discount and accumulate
        value += period_cf * discount_factors[j] - coupon_pvs[j]

        # Default leg contribution
        value += (F_j - recovery) * dpv_increments[j]

    return value


# ---- Instrument class ----

class TotalReturnSwapLou:
    """Total Return Swap instrument (Lou 2018 framework).

    Trade-level constructor for equity or bond TRS with repo financing.
    Supports full CSA and repo-style margining.

        trs = TotalReturnSwapLou(
            spot=100.0, funding_rate=0.12, repo_spread=0.02,
            maturity=1.0, sigma=0.30, notional=10_000_000)
        result = trs.price(curve)
        portfolio.add(Trade(trs))

    Args:
        spot: current underlying price.
        funding_rate: rf = Libor + spread.
        repo_spread: rs - r (repo vs OIS spread).
        maturity: T in years.
        sigma: underlying volatility.
        notional: funding notional M0 (defaults to spot).
        div_yield: continuous dividend yield.
    """

    def __init__(
        self,
        spot: float,
        funding_rate: float,
        repo_spread: float = 0.0,
        maturity: float = 1.0,
        sigma: float = 0.20,
        notional: float | None = None,
        div_yield: float = 0.0,
    ):
        self.spot = spot
        self.funding_rate = funding_rate
        self.repo_spread = repo_spread
        self.maturity = maturity
        self.sigma = sigma
        self.notional = notional if notional is not None else spot
        self.div_yield = div_yield

    def price(self, curve) -> TRSResult:
        """Price the TRS using a discount curve (full CSA)."""
        from pricebook.day_count import year_fraction, DayCountConvention
        from datetime import timedelta

        ref = curve.reference_date
        mat_date = ref + timedelta(days=int(self.maturity * 365))
        D = curve.df(mat_date)

        return trs_equity_full_csa(
            self.spot, self.spot, self.funding_rate, self.maturity,
            0.0, D, rs_minus_r=self.repo_spread, t=0.0, M_0=self.notional)

    def greeks(self, curve) -> dict[str, float]:
        """Bump-and-reprice delta and repo sensitivity."""
        base = self.price(curve)

        # Delta: dV/dS
        bump = self.spot * 0.01
        old_spot = self.spot
        self.spot = old_spot + bump
        up = self.price(curve)
        self.spot = old_spot - bump
        dn = self.price(curve)
        self.spot = old_spot
        delta = (up.value - dn.value) / (2 * bump)

        # Repo sensitivity: dV/d(rs-r)
        old_repo = self.repo_spread
        self.repo_spread = old_repo + 0.0001
        up_repo = self.price(curve)
        self.repo_spread = old_repo
        repo_sens = (up_repo.value - base.value) / 0.0001

        return {"delta": delta, "repo_sensitivity": repo_sens, "fva": base.fva}

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv()."""
        result = self.price(ctx.discount_curve)
        return result.value
