"""Bootstrap hazard rates from bond prices when no CDS curve is available.

The problem: given N risky bonds of the same issuer at observed market prices,
extract a piecewise constant hazard rate term structure that reprices all bonds.

This is harder than CDS bootstrapping because:
1. Bonds have large intermediate cashflows (coupons), not just protection/premium legs
2. Coupon dates may not align across bonds
3. Recovery assumption has more impact (recovery of par vs recovery of market value)
4. Illiquid bonds have wide bid-ask → noisy input prices
5. Bonds may have embedded options (callable), accrued interest, settlement conventions
6. The problem is over-determined if N > number of hazard pillars (need fitting, not exact)

    from pricebook.credit.bond_hazard_bootstrap import (
        bootstrap_hazard_from_bonds, HazardBootstrapResult,
    )

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives, Ch 6.
    Duffie & Singleton (1999). Modeling Term Structures of Defaultable Bonds.
    Hull, Predescu & White (2004). Bond Prices, Default Probabilities and Risk Premiums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize, brentq

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency, generate_schedule


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class BondInput:
    """A single bond observation for hazard rate calibration."""
    maturity: date
    coupon: float          # annual coupon rate (e.g. 0.05 = 5%)
    market_price: float    # dirty price per 100 face
    frequency: int = 2     # coupons per year
    recovery: float = 0.40 # assumed recovery on default (fraction of par)
    weight: float = 1.0    # fitting weight (lower for illiquid bonds)

    def to_dict(self) -> dict:
        return {**vars(self), "maturity": self.maturity.isoformat()}


@dataclass
class HazardBootstrapResult:
    """Result of hazard rate calibration from bond prices."""
    survival_curve: SurvivalCurve
    pillar_dates: list[date]
    pillar_hazards: list[float]   # piecewise constant h between pillars
    fitted_prices: list[float]    # model price per bond
    market_prices: list[float]
    residuals_bp: list[float]     # (model - market) in bp of par
    rmse_bp: float                # root mean square error in bp
    max_error_bp: float
    n_bonds: int
    method: str
    converged: bool

    def to_dict(self) -> dict:
        return {
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "pillar_hazards": self.pillar_hazards,
            "fitted_prices": self.fitted_prices,
            "market_prices": self.market_prices,
            "residuals_bp": self.residuals_bp,
            "rmse_bp": self.rmse_bp,
            "max_error_bp": self.max_error_bp,
            "n_bonds": self.n_bonds,
            "method": self.method,
            "converged": self.converged,
        }


# ═══════════════════════════════════════════════════════════════
# Risky Bond Pricing (internal)
# ═══════════════════════════════════════════════════════════════

def _price_risky_bond(
    reference_date: date,
    maturity: date,
    coupon: float,
    frequency: int,
    recovery: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """Price a risky bond given discount and survival curves.

    PV = Σ c × τ × df(t_i) × Q(t_i)                   [coupon leg]
       + 1.0 × df(T) × Q(T)                             [principal leg]
       + R × Σ df(t_mid) × (Q(t_{i-1}) - Q(t_i))       [recovery leg]

    where Q(t) = survival probability, df(t) = discount factor.
    """
    freq_map = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL, 4: Frequency.QUARTERLY}
    freq = freq_map.get(frequency, Frequency.SEMI_ANNUAL)

    schedule = generate_schedule(reference_date, maturity, freq)
    dc = DayCountConvention.ACT_365_FIXED

    pv = 0.0
    prev_surv = 1.0

    for i in range(1, len(schedule)):
        t_start = schedule[i - 1]
        t_end = schedule[i]
        yf = year_fraction(t_start, t_end, dc)
        df = discount_curve.df(t_end)
        surv = survival_curve.survival(t_end)

        # Coupon (survival-weighted)
        pv += coupon * yf * df * surv * 100.0

        # Recovery on default (mid-period)
        t_mid_yf = year_fraction(reference_date, t_start, dc) + 0.5 * yf
        t_mid = date.fromordinal(
            reference_date.toordinal() + int(t_mid_yf * 365)
        )
        df_mid = discount_curve.df(t_mid)
        default_prob = prev_surv - surv
        pv += recovery * 100.0 * df_mid * default_prob

        prev_surv = surv

    # Principal at maturity (survival-weighted)
    pv += 100.0 * discount_curve.df(maturity) * survival_curve.survival(maturity)

    return pv


# ═══════════════════════════════════════════════════════════════
# Sequential Bootstrap (exact fit, N bonds → N hazard pillars)
# ═══════════════════════════════════════════════════════════════

def _bootstrap_sequential(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
) -> HazardBootstrapResult:
    """Sequential bootstrap: one bond per maturity, exact fit.

    Like CDS bootstrapping but from bond prices.
    Bonds MUST be sorted by maturity. Each bond adds one hazard pillar.

    Limitation: requires exactly one bond per target maturity.
    Works best with 3-8 bonds spanning the curve.
    """
    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)
    n = len(sorted_bonds)

    pillar_dates = [reference_date]
    pillar_survivals = [1.0]
    fitted_prices = []

    for i, bond in enumerate(sorted_bonds):
        # Build trial survival curve with prior pillars + new pillar guess
        def objective(q_new: float) -> float:
            trial_dates = pillar_dates + [bond.maturity]
            trial_survivals = pillar_survivals + [max(q_new, 1e-10)]
            trial_curve = SurvivalCurve(reference_date, trial_dates[1:], trial_survivals[1:])
            model_price = _price_risky_bond(
                reference_date, bond.maturity, bond.coupon, bond.frequency,
                bond.recovery, discount_curve, trial_curve,
            )
            return model_price - bond.market_price

        # Solve for Q(T) that reprices this bond
        try:
            q_solved = brentq(objective, 1e-10, 1.0 - 1e-10)
        except ValueError:
            # Bracket failure — bond price inconsistent with model
            q_solved = max(pillar_survivals[-1] * 0.95, 1e-6)

        pillar_dates.append(bond.maturity)
        pillar_survivals.append(q_solved)

        # Compute fitted price
        curve = SurvivalCurve(reference_date, pillar_dates[1:], pillar_survivals[1:])
        fitted = _price_risky_bond(
            reference_date, bond.maturity, bond.coupon, bond.frequency,
            bond.recovery, discount_curve, curve,
        )
        fitted_prices.append(fitted)

    # Build final survival curve
    final_curve = SurvivalCurve(reference_date, pillar_dates[1:], pillar_survivals[1:])

    # Extract piecewise hazards
    hazards = []
    dc = DayCountConvention.ACT_365_FIXED
    for i in range(1, len(pillar_dates)):
        dt = year_fraction(pillar_dates[i-1], pillar_dates[i], dc)
        if dt > 0 and pillar_survivals[i] > 0 and pillar_survivals[i-1] > 0:
            h = -math.log(pillar_survivals[i] / pillar_survivals[i-1]) / dt
        else:
            h = 0.0
        hazards.append(max(h, 0.0))

    # Compute residuals
    market_prices = [b.market_price for b in sorted_bonds]
    residuals = [(f - m) * 100 for f, m in zip(fitted_prices, market_prices)]
    rmse = math.sqrt(sum(r**2 for r in residuals) / n) if n > 0 else 0.0

    return HazardBootstrapResult(
        survival_curve=final_curve,
        pillar_dates=pillar_dates[1:],
        pillar_hazards=hazards,
        fitted_prices=fitted_prices,
        market_prices=market_prices,
        residuals_bp=residuals,
        rmse_bp=rmse,
        max_error_bp=max(abs(r) for r in residuals) if residuals else 0.0,
        n_bonds=n,
        method="sequential",
        converged=True,
    )


# ═══════════════════════════════════════════════════════════════
# Global Fit (least-squares, N bonds → M hazard pillars, M ≤ N)
# ═══════════════════════════════════════════════════════════════

def _bootstrap_global(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
    n_pillars: int | None = None,
) -> HazardBootstrapResult:
    """Global least-squares fit: fit M hazard pillars to N bond prices.

    More robust than sequential when bonds are noisy or have gaps.
    Minimises weighted sum of squared pricing errors.

    Args:
        n_pillars: number of hazard rate segments. Default = min(N, 5).
    """
    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)
    n = len(sorted_bonds)
    if n_pillars is None:
        n_pillars = min(n, 5)

    # Create pillar dates evenly spaced across bond maturities
    dc = DayCountConvention.ACT_365_FIXED
    t_max = year_fraction(reference_date, sorted_bonds[-1].maturity, dc)
    pillar_times = [t_max * (i + 1) / n_pillars for i in range(n_pillars)]
    pillar_dates = [
        date.fromordinal(reference_date.toordinal() + int(t * 365))
        for t in pillar_times
    ]

    def objective(hazard_rates: np.ndarray) -> float:
        """Weighted sum of squared pricing errors."""
        # Build survival curve from hazard rates
        survivals = []
        cum_surv = 1.0
        prev_t = 0.0
        for i, t in enumerate(pillar_times):
            dt = t - prev_t
            cum_surv *= math.exp(-max(hazard_rates[i], 0.0) * dt)
            survivals.append(cum_surv)
            prev_t = t

        try:
            curve = SurvivalCurve(reference_date, pillar_dates, survivals)
        except Exception:
            return 1e10

        total_err = 0.0
        for bond in sorted_bonds:
            model = _price_risky_bond(
                reference_date, bond.maturity, bond.coupon, bond.frequency,
                bond.recovery, discount_curve, curve,
            )
            err_bp = (model - bond.market_price) * 100  # in bp of par
            total_err += bond.weight * err_bp ** 2
        return total_err

    # Initial guess: flat hazard from average Z-spread
    avg_spread = 0.0
    for b in sorted_bonds:
        t = year_fraction(reference_date, b.maturity, dc)
        if t > 0 and b.market_price > 0:
            implied_yield = b.coupon + (100 - b.market_price) / (t * 100)
            rf = -math.log(discount_curve.df(b.maturity)) / t if t > 0 else 0.0
            avg_spread += max(implied_yield - rf, 0.0) / (1 - b.recovery)
    avg_spread /= max(n, 1)
    x0 = np.full(n_pillars, max(avg_spread, 0.001))

    # Optimise with bounds (hazard >= 0)
    bounds = [(0.0, 2.0)] * n_pillars  # cap at 200% annual hazard
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12})

    # Build final curve from optimised hazard rates
    hazard_rates = result.x
    survivals = []
    cum_surv = 1.0
    prev_t = 0.0
    for i, t in enumerate(pillar_times):
        dt = t - prev_t
        cum_surv *= math.exp(-max(hazard_rates[i], 0.0) * dt)
        survivals.append(cum_surv)
        prev_t = t

    final_curve = SurvivalCurve(reference_date, pillar_dates, survivals)

    # Compute fitted prices and residuals
    fitted_prices = []
    for bond in sorted_bonds:
        fitted = _price_risky_bond(
            reference_date, bond.maturity, bond.coupon, bond.frequency,
            bond.recovery, discount_curve, final_curve,
        )
        fitted_prices.append(fitted)

    market_prices = [b.market_price for b in sorted_bonds]
    residuals = [(f - m) * 100 for f, m in zip(fitted_prices, market_prices)]
    rmse = math.sqrt(sum(r**2 for r in residuals) / n) if n > 0 else 0.0

    return HazardBootstrapResult(
        survival_curve=final_curve,
        pillar_dates=pillar_dates,
        pillar_hazards=[float(h) for h in hazard_rates],
        fitted_prices=fitted_prices,
        market_prices=market_prices,
        residuals_bp=residuals,
        rmse_bp=rmse,
        max_error_bp=max(abs(r) for r in residuals) if residuals else 0.0,
        n_bonds=n,
        method="global_ls",
        converged=result.success,
    )


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def bootstrap_hazard_from_bonds(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
    method: str = "auto",
    n_pillars: int | None = None,
) -> HazardBootstrapResult:
    """Bootstrap a survival/hazard curve from risky bond prices.

    Args:
        reference_date: valuation date.
        bonds: list of BondInput with market prices.
        discount_curve: risk-free OIS discount curve.
        method: "sequential" (exact, 1 bond per pillar),
                "global" (least-squares, N bonds → M pillars),
                "auto" (sequential if N ≤ 8 with distinct maturities, else global).
        n_pillars: for global method, number of hazard segments.

    Returns:
        HazardBootstrapResult with survival curve and diagnostics.

    Minimum bonds: 1 (flat hazard), 2 (2-segment), recommended 3-8.

    Edge cases:
        - 1 bond: flat hazard rate (single pillar)
        - Bonds with same maturity: global method required (over-determined)
        - Very high spreads (distressed): solver may struggle, use global
        - Negative hazard: clamped to 0 (bond trades above risk-free par)
    """
    if not bonds:
        raise ValueError("At least one bond required")

    n = len(bonds)
    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)

    # Check for duplicate maturities
    maturities = [b.maturity for b in sorted_bonds]
    unique_mats = len(set(maturities))

    if method == "auto":
        if unique_mats == n and n <= 8:
            method = "sequential"
        else:
            method = "global"

    if method == "sequential":
        if unique_mats < n:
            raise ValueError(
                f"Sequential method requires distinct maturities, "
                f"got {n} bonds with {unique_mats} unique maturities. Use method='global'."
            )
        return _bootstrap_sequential(reference_date, sorted_bonds, discount_curve)
    else:
        return _bootstrap_global(reference_date, sorted_bonds, discount_curve, n_pillars)


def implied_hazard_from_spread(
    spread_bp: float,
    recovery: float = 0.40,
) -> float:
    """Quick approximation: constant hazard from spread.

    h ≈ spread / (1 - R)

    This is the starting point, not a calibration.
    """
    return spread_bp / 10_000 / (1 - recovery)


def minimum_bonds_for_calibration(
    target_maturities: list[float],
) -> dict:
    """Guidance on minimum bond requirements for robust calibration.

    Returns recommended number and maturity distribution.
    """
    max_t = max(target_maturities) if target_maturities else 10
    return {
        "minimum": 1,
        "recommended": max(3, min(len(target_maturities), 8)),
        "ideal_spread": f"2Y, 5Y, 10Y (or evenly across 0-{max_t:.0f}Y)",
        "notes": [
            "1 bond: flat hazard only — no term structure",
            "2 bonds: linear hazard — captures slope but not curvature",
            "3-5 bonds: good term structure — covers short/medium/long",
            "6-8 bonds: rich structure — can capture humps/inversions",
            "8+ bonds: use global fit (over-determined, least-squares)",
            "Bonds at similar maturities: use global fit (sequential fails)",
            "Wide bid-ask: lower weight in global fit for noisy prices",
            "Distressed (spread > 1000bp): use global fit with tight bounds",
        ],
    }
