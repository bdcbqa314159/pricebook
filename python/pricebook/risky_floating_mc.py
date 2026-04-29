"""Correlated rate-credit Monte Carlo for risky floating payments.

Joint simulation of short rate r(t) and hazard rate λ(t) with
correlation ρ. Captures wrong-way risk in floating rate pricing.

    from pricebook.risky_floating_mc import price_risky_frn_mc

    result = price_risky_frn_mc(
        frn, spot_rate=0.03, spot_hazard=0.02,
        rate_vol=0.01, hazard_vol=0.15, correlation=-0.3,
        n_paths=100_000)

References:
    Brigo & Mercurio (2006). Interest Rate Models. Ch. 22.
    Schönbucher (2003). Credit Derivatives Pricing Models. Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.risky_floating import CreditRiskyFRN


@dataclass
class RiskyFloatingMCResult:
    """Result of correlated rate-credit MC pricing."""
    price: float                    # MC price per 100 face
    price_independent: float        # price with ρ=0 (for comparison)
    wrong_way_adjustment: float     # correlated - independent
    std_error: float
    n_paths: int
    avg_rate_path: list[float]
    avg_hazard_path: list[float]

    def to_dict(self) -> dict:
        return {"price": self.price, "price_independent": self.price_independent,
                "wrong_way_adjustment": self.wrong_way_adjustment,
                "std_error": self.std_error, "n_paths": self.n_paths}


def price_risky_frn_mc(
    frn: CreditRiskyFRN,
    spot_rate: float,
    spot_hazard: float,
    rate_mean_reversion: float = 0.5,
    rate_long_run: float | None = None,
    rate_vol: float = 0.01,
    hazard_mean_reversion: float = 0.5,
    hazard_long_run: float | None = None,
    hazard_vol: float = 0.15,
    correlation: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 42,
) -> RiskyFloatingMCResult:
    """Price a credit-risky FRN via correlated rate-credit MC.

    Models:
        dr = a_r (θ_r - r) dt + σ_r dW₁       (Vasicek for rate)
        dλ = a_λ (θ_λ - λ) dt + σ_λ √λ dW₂    (CIR for hazard)
        dW₁ · dW₂ = ρ dt

    For each path:
        - Simulate (r, λ) at each coupon date
        - Compute forward rate from r
        - Compute survival from cumulative λ
        - Default time: first time cumulative hazard > -ln(U)
        - PV = discounted coupons until default + recovery at default

    Args:
        frn: the CreditRiskyFRN instrument.
        spot_rate: current short rate.
        spot_hazard: current hazard rate.
        correlation: rate-hazard correlation. ρ<0 = wrong-way risk.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {correlation}")

    ref_date = frn.start
    cashflows = frn.floating_leg.cashflows
    n_periods = len(cashflows)

    if n_periods == 0:
        return RiskyFloatingMCResult(0, 0, 0, 0, 0, [], [])

    # Time grid from cashflow dates
    times = [year_fraction(ref_date, cf.payment_date, DayCountConvention.ACT_365_FIXED)
             for cf in cashflows]

    theta_r = rate_long_run if rate_long_run is not None else spot_rate
    theta_lam = hazard_long_run if hazard_long_run is not None else spot_hazard

    rng = np.random.default_rng(seed)

    def simulate_paths(rho: float) -> tuple[np.ndarray, list[float], list[float]]:
        """Simulate and return PV per path + avg paths."""
        r = np.full(n_paths, spot_rate)
        lam = np.full(n_paths, spot_hazard)
        cum_hazard = np.zeros(n_paths)
        cum_rate = np.zeros(n_paths)
        # Random default threshold per path
        U = rng.random(n_paths)
        default_threshold = -np.log(np.maximum(U, 1e-15))

        alive = np.ones(n_paths, dtype=bool)
        pv_paths = np.zeros(n_paths)
        avg_r = []
        avg_lam = []

        t_prev = 0.0
        for i, t in enumerate(times):
            dt = t - t_prev
            sqrt_dt = math.sqrt(max(dt, 1e-10))

            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w1 = z1
            w2 = rho * z1 + math.sqrt(1 - rho**2) * z2

            # Rate: Vasicek
            r = r + rate_mean_reversion * (theta_r - r) * dt + rate_vol * sqrt_dt * w1

            # Hazard: CIR (floor at 0)
            lam_safe = np.maximum(lam, 0.0)
            lam = lam + hazard_mean_reversion * (theta_lam - lam_safe) * dt \
                  + hazard_vol * np.sqrt(lam_safe) * sqrt_dt * w2
            lam = np.maximum(lam, 0.0)

            cum_hazard += lam * dt
            cum_rate += r * dt

            # Default check
            just_defaulted = alive & (cum_hazard > default_threshold)

            # Coupon: paid if alive
            cf = cashflows[i]
            fwd = r + frn.spread  # r is the short rate, approximate forward
            yf = cf.year_frac
            df = np.exp(-cum_rate)

            # Alive paths: receive coupon
            coupon = frn.notional * fwd * yf * df
            pv_paths += coupon * alive * ~just_defaulted

            # Just defaulted: receive recovery
            recovery_pv = frn.recovery * frn.notional * df
            pv_paths += recovery_pv * just_defaulted

            alive &= ~just_defaulted

            avg_r.append(float(r.mean()))
            avg_lam.append(float(lam.mean()))
            t_prev = t

        # Terminal: alive paths get principal
        df_T = np.exp(-cum_rate)
        pv_paths += frn.notional * df_T * alive

        return pv_paths, avg_r, avg_lam

    # Correlated simulation
    pv_corr, avg_r, avg_lam = simulate_paths(correlation)
    price_corr = float(pv_corr.mean()) / frn.notional * 100

    # Independent simulation (ρ=0) for comparison
    rng2 = np.random.default_rng(seed + 1)
    pv_indep, _, _ = simulate_paths(0.0)
    price_indep = float(pv_indep.mean()) / frn.notional * 100

    std_err = float(pv_corr.std(ddof=1) / math.sqrt(n_paths)) / frn.notional * 100

    return RiskyFloatingMCResult(
        price=price_corr,
        price_independent=price_indep,
        wrong_way_adjustment=price_corr - price_indep,
        std_error=std_err,
        n_paths=n_paths,
        avg_rate_path=avg_r,
        avg_hazard_path=avg_lam,
    )
