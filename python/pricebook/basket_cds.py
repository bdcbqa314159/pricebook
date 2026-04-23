"""
Basket CDS and exotic CLN via Gaussian copula.

Gaussian copula: correlated default times from a one-factor model.
    Each name: Z_i = sqrt(rho)*M + sqrt(1-rho)*epsilon_i
    Default if Z_i < Phi^{-1}(1 - Q_i(T))

First-to-default (FTD): protection triggered by first default.
Nth-to-default (NTD): protection triggered by Nth default.

Exotic CLN: leveraged notional, digital recovery.

    ftd_spread = ftd_basket_spread(survival_curves, discount_curve, rho=0.3, T=5)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

from pricebook.day_count import DayCountConvention, year_fraction, date_from_year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


def simulate_defaults_copula(
    survival_curves: list[SurvivalCurve],
    T: float,
    rho: float,
    n_sims: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate default indicators at time T using Gaussian copula.

    Returns:
        Boolean array of shape (n_sims, n_names). True = defaulted by T.
    """
    n_names = len(survival_curves)
    rng = np.random.default_rng(seed)

    # Systematic factor
    M = rng.standard_normal(n_sims)

    # Idiosyncratic factors
    eps = rng.standard_normal((n_sims, n_names))

    # Correlated normals
    sqrt_rho = math.sqrt(max(rho, 0.0))
    sqrt_1_rho = math.sqrt(max(1.0 - rho, 0.0))
    Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

    # Default thresholds from survival probabilities
    ref = survival_curves[0].reference_date
    T_date = date_from_year_fraction(ref, T)

    thresholds = np.array([
        norm.ppf(1 - sc.survival(T_date)) for sc in survival_curves
    ])

    # Default if Z_i < threshold_i
    return Z < thresholds[np.newaxis, :]


def count_defaults(defaults: np.ndarray) -> np.ndarray:
    """Count number of defaults per simulation. Shape: (n_sims,)."""
    return defaults.sum(axis=1)


def ftd_spread(
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    rho: float,
    T: float,
    recovery: float = 0.4,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """First-to-default basket spread via MC simulation.

    Thin wrapper around ntd_spread with n=1.
    """
    return ntd_spread(
        survival_curves, discount_curve, rho, T,
        n=1, recovery=recovery, n_sims=n_sims, seed=seed,
    )


def ntd_spread(
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    rho: float,
    T: float,
    n: int,
    recovery: float = 0.4,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """Nth-to-default basket spread.

    Args:
        n: trigger on the Nth default (1 = FTD).
    """
    # Simulate defaults at multiple time points for proper timing
    ref = survival_curves[0].reference_date
    n_years = max(1, int(T))
    annual_times = [min(yr, T) for yr in range(1, n_years + 1)]

    # Simulate at each annual time point
    n_names = len(survival_curves)
    rng = np.random.default_rng(seed)
    M = rng.standard_normal(n_sims)
    eps = rng.standard_normal((n_sims, n_names))
    sqrt_rho = math.sqrt(max(rho, 0.0))
    sqrt_1_rho = math.sqrt(max(1.0 - rho, 0.0))
    Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

    # For each time point, check if nth default has occurred
    ntd_by_time = []
    for t in annual_times:
        T_date = date_from_year_fraction(ref, t)
        thresholds = np.array([
            norm.ppf(max(1 - sc.survival(T_date), 1e-15)) for sc in survival_curves
        ])
        defaults_t = Z < thresholds[np.newaxis, :]
        n_defaults_t = defaults_t.sum(axis=1)
        ntd_by_time.append(n_defaults_t >= n)

    # Protection leg: (1-R) * df(T) * P(ntd triggered by T)
    T_date = date_from_year_fraction(ref, T)
    df_T = discount_curve.df(T_date)
    ntd_final = ntd_by_time[-1]
    protection = (1 - recovery) * df_T * ntd_final.mean()

    # Risky annuity: per-simulation survival at each annual point
    annuity = 0.0
    for i, t in enumerate(annual_times):
        d = date_from_year_fraction(ref, t)
        df = discount_curve.df(d)
        # Basket survival = fraction of sims where nth default hasn't triggered yet
        basket_surv = 1.0 - ntd_by_time[i].mean()
        basket_surv = max(basket_surv, 0.001)
        annuity += df * basket_surv

    if annuity <= 0:
        return 0.0
    return protection / annuity


class LeveragedCLN:
    """Credit-linked note with leveraged notional.

    The investor's funded amount is `notional`, but credit exposure
    is `leverage * notional`. Higher leverage amplifies credit risk.

    Args:
        notional: funded amount.
        leverage: credit exposure multiplier.
        coupon_rate: annual coupon.
        recovery: recovery rate on default.
    """

    def __init__(
        self,
        notional: float = 100.0,
        leverage: float = 1.0,
        coupon_rate: float = 0.06,
        recovery: float = 0.4,
        T: float = 5.0,
    ):
        self.notional = notional
        self.leverage = leverage
        self.coupon_rate = coupon_rate
        self.recovery = recovery
        self.T = T

    def pv(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """PV of the leveraged CLN.

        Coupons: notional * coupon_rate * df * survival (per year)
        Default loss: leverage * notional * (1-R) * default_prob * df
        Principal: notional * df_T * survival_T
        """
        ref = discount_curve.reference_date
        pv = 0.0
        n_years = max(1, int(self.T))

        for yr in range(1, n_years + 1):
            t = min(yr, self.T)
            d = ref + relativedelta(years=int(t))
            d_prev = ref + relativedelta(years=max(0, int(t) - 1))
            df = discount_curve.df(d)
            surv = survival_curve.survival(d)
            surv_prev = survival_curve.survival(d_prev)
            default_prob = surv_prev - surv

            # Coupon (funded amount)
            pv += self.notional * self.coupon_rate * df * surv

            # Default loss (leveraged amount)
            loss = self.leverage * self.notional * (1 - self.recovery) * default_prob
            pv -= loss * df

        # Principal return
        d_T = date_from_year_fraction(ref, self.T)
        pv += self.notional * discount_curve.df(d_T) * survival_curve.survival(d_T)

        return pv
