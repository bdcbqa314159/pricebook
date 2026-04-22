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

from pricebook.day_count import DayCountConvention, year_fraction
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
    T_date = date.fromordinal(ref.toordinal() + int(T * 365))

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
    defaults = simulate_defaults_copula(survival_curves, T, rho, n_sims, seed)
    n_defaults = count_defaults(defaults)

    ntd_triggered = n_defaults >= n

    ref = survival_curves[0].reference_date
    T_date = date.fromordinal(ref.toordinal() + int(T * 365))
    df_T = discount_curve.df(T_date)

    protection = (1 - recovery) * df_T * ntd_triggered.mean()

    # Risky annuity (simplified)
    annuity = 0.0
    n_years = max(1, int(T))
    for yr in range(1, n_years + 1):
        t = min(yr, T)
        d = ref + relativedelta(years=int(t))
        df = discount_curve.df(d)
        surv_prob = 1.0 - ntd_triggered.mean() * (t / T)
        surv_prob = max(surv_prob, 0.01)
        annuity += df * surv_prob

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
        d_T = date.fromordinal(ref.toordinal() + int(self.T * 365))
        pv += self.notional * discount_curve.df(d_T) * survival_curve.survival(d_T)

        return pv
