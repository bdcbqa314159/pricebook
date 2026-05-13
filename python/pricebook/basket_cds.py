"""
Basket CDS, exotic CLN, and bespoke tranches via Gaussian copula.

Gaussian copula: correlated default times from a one-factor model.
    Each name: Z_i = sqrt(rho)*M + sqrt(1-rho)*epsilon_i
    Default if Z_i < Phi^{-1}(1 - Q_i(T))

First-to-default (FTD): protection triggered by first default.
Nth-to-default (NTD): protection triggered by Nth default.
Bespoke tranche: custom [attach, detach] on a bespoke portfolio.

Exotic CLN: leveraged notional, digital recovery.

    ftd_spread = ftd_basket_spread(survival_curves, discount_curve, rho=0.3, T=5)
    bt = bespoke_tranche(pds, attach=0.03, detach=0.07, rho=0.3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------


def simulate_defaults_copula_via_engine(
    survival_curves: list[SurvivalCurve],
    T: float,
    rho: float,
    n_sims: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """``simulate_defaults_copula`` via unified MC engine (correlated normals)."""
    from pricebook.mc_migrate import correlated_gbm_paths  # noqa: lazy

    # Gaussian copula only needs correlated normals, not asset paths.
    # The engine's correlated-GBM helper gives correlated Brownian increments
    # from which we extract the terminal Z values (single time-step).
    n_names = len(survival_curves)
    # Build uniform-correlation matrix
    corr = np.full((n_names, n_names), rho)
    np.fill_diagonal(corr, 1.0)

    # One-step "GBM" with zero drift / unit vol → terminal log = Z * √T
    spots = [1.0] * n_names
    rates = [0.0] * n_names
    vols = [1.0] * n_names  # unit vol so log(S_T/S_0) = -0.5*T + Z*√T

    paths = correlated_gbm_paths(spots, rates, vols, corr,
                                  T=1.0, n_steps=1, n_paths=n_sims, seed=seed)
    # Extract standardised normals: log(S_T) = log(1) + (-0.5 + Z) → Z = log(S_T) + 0.5
    Z = np.log(paths[:, -1, :]) + 0.5

    ref = survival_curves[0].reference_date
    T_date = date_from_year_fraction(ref, T)
    thresholds = np.array([
        norm.ppf(1 - sc.survival(T_date)) for sc in survival_curves
    ])
    return Z < thresholds[np.newaxis, :]


# ---------------------------------------------------------------------------
# Bespoke tranche
# ---------------------------------------------------------------------------

@dataclass
class BespokeTrancheResult:
    """Bespoke tranche pricing result."""
    expected_loss: float        # expected tranche loss (fraction of tranche width)
    tranche_spread: float       # fair spread (annualised)
    attach: float
    detach: float
    portfolio_el: float         # portfolio expected loss
    n_names: int

    def to_dict(self) -> dict:
        return vars(self)


def bespoke_tranche(
    marginal_pds: list[float],
    attach: float,
    detach: float,
    rho: float = 0.30,
    lgd: float = 0.60,
    T: float = 5.0,
    rate: float = 0.04,
    n_sims: int = 50_000,
    seed: int | None = 42,
) -> BespokeTrancheResult:
    """Bespoke tranche: custom [attach, detach] on a bespoke credit portfolio.

    Uses one-factor Gaussian copula to simulate correlated defaults.
    Each name has its own marginal PD; correlation is flat.

    Tranche loss = max(0, min(portfolio_loss - attach, detach - attach))
                   / (detach - attach)

    The fair spread equates the PV of expected tranche loss payments
    to the PV of spread payments on surviving tranche notional.

    Args:
        marginal_pds: list of marginal default probabilities (one per name).
        attach: attachment point (e.g. 0.03 for 3%).
        detach: detachment point (e.g. 0.07 for 7%).
        rho: flat pairwise correlation.
        lgd: loss given default (uniform across names).
    """
    if attach >= detach:
        raise ValueError(f"attach ({attach}) must be < detach ({detach})")
    if not marginal_pds:
        raise ValueError("marginal_pds must not be empty")

    n_names = len(marginal_pds)
    rng = np.random.default_rng(seed)

    # One-factor Gaussian copula
    sqrt_rho = math.sqrt(max(rho, 0.0))
    sqrt_1_rho = math.sqrt(max(1 - rho, 0.0))

    # Clamp PDs to (0, 1) to avoid inf from norm.ppf
    clamped_pds = [max(1e-10, min(1 - 1e-10, pd)) for pd in marginal_pds]
    thresholds = np.array([norm.ppf(pd) for pd in clamped_pds])

    # Simulate
    M = rng.standard_normal(n_sims)  # systematic factor
    eps = rng.standard_normal((n_sims, n_names))  # idiosyncratic
    Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps
    defaults = Z < thresholds[np.newaxis, :]  # (n_sims, n_names)

    # Portfolio loss fraction
    portfolio_loss = defaults.sum(axis=1) * lgd / n_names
    portfolio_el = float(portfolio_loss.mean())

    # Tranche loss
    width = detach - attach
    tranche_loss = np.maximum(0, np.minimum(portfolio_loss - attach, width)) / width
    el = float(tranche_loss.mean())

    # Fair spread: EL / risky annuity (weighted by tranche survival)
    n_annual = max(int(T), 1)
    risky_annuity = 0.0
    for yr in range(1, n_annual + 1):
        # Fraction of time up to yr
        t_yr = yr * T / n_annual
        # Portfolio loss up to this point (approximate: scale by time fraction)
        loss_at_yr = portfolio_loss * (yr / n_annual)
        tranche_loss_yr = np.maximum(0, np.minimum(loss_at_yr - attach, width)) / width
        tranche_survival = 1.0 - tranche_loss_yr
        risky_annuity += math.exp(-rate * t_yr) * float(tranche_survival.mean())

    tranche_spread = el / risky_annuity if risky_annuity > 0 else 0.0

    return BespokeTrancheResult(
        expected_loss=el,
        tranche_spread=float(tranche_spread),
        attach=attach,
        detach=detach,
        portfolio_el=portfolio_el,
        n_names=n_names,
    )
