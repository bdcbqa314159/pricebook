"""Catastrophe bonds and insurance-linked securities (ILS).

Pricing and risk analytics for cat bonds, parametric and indemnity triggers,
ILS portfolio construction, and seasonal adjustments.

* :class:`CatBondResult`        — cat bond pricing output.
* :class:`ILSPortfolioResult`   — ILS portfolio analytics.
* :class:`PeriodType`           — risk period enumeration.
* :func:`cat_bond_price`        — price a cat bond given expected loss and recovery.
* :func:`parametric_trigger_prob` — Gumbel EVT-based trigger probability.
* :func:`indemnity_trigger_loss`  — MC simulation of indemnity trigger payouts.
* :func:`cat_bond_spread_decomposition` — decompose total spread into components.
* :func:`ils_portfolio`         — correlated ILS portfolio via Gaussian copula.
* :func:`seasonal_adjustment`   — adjust probability for remaining risk period.

References:
    Cummins, J.D. & Weiss, M.A. (2009). Convergence of Insurance and Financial
        Markets: Hybrid and Securitized Risk-Transfer Solutions. *Journal of Risk
        and Insurance*, 76(3), 493–545.
    Lane, M.N. (2000). Pricing Risk Transfer Transactions. *ASTIN Bulletin*,
        30(2), 259–293.
    Braun, A. (2016). Pricing in the Primary Market for Cat Bonds: New Empirical
        Evidence. *Journal of Risk and Insurance*, 83(4), 811–847.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CatBondResult:
    """Cat bond pricing and risk output."""
    price: float                # clean price per 100 notional
    spread: float               # total spread over risk-free (annual)
    expected_loss: float        # annual expected loss rate
    probability_of_loss: float  # probability of any principal loss
    coupon: float               # annual coupon rate (risk-free + spread)
    exhaustion_prob: float      # probability of full principal loss

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class ILSPortfolioResult:
    """ILS portfolio analytics."""
    portfolio_expected_loss: float          # portfolio-level EL
    diversification_benefit: float          # sum(EL_i) - portfolio EL
    var_99: float                           # 99th-percentile loss (fraction of notional)
    constituent_contributions: list[float]  # each bond's marginal contribution to portfolio EL

    def to_dict(self) -> dict:
        return {
            "portfolio_expected_loss": self.portfolio_expected_loss,
            "diversification_benefit": self.diversification_benefit,
            "var_99": self.var_99,
            "constituent_contributions": self.constituent_contributions,
        }


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PeriodType(Enum):
    """Risk period type for seasonal adjustment."""
    ANNUAL = "annual"
    HURRICANE_SEASON = "hurricane_season"   # Jun–Nov (6 months)
    EARTHQUAKE = "earthquake"               # uniform throughout year


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discount_factor(rate: float, T: float) -> float:
    return math.exp(-rate * T)


def _gumbel_cdf(x: float, mu: float, beta: float) -> float:
    """Gumbel (Type-I extreme value) CDF: exp(-exp(-(x-mu)/beta))."""
    return math.exp(-math.exp(-(x - mu) / beta))


# ---------------------------------------------------------------------------
# 1. Cat bond price
# ---------------------------------------------------------------------------

def cat_bond_price(
    notional: float,
    coupon_spread: float,
    risk_free_rate: float,
    expected_loss: float,
    T: float,
    recovery_if_triggered: float = 0.0,
) -> CatBondResult:
    """Price a catastrophe bond.

    The coupon is paid from a collateral trust; the principal is at risk.
    Simple single-period (or continuously-compounded multi-year) model:

        price = PV(coupons) × (1 - prob_loss) + PV(recovery) × prob_loss

    The spread is set by the market at:

        spread = risk_free + EL / (1 - EL) + risk_premium

    where risk_premium is calibrated so spread ≈ 2× EL (Lane 2000 rule-of-thumb).

    Args:
        notional:               Face value.
        coupon_spread:          Spread over risk-free (annualised).
        risk_free_rate:         Risk-free rate (annualised).
        expected_loss:          Annual expected loss rate (fraction of notional).
        T:                      Term in years.
        recovery_if_triggered:  Recovery fraction of notional if triggered (0–1).

    Returns:
        CatBondResult with price per 100.
    """
    coupon = risk_free_rate + coupon_spread
    prob_loss = 1.0 - math.exp(-expected_loss * T)       # Poisson arrival
    exhaustion_prob = prob_loss * (1.0 - recovery_if_triggered)

    # PV of coupon stream assuming coupons paid annually, loss occurs mid-period.
    # Fix T4-STRUCT: pre-fix ``range(1, int(T) + 1)`` silently dropped any
    # non-integer remainder of ``T``.  For T=0.5 (6mo bond) the loop was
    # empty → zero coupon PV.  For T=3.5 the final 0.5y accrual was
    # missed.  Add a fractional final accrual when T has non-integer part.
    pv_coupons = 0.0
    n_full = int(T)
    for t in range(1, n_full + 1):
        df_t = _discount_factor(risk_free_rate, t)
        # weight: probability bond is still alive at t (geometric approximation)
        survival = math.exp(-expected_loss * t)
        pv_coupons += coupon * notional * df_t * survival
    remainder = T - n_full
    if remainder > 1e-9:
        df_rem = _discount_factor(risk_free_rate, T)
        survival_rem = math.exp(-expected_loss * T)
        pv_coupons += coupon * notional * remainder * df_rem * survival_rem

    df_T = _discount_factor(risk_free_rate, T)
    pv_principal_no_loss = notional * df_T * (1.0 - prob_loss)
    pv_recovery = notional * recovery_if_triggered * df_T * prob_loss
    price_abs = pv_coupons + pv_principal_no_loss + pv_recovery
    price_per_100 = 100.0 * price_abs / notional

    # Spread decomposition (Lane 2000)
    risk_premium = expected_loss  # market charges ~1× EL as risk premium
    spread = risk_free_rate + expected_loss / max(1.0 - expected_loss, 1e-9) + risk_premium

    return CatBondResult(
        price=price_per_100,
        spread=spread,
        expected_loss=expected_loss,
        probability_of_loss=prob_loss,
        coupon=coupon,
        exhaustion_prob=exhaustion_prob,
    )


# ---------------------------------------------------------------------------
# 2. Parametric trigger probability (Gumbel EVT)
# ---------------------------------------------------------------------------

def parametric_trigger_prob(
    threshold: float,
    location_mu: float,
    scale_sigma: float,
    historical_events: list[float] | None = None,
) -> float:
    """Probability that a parametric trigger is breached using Gumbel EVT.

    Fits a Gumbel distribution to annual maxima of the hazard variable
    (earthquake magnitude, wind speed, etc.).  If ``historical_events`` is
    provided, method-of-moments re-estimates mu and sigma.

    P(breach) = 1 - CDF(threshold; mu, sigma)

    Args:
        threshold:          Trigger level (e.g. Richter magnitude 7.0).
        location_mu:        Gumbel location parameter (mode of distribution).
        scale_sigma:        Gumbel scale parameter (> 0).
        historical_events:  Optional list of annual-maximum observations for
                            MOM calibration.

    Returns:
        Probability of breaching the threshold in one year.
    """
    if scale_sigma <= 0:
        raise ValueError("scale_sigma must be positive")

    mu = location_mu
    beta = scale_sigma

    if historical_events is not None and len(historical_events) >= 2:
        arr = np.asarray(historical_events, dtype=float)
        # Method of moments: beta = std * sqrt(6) / pi, mu = mean - euler * beta
        euler = 0.5772156649  # Euler–Mascheroni constant
        beta = float(np.std(arr, ddof=1)) * math.sqrt(6.0) / math.pi
        mu = float(np.mean(arr)) - euler * beta

    prob_no_breach = _gumbel_cdf(threshold, mu, max(beta, 1e-12))
    return 1.0 - prob_no_breach


# ---------------------------------------------------------------------------
# 3. Indemnity trigger loss (Monte Carlo)
# ---------------------------------------------------------------------------

def indemnity_trigger_loss(
    attachment: float,
    exhaustion: float,
    loss_distribution_mean: float,
    loss_distribution_cv: float,
    n_simulations: int = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    """Monte Carlo simulation of indemnity trigger payouts.

    Industry loss follows a lognormal distribution.  Payout fraction is:

        payout = min(max(loss - attachment, 0) / (exhaustion - attachment), 1)

    Args:
        attachment:              Lower trigger level (attach point).
        exhaustion:              Upper trigger level (full loss above here).
        loss_distribution_mean:  Mean of industry loss distribution.
        loss_distribution_cv:    Coefficient of variation (std / mean).
        n_simulations:           Number of MC paths.
        seed:                    Random seed for reproducibility.

    Returns:
        Dictionary with expected_loss, prob_of_loss, prob_of_exhaustion,
        var_99 (99th-percentile payout), mean_payout.
    """
    if exhaustion <= attachment:
        raise ValueError("exhaustion must be greater than attachment")
    if loss_distribution_cv <= 0:
        raise ValueError("loss_distribution_cv must be positive")

    rng = np.random.default_rng(seed)
    sigma_ln = math.sqrt(math.log(1.0 + loss_distribution_cv**2))
    mu_ln = math.log(loss_distribution_mean) - 0.5 * sigma_ln**2
    losses = rng.lognormal(mu_ln, sigma_ln, n_simulations)

    layer = exhaustion - attachment
    raw = np.clip((losses - attachment) / layer, 0.0, 1.0)

    return {
        "expected_loss": float(np.mean(raw)),
        "prob_of_loss": float(np.mean(raw > 0.0)),
        "prob_of_exhaustion": float(np.mean(raw >= 1.0)),
        "var_99": float(np.percentile(raw, 99)),
        "mean_payout": float(np.mean(raw)),
    }


# ---------------------------------------------------------------------------
# 4. Spread decomposition
# ---------------------------------------------------------------------------

def cat_bond_spread_decomposition(
    total_spread: float,
    expected_loss: float,
    expense_load: float = 0.01,
) -> dict[str, float]:
    """Decompose total cat bond spread into components.

    total_spread = expected_loss + risk_premium + expense_load

    Risk multiple (Sharpe ratio proxy) = total_spread / expected_loss.

    Args:
        total_spread:   Observed market spread over risk-free.
        expected_loss:  Modelled expected annual loss rate.
        expense_load:   Issuer / SPV expense load (default 1%).

    Returns:
        Dictionary with expected_loss, risk_premium, expense_load,
        risk_multiple, and implied_loss_given_trigger.
    """
    risk_premium = total_spread - expected_loss - expense_load
    risk_multiple = total_spread / max(expected_loss, 1e-12)

    return {
        "expected_loss": expected_loss,
        "risk_premium": risk_premium,
        "expense_load": expense_load,
        "total_spread": total_spread,
        "risk_multiple": risk_multiple,
    }


# ---------------------------------------------------------------------------
# 5. ILS portfolio (Gaussian copula)
# ---------------------------------------------------------------------------

def ils_portfolio(
    bonds: list[dict],
    correlation: float = 0.1,
    n_simulations: int = 20_000,
    seed: int = 42,
) -> ILSPortfolioResult:
    """Portfolio of cat bonds with loss correlation via Gaussian copula.

    Each bond in ``bonds`` is a dict with keys:
        * ``expected_loss``   — annual EL as fraction of notional.
        * ``notional``        — bond notional.
        * ``recovery``        — recovery fraction (default 0.0).

    Losses are simulated by mapping correlated Gaussian shocks to uniform
    marginals, then inverting the per-bond Bernoulli (Poisson) trigger.

    Args:
        bonds:          List of bond parameter dicts.
        correlation:    Pairwise loss correlation (constant, 0–1).
        n_simulations:  MC paths.
        seed:           Random seed.

    Returns:
        ILSPortfolioResult.
    """
    n = len(bonds)
    if n == 0:
        raise ValueError("bonds list is empty")

    rng = np.random.default_rng(seed)

    # Cholesky of equicorrelation matrix
    corr_matrix = np.full((n, n), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    L = np.linalg.cholesky(corr_matrix)

    # Simulate correlated uniforms via Gaussian copula
    z = rng.standard_normal((n_simulations, n))
    z_corr = z @ L.T
    from scipy.stats import norm as _scipy_norm  # noqa: PLC0415
    u = _scipy_norm.cdf(z_corr)  # shape (n_simulations, n)

    # Per-bond: loss occurs when uniform < prob_loss
    total_notional = sum(b.get("notional", 1.0) for b in bonds)
    portfolio_losses = np.zeros(n_simulations)
    standalone_els: list[float] = []
    contributions: list[float] = []

    for i, bond in enumerate(bonds):
        el = float(bond.get("expected_loss", 0.0))
        notional_i = float(bond.get("notional", 1.0))
        recovery_i = float(bond.get("recovery", 0.0))
        prob_loss_i = el  # annual probability ≈ EL for small EL

        triggered = u[:, i] < prob_loss_i
        loss_given_trigger = notional_i * (1.0 - recovery_i)
        bond_losses = triggered * loss_given_trigger
        portfolio_losses += bond_losses

        standalone_el = el * notional_i * (1.0 - recovery_i) / total_notional
        standalone_els.append(standalone_el)

    portfolio_el = float(np.mean(portfolio_losses)) / total_notional
    sum_standalone = sum(standalone_els)
    diversification_benefit = sum_standalone - portfolio_el
    var_99 = float(np.percentile(portfolio_losses, 99)) / total_notional

    # Marginal contributions: correlation-weighted
    for i, bond in enumerate(bonds):
        el = float(bond.get("expected_loss", 0.0))
        notional_i = float(bond.get("notional", 1.0))
        recovery_i = float(bond.get("recovery", 0.0))
        contrib = el * notional_i * (1.0 - recovery_i) / total_notional * (
            portfolio_el / max(sum_standalone, 1e-12)
        )
        contributions.append(contrib)

    return ILSPortfolioResult(
        portfolio_expected_loss=portfolio_el,
        diversification_benefit=diversification_benefit,
        var_99=var_99,
        constituent_contributions=contributions,
    )


# ---------------------------------------------------------------------------
# 6. Seasonal adjustment
# ---------------------------------------------------------------------------

def seasonal_adjustment(
    annual_prob: float,
    period_type: PeriodType,
    remaining_fraction: float,
) -> float:
    """Adjust annual trigger probability for time remaining in the risk period.

    Hurricane season is concentrated in 6 months (Jun–Nov); roughly 90% of
    annual hurricane activity falls within the season.  Earthquakes are
    approximately uniform throughout the year.

    Args:
        annual_prob:        Annual trigger probability.
        period_type:        Risk period type (PeriodType enum).
        remaining_fraction: Fraction of the risk period still remaining (0–1).

    Returns:
        Adjusted probability for the remaining risk period.
    """
    if not 0.0 <= remaining_fraction <= 1.0:
        raise ValueError("remaining_fraction must be in [0, 1]")

    if period_type == PeriodType.ANNUAL:
        seasonal_concentration = 1.0
    elif period_type == PeriodType.HURRICANE_SEASON:
        # Season is ~6 months; ~90% of risk concentrated in season
        seasonal_concentration = 0.90 / 0.5  # annualised factor within season
    elif period_type == PeriodType.EARTHQUAKE:
        seasonal_concentration = 1.0          # uniform
    else:
        seasonal_concentration = 1.0

    # Convert annual prob to instantaneous rate, scale, convert back
    annual_rate = -math.log(max(1.0 - annual_prob, 1e-12))
    period_rate = annual_rate * seasonal_concentration * remaining_fraction
    return 1.0 - math.exp(-period_rate)
