"""Stochastic correlation for multi-name credit structures.

Time-varying correlation for CDO/CLO tranche pricing, with
regime-switching and correlation smile calibration.

* :class:`StochasticCorrelationResult` — pricing result.
* :func:`regime_switching_correlation` — two-regime correlation.
* :func:`correlation_smile` — implied correlation across tranches.
* :func:`stochastic_corr_tranche` — tranche pricing with stoch corr.

References:
    Burtschell, Gregory & Laurent, *A Comparative Analysis of CDO
    Pricing Models*, 2009.
    Andersen & Sidenius, *Extensions to the Gaussian Copula*, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class StochasticCorrelationResult:
    """Tranche pricing with stochastic correlation."""
    tranche_spread: float
    expected_loss_pct: float
    base_correlation: float
    regime_probs: list[float]
    regime_correlations: list[float]
    attachment: float
    detachment: float

    def to_dict(self) -> dict:
        return {
            "tranche_spread": self.tranche_spread,
            "expected_loss_pct": self.expected_loss_pct,
            "base_correlation": self.base_correlation,
            "attachment": self.attachment,
            "detachment": self.detachment,
            "n_regimes": len(self.regime_correlations),
        }


def regime_switching_correlation(
    avg_pd: float,
    avg_lgd: float,
    attachment: float,
    detachment: float,
    correlations: list[float],
    regime_probs: list[float],
    maturity_years: float = 5.0,
    rate: float = 0.04,
) -> StochasticCorrelationResult:
    """Tranche pricing with regime-switching correlation.

    Two (or more) correlation regimes with given probabilities.
    Tranche EL = Σ prob_k × EL(correlation_k).

    This captures the correlation smile: senior tranches see
    higher implied correlation than mezzanine.

    Args:
        avg_pd: portfolio average default probability.
        avg_lgd: portfolio average LGD.
        attachment: lower attachment.
        detachment: upper detachment.
        correlations: correlation per regime.
        regime_probs: probability of each regime.
    """
    width = detachment - attachment
    if width <= 0:
        return StochasticCorrelationResult(0, 0, 0, regime_probs, correlations, attachment, detachment)

    # Weighted tranche EL across regimes
    total_el = 0.0
    for corr, prob in zip(correlations, regime_probs):
        el = _vasicek_tranche_el(avg_pd, avg_lgd, corr, attachment, detachment)
        total_el += prob * el

    # Spread from EL
    annuity = sum(math.exp(-rate * t) for t in np.arange(0.25, maturity_years + 0.01, 0.25)) * 0.25
    spread = total_el / (width * annuity) if width * annuity > 0 else 0

    avg_corr = sum(c * p for c, p in zip(correlations, regime_probs))

    return StochasticCorrelationResult(
        tranche_spread=spread * 10_000,
        expected_loss_pct=total_el / width * 100,
        base_correlation=avg_corr,
        regime_probs=regime_probs,
        regime_correlations=correlations,
        attachment=attachment,
        detachment=detachment,
    )


@dataclass
class CorrelationSmilePoint:
    """Single point on the correlation smile."""
    attachment: float
    detachment: float
    implied_correlation: float
    tranche_spread: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def correlation_smile(
    avg_pd: float,
    avg_lgd: float,
    tranche_spreads_bp: list[float],
    attachments: list[float],
    detachments: list[float],
    maturity_years: float = 5.0,
    rate: float = 0.04,
) -> list[CorrelationSmilePoint]:
    """Calibrate implied correlation smile from tranche spreads.

    For each tranche, finds the flat correlation that matches
    the observed spread. The resulting curve (correlation vs
    attachment) is the correlation smile.

    Args:
        tranche_spreads_bp: observed tranche spreads in bp.
        attachments: lower attachment per tranche.
        detachments: upper detachment per tranche.
    """
    results = []
    for spread_bp, att, det in zip(tranche_spreads_bp, attachments, detachments):
        corr = _calibrate_single_corr(avg_pd, avg_lgd, att, det, spread_bp, maturity_years, rate)
        results.append(CorrelationSmilePoint(att, det, corr, spread_bp))
    return results


def stochastic_corr_tranche(
    avg_pd: float,
    avg_lgd: float,
    attachment: float,
    detachment: float,
    corr_mean: float = 0.30,
    corr_vol: float = 0.10,
    maturity_years: float = 5.0,
    rate: float = 0.04,
    n_sims: int = 10_000,
    seed: int = 42,
) -> StochasticCorrelationResult:
    """Tranche pricing with continuously stochastic correlation.

    Correlation drawn from a beta distribution with given mean and vol.

    Args:
        corr_mean: mean correlation.
        corr_vol: vol of correlation.
    """
    rng = np.random.default_rng(seed)
    width = detachment - attachment

    # Beta distribution parameters from mean and vol
    a_param, b_param = _beta_params(corr_mean, corr_vol)

    total_el = 0.0
    for _ in range(n_sims):
        corr = float(rng.beta(a_param, b_param))
        corr = max(0.01, min(corr, 0.99))
        el = _vasicek_tranche_el(avg_pd, avg_lgd, corr, attachment, detachment)
        total_el += el

    avg_el = total_el / n_sims
    annuity = sum(math.exp(-rate * t) for t in np.arange(0.25, maturity_years + 0.01, 0.25)) * 0.25
    spread = avg_el / (width * annuity) if width * annuity > 0 else 0

    return StochasticCorrelationResult(
        tranche_spread=spread * 10_000,
        expected_loss_pct=avg_el / width * 100 if width > 0 else 0,
        base_correlation=corr_mean,
        regime_probs=[1.0],
        regime_correlations=[corr_mean],
        attachment=attachment,
        detachment=detachment,
    )


# ---- Internal helpers ----

def _vasicek_tranche_el(
    avg_pd: float, avg_lgd: float, corr: float,
    attachment: float, detachment: float,
    n_points: int = 200,
) -> float:
    """Vasicek one-factor tranche expected loss."""
    rho = max(min(corr, 0.999), 0.001)
    sqrt_rho = math.sqrt(rho)
    sqrt_1_rho = math.sqrt(1 - rho)
    width = detachment - attachment

    m_grid = np.linspace(-4, 4, n_points)
    dm = m_grid[1] - m_grid[0]
    pdf_m = np.exp(-0.5 * m_grid**2) / math.sqrt(2 * math.pi)

    tranche_el = 0.0
    for m, pdf in zip(m_grid, pdf_m):
        cond_pd = norm.cdf((norm.ppf(avg_pd) - sqrt_rho * m) / sqrt_1_rho)
        cond_loss = cond_pd * avg_lgd
        tranche_loss = min(max(cond_loss - attachment, 0), width)
        tranche_el += tranche_loss * pdf * dm

    return tranche_el


def _calibrate_single_corr(
    avg_pd, avg_lgd, att, det, target_bp, mat, rate,
    tol=0.5, max_iter=50,
):
    """Bisection for single tranche implied correlation."""
    lo, hi = 0.01, 0.99
    width = det - att
    annuity = sum(math.exp(-rate * t) for t in np.arange(0.25, mat + 0.01, 0.25)) * 0.25

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        el = _vasicek_tranche_el(avg_pd, avg_lgd, mid, att, det)
        spread_bp = el / (width * annuity) * 10_000 if width * annuity > 0 else 0

        if abs(spread_bp - target_bp) < tol:
            return mid
        if spread_bp > target_bp:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def _beta_params(mean: float, vol: float) -> tuple[float, float]:
    """Beta distribution parameters from mean and vol."""
    mean = max(0.01, min(mean, 0.99))
    var = min(vol**2, mean * (1 - mean) * 0.99)
    if var <= 0:
        return (10.0, 10.0 * (1 - mean) / mean)
    common = mean * (1 - mean) / var - 1
    a = mean * common
    b = (1 - mean) * common
    return (max(a, 0.1), max(b, 0.1))
