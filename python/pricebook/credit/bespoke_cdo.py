"""Bespoke/single-tranche CDO: custom portfolio, correlation calibration.

Bespoke CDO with arbitrary portfolio, calibration of base correlation
to tranche quotes, leveraged super-senior (LSS) tranches, and
tranche Greeks.

* :class:`BespokeCDOResult` — pricing result with tranche Greeks.
* :func:`bespoke_tranche_price` — price bespoke tranche.
* :func:`calibrate_bespoke_correlation` — calibrate to market quote.
* :func:`leveraged_super_senior` — LSS tranche pricing.
* :func:`tranche_greeks` — delta, gamma, vega for tranche.

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 10-11, 2008.
    Andersen & Sidenius, *Extensions to the Gaussian Copula*, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BespokeCDOResult:
    """Bespoke CDO tranche pricing result."""
    tranche_spread: float       # fair tranche spread (bp)
    expected_loss_pct: float
    attachment: float
    detachment: float
    base_correlation: float
    n_names: int
    # Greeks
    spread_delta: float = 0.0   # ∂spread / ∂portfolio_spread (per bp)
    correlation_delta: float = 0.0  # ∂spread / ∂correlation

    def to_dict(self) -> dict:
        return vars(self)


def _vasicek_loss_dist(
    pds: list[float],
    lgds: list[float],
    correlation: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Large-pool loss distribution via Vasicek one-factor model.

    Returns (loss_levels, probabilities).
    """
    from scipy.stats import norm

    n = len(pds)
    avg_pd = sum(pds) / n
    avg_lgd = sum(lgds) / n

    # Loss grid
    max_loss = avg_lgd
    loss_grid = np.linspace(0, max_loss, n_points)

    rho = max(min(correlation, 0.999), 0.001)
    sqrt_rho = math.sqrt(rho)
    sqrt_1_rho = math.sqrt(1 - rho)

    # Conditional default probability as function of systematic factor M
    m_grid = np.linspace(-4, 4, 200)
    pdf_m = np.exp(-0.5 * m_grid ** 2) / math.sqrt(2 * math.pi)
    dm = m_grid[1] - m_grid[0]

    loss_probs = np.zeros(n_points)
    for i_m, m in enumerate(m_grid):
        # Conditional PD
        cond_pd = norm.cdf((norm.ppf(avg_pd) - sqrt_rho * m) / sqrt_1_rho)
        # Conditional expected loss
        cond_el = cond_pd * avg_lgd
        # Map to loss grid
        idx = min(int(cond_el / max_loss * (n_points - 1)), n_points - 1)
        loss_probs[idx] += pdf_m[i_m] * dm

    # Normalise
    total = loss_probs.sum()
    if total > 0:
        loss_probs /= total

    return loss_grid, loss_probs


def bespoke_tranche_price(
    pds: list[float],
    lgds: list[float],
    notionals: list[float],
    attachment: float,
    detachment: float,
    correlation: float,
    maturity_years: float = 5.0,
    rate: float = 0.04,
) -> BespokeCDOResult:
    """Price a bespoke CDO tranche.

    Tranche loss = min(max(portfolio_loss − attachment, 0), detachment − attachment).

    Args:
        pds: per-name probability of default.
        lgds: per-name loss-given-default.
        notionals: per-name notional.
        attachment: lower attachment point (fraction of portfolio).
        detachment: upper detachment point.
        correlation: equi-correlation.
        maturity_years: tranche maturity.
        rate: risk-free rate.
    """
    total_notional = sum(notionals)
    width = detachment - attachment
    if width <= 0:
        return BespokeCDOResult(0, 0, attachment, detachment, correlation, len(pds))

    # Notional-weighted average PD and LGD
    avg_pd = sum(pd * n for pd, n in zip(pds, notionals)) / total_notional
    avg_lgd = sum(lgd * n for lgd, n in zip(lgds, notionals)) / total_notional

    # Loss distribution using notional-weighted averages
    loss_grid, loss_probs = _vasicek_loss_dist([avg_pd], [avg_lgd], correlation)

    # Tranche expected loss
    tranche_el = 0.0
    for loss, prob in zip(loss_grid, loss_probs):
        tranche_loss = min(max(loss - attachment, 0), width)
        tranche_el += tranche_loss * prob

    # Tranche spread ≈ EL / (width × annuity)
    annuity = sum(math.exp(-rate * t) for t in np.arange(0.25, maturity_years + 0.01, 0.25)) * 0.25
    tranche_spread_decimal = tranche_el / (width * annuity) if width * annuity > 0 else 0

    return BespokeCDOResult(
        tranche_spread=tranche_spread_decimal * 10_000,
        expected_loss_pct=tranche_el / width * 100 if width > 0 else 0,
        attachment=attachment,
        detachment=detachment,
        base_correlation=correlation,
        n_names=len(pds),
    )


def calibrate_bespoke_correlation(
    pds: list[float],
    lgds: list[float],
    notionals: list[float],
    attachment: float,
    detachment: float,
    market_spread_bp: float,
    maturity_years: float = 5.0,
    rate: float = 0.04,
    tol: float = 0.5,
    max_iter: int = 50,
) -> float:
    """Calibrate base correlation to match market tranche spread.

    Uses bisection to find the correlation that produces the
    observed market spread.

    Args:
        market_spread_bp: observed tranche spread in bp.

    Returns:
        Calibrated base correlation.
    """
    lo, hi = 0.01, 0.99

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        result = bespoke_tranche_price(
            pds, lgds, notionals, attachment, detachment, mid,
            maturity_years, rate,
        )
        if abs(result.tranche_spread - market_spread_bp) < tol:
            return mid
        if result.tranche_spread > market_spread_bp:
            lo = mid  # higher correlation → lower spread for senior tranches
        else:
            hi = mid

    return (lo + hi) / 2


@dataclass
class LSSResult:
    """Leveraged super-senior result."""
    leveraged_spread: float     # LSS spread in bp
    unleveraged_spread: float   # base super-senior spread
    leverage: float
    attachment: float
    detachment: float
    gap_risk_pct: float         # risk of loss exceeding collateral

    def to_dict(self) -> dict:
        return vars(self)


def leveraged_super_senior(
    pds: list[float],
    lgds: list[float],
    notionals: list[float],
    attachment: float,
    leverage: float = 10.0,
    correlation: float = 0.30,
    maturity_years: float = 5.0,
    rate: float = 0.04,
) -> LSSResult:
    """Price a leveraged super-senior (LSS) tranche.

    LSS: investor posts collateral = notional/leverage.
    Receives leveraged spread but bears first loss on their collateral.

    Args:
        attachment: super-senior attachment (e.g. 0.15 for 15-100%).
        leverage: leverage ratio (e.g. 10× means 10% collateral).
    """
    base = bespoke_tranche_price(
        pds, lgds, notionals, attachment, 1.0, correlation,
        maturity_years, rate,
    )

    leveraged_spread = base.tranche_spread * leverage

    # Gap risk: probability that losses exceed collateral
    collateral_pct = 1.0 / leverage
    loss_grid, loss_probs = _vasicek_loss_dist(pds, lgds, correlation)
    gap_prob = sum(p for l, p in zip(loss_grid, loss_probs) if l > attachment + collateral_pct * (1 - attachment))

    return LSSResult(
        leveraged_spread=leveraged_spread,
        unleveraged_spread=base.tranche_spread,
        leverage=leverage,
        attachment=attachment,
        detachment=1.0,
        gap_risk_pct=gap_prob * 100,
    )


def tranche_greeks(
    pds: list[float],
    lgds: list[float],
    notionals: list[float],
    attachment: float,
    detachment: float,
    correlation: float,
    maturity_years: float = 5.0,
    rate: float = 0.04,
) -> BespokeCDOResult:
    """Compute tranche Greeks: spread delta and correlation delta."""
    base = bespoke_tranche_price(
        pds, lgds, notionals, attachment, detachment, correlation,
        maturity_years, rate,
    )

    # Spread delta: bump all PDs by 1bp equivalent
    bump_factor = 1.01
    bumped_pds = [pd * bump_factor for pd in pds]
    up = bespoke_tranche_price(
        bumped_pds, lgds, notionals, attachment, detachment, correlation,
        maturity_years, rate,
    )
    base.spread_delta = up.tranche_spread - base.tranche_spread

    # Correlation delta
    dc = 0.01
    corr_up = bespoke_tranche_price(
        pds, lgds, notionals, attachment, detachment,
        min(correlation + dc, 0.99), maturity_years, rate,
    )
    base.correlation_delta = (corr_up.tranche_spread - base.tranche_spread) / dc

    return base
