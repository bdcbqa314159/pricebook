"""Kelly criterion: optimal bet sizing for long-run growth.

* :func:`kelly_fraction` — single-asset Kelly.
* :func:`fractional_kelly` — conservative Kelly with fraction.
* :func:`multi_asset_kelly` — portfolio Kelly via mean-variance.

References:
    Kelly, *A New Interpretation of Information Rate*, Bell Sys. Tech. J., 1956.
    Thorp, *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class KellyResult:
    """Kelly criterion result."""
    kelly_fraction: float
    expected_growth: float      # log growth rate
    expected_return: float
    volatility: float
    sharpe_ratio: float

    def to_dict(self) -> dict:
        return vars(self)


def kelly_fraction(
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> KellyResult:
    """Single-asset Kelly fraction: f* = μ/σ².

    For a single risky asset with return μ and variance σ²,
    the Kelly fraction maximises E[log(wealth)].

    f* = (μ − rf) / σ²

    Args:
        expected_return: expected arithmetic return.
        volatility: return standard deviation.
        risk_free_rate: risk-free rate.
    """
    excess = expected_return - risk_free_rate
    var = volatility ** 2
    f_star = excess / var if var > 0 else 0

    # Expected log growth at Kelly
    growth = risk_free_rate + f_star * excess - 0.5 * f_star**2 * var

    sharpe = excess / volatility if volatility > 0 else 0

    return KellyResult(
        kelly_fraction=f_star,
        expected_growth=growth,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def fractional_kelly(
    expected_return: float,
    volatility: float,
    fraction: float = 0.5,
    risk_free_rate: float = 0.0,
) -> KellyResult:
    """Fractional Kelly: f = fraction × f*.

    In practice, full Kelly is too aggressive. Half-Kelly (f=0.5)
    achieves ~75% of the growth rate with ~50% of the drawdown.

    Args:
        fraction: Kelly fraction (0.5 = half-Kelly).
    """
    full = kelly_fraction(expected_return, volatility, risk_free_rate)
    f = fraction * full.kelly_fraction

    excess = expected_return - risk_free_rate
    growth = risk_free_rate + f * excess - 0.5 * f**2 * volatility**2

    return KellyResult(
        kelly_fraction=f,
        expected_growth=growth,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=full.sharpe_ratio,
    )


def multi_asset_kelly(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
    fraction: float = 1.0,
    max_leverage: float = 5.0,
) -> dict:
    """Multi-asset Kelly: f* = Σ⁻¹(μ − rf).

    For N assets: f* = Σ⁻¹ × excess_returns.
    This is identical to the maximum-growth portfolio.

    Args:
        mu: expected returns (N,).
        cov: covariance matrix (N, N).
        fraction: Kelly fraction (1.0 = full).
        max_leverage: cap total position.
    """
    excess = mu - risk_free_rate

    try:
        inv_cov = np.linalg.inv(cov)
        f_star = inv_cov @ excess
    except np.linalg.LinAlgError:
        f_star = excess / np.diag(cov)

    # Apply fraction
    f = fraction * f_star

    # Cap leverage
    total_leverage = float(np.sum(np.abs(f)))
    if total_leverage > max_leverage:
        f *= max_leverage / total_leverage

    # Expected growth
    growth = risk_free_rate + float(f @ excess) - 0.5 * float(f @ cov @ f)

    return {
        "weights": f.tolist(),
        "kelly_fraction": fraction,
        "expected_growth": growth,
        "leverage": float(np.sum(np.abs(f))),
        "n_assets": len(mu),
    }
