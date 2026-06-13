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
    # Fix T4-RISK11: pre-fix silently returned f=0 when volatility=0,
    # masking two genuinely different degenerate cases: (i) excess > 0
    # with σ=0 is a deterministic free-money arbitrage (Kelly is +∞);
    # (ii) excess < 0 with σ=0 is a deterministic loss (Kelly = −∞,
    # short to infinity).  Returning 0 in either case is misleading
    # — it pretends the asset is unattractive when it's actually an
    # arbitrage.  Now raises ValueError so the caller is forced to
    # handle the degenerate input explicitly.
    if volatility <= 0:
        raise ValueError(
            f"kelly_fraction requires volatility > 0 (got {volatility}); "
            "a zero-volatility asset with non-zero excess return is a "
            "deterministic arbitrage where Kelly is undefined / infinite."
        )
    excess = expected_return - risk_free_rate
    var = volatility ** 2
    f_star = excess / var

    # Expected log growth at Kelly
    growth = risk_free_rate + f_star * excess - 0.5 * f_star**2 * var

    sharpe = excess / volatility

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
    # Fix T4-RISK12: pre-fix validated nothing and fell back to a
    # diagonal-only inverse on singular Σ.  Two problems:
    #   - No square/symmetry check: caller could pass a (3, 4) matrix
    #     or a non-symmetric one and get garbage out.
    #   - Diagonal-only fallback drops correlation entirely.  For a
    #     near-singular Σ from highly-correlated assets, the true Kelly
    #     concentrates weight on the best risk-adjusted asset; the
    #     diagonal fallback spreads incorrectly across all assets.
    # Now validates input shape and uses np.linalg.pinv (Moore-Penrose
    # pseudoinverse) which handles singular Σ gracefully and preserves
    # the correlation-aware structure.
    cov = np.asarray(cov, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2D matrix; got shape {cov.shape}")
    if cov.shape[0] != mu_arr.shape[0]:
        raise ValueError(
            f"cov shape {cov.shape} incompatible with mu length {mu_arr.shape[0]}"
        )
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("cov must be symmetric (within 1e-10 tolerance)")

    excess = mu_arr - risk_free_rate

    try:
        inv_cov = np.linalg.inv(cov)
        f_star = inv_cov @ excess
    except np.linalg.LinAlgError:
        # Pseudoinverse preserves correlation structure better than
        # diagonal-only — for rank-deficient Σ it returns the minimum-
        # norm solution to the underdetermined system.
        inv_cov = np.linalg.pinv(cov)
        f_star = inv_cov @ excess

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
