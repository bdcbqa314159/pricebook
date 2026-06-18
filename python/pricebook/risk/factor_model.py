"""Cross-asset factor models: construction, attribution, covariance, timing.

    from pricebook.risk.factor_model import (
        build_factors, factor_attribution, factor_covariance, factor_timing,
    )

    factors = build_factors(returns, factor_type="momentum", window=60)
    attrib = factor_attribution(portfolio_returns, factor_returns)
    cov = factor_covariance(factor_returns, method="shrinkage")

References:
    Ang, *Asset Management*, OUP, 2014.
    Ilmanen, *Expected Returns*, Wiley, 2011.
    Asness, Moskowitz & Pedersen, *Value and Momentum Everywhere*, JF, 2013.
    Ledoit & Wolf, *A Well-Conditioned Estimator for Large Covariance Matrices*, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ============================================================================
# Factor construction
# ============================================================================

@dataclass
class FactorResult:
    """Constructed factor series."""
    name: str
    values: np.ndarray        # factor signal at each time step
    z_scores: np.ndarray      # standardised signal
    percentiles: np.ndarray   # percentile rank



    def to_dict(self) -> dict:
        return dict(vars(self))
def zscore(x: np.ndarray, window: int | None = None) -> np.ndarray:
    """Rolling z-score. If window=None, use full-sample."""
    if window is None:
        mu = x.mean()
        sigma = x.std()
        return (x - mu) / sigma if sigma > 1e-10 else np.zeros_like(x)
    z = np.zeros_like(x)
    for i in range(window, len(x)):
        w = x[i - window:i]
        mu = w.mean()
        sigma = w.std()
        z[i] = (x[i] - mu) / sigma if sigma > 1e-10 else 0.0
    return z


def percentile_rank(x: np.ndarray, window: int | None = None) -> np.ndarray:
    """Rolling percentile rank (0 to 1)."""
    from scipy.stats import rankdata
    if window is None:
        ranks = rankdata(x)
        return ranks / len(x)
    pct = np.zeros_like(x)
    for i in range(window, len(x)):
        w = x[i - window:i + 1]
        ranks = rankdata(w)
        pct[i] = ranks[-1] / len(w)
    return pct


def build_factor(
    returns: np.ndarray,
    factor_type: str,
    window: int = 60,
    name: str | None = None,
) -> FactorResult:
    """Build a single factor from return series.

    Factor types:
        - "momentum": cumulative return over window
        - "carry": mean return over window (proxy for yield)
        - "value": negative of cumulative return (mean reversion)
        - "vol": rolling volatility (low vol = attractive)
        - "quality": Sharpe-like (mean/vol over window)

        factor = build_factor(returns, "momentum", window=120)
    """
    n = len(returns)
    values = np.zeros(n)

    if factor_type == "momentum":
        for i in range(window, n):
            values[i] = returns[i - window:i].sum()
    elif factor_type == "carry":
        for i in range(window, n):
            values[i] = returns[i - window:i].mean()
    elif factor_type == "value":
        for i in range(window, n):
            values[i] = -returns[i - window:i].sum()
    elif factor_type == "vol":
        for i in range(window, n):
            values[i] = -returns[i - window:i].std()  # negative: low vol = high signal
    elif factor_type == "quality":
        for i in range(window, n):
            w = returns[i - window:i]
            mu = w.mean()
            sigma = w.std()
            values[i] = mu / sigma if sigma > 1e-10 else 0.0
    else:
        raise ValueError(f"Unknown factor_type: {factor_type}. "
                         f"Use 'momentum', 'carry', 'value', 'vol', 'quality'.")

    z = zscore(values, window)
    pct = percentile_rank(values, window)

    return FactorResult(
        name=name or factor_type,
        values=values,
        z_scores=z,
        percentiles=pct,
    )


def build_multi_asset_factors(
    asset_returns: dict[str, np.ndarray],
    factor_types: list[str] | None = None,
    window: int = 60,
) -> dict[str, dict[str, FactorResult]]:
    """Build factors across multiple assets.

        factors = build_multi_asset_factors(
            {"SPX": spx_ret, "UST10Y": ust_ret, "EURUSD": fx_ret},
            factor_types=["momentum", "carry", "vol"])
    """
    if factor_types is None:
        factor_types = ["momentum", "carry", "value", "vol", "quality"]

    result = {}
    for ft in factor_types:
        result[ft] = {}
        for asset, ret in asset_returns.items():
            result[ft][asset] = build_factor(ret, ft, window, f"{asset}_{ft}")

    return result


# ============================================================================
# Factor attribution
# ============================================================================

@dataclass
class FactorAttributionResult:
    """Factor P&L attribution."""
    alpha: float                        # intercept (unexplained return)
    betas: dict[str, float]             # factor loadings
    r_squared: float                    # fraction of variance explained
    factor_contributions: dict[str, float]  # beta_i × mean(factor_i)
    residual_vol: float



    def to_dict(self) -> dict:
        return dict(vars(self))
def factor_attribution(
    portfolio_returns: np.ndarray,
    factor_returns: dict[str, np.ndarray],
) -> FactorAttributionResult:
    """Decompose portfolio returns by factor exposures via OLS.

    portfolio_return = α + Σ β_i × factor_return_i + ε

        attrib = factor_attribution(port_ret, {"momentum": mom_ret, "carry": carry_ret})
    """
    n = len(portfolio_returns)
    factor_names = list(factor_returns.keys())
    k = len(factor_names)

    # Build design matrix [1, f1, f2, ...]
    X = np.column_stack([np.ones(n)] + [factor_returns[f][:n] for f in factor_names])
    y = portfolio_returns[:n]

    # OLS: β = (X'X)^{-1} X'y
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        coeffs = np.zeros(k + 1)

    alpha = float(coeffs[0])
    betas = {f: float(coeffs[i + 1]) for i, f in enumerate(factor_names)}

    # R-squared
    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    # Contributions
    contributions = {}
    for f in factor_names:
        contributions[f] = betas[f] * float(factor_returns[f][:n].mean())

    residual_vol = float(np.std(y - y_hat)) if n > 1 else 0.0

    return FactorAttributionResult(alpha, betas, r2, contributions, residual_vol)


# ============================================================================
# Factor covariance
# ============================================================================

@dataclass
class FactorCovarianceResult:
    """Factor covariance matrix."""
    covariance: np.ndarray
    correlation: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    method: str
    names: list[str]



    def to_dict(self) -> dict:
        return dict(vars(self))
def factor_covariance(
    factor_returns: dict[str, np.ndarray],
    method: str = "sample",
    shrinkage_target: str = "identity",
) -> FactorCovarianceResult:
    """Compute factor covariance matrix.

    Methods:
        - "sample": standard sample covariance
        - "shrinkage": Ledoit-Wolf shrinkage toward target

        cov = factor_covariance({"mom": mom_ret, "carry": carry_ret}, method="shrinkage")
    """
    names = list(factor_returns.keys())
    n_factors = len(names)
    data = np.column_stack([factor_returns[f] for f in names])
    n = data.shape[0]

    if method == "sample":
        cov = np.cov(data, rowvar=False)
    elif method == "shrinkage":
        # Fix T4-RISK7: pre-fix used the ad-hoc intensity
        # ``alpha_lw = 1 / (n · delta)`` where ``delta = ||S − F||²``
        # — this has NO basis in Ledoit-Wolf (2004) and is dimensionally
        # inconsistent (units of (cov-squared)^-1).  The result was not
        # actually a Ledoit-Wolf shrunk estimator.  Now uses the
        # *correct* LW formula for the μ·I target (Ledoit & Wolf 2004,
        # §3.2):
        #   π̂ = (1/T) Σ_t Σ_{i,j} (x̃_ti x̃_tj − s_ij)²   (sum form)
        #   γ̂² = ||F − S||²_F
        #   ρ̂ = trace of the π̂ matrix   (identity-target case)
        #   κ = (π̂ − ρ̂) / γ̂²
        #   δ* = max(0, min(κ / T, 1))
        # Returns sample_cov when γ̂² is ~0 (sample equals target).
        sample_cov = np.cov(data, rowvar=False)
        if shrinkage_target == "identity":
            target = np.eye(n_factors) * np.trace(sample_cov) / n_factors
        elif shrinkage_target == "diagonal":
            target = np.diag(np.diag(sample_cov))
        else:
            target = np.eye(n_factors) * np.trace(sample_cov) / n_factors

        T_obs = n
        X_centered = data - data.mean(axis=0)
        # MLE sample cov (T divisor, not T-1) — what the LW derivation uses.
        S_mle = (X_centered.T @ X_centered) / T_obs
        # π̂ matrix vectorised:
        #   π̂[i,j] = (1/T) Σ_t (x̃_ti x̃_tj − S_mle[i,j])²
        #          = (1/T) Σ_t (x̃_ti x̃_tj)² − S_mle[i,j]²   (cross-term cancels)
        X_sq = X_centered ** 2
        pi_mat = (X_sq.T @ X_sq) / T_obs - S_mle ** 2
        pi_hat = pi_mat.sum()
        rho_hat = np.trace(pi_mat)  # identity-target simplification
        gamma_sq = float(np.sum((target - S_mle) ** 2))

        if gamma_sq < 1e-15:
            delta_star = 0.0
        else:
            kappa = (pi_hat - rho_hat) / gamma_sq
            delta_star = max(0.0, min(kappa / T_obs, 1.0))

        cov = delta_star * target + (1.0 - delta_star) * sample_cov
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure symmetry
    cov = (cov + cov.T) / 2

    vols = np.sqrt(np.diag(cov))
    outer = np.outer(vols, vols)
    corr = cov / outer if np.all(outer > 1e-10) else np.eye(n_factors)

    eigvals = np.linalg.eigvalsh(cov)
    cond = float(eigvals.max() / eigvals.min()) if eigvals.min() > 1e-15 else float('inf')

    return FactorCovarianceResult(cov, corr, eigvals, cond, method, names)


# ============================================================================
# Factor timing
# ============================================================================

@dataclass
class FactorTimingResult:
    """Factor timing signal."""
    factor: str
    current_z: float
    signal: str              # "overweight", "neutral", "underweight"
    historical_hit: float    # hit rate of this signal historically



    def to_dict(self) -> dict:
        return dict(vars(self))
def factor_timing(
    factor_values: np.ndarray,
    factor_returns: np.ndarray,
    current_value: float | None = None,
    z_threshold: float = 1.0,
) -> FactorTimingResult:
    """Factor timing: over/underweight based on z-score of factor value.

    Contrarian convention (Asness, Ilmanen):
      - Low z-score ⇒ factor value is depressed vs history ⇒ "cheap"
        ⇒ OVERWEIGHT (expect mean reversion higher).
      - High z-score ⇒ "expensive" ⇒ UNDERWEIGHT.

        timing = factor_timing(mom_values, mom_returns)

    Fix T4-RISK8: pre-fix used the opposite (momentum-style) signal
    direction: ``if current_z > z_threshold: signal = "overweight"``.
    That contradicted the function's own docstring ("low z-score,
    overweight").  The hit-rate calculation was also reversed
    accordingly.  Both have been swapped to match the contrarian
    convention.
    """
    z = zscore(factor_values)
    current_z = float(z[-1]) if current_value is None else float(
        (current_value - factor_values.mean()) / factor_values.std()
        if factor_values.std() > 1e-10 else 0.0
    )

    if current_z < -z_threshold:
        signal = "overweight"
    elif current_z > z_threshold:
        signal = "underweight"
    else:
        signal = "neutral"

    # Historical hit rate for this signal — contrarian view:
    #   z < -threshold (overweight) is "correct" if next return is positive.
    #   z > +threshold (underweight) is "correct" if next return is negative.
    n = len(factor_returns)
    correct = 0
    total = 0
    for i in range(1, n):
        if z[i - 1] < -z_threshold:  # overweight signal
            total += 1
            if factor_returns[i] > 0:
                correct += 1
        elif z[i - 1] > z_threshold:  # underweight signal
            total += 1
            if factor_returns[i] < 0:
                correct += 1
    hit = correct / total if total > 0 else 0.5

    return FactorTimingResult("factor", current_z, signal, hit)
