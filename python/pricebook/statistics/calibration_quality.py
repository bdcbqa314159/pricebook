"""Information-theoretic calibration quality metrics.

Applies entropy, KL divergence, and Fisher information to assess
the quality of model calibrations (vol surfaces, curves, etc.).

    from pricebook.statistics.calibration_quality import (
        calibration_entropy, calibration_kl, parameter_stability,
        model_comparison, CalibrationQualityResult,
    )

Key questions answered:
- How much information does the calibration extract from market data?
- How stable are calibrated parameters across recalibrations?
- Which model better represents the market (model selection via KL)?
- How well-identified are model parameters (Fisher information)?

References:
    Cont & Tankov (2004). Financial Modelling with Jump Processes.
    Avellaneda (1998). Minimum-Entropy Calibration of Asset-Pricing Models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.statistics.information_theory import (
    shannon_entropy, kl_divergence, fisher_information_matrix,
    cramer_rao_bound,
)


@dataclass
class CalibrationQualityResult:
    """Assessment of a model calibration."""
    rmse: float
    max_error: float
    mean_abs_error: float
    r_squared: float
    information_ratio: float
    entropy_residual: float
    n_instruments: int
    n_parameters: int
    degrees_of_freedom: int

    def to_dict(self) -> dict:
        return vars(self)

    @property
    def is_overfit(self) -> bool:
        """Heuristic: overfit if dof < 3 or info ratio < 1."""
        return self.degrees_of_freedom < 3 or self.information_ratio < 1.0


@dataclass
class ModelComparisonResult:
    """Result of comparing two model calibrations."""
    model_a_name: str
    model_b_name: str
    kl_a_to_b: float
    kl_b_to_a: float
    js_divergence: float
    preferred: str
    aic_a: float
    aic_b: float
    bic_a: float
    bic_b: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ParameterStabilityResult:
    """How stable calibrated parameters are across windows."""
    param_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    cv: np.ndarray  # coefficient of variation
    max_drift: np.ndarray  # max abs change between consecutive calibrations
    stability_score: float  # 0-1, higher = more stable

    def to_dict(self) -> dict:
        return {
            "param_names": self.param_names,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "cv": self.cv.tolist(),
            "stability_score": self.stability_score,
        }


def calibration_entropy(
    market_prices: np.ndarray,
    model_prices: np.ndarray,
    market_vols: np.ndarray | None = None,
    model_vols: np.ndarray | None = None,
) -> CalibrationQualityResult:
    """Assess calibration quality using information-theoretic metrics.

    Args:
        market_prices: (n,) observed market prices.
        model_prices: (n,) model-implied prices.
        market_vols: (n,) market implied vols (optional, used for vol-space metrics).
        model_vols: (n,) model implied vols (optional).
    """
    market_prices = np.asarray(market_prices, dtype=float)
    model_prices = np.asarray(model_prices, dtype=float)
    if len(market_prices) != len(model_prices):
        raise ValueError(f"market_prices ({len(market_prices)}) and model_prices ({len(model_prices)}) must have same length")
    errors = model_prices - market_prices
    n = len(errors)
    if n < 1:
        raise ValueError("Need at least 1 price for calibration assessment")

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(np.abs(errors)))
    mae = float(np.mean(np.abs(errors)))

    # R-squared
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((market_prices - np.mean(market_prices)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-20)

    # Entropy of residuals — lower = more structured residuals = worse
    # Good calibration: residuals are pure noise → high entropy
    abs_errors = np.abs(errors)
    if np.sum(abs_errors) > 0:
        p_residual = abs_errors / np.sum(abs_errors)
        ent = shannon_entropy(p_residual)
        # Normalise to [0, 1] vs uniform
        max_ent = math.log(n)
        entropy_ratio = ent / max_ent if max_ent > 0 else 1.0
    else:
        entropy_ratio = 1.0

    # Vol-space metrics if available
    if market_vols is not None and model_vols is not None:
        vol_errors = np.array(model_vols) - np.array(market_vols)
        vol_rmse = float(np.sqrt(np.mean(vol_errors ** 2)))
    else:
        vol_rmse = rmse

    return CalibrationQualityResult(
        rmse=rmse,
        max_error=max_err,
        mean_abs_error=mae,
        r_squared=r2,
        information_ratio=entropy_ratio * n,  # effective instruments
        entropy_residual=entropy_ratio,
        n_instruments=n,
        n_parameters=0,  # caller can set
        degrees_of_freedom=n,
    )


def calibration_kl(
    market_prices: np.ndarray,
    model_a_prices: np.ndarray,
    model_b_prices: np.ndarray,
) -> float:
    """KL-based comparison: which model is closer to market?

    Constructs empirical distributions from pricing errors and
    computes KL divergence. Lower KL = better fit.

    Returns KL(A || market) - KL(B || market):
        negative → model A is better
        positive → model B is better
    """
    err_a = np.abs(np.array(model_a_prices) - np.array(market_prices))
    err_b = np.abs(np.array(model_b_prices) - np.array(market_prices))

    # Convert to probability-like weights (softmax of negative errors)
    def _to_probs(x):
        x = np.maximum(x, 1e-15)
        p = x / x.sum()
        return np.maximum(p, 1e-15)

    # Uniform = ideal (all errors equal)
    uniform = np.ones(len(market_prices)) / len(market_prices)
    p_a = _to_probs(err_a)
    p_b = _to_probs(err_b)

    kl_a = kl_divergence(p_a, uniform)
    kl_b = kl_divergence(p_b, uniform)

    return kl_a - kl_b


def parameter_stability(
    param_history: np.ndarray,
    param_names: list[str] | None = None,
) -> ParameterStabilityResult:
    """Assess stability of calibrated parameters across time windows.

    Args:
        param_history: (T, n_params) matrix of calibrated parameters over T windows.
        param_names: optional parameter names.
    """
    param_history = np.atleast_2d(param_history)
    T, n = param_history.shape

    if param_names is None:
        param_names = [f"param_{i}" for i in range(n)]

    means = np.mean(param_history, axis=0)
    stds = np.std(param_history, axis=0)
    cv = stds / np.maximum(np.abs(means), 1e-10)

    # Max consecutive change
    if T > 1:
        diffs = np.abs(np.diff(param_history, axis=0))
        max_drift = np.max(diffs, axis=0)
    else:
        max_drift = np.zeros(n)

    # Stability score: average 1/(1 + cv)
    stability = float(np.mean(1.0 / (1.0 + cv)))

    return ParameterStabilityResult(
        param_names=param_names,
        means=means,
        stds=stds,
        cv=cv,
        max_drift=max_drift,
        stability_score=stability,
    )


def model_comparison(
    market_prices: np.ndarray,
    model_a_prices: np.ndarray,
    model_b_prices: np.ndarray,
    n_params_a: int,
    n_params_b: int,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> ModelComparisonResult:
    """Compare two model calibrations using AIC, BIC, and KL divergence.

    Uses Akaike and Bayesian Information Criteria alongside
    information-theoretic divergence measures.
    """
    market = np.asarray(market_prices, dtype=float)
    model_a = np.asarray(model_a_prices, dtype=float)
    model_b = np.asarray(model_b_prices, dtype=float)
    n = len(market)
    if n < 2:
        raise ValueError("Need at least 2 prices for model comparison")
    if len(model_a) != n or len(model_b) != n:
        raise ValueError("All price arrays must have same length")
    if n_params_a < 0 or n_params_b < 0:
        raise ValueError("n_params must be non-negative")

    err_a = model_a - market
    err_b = model_b - market

    sse_a = float(np.sum(err_a ** 2))
    sse_b = float(np.sum(err_b ** 2))

    # Log-likelihood (Gaussian errors)
    ll_a = -n / 2 * math.log(2 * math.pi) - n / 2 * math.log(max(sse_a / n, 1e-20)) - n / 2
    ll_b = -n / 2 * math.log(2 * math.pi) - n / 2 * math.log(max(sse_b / n, 1e-20)) - n / 2

    # AIC = -2 LL + 2k
    aic_a = -2 * ll_a + 2 * n_params_a
    aic_b = -2 * ll_b + 2 * n_params_b

    # BIC = -2 LL + k ln(n)
    bic_a = -2 * ll_a + n_params_a * math.log(n)
    bic_b = -2 * ll_b + n_params_b * math.log(n)

    # KL divergences (error distributions)
    abs_a = np.maximum(np.abs(err_a), 1e-15)
    abs_b = np.maximum(np.abs(err_b), 1e-15)
    p_a = abs_a / abs_a.sum()
    p_b = abs_b / abs_b.sum()
    p_a = np.maximum(p_a, 1e-15)
    p_b = np.maximum(p_b, 1e-15)

    kl_ab = kl_divergence(p_a, p_b)
    kl_ba = kl_divergence(p_b, p_a)
    js = 0.5 * kl_divergence(p_a, 0.5 * (p_a + p_b)) + 0.5 * kl_divergence(p_b, 0.5 * (p_a + p_b))

    preferred = model_a_name if bic_a < bic_b else model_b_name

    return ModelComparisonResult(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        kl_a_to_b=kl_ab,
        kl_b_to_a=kl_ba,
        js_divergence=js,
        preferred=preferred,
        aic_a=aic_a,
        aic_b=aic_b,
        bic_a=bic_a,
        bic_b=bic_b,
    )


def fisher_parameter_quality(
    model_func,
    params: np.ndarray,
    data: np.ndarray,
    param_names: list[str] | None = None,
) -> dict:
    """Assess parameter identification quality via Fisher information.

    Computes the Fisher Information Matrix (FIM) and Cramer-Rao bounds.
    Well-identified parameters have high FIM diagonal entries and tight CR bounds.

    Args:
        model_func: callable(params) → (n,) model predictions.
        params: (k,) calibrated parameter values.
        data: (n,) observed data.
        param_names: optional parameter names.
    """
    params = np.asarray(params, dtype=float)
    k = len(params)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(k)]

    residuals = np.array(data) - np.array(model_func(params))
    sigma2 = float(np.var(residuals))

    # Log-likelihood gradient: d/dθ log L = (1/σ²) Σ (y_i - f_i) df_i/dθ
    # FIM = (1/σ²) J'J where J is the Jacobian of model_func
    from pricebook.numerical._differentiate import jacobian, DiffMethod
    J_result = jacobian(model_func, params, DiffMethod.CENTRAL)
    J = J_result.value

    fim = J.T @ J / max(sigma2, 1e-20)
    cr = cramer_rao_bound(fim)

    # Condition number of FIM (high = ill-conditioned = poorly identified)
    fim_cond = float(np.linalg.cond(fim)) if np.linalg.matrix_rank(fim) == k else float('inf')

    return {
        "param_names": param_names,
        "fim_diagonal": np.diag(fim).tolist(),
        "cramer_rao_bounds": np.diag(cr).tolist(),
        "fim_condition": fim_cond,
        "well_identified": fim_cond < 1e6,
        "sigma2": sigma2,
    }
