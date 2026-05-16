"""Advanced regression: Ridge, Lasso, Elastic Net, quantile, robust.

    from pricebook.statistics.regression import ridge, lasso, elastic_net, quantile_regression, robust_regression

References:
    Hoerl & Kennard (1970). Ridge Regression.
    Tibshirani (1996). Regression Shrinkage and Selection via the Lasso.
    Koenker & Bassett (1978). Regression Quantiles.
    Huber (1973). Robust Regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionResult:
    """Regression fit result."""
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    residuals: np.ndarray
    method: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "r_squared": self.r_squared,
        }


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Add column of ones for intercept."""
    return np.column_stack([np.ones(X.shape[0]), X])


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-15)


def ols(X: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Ordinary Least Squares: beta = (X'X)^{-1} X'y."""
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(y, dtype=float)
    X_int = _add_intercept(X)
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    y_hat = X_int @ beta
    return RegressionResult(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        r_squared=_r_squared(y, y_hat),
        residuals=y - y_hat,
        method="ols",
    )


def ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> RegressionResult:
    """Ridge regression: beta = (X'X + alpha I)^{-1} X'y.

    L2 penalty shrinks coefficients toward zero.
    Larger alpha → more regularisation.
    """
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(y, dtype=float)
    X_int = _add_intercept(X)
    n, p = X_int.shape
    penalty = alpha * np.eye(p)
    penalty[0, 0] = 0  # don't penalise intercept
    beta = np.linalg.solve(X_int.T @ X_int + penalty, X_int.T @ y)
    y_hat = X_int @ beta
    return RegressionResult(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        r_squared=_r_squared(y, y_hat),
        residuals=y - y_hat,
        method="ridge",
    )


def lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> RegressionResult:
    """Lasso regression via coordinate descent.

    L1 penalty: some coefficients become exactly zero (feature selection).
    """
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(y, dtype=float)
    X_int = _add_intercept(X)
    n, p = X_int.shape

    beta = np.zeros(p)
    beta[0] = y.mean()

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r = y - X_int @ beta + X_int[:, j] * beta[j]
            rho = X_int[:, j] @ r / n
            if j == 0:
                beta[j] = rho  # no penalty on intercept
            else:
                beta[j] = _soft_threshold(rho, alpha / n)
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    y_hat = X_int @ beta
    return RegressionResult(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        r_squared=_r_squared(y, y_hat),
        residuals=y - y_hat,
        method="lasso",
    )


def _soft_threshold(x: float, lam: float) -> float:
    """Soft thresholding operator for Lasso."""
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    return 0.0


def elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> RegressionResult:
    """Elastic Net: l1_ratio × Lasso + (1-l1_ratio) × Ridge.

    Combines L1 (sparsity) and L2 (grouping) penalties.
    """
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(y, dtype=float)
    X_int = _add_intercept(X)
    n, p = X_int.shape

    beta = np.zeros(p)
    beta[0] = y.mean()
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r = y - X_int @ beta + X_int[:, j] * beta[j]
            rho = X_int[:, j] @ r / n
            if j == 0:
                beta[j] = rho
            else:
                xj_sq = (X_int[:, j] ** 2).sum() / n
                beta[j] = _soft_threshold(rho, l1 / n) / (xj_sq + l2 / n)
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    y_hat = X_int @ beta
    return RegressionResult(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        r_squared=_r_squared(y, y_hat),
        residuals=y - y_hat,
        method="elastic_net",
    )


def quantile_regression(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.5,
    max_iter: int = 100,
) -> RegressionResult:
    """Quantile regression via iteratively reweighted least squares.

    Minimises: sum rho_q(y - X beta) where rho_q(u) = u(q - I(u<0)).
    q=0.5 gives median regression (robust to outliers).
    """
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(y, dtype=float)
    X_int = _add_intercept(X)
    n, p = X_int.shape

    # Start from OLS
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X_int @ beta
        # Weights for IRLS
        w = np.where(resid >= 0, quantile, 1 - quantile)
        w /= np.maximum(np.abs(resid), 1e-6)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X_int.T @ W @ X_int, X_int.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            break
        beta = beta_new

    y_hat = X_int @ beta
    return RegressionResult(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        r_squared=_r_squared(y, y_hat),
        residuals=y - y_hat,
        method=f"quantile_{quantile}",
    )


def robust_regression(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "huber",
    max_iter: int = 50,
    c: float = 1.345,
) -> RegressionResult:
    """Robust regression via iteratively reweighted least squares.

    Args:
        method: 'huber' (default) or 'tukey' (bisquare).
        c: tuning constant (1.345 for Huber, 4.685 for Tukey).
    """
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(y, dtype=float)
    X_int = _add_intercept(X)
    n, p = X_int.shape

    # Start from OLS
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]

    if method == "tukey":
        c = 4.685

    for _ in range(max_iter):
        resid = y - X_int @ beta
        sigma = np.median(np.abs(resid)) * 1.4826  # MAD estimate
        sigma = max(sigma, 1e-10)
        u = resid / sigma

        if method == "huber":
            w = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
        elif method == "tukey":
            w = np.where(np.abs(u) <= c, (1 - (u / c) ** 2) ** 2, 0.0)
        else:
            raise ValueError(f"unknown method: {method!r}")

        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X_int.T @ W @ X_int, X_int.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            break
        beta = beta_new

    y_hat = X_int @ beta
    return RegressionResult(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        r_squared=_r_squared(y, y_hat),
        residuals=y - y_hat,
        method=f"robust_{method}",
    )
