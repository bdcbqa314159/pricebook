"""Information theory for quantitative finance.

Entropy, divergence, mutual information, Fisher information — tools for
model risk quantification, feature selection, and calibration quality.

    from pricebook.statistics.information_theory import (
        shannon_entropy, kl_divergence, mutual_information,
        fisher_information_matrix, cramer_rao_bound,
    )

References:
    Cover & Thomas (2006). Elements of Information Theory.
    Buchen & Kelly (1996). The Maximum Entropy Distribution of an Asset.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════
# 2.1: Entropy and Divergence
# ═══════════════════════════════════════════════════════════════


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution.

    H(P) = -Σ pᵢ log pᵢ (in nats, base e).
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # skip zero entries
    return float(-np.dot(p, np.log(p)))


def differential_entropy(
    samples: np.ndarray,
    method: str = "kde",
    n_bins: int = 50,
) -> float:
    """Differential (continuous) entropy from samples.

    Args:
        samples: (N,) sample array.
        method: "kde" (kernel density) or "histogram".

    Returns:
        Estimated entropy in nats.
    """
    x = np.asarray(samples, dtype=float)
    if method == "histogram":
        counts, edges = np.histogram(x, bins=n_bins, density=True)
        dx = edges[1] - edges[0]
        p = counts * dx
        p = p[p > 0]
        return float(-np.dot(p, np.log(p / dx)) * dx)
    else:
        # KDE with Silverman bandwidth
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x)
        eval_points = np.linspace(x.min() - 3 * x.std(), x.max() + 3 * x.std(), 500)
        pdf = kde(eval_points)
        dx = eval_points[1] - eval_points[0]
        pdf = pdf[pdf > 0]
        return float(-np.sum(pdf * np.log(pdf)) * dx)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence: KL(P‖Q) = Σ pᵢ log(pᵢ/qᵢ).

    Measures how much P diverges from Q. Not symmetric.
    KL = 0 iff P = Q. KL > 0 always.

    Use case: model risk = KL(historical ‖ calibrated).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (p > 0) & (q > 0)
    return float(np.dot(p[mask], np.log(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded).

    JS(P‖Q) = 0.5 × KL(P‖M) + 0.5 × KL(Q‖M), where M = 0.5(P+Q).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy: H(P, Q) = -Σ pᵢ log qᵢ.

    H(P, Q) = H(P) + KL(P‖Q). Minimising cross-entropy = minimising KL.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (p > 0) & (q > 0)
    return float(-np.dot(p[mask], np.log(q[mask])))


def wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein-1 distance (earth mover's distance) for 1D distributions.

    W₁(P, Q) = ∫|F_P(x) - F_Q(x)| dx ≈ Σ|CDF_P - CDF_Q| × Δx.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


# ═══════════════════════════════════════════════════════════════
# 2.2: Mutual Information and Feature Selection
# ═══════════════════════════════════════════════════════════════


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y).

    Estimated via histogram binning.

    Use case: which financial ratios best predict default?
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_xy / hist_xy.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)
    h_xy = shannon_entropy(p_xy.flatten())

    return max(h_x + h_y - h_xy, 0.0)


def conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Conditional mutual information: I(X;Y|Z).

    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).

    Use case: does X add information about Y beyond what Z provides?
    """
    x, y, z = [np.asarray(a, dtype=float) for a in [x, y, z]]

    hist_xz, _ = np.histogramdd(np.column_stack([x, z]), bins=n_bins)
    hist_yz, _ = np.histogramdd(np.column_stack([y, z]), bins=n_bins)
    hist_xyz, _ = np.histogramdd(np.column_stack([x, y, z]), bins=n_bins)
    hist_z, _ = np.histogram(z, bins=n_bins)

    def h(counts):
        p = counts.flatten() / max(counts.sum(), 1e-15)
        return shannon_entropy(p)

    return max(h(hist_xz) + h(hist_yz) - h(hist_xyz) - h(hist_z), 0.0)


def information_gain(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Rank features by mutual information with target.

    Args:
        features: (N, D) feature matrix.
        target: (N,) target variable (e.g., default indicator, regime label).
        feature_names: optional names for features.

    Returns list sorted by MI (highest first).
    """
    features = np.asarray(features)
    target = np.asarray(target)
    n_features = features.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    results = []
    for i in range(n_features):
        mi = mutual_information(features[:, i], target)
        results.append({
            "feature": feature_names[i],
            "mutual_information": mi,
        })

    return sorted(results, key=lambda r: -r["mutual_information"])


# ═══════════════════════════════════════════════════════════════
# 2.3: Fisher Information and Parameter Uncertainty
# ═══════════════════════════════════════════════════════════════


def fisher_information_matrix(
    log_likelihood_fn: callable,
    params: np.ndarray,
    dx: float = 1e-5,
) -> np.ndarray:
    """Numerical Fisher information matrix (observed information).

    FIM = -H(ℓ) where H is the Hessian of the log-likelihood.

    Args:
        log_likelihood_fn: callable(params) → float (total log-likelihood).
        params: (D,) parameter vector at MLE.
        dx: finite-difference step size.

    Returns:
        (D, D) Fisher information matrix.
    """
    params = np.asarray(params, dtype=float)
    d = len(params)
    H = np.zeros((d, d))
    f0 = log_likelihood_fn(params)

    for i in range(d):
        for j in range(i, d):
            p_pp = params.copy()
            p_pm = params.copy()
            p_mp = params.copy()
            p_mm = params.copy()
            p_pp[i] += dx
            p_pp[j] += dx
            p_pm[i] += dx
            p_pm[j] -= dx
            p_mp[i] -= dx
            p_mp[j] += dx
            p_mm[i] -= dx
            p_mm[j] -= dx
            H[i, j] = (log_likelihood_fn(p_pp) - log_likelihood_fn(p_pm)
                        - log_likelihood_fn(p_mp) + log_likelihood_fn(p_mm)) / (4 * dx**2)
            H[j, i] = H[i, j]

    return -H  # FIM = -Hessian


def cramer_rao_bound(fim: np.ndarray) -> np.ndarray:
    """Cramér-Rao lower bound on estimator variance.

    CRB = diag(FIM⁻¹) — minimum variance for any unbiased estimator.

    Returns:
        (D,) minimum variance per parameter.
    """
    try:
        fim_inv = np.linalg.inv(fim)
        return np.diag(fim_inv)
    except np.linalg.LinAlgError:
        return np.full(fim.shape[0], np.inf)


def parameter_confidence_intervals(
    fim: np.ndarray,
    params: np.ndarray,
    confidence: float = 0.95,
) -> list[dict]:
    """Confidence intervals from Fisher information.

    CI = params ± z × √(CRB)

    Args:
        fim: Fisher information matrix.
        params: MLE parameter values.
        confidence: confidence level (default 95%).
    """
    z = norm.ppf(0.5 + confidence / 2)
    crb = cramer_rao_bound(fim)
    results = []
    for i, (p, v) in enumerate(zip(params, crb)):
        se = math.sqrt(max(v, 0))
        results.append({
            "param_idx": i,
            "estimate": float(p),
            "std_error": se,
            "ci_lower": float(p - z * se),
            "ci_upper": float(p + z * se),
            "confidence": confidence,
        })
    return results
