"""Monte Carlo convergence diagnostics.

    from pricebook.models.mc_diagnostics import (
        batch_means, effective_sample_size, convergence_table,
    )

    se = batch_means(payoff_values, n_batches=20)
    ess = effective_sample_size(payoff_values)

References:
    Glasserman (2003). Monte Carlo Methods in Financial Engineering, Ch. 2.
    Jones et al. (2006). Fixed-Width Output Analysis for Markov Chain MC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BatchMeansResult:
    """Result of batch means standard error estimation."""
    mean: float
    se: float               # standard error (robust)
    n_batches: int
    batch_size: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def batch_means(
    values: np.ndarray,
    n_batches: int = 20,
) -> BatchMeansResult:
    """Batch means standard error estimation.

    Splits values into n_batches equal-sized batches, computes
    the mean of each batch, then estimates SE from inter-batch variance.

    More robust than naive SE when values are autocorrelated.

    Args:
        values: 1D array of MC payoff samples.
        n_batches: number of batches (default 20).

    Returns:
        BatchMeansResult with mean and standard error.
    """
    n = len(values)
    batch_size = n // n_batches
    if batch_size < 2:
        return BatchMeansResult(float(np.mean(values)), float(np.std(values) / math.sqrt(n)),
                                 n_batches, batch_size)

    # Trim to exact multiple
    trimmed = values[:batch_size * n_batches]
    batches = trimmed.reshape(n_batches, batch_size)
    batch_means_arr = batches.mean(axis=1)

    overall_mean = float(batch_means_arr.mean())
    se = float(batch_means_arr.std(ddof=1) / math.sqrt(n_batches))

    return BatchMeansResult(overall_mean, se, n_batches, batch_size)


def effective_sample_size(
    values: np.ndarray,
    max_lag: int = 100,
) -> float:
    """Effective sample size accounting for autocorrelation.

    ESS = N / (1 + 2 Σ_{k=1}^{K} ρ(k))

    where ρ(k) is the autocorrelation at lag k.
    For iid samples, ESS = N. For correlated, ESS < N.

    Args:
        values: 1D array of samples.
        max_lag: maximum lag for autocorrelation sum.

    Returns:
        Effective sample size (float).
    """
    n = len(values)
    if n < 3:
        return float(n)

    # Centre
    x = values - values.mean()
    var = float(x.var())
    if var < 1e-30:
        return float(n)

    # Autocorrelation via FFT (fast)
    f = np.fft.fft(x, n=2 * n)
    acf_full = np.fft.ifft(f * np.conj(f)).real[:n] / (var * n)

    # Sum positive autocorrelations (initial monotone sequence estimator)
    tau = 1.0
    for k in range(1, min(max_lag, n)):
        rho_k = acf_full[k]
        if rho_k < 0.05:  # stop at first insignificant lag
            break
        tau += 2 * rho_k

    return n / max(tau, 1.0)


@dataclass
class ConvergenceEntry:
    """Single entry in convergence table."""
    n_samples: int
    mean: float
    se: float
    relative_error_pct: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def convergence_table(
    values: np.ndarray,
    checkpoints: list[int] | None = None,
) -> list[ConvergenceEntry]:
    """Running convergence table at various sample counts.

    Shows how mean and SE stabilise as N increases.

    Args:
        values: full sample array.
        checkpoints: sample counts (default: 1k, 5k, 10k, 50k, 100k, N).

    Returns:
        List of ConvergenceEntry.
    """
    n = len(values)
    if checkpoints is None:
        checkpoints = [c for c in [1_000, 5_000, 10_000, 50_000, 100_000, n]
                        if c <= n]
        if n not in checkpoints:
            checkpoints.append(n)

    entries = []
    for cp in checkpoints:
        subset = values[:cp]
        mean = float(subset.mean())
        se = float(subset.std(ddof=1) / math.sqrt(cp))
        rel = abs(se / mean) * 100 if abs(mean) > 1e-10 else 0.0
        entries.append(ConvergenceEntry(cp, mean, se, rel))

    return entries


# ═══════════════════════════════════════════════════════════════
# Extended diagnostics (P4)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MCFullDiagnostics:
    """Complete MC diagnostics with VRE and convergence rate."""
    n_samples: int
    mean: float
    std_error: float
    std_error_batch: float
    ess: float
    ess_ratio: float
    vre: float                  # variance reduction efficiency
    convergence_rate: float     # estimated order
    ci_95: tuple[float, float]
    ci_99: tuple[float, float]
    skewness: float
    kurtosis: float
    relative_error_pct: float
    is_converged: bool

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "mean": self.mean,
            "std_error": self.std_error,
            "std_error_batch": self.std_error_batch,
            "ess": round(self.ess, 1),
            "ess_ratio": round(self.ess_ratio, 3),
            "vre": round(self.vre, 2),
            "convergence_rate": round(self.convergence_rate, 3),
            "ci_95": [round(x, 6) for x in self.ci_95],
            "relative_error_pct": round(self.relative_error_pct, 4),
            "is_converged": self.is_converged,
        }


def full_diagnostics(
    values: np.ndarray,
    values_crude: np.ndarray | None = None,
    n_batches: int = 20,
    convergence_threshold_pct: float = 1.0,
) -> MCFullDiagnostics:
    """Run complete diagnostics on MC output.

    Args:
        values: discounted payoff values (after VR).
        values_crude: values WITHOUT variance reduction (for VRE).
        n_batches: batches for batch means SE.
        convergence_threshold_pct: relative error threshold for convergence.
    """
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))

    bm = batch_means(values, n_batches)
    ess = effective_sample_size(values)
    ess_ratio = ess / n if n > 0 else 0

    vre = variance_reduction_efficiency(values_crude, values) if values_crude is not None else 1.0

    ci_95 = (mean - 1.96 * se, mean + 1.96 * se)
    ci_99 = (mean - 2.576 * se, mean + 2.576 * se)

    # Higher moments
    if n > 3:
        centered = values - mean
        m2 = float(np.mean(centered**2))
        m3 = float(np.mean(centered**3))
        m4 = float(np.mean(centered**4))
        skew = m3 / (m2**1.5) if m2 > 1e-15 else 0
        kurt = m4 / (m2**2) - 3.0 if m2 > 1e-15 else 0
    else:
        skew, kurt = 0.0, 0.0

    rel_err = abs(se / mean) * 100 if abs(mean) > 1e-10 else 0

    return MCFullDiagnostics(
        n_samples=n, mean=mean, std_error=se,
        std_error_batch=bm.se, ess=ess, ess_ratio=ess_ratio,
        vre=vre, convergence_rate=0.5,
        ci_95=ci_95, ci_99=ci_99,
        skewness=skew, kurtosis=kurt,
        relative_error_pct=rel_err,
        is_converged=rel_err < convergence_threshold_pct,
    )


def variance_reduction_efficiency(
    values_crude: np.ndarray,
    values_reduced: np.ndarray,
) -> float:
    """VRE = Var(crude) / Var(reduced). >1 means VR helped."""
    var_crude = float(np.var(values_crude))
    var_reduced = float(np.var(values_reduced))
    if var_reduced < 1e-15:
        return float('inf') if var_crude > 0 else 1.0
    return var_crude / var_reduced


def estimate_convergence_rate(
    prices_at_n: list[tuple[int, float]],
) -> float:
    """Estimate convergence order from prices at different N.

    MC: rate ≈ 0.5 (error ~ 1/√N). QMC: rate ≈ 1.0 (error ~ 1/N).
    Fits log(error) = -rate × log(N) + const.
    """
    if len(prices_at_n) < 3:
        return 0.5

    sorted_pairs = sorted(prices_at_n, key=lambda x: x[0])
    true_price = sorted_pairs[-1][1]

    log_n, log_err = [], []
    for n, p in sorted_pairs[:-1]:
        err = abs(p - true_price)
        if err > 1e-15 and n > 0:
            log_n.append(math.log(n))
            log_err.append(math.log(err))

    if len(log_n) < 2:
        return 0.5

    x = np.array(log_n)
    y = np.array(log_err)
    coeffs = np.polyfit(x, y, 1)
    rate = -coeffs[0]
    return max(0.1, min(rate, 2.0))
