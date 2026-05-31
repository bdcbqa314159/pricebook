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
        return vars(self)


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
        return vars(self)


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
