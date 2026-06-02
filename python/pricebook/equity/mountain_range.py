"""Mountain range options: Napoleon, Everest, Atlas, Altiplano.

Multi-asset exotic options with correlation dependence.
All priced via Monte Carlo with correlated GBM.

* :func:`napoleon_option` — worst-of cliquet with local caps/floors.
* :func:`everest_option` — call/put on worst performer.
* :func:`atlas_option` — remove best and worst, payoff on remainder.
* :func:`altiplano_option` — digital basket (pays if all above barrier).

References:
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.
    Overhaus et al., *Equity Derivatives*, Wiley, 2002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MountainRangeResult:
    """Mountain range option pricing result."""
    price: float
    delta_avg: float            # average delta across assets
    expected_payoff: float
    payoff_std: float
    n_assets: int
    product: str

    def to_dict(self) -> dict:
        return vars(self)


def _correlated_gbm(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    rate: float,
    div_yields: list[float],
    T: float,
    n_steps: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate correlated GBM paths.

    Returns: (n_sims, n_steps+1, n_assets) array.
    """
    n = len(spots)
    dt = T / n_steps
    L = np.linalg.cholesky(np.array(correlations))

    paths = np.zeros((n_sims, n_steps + 1, n))
    paths[:, 0, :] = spots

    for t in range(n_steps):
        Z = rng.standard_normal((n_sims, n))
        corr_Z = Z @ L.T
        for j in range(n):
            drift = (rate - div_yields[j] - 0.5 * vols[j]**2) * dt
            diffusion = vols[j] * math.sqrt(dt) * corr_Z[:, j]
            paths[:, t + 1, j] = paths[:, t, j] * np.exp(drift + diffusion)

    return paths


def napoleon_option(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    observation_dates: list[float],
    local_floor: float = -0.10,
    local_cap: float = 0.10,
    rate: float = 0.04,
    div_yields: list[float] | None = None,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
) -> MountainRangeResult:
    """Napoleon option: worst-of cliquet with local caps and floors.

    Each period: return_i = min(cap, max(floor, worst_performance_i)).
    Payoff = notional × (1 + Σ return_i).

    Args:
        observation_dates: observation times (years).
        local_floor: floor on each period's return.
        local_cap: cap on each period's return.
    """
    n = len(spots)
    divs = div_yields or [0.02] * n
    rng = np.random.default_rng(seed)
    T = observation_dates[-1]
    n_obs = len(observation_dates)

    # Simulate at observation dates
    all_times = [0.0] + observation_dates
    n_steps = len(all_times) - 1

    paths = _correlated_gbm(spots, vols, correlations, rate, divs, T, n_steps, n_sims, rng)

    payoffs = np.zeros(n_sims)
    for sim in range(n_sims):
        total_return = 0.0
        for t in range(1, n_steps + 1):
            perfs = paths[sim, t, :] / paths[sim, t - 1, :] - 1
            worst = float(np.min(perfs))
            capped = max(local_floor, min(local_cap, worst))
            total_return += capped
        payoffs[sim] = notional * (1 + total_return)

    df = math.exp(-rate * T)
    payoffs = np.maximum(payoffs, 0)
    price = float(np.mean(payoffs)) * df

    return MountainRangeResult(
        price=price, delta_avg=0, expected_payoff=float(np.mean(payoffs)),
        payoff_std=float(np.std(payoffs)), n_assets=n, product="napoleon",
    )


def everest_option(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    T: float = 5.0,
    rate: float = 0.04,
    div_yields: list[float] | None = None,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
) -> MountainRangeResult:
    """Everest option: payoff based on worst performer at maturity.

    Payoff = notional × max(0, worst_performance).
    """
    n = len(spots)
    divs = div_yields or [0.02] * n
    rng = np.random.default_rng(seed)

    paths = _correlated_gbm(spots, vols, correlations, rate, divs, T, 1, n_sims, rng)
    final = paths[:, -1, :]
    perfs = final / np.array(spots)
    worst = np.min(perfs, axis=1)

    payoffs = notional * np.maximum(worst - 1, 0)
    df = math.exp(-rate * T)
    price = float(np.mean(payoffs)) * df

    return MountainRangeResult(
        price=price, delta_avg=0, expected_payoff=float(np.mean(payoffs)),
        payoff_std=float(np.std(payoffs)), n_assets=n, product="everest",
    )


def atlas_option(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    n_remove: int = 2,
    T: float = 5.0,
    rate: float = 0.04,
    div_yields: list[float] | None = None,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
) -> MountainRangeResult:
    """Atlas option: remove best and worst, payoff on remainder average.

    Payoff = notional × max(0, avg_middle_performance − 1).

    Args:
        n_remove: number of best and worst performers to remove.
    """
    n = len(spots)
    divs = div_yields or [0.02] * n
    rng = np.random.default_rng(seed)

    paths = _correlated_gbm(spots, vols, correlations, rate, divs, T, 1, n_sims, rng)
    final = paths[:, -1, :]
    perfs = final / np.array(spots)

    # Remove n_remove best and worst
    sorted_perfs = np.sort(perfs, axis=1)
    middle = sorted_perfs[:, n_remove:-n_remove] if 2 * n_remove < n else sorted_perfs
    avg_middle = np.mean(middle, axis=1)

    payoffs = notional * np.maximum(avg_middle - 1, 0)
    df = math.exp(-rate * T)
    price = float(np.mean(payoffs)) * df

    return MountainRangeResult(
        price=price, delta_avg=0, expected_payoff=float(np.mean(payoffs)),
        payoff_std=float(np.std(payoffs)), n_assets=n, product="atlas",
    )


def altiplano_option(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    barrier: float = 0.80,
    coupon: float = 0.10,
    T: float = 1.0,
    rate: float = 0.04,
    div_yields: list[float] | None = None,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
) -> MountainRangeResult:
    """Altiplano option: digital basket, pays coupon if ALL above barrier.

    Payoff = coupon × notional if min(S_i/S_i0) ≥ barrier, else 0.

    Args:
        barrier: fraction of initial spot (e.g. 0.80).
        coupon: coupon rate paid if condition met.
    """
    n = len(spots)
    divs = div_yields or [0.02] * n
    rng = np.random.default_rng(seed)

    paths = _correlated_gbm(spots, vols, correlations, rate, divs, T, 1, n_sims, rng)
    final = paths[:, -1, :]
    perfs = final / np.array(spots)
    worst = np.min(perfs, axis=1)

    payoffs = np.where(worst >= barrier, coupon * notional, 0.0)
    df = math.exp(-rate * T)
    price = float(np.mean(payoffs)) * df

    return MountainRangeResult(
        price=price, delta_avg=0, expected_payoff=float(np.mean(payoffs)),
        payoff_std=float(np.std(payoffs)), n_assets=n, product="altiplano",
    )
