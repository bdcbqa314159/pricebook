"""Hawkes-driven survival curves for CDS pricing.

Converts Hawkes intensity paths into survival probabilities and
pricebook-compatible SurvivalCurve objects.

    from pricebook.credit.hawkes_survival import (
        HawkesSurvivalCurve, hawkes_survival_mc,
    )

The survival probability under stochastic intensity:
    Q(T) = E[exp(-∫₀ᵀ λ(s) ds)]

Computed via Monte Carlo over simulated intensity paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.survival_curve import SurvivalCurve


@dataclass
class HawkesSurvivalResult:
    """Survival probabilities from Hawkes intensity simulation."""
    times: np.ndarray
    survival_probs: np.ndarray       # (n_grid,) — mean Q(t)
    survival_std: np.ndarray         # (n_grid,) — std of Q(t)
    mean_intensity: np.ndarray       # (n_grid,) — E[λ(t)]
    implied_hazard: np.ndarray       # (n_grid,) — -d/dt ln Q(t)
    n_paths: int

    def to_dict(self) -> dict:
        return {
            "n_paths": self.n_paths,
            "mean_intensity_avg": float(np.mean(self.mean_intensity)),
            "survival_at_1Y": float(np.interp(1.0, self.times, self.survival_probs)),
            "survival_at_5Y": float(np.interp(5.0, self.times, self.survival_probs)),
        }


def hawkes_survival_mc(
    intensities: np.ndarray,
    times: np.ndarray,
) -> HawkesSurvivalResult:
    """Compute survival probabilities from Hawkes intensity paths.

    Args:
        intensities: (n_paths, n_grid) intensity paths from Hawkes simulation.
        times: (n_grid,) time grid.

    Returns:
        HawkesSurvivalResult with survival curve, mean intensity, implied hazard.
    """
    n_paths, n_grid = intensities.shape
    dt = np.diff(times, prepend=0)

    # Integrated intensity: Λ(t) = ∫₀ᵗ λ(s) ds
    cum_intensity = np.cumsum(intensities * dt[np.newaxis, :], axis=1)

    # Survival: Q(t) = exp(-Λ(t)) per path
    survival_paths = np.exp(-cum_intensity)

    # Mean and std across paths
    survival_mean = np.mean(survival_paths, axis=0)
    survival_std = np.std(survival_paths, axis=0)
    mean_intensity = np.mean(intensities, axis=0)

    # Implied hazard: h(t) = -d/dt ln Q(t) ≈ Δ(-ln Q) / Δt
    log_surv = np.log(np.maximum(survival_mean, 1e-15))
    implied_hazard = np.zeros(n_grid)
    implied_hazard[1:] = -np.diff(log_surv) / np.diff(times)
    implied_hazard[0] = mean_intensity[0]

    return HawkesSurvivalResult(
        times=times,
        survival_probs=survival_mean,
        survival_std=survival_std,
        mean_intensity=mean_intensity,
        implied_hazard=implied_hazard,
        n_paths=n_paths,
    )


class HawkesSurvivalCurve:
    """Survival curve driven by a Hawkes process.

    Wraps Hawkes simulation → MC survival → pricebook SurvivalCurve.

    Usage:
        from pricebook.models.hawkes_credit import FractionalHawkesProcess, HawkesKernel
        hawkes = FractionalHawkesProcess(mu=0.02, kernel=HawkesKernel.POWER_LAW,
                                          kernel_params={"alpha": 0.01, "H": 0.3})
        hsc = HawkesSurvivalCurve(hawkes, T_max=10.0, n_paths=10_000)
        q5 = hsc.survival(5.0)
        curve = hsc.to_pricebook_curve(reference_date)
    """

    def __init__(
        self,
        hawkes_process,
        T_max: float = 10.0,
        n_paths: int = 10_000,
        n_grid: int = 200,
        seed: int | None = 42,
    ):
        self._hawkes = hawkes_process
        self._T_max = T_max

        # Simulate
        result = hawkes_process.simulate(T_max, n_paths, n_grid, seed)
        self._sim_result = result

        # Compute survival
        self._surv_result = hawkes_survival_mc(result.intensities, result.times)

    @property
    def result(self) -> HawkesSurvivalResult:
        return self._surv_result

    def survival(self, T: float) -> float:
        """MC survival probability at time T."""
        if T <= 0:
            return 1.0
        if T >= self._T_max:
            return float(self._surv_result.survival_probs[-1])
        return float(np.interp(T, self._surv_result.times,
                               self._surv_result.survival_probs))

    def hazard_rate(self, T: float) -> float:
        """Instantaneous hazard rate at time T."""
        if T <= 0:
            return float(self._surv_result.mean_intensity[0])
        return float(np.interp(T, self._surv_result.times,
                               self._surv_result.implied_hazard))

    def mean_intensity(self, T: float) -> float:
        """Expected intensity at time T."""
        return float(np.interp(T, self._surv_result.times,
                               self._surv_result.mean_intensity))

    def to_pricebook_curve(self, reference_date: date) -> SurvivalCurve:
        """Convert to pricebook's SurvivalCurve for CDS pricing.

        Samples survival at annual intervals out to T_max.
        """
        from dateutil.relativedelta import relativedelta

        n_years = int(self._T_max)
        pillar_dates = [reference_date + relativedelta(years=i) for i in range(1, n_years + 1)]
        pillar_survs = [self.survival(float(i)) for i in range(1, n_years + 1)]

        return SurvivalCurve(reference_date, pillar_dates, pillar_survs)

    def to_dict(self) -> dict:
        return {
            "T_max": self._T_max,
            "branching_ratio": self._sim_result.branching_ratio,
            **self._surv_result.to_dict(),
        }
