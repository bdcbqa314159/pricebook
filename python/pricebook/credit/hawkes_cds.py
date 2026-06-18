"""CDS pricing under Hawkes self-exciting intensity.

Single-name CDS where the hazard rate follows a Hawkes process,
capturing self-excitation (own credit events raise future intensity)
and contagion (other names' events raise this name's intensity).

    from pricebook.credit.hawkes_cds import (
        hawkes_cds_spread, hawkes_cds_pv, HawkesCDSResult,
    )

The par CDS spread is the spread S that makes:
    Premium Leg PV = Protection Leg PV

Under Hawkes intensity, both legs are computed via MC over simulated
survival paths.

References:
    Errais, Giesecke & Goldberg (2010). Affine Point Processes and
        Portfolio Credit Risk. SIAM J. Financial Math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.hawkes_credit import FractionalHawkesProcess, HawkesKernel
from pricebook.credit.hawkes_survival import HawkesSurvivalCurve


@dataclass
class HawkesCDSResult:
    """CDS pricing result under Hawkes intensity."""
    par_spread: float           # par CDS spread
    par_spread_bp: float        # in basis points
    protection_pv: float        # PV of protection leg (per unit notional)
    premium_pv01: float         # risky annuity (PV of 1bp running)
    survival_5y: float          # Q(5Y) from Hawkes
    mean_intensity: float       # average intensity over life
    branching_ratio: float
    kernel: str
    n_paths: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def hawkes_cds_spread(
    hawkes_process: FractionalHawkesProcess,
    maturity: float = 5.0,
    recovery: float = 0.4,
    discount_rate: float = 0.03,
    n_paths: int = 20_000,
    frequency: int = 4,
    seed: int | None = 42,
) -> HawkesCDSResult:
    """Compute par CDS spread under Hawkes intensity.

    Simulates intensity paths, computes survival probabilities,
    then uses standard CDS pricing:
        S = (1-R) × Protection PV / Risky Annuity

    Args:
        hawkes_process: configured FractionalHawkesProcess.
        maturity: CDS maturity in years.
        recovery: recovery rate.
        discount_rate: flat risk-free rate.
        n_paths: MC paths for intensity simulation.
        frequency: premium payment frequency (4 = quarterly).
        seed: random seed.
    """
    if not 0 <= recovery < 1:
        raise ValueError(f"recovery must be in [0, 1), got {recovery}")
    if maturity <= 0:
        raise ValueError(f"maturity must be positive, got {maturity}")

    # Simulate Hawkes survival curve
    n_grid = max(200, int(maturity * 50))
    hsc = HawkesSurvivalCurve(hawkes_process, T_max=maturity, n_paths=n_paths,
                               n_grid=n_grid, seed=seed)

    # Premium leg: risky annuity = Σ τᵢ × df(tᵢ) × Q(tᵢ)
    dt = maturity / (maturity * frequency)
    payment_times = [i * dt for i in range(1, int(maturity * frequency) + 1)]

    annuity = 0.0
    for t in payment_times:
        df = math.exp(-discount_rate * t)
        q = hsc.survival(t)
        annuity += dt * df * q

    # Protection leg: (1-R) × Σ df(tᵢ) × ΔPD(tᵢ)
    # ΔPD = Q(tᵢ₋₁) - Q(tᵢ)
    protection = 0.0
    q_prev = 1.0
    for t in payment_times:
        df = math.exp(-discount_rate * t)
        q = hsc.survival(t)
        dpd = q_prev - q
        protection += (1 - recovery) * df * dpd
        q_prev = q

    # Par spread: S such that S × annuity = protection
    if annuity > 1e-15:
        par_spread = protection / annuity
    else:
        par_spread = 0.0

    return HawkesCDSResult(
        par_spread=par_spread,
        par_spread_bp=par_spread * 10_000,
        protection_pv=protection,
        premium_pv01=annuity,
        survival_5y=hsc.survival(min(5.0, maturity)),
        mean_intensity=float(np.mean(hsc.result.mean_intensity)),
        branching_ratio=hawkes_process.branching_ratio_value,
        kernel=hawkes_process.kernel.value,
        n_paths=n_paths,
    )


def hawkes_cds_pv(
    hawkes_process: FractionalHawkesProcess,
    spread: float,
    maturity: float = 5.0,
    recovery: float = 0.4,
    discount_rate: float = 0.03,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> float:
    """PV of a CDS at given spread under Hawkes intensity.

    PV = Protection PV - S × Risky Annuity
    """
    result = hawkes_cds_spread(hawkes_process, maturity, recovery, discount_rate,
                                n_paths, seed=seed)
    return result.protection_pv - spread * result.premium_pv01


def hawkes_cds_spread_comparison(
    mu: float = 0.02,
    alpha_values: list[float] | None = None,
    beta: float = 1.0,
    maturity: float = 5.0,
    recovery: float = 0.4,
    n_paths: int = 10_000,
) -> list[dict]:
    """Compare CDS spreads across different self-excitation levels.

    Shows how self-excitation (α) widens the CDS spread beyond
    the baseline Poisson intensity (α=0).
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    for alpha in alpha_values:
        if alpha == 0:
            # Pure Poisson: closed-form
            h = mu
            q5 = math.exp(-h * maturity)
            spread = (1 - recovery) * h
            results.append({
                "alpha": 0.0, "branching_ratio": 0.0,
                "par_spread_bp": spread * 10_000,
                "survival_5y": q5, "label": "Poisson (no excitation)",
            })
        else:
            hp = FractionalHawkesProcess(
                mu=mu, kernel=HawkesKernel.EXPONENTIAL,
                kernel_params={"alpha": alpha, "beta": beta},
            )
            r = hawkes_cds_spread(hp, maturity, recovery, n_paths=n_paths)
            results.append({
                "alpha": alpha,
                "branching_ratio": r.branching_ratio,
                "par_spread_bp": r.par_spread_bp,
                "survival_5y": r.survival_5y,
                "label": f"Hawkes (α={alpha})",
            })

    return results
