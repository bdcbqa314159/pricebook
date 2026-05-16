"""MC exposure engine for XVA computation.

Simulates mark-to-market paths and computes exposure profiles
(EPE, ENE, PFE) for CVA/DVA/FVA calculation.

Replaces 8+ desk-specific XVA simulation implementations with
a single parametric engine.

    from pricebook.models.mc_exposure import ExposureEngine, ExposureResult

    engine = ExposureEngine(
        process=BlackScholesProcess(100, 0.05, 0.20),
        time_grid=TimeGrid.uniform(5.0, 60),  # quarterly for 5 years
        n_paths=10_000,
        revalue=lambda paths, t_idx: paths[:, t_idx] - strike,
    )
    result = engine.compute()
    # result.epe → expected positive exposure profile
    # result.cva → unilateral CVA

References:
    Gregory (2020). The xVA Challenge, Wiley, 4th ed.
    Green (2015). XVA: Credit, Funding and Capital, Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid, ProcessSpec


@dataclass
class ExposureResult:
    """Exposure simulation result."""
    times: list[float]           # time points
    epe: list[float]             # expected positive exposure at each time
    ene: list[float]             # expected negative exposure at each time
    pfe_95: list[float]          # 95th percentile future exposure
    pfe_99: list[float]          # 99th percentile
    ee_peak: float               # peak expected exposure
    effective_epe: float         # time-averaged EPE (for CVA)
    cva: float                   # unilateral CVA
    dva: float                   # DVA
    fva: float                   # FVA

    def to_dict(self) -> dict:
        return {
            "times": self.times, "epe": self.epe, "ene": self.ene,
            "pfe_95": self.pfe_95, "pfe_99": self.pfe_99,
            "ee_peak": self.ee_peak, "effective_epe": self.effective_epe,
            "cva": self.cva, "dva": self.dva, "fva": self.fva,
        }


class ExposureEngine:
    """Monte Carlo exposure simulation for XVA.

    Generates market factor paths, revalues the portfolio at each time
    point, and computes exposure profiles.

    Args:
        process: SDE for the underlying market factor(s).
        time_grid: exposure monitoring dates.
        n_paths: number of simulation paths.
        revalue: callable(paths, step_index) → array of mark-to-market values.
            Takes the full path array and a step index, returns (n_paths,) MTM.
        counterparty_spread: counterparty CDS spread (for CVA).
        own_spread: own CDS spread (for DVA).
        funding_spread: unsecured funding spread (for FVA).
        recovery: counterparty recovery rate.
        seed: random seed.
    """

    def __init__(
        self,
        process: ProcessSpec,
        time_grid: TimeGrid,
        n_paths: int = 10_000,
        revalue=None,
        counterparty_spread: float = 0.01,
        own_spread: float = 0.005,
        funding_spread: float = 0.002,
        recovery: float = 0.40,
        seed: int = 42,
    ):
        self.process = process
        self.time_grid = time_grid
        self.n_paths = n_paths
        self.revalue = revalue
        self.counterparty_spread = counterparty_spread
        self.own_spread = own_spread
        self.funding_spread = funding_spread
        self.recovery = recovery
        self.seed = seed

    def compute(self) -> ExposureResult:
        """Run exposure simulation and compute all XVA metrics."""
        # Generate paths
        engine = MCEngine(self.process, self.time_grid, self.n_paths, self.seed)
        paths = engine.paths
        times = self.time_grid.times

        n_steps = self.time_grid.n_steps
        lgd = 1 - self.recovery

        epe = []
        ene = []
        pfe_95 = []
        pfe_99 = []

        for step in range(n_steps + 1):
            if self.revalue is not None:
                mtm = self.revalue(paths, step)
            else:
                # Default: MTM = spot value - initial value
                if paths.ndim == 2:
                    mtm = np.exp(paths[:, step]) - np.exp(paths[:, 0])
                else:
                    mtm = np.exp(paths[:, step, 0]) - np.exp(paths[:, 0, 0])

            positive = np.maximum(mtm, 0.0)
            negative = np.minimum(mtm, 0.0)

            epe.append(float(np.mean(positive)))
            ene.append(float(np.mean(np.abs(negative))))
            pfe_95.append(float(np.percentile(mtm, 95)))
            pfe_99.append(float(np.percentile(mtm, 99)))

        # Peak exposure
        ee_peak = max(epe)

        # Effective EPE: non-decreasing time-average
        effective_epe_profile = []
        running_max = 0.0
        for ep in epe:
            running_max = max(running_max, ep)
            effective_epe_profile.append(running_max)
        effective_epe = float(np.mean(effective_epe_profile))

        # CVA: (1-R) × ∫ EPE(t) × h(t) dt ≈ LGD × Σ EPE_i × spread × Δt
        dt_arr = np.diff(times)
        cva = 0.0
        dva = 0.0
        fva = 0.0

        for i in range(n_steps):
            dt_i = float(dt_arr[i])
            cva += lgd * epe[i + 1] * self.counterparty_spread * dt_i
            dva += lgd * ene[i + 1] * self.own_spread * dt_i
            fva += epe[i + 1] * self.funding_spread * dt_i

        return ExposureResult(
            times=[float(t) for t in times],
            epe=epe, ene=ene,
            pfe_95=pfe_95, pfe_99=pfe_99,
            ee_peak=ee_peak,
            effective_epe=effective_epe,
            cva=cva, dva=dva, fva=fva,
        )
