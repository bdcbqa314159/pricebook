"""Curve diffusion models — evolve bootstrapped curves forward via HJM/LMM.

Each simulated path produces standard DiscountCurve objects, so all
existing pricing code works unchanged on simulated future curves.

    from pricebook.curves.curve_diffusion import (
        CurveDiffusionEngine, CurveDiffusionConfig, CurveDiffusionResult,
    )

References:
    Heath, Jarrow & Morton (1992). Bond Pricing and the Term Structure
    of Interest Rates: A New Methodology. Econometrica.
    Brace, Gatarek & Musiela (1997). The Market Model of Interest Rate Dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class CurveDiffusionConfig:
    """Configuration for curve diffusion simulation."""
    n_factors: int = 2           # number of volatility factors
    vol_levels: list[float] = field(default_factory=lambda: [0.01, 0.005])
    vol_decays: list[float] = field(default_factory=lambda: [0.5, 1.5])
    n_paths: int = 1000
    n_steps: int = 12            # time steps per year
    horizon_years: float = 1.0
    tenor_grid: list[float] = field(default_factory=lambda: [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    seed: int = 42

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class CurveDiffusionResult:
    """Result of curve diffusion simulation."""
    time_grid: list[float]
    curves: list[list[DiscountCurve]]  # [step][path] → DiscountCurve
    forward_rate_mean: np.ndarray      # (n_steps, n_tenors) mean forward rates
    forward_rate_std: np.ndarray       # (n_steps, n_tenors) std of forward rates
    n_paths: int
    n_steps: int

    def scenario_curves(self, step: int) -> list[DiscountCurve]:
        """All paths at a given time step."""
        return self.curves[step]

    def to_dict(self) -> dict:
        return {
            "time_grid": self.time_grid,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "forward_rate_mean": self.forward_rate_mean.tolist(),
        }


class CurveDiffusionEngine:
    """Evolve a bootstrapped curve forward using HJM dynamics.

    Multi-factor HJM: df(t, T) = f(0, T-t) + drift + Σ σ_k(T-t) × W_k(t)

    Each factor has exponentially decaying vol: σ_k(x) = level_k × exp(-decay_k × x).

    The drift is determined by the no-arbitrage HJM drift condition:
        μ(t, T) = σ(t, T) × ∫_t^T σ(t, s) ds
    """

    def __init__(self, initial_curve: DiscountCurve, config: CurveDiffusionConfig):
        self.initial_curve = initial_curve
        self.config = config
        self._initial_forwards = self._extract_forwards()

    def _extract_forwards(self) -> np.ndarray:
        """Extract initial forward rates at the tenor grid."""
        ref = self.initial_curve.reference_date
        dc = DayCountConvention.ACT_365_FIXED
        forwards = []
        for tau in self.config.tenor_grid:
            d = date.fromordinal(ref.toordinal() + int(tau * 365))
            f = self.initial_curve.instantaneous_forward(tau)
            forwards.append(f)
        return np.array(forwards)

    def simulate(self) -> CurveDiffusionResult:
        """Run the curve diffusion simulation."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        n_tenors = len(cfg.tenor_grid)
        dt = cfg.horizon_years / cfg.n_steps
        sqrt_dt = math.sqrt(dt)
        tenors = np.array(cfg.tenor_grid)

        # Factor volatilities: σ_k(x) = level_k × exp(-decay_k × x)
        n_factors = min(cfg.n_factors, len(cfg.vol_levels))

        # Forward rate paths: (n_paths, n_steps+1, n_tenors)
        f_paths = np.zeros((cfg.n_paths, cfg.n_steps + 1, n_tenors))
        f_paths[:, 0, :] = self._initial_forwards

        for step in range(cfg.n_steps):
            t = step * dt
            # Random shocks for each factor
            dW = rng.standard_normal((cfg.n_paths, n_factors)) * sqrt_dt

            for j in range(n_tenors):
                tau = tenors[j]
                remaining = max(tau - t, 0.01)

                # HJM drift: μ = Σ σ_k(x) × ∫₀ˣ σ_k(s) ds
                drift = 0.0
                diffusion = np.zeros(cfg.n_paths)

                for k in range(n_factors):
                    level = cfg.vol_levels[k]
                    decay = cfg.vol_decays[k]
                    sigma = level * math.exp(-decay * remaining)

                    # Integral of sigma from 0 to remaining
                    if decay > 0:
                        integral = level / decay * (1 - math.exp(-decay * remaining))
                    else:
                        integral = level * remaining

                    drift += sigma * integral
                    diffusion += sigma * dW[:, k]

                f_paths[:, step + 1, j] = f_paths[:, step, j] + drift * dt + diffusion

        # Build DiscountCurve objects for each (step, path)
        ref = self.initial_curve.reference_date
        time_grid = [i * dt for i in range(cfg.n_steps + 1)]
        curves = []

        for step in range(cfg.n_steps + 1):
            step_curves = []
            sim_time = step * dt

            for path in range(cfg.n_paths):
                forwards = f_paths[path, step, :]
                # Convert forwards to DFs: df(τ) = exp(-∫₀ᵗ f(s) ds) ≈ exp(-f × τ)
                pillar_dates = []
                pillar_dfs = []
                for j, tau in enumerate(tenors):
                    remaining = max(tau - sim_time, 0.01)
                    d = date.fromordinal(ref.toordinal() + int(tau * 365))
                    df = math.exp(-max(forwards[j], -0.05) * remaining)
                    pillar_dates.append(d)
                    pillar_dfs.append(max(df, 1e-15))

                step_curves.append(DiscountCurve(ref, pillar_dates, pillar_dfs))

            curves.append(step_curves)

        # Statistics
        fwd_mean = np.mean(f_paths, axis=0)
        fwd_std = np.std(f_paths, axis=0)

        return CurveDiffusionResult(
            time_grid=time_grid,
            curves=curves,
            forward_rate_mean=fwd_mean,
            forward_rate_std=fwd_std,
            n_paths=cfg.n_paths,
            n_steps=cfg.n_steps,
        )
