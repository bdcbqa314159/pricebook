"""Regression for L2 Wave-2 audit — `fokker_planck_1d` Crank-Nicolson
implicit-matrix coefficients.

Pre-fix the implicit step had:
    diag = 1 + dt·diff[i] / dx²              (extra 2× factor)
    lower = upper = -0.25·dt·diff[i] / dx²   (using diff[i] not diff[i±1])

The correct CN implicit for L_diff[p] = (½/dx²)·(diff_{i+1}·p_{i+1} -
2·diff_i·p_i + diff_{i-1}·p_{i-1}):

    diag[i] = 1 + 0.5·dt·diff[i] / dx²
    lower[i] = -0.25·dt·diff[i-1] / dx²
    upper[i] = -0.25·dt·diff[i+1] / dx²

Pre-fix the 2× over-stated diagonal damped the density artificially.  On
a vanilla BS lognormal (S₀=100, r=5 %, σ=20 %, T=1 y), the FP variance
came out 19 % BELOW the analytical lognormal value.  Post-fix variance
matches to 2 % (the residual is convection treatment, separate issue).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.models.fokker_planck import fokker_planck_1d


class TestFokkerPlanckLognormal:
    def test_variance_matches_analytical_lognormal(self):
        """For constant-vol BS, the FP density at T should be lognormal.
        Variance must match Var[S_T] = S₀²·exp(2rT)·(exp(σ²T) − 1).
        Pre-fix was 81% of true variance; post-fix ≥ 95%."""
        S0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
        res = fokker_planck_1d(
            spot=S0, rate=r, vol=sigma, T=T,
            n_space=400, n_time=400,
        )
        exact_var = S0**2 * math.exp(2*r*T) * (math.exp(sigma**2 * T) - 1)
        ratio = res.variance / exact_var
        assert ratio > 0.95, (
            f"FP variance / exact = {ratio:.4f}, expected > 0.95. "
            f"Pre-fix was ~0.81 (the 2× diagonal damped variance artificially)."
        )

    def test_mean_close_to_exact(self):
        """Mean should be ≈ S₀·exp(rT) for risk-neutral BS."""
        S0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
        res = fokker_planck_1d(
            spot=S0, rate=r, vol=sigma, T=T,
            n_space=400, n_time=400,
        )
        exact_mean = S0 * math.exp(r * T)
        rel = abs(res.mean - exact_mean) / exact_mean
        # Residual bias comes from explicit convection treatment, not the
        # implicit-matrix fix. Pre-fix was already in this range; the
        # variance fix doesn't degrade it.
        assert rel < 0.03

    def test_density_normalised(self):
        """Total mass under the FP density should be 1 (probability)."""
        S0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
        res = fokker_planck_1d(
            spot=S0, rate=r, vol=sigma, T=T,
            n_space=300, n_time=300,
        )
        # Integrate density over the grid.
        mass = float(np.trapezoid(res.density, res.grid))
        assert abs(mass - 1.0) < 0.02

    def test_short_T_density_concentrated_near_spot(self):
        """For very short T, the density should be a narrow peak around
        the initial spot.  Mode within a small range of spot."""
        S0, r, sigma, T = 100.0, 0.05, 0.20, 0.05  # ~ 2 weeks
        res = fokker_planck_1d(
            spot=S0, rate=r, vol=sigma, T=T,
            n_space=400, n_time=100,
        )
        mode_idx = int(np.argmax(res.density))
        mode_S = res.grid[mode_idx]
        # The mode should be near S0 (within a few %).
        assert abs(mode_S - S0) < 5.0
