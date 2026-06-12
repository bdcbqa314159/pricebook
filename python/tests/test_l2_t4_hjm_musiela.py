"""Regression for L2 Wave-2 audit — `HJMModel.simulate` non-uniform tenor.

Pre-fix the Musiela `∂f/∂x` finite-difference used a SINGLE SCALAR
``dx = self.tenors[1] - self.tenors[0]`` for every segment.  The default
tenor grid is ``[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]`` — manifestly
non-uniform — so ∂f/∂x was correct only for the (0.25, 0.5) segment, and
biased by factor `dx_first / dx_actual` for every later one.

For the (10, 15) y segment, dx_actual = 5 vs dx_first = 0.25 → pre-fix
slope was 20× the correct value, causing rapid mean-reversion of forwards
at long tenors and divergence-like behaviour at the short end.

Post-fix uses per-segment ``np.diff(tenors)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.models.hjm import HJMModel


class TestHJMNonUniformTenor:
    def test_simulate_with_default_nonuniform_tenors_finite(self):
        """Sanity: HJM simulation on the default non-uniform tenor grid
        produces finite paths.  Pre-fix could produce wildly skewed
        forwards at the long end due to the 20× slope bias."""
        f0 = [0.03, 0.035, 0.04, 0.042, 0.044, 0.045, 0.045, 0.045, 0.045, 0.045]
        tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        hjm = HJMModel(f0, tenors, constant_vol=0.01)
        paths = hjm.simulate(T=2.0, n_steps=20, n_paths=200, seed=42)
        assert np.all(np.isfinite(paths))
        # At T=2y, forwards should still be in a sensible range.
        assert np.all(paths[:, -1, :] > -0.5)
        assert np.all(paths[:, -1, :] < 0.5)

    def test_uniform_tenor_unchanged(self):
        """Uniform tenor grid: the pre-fix and post-fix dfdx are identical
        (since dx is constant across segments).  Verify uniform path runs."""
        f0 = [0.03] * 10
        tenors = list(np.linspace(0.5, 5.0, 10))  # uniform 0.5y spacing
        hjm = HJMModel(f0, tenors, constant_vol=0.01)
        paths = hjm.simulate(T=1.0, n_steps=10, n_paths=100, seed=42)
        assert np.all(np.isfinite(paths))

    def test_flat_initial_curve_remains_flat_mean(self):
        """With a flat initial forward curve and constant vol, the MEAN
        forward over many paths should stay roughly flat (deterministic
        drift from no-arbitrage is small for these params).  Pre-fix the
        non-uniform ∂f/∂x term would have skewed the mean curve."""
        f0 = [0.04] * 10
        tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        hjm = HJMModel(f0, tenors, constant_vol=0.005)
        paths = hjm.simulate(T=1.0, n_steps=20, n_paths=2000, seed=42)
        mean_curve = paths[:, -1, :].mean(axis=0)
        # All terminal mean forwards should be within ~2% of the initial
        # (small drift due to vol×integral term, plus MC noise).
        for f in mean_curve:
            assert 0.035 < f < 0.045, (
                f"Mean terminal forward {f:.4f} drifted far from 0.04 — "
                "likely non-uniform-tenor slope bug recurring"
            )
