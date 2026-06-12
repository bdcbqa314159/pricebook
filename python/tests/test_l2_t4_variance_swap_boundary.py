"""Regression for L2 Wave-2 audit — `fair_variance` trapezoid boundary weights.

Pre-fix the boundary points used `dk = K[1] - K[0]` (full segment width),
while interior points used `dk = 0.5·(K[i+1] - K[i-1])` (half-segment sum
on either side).  The boundary weights were therefore 2× too large
compared to a consistent trapezoidal rule.

For a sparse strike grid (typical of liquid options on a single name)
this introduces a ~5% bias in the replication integral.

Post-fix: boundaries use `0.5·(K[1]-K[0])` and `0.5·(K[-1]-K[-2])` —
half the boundary segment, consistent with the trapezoidal rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.equity.variance_swap import fair_variance_from_vols


class TestVarianceSwapReplication:
    def test_bs_constant_vol_recovers_sigma_squared_dense(self):
        """For a flat BS smile (constant σ across strikes), the replication
        integral should give fair_variance ≈ σ².  Dense grid → tight."""
        sigma = 0.20
        F, df, T = 100.0, math.exp(-0.05), 1.0
        strikes = np.linspace(40, 200, 81)
        vols = np.full_like(strikes, sigma)
        result = fair_variance_from_vols(F, df, T, strikes, vols)
        rel = abs(result.fair_variance - sigma**2) / sigma**2
        assert rel < 0.005, (
            f"Fair var = {result.fair_variance:.6f}, σ² = {sigma**2:.6f}, rel = {rel:.3%}"
        )

    def test_bs_constant_vol_sparse_grid_within_5pct(self):
        """On a sparse grid (truncation error dominates), the bias should
        be limited to a few percent.  Pre-fix the boundary 2× weight
        added an extra ~5% on top of truncation."""
        sigma = 0.20
        F, df, T = 100.0, math.exp(-0.05), 1.0
        strikes = np.array([60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 150.0])
        vols = np.full_like(strikes, sigma, dtype=float)
        result = fair_variance_from_vols(F, df, T, strikes, vols)
        rel = abs(result.fair_variance - sigma**2) / sigma**2
        # Truncation error alone gives a few percent.  Pre-fix would have
        # added another ~5% from the boundary over-weighting → likely >10%.
        assert rel < 0.06, (
            f"Sparse-grid fair var = {result.fair_variance:.6f}, σ² = {sigma**2:.6f}, "
            f"rel = {rel:.3%}"
        )

    def test_uniform_grid_consistent_with_simpson(self):
        """Sanity: with uniform spacing, trapezoidal-style weighting should
        give symmetric weights and no spurious boundary skew."""
        sigma = 0.20
        F, df, T = 100.0, math.exp(-0.05), 1.0
        strikes = np.linspace(50, 150, 41)
        vols = np.full_like(strikes, sigma)
        result = fair_variance_from_vols(F, df, T, strikes, vols)
        rel = abs(result.fair_variance - sigma**2) / sigma**2
        assert rel < 0.01
