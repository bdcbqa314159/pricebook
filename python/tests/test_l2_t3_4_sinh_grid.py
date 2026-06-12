"""Regression for L2 Tier-3 T3.4 — SINH grid must land on [s_min, s_max]
exactly even when ``concentration_point`` ≠ midpoint.

Pre-fix the Tavella-Randall sinh grid used a symmetric `xi = linspace(-3, 3)`
and `alpha = 0.5 * (s_max - s_min) / sinh(3)`.  When `concentration_point`
was not the midpoint, the symmetric xi range pushed the grid endpoints past
`s_min` / `s_max` — for c < midpoint the grid extended BELOW s_min and could
go NEGATIVE when s_min was small (e.g. for the BS PDE with s_min = 0.01 *
spot).

Post-fix the routine picks `alpha` from the larger half-distance and solves
for xi_min, xi_max so the grid endpoints are exactly [s_min, s_max].
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._pde import GridType, build_grid


class TestSinhGridBounds:
    def test_midpoint_concentration_endpoints_exact(self):
        """Sanity: concentration at midpoint should still hit endpoints."""
        s_min, s_max = 1.0, 500.0
        grid = build_grid(s_min, s_max, 100, GridType.SINH,
                          concentration_point=(s_min + s_max) / 2)
        assert grid[0] == pytest.approx(s_min, abs=1e-10)
        assert grid[-1] == pytest.approx(s_max, abs=1e-10)

    def test_off_midpoint_no_negative_values(self):
        """The canonical bug: concentration BELOW the midpoint must not push
        the grid below s_min (and certainly not negative)."""
        s_min, s_max = 1.0, 500.0
        # Concentrate near strike (low end of the range).
        c = 50.0  # well below midpoint of 250.5
        grid = build_grid(s_min, s_max, 200, GridType.SINH, concentration_point=c)
        assert (grid > 0).all(), (
            f"SINH grid has non-positive values: min(grid) = {grid.min():.6f}"
        )
        # Endpoints should still be at s_min and s_max.
        assert grid[0] == pytest.approx(s_min, abs=1e-8)
        assert grid[-1] == pytest.approx(s_max, abs=1e-8)

    def test_concentration_above_midpoint_endpoints_exact(self):
        """Symmetric case: concentration above midpoint shouldn't push grid
        above s_max."""
        s_min, s_max = 1.0, 500.0
        c = 400.0  # well above midpoint
        grid = build_grid(s_min, s_max, 200, GridType.SINH, concentration_point=c)
        assert grid[0] == pytest.approx(s_min, abs=1e-8)
        assert grid[-1] == pytest.approx(s_max, abs=1e-8)

    def test_grid_is_monotone(self):
        """The SINH grid must be strictly monotone increasing."""
        for c in [10.0, 100.0, 250.0, 400.0]:
            grid = build_grid(1.0, 500.0, 150, GridType.SINH,
                              concentration_point=c)
            diffs = np.diff(grid)
            assert (diffs > 0).all(), (
                f"SINH grid not monotone for c={c}: min(diff) = {diffs.min():.4e}"
            )

    def test_concentration_density(self):
        """Density should be highest near the concentration point."""
        c = 100.0
        grid = build_grid(1.0, 500.0, 200, GridType.SINH, concentration_point=c)
        # Find nearest gridpoint to c.
        idx = int(np.argmin(np.abs(grid - c)))
        # Local spacing near c should be smaller than far from c.
        if 1 < idx < len(grid) - 2:
            local_spacing = grid[idx + 1] - grid[idx]
            far_spacing = grid[-1] - grid[-2]
            assert local_spacing < far_spacing, (
                f"density not concentrated near c={c}: "
                f"local={local_spacing:.3f}, far={far_spacing:.3f}"
            )
