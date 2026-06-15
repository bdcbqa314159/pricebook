"""Regression for L2 T4 audit of `options.swaption_vol_cube._lookup_atm`:

Pre-fix the function used ``np.searchsorted`` + ``grid[i, j]`` —
lookup-with-round-up — rather than bilinear interpolation.  For any
query strictly between two pillars on either axis, the function
returned the UPPER-RIGHT cell's value instead of the interpolated one.

Used by ``build_swaption_vol_cube`` to populate each SABR node's
``atm_vol``, so off-pillar nodes carried a wrong ATM that leaked out
via the exactly-strike-equals-forward fast-path in ``SABRNode.vol``.

Fix (T4-SVC1): true bilinear interpolation matching ``_interp_atm``.
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.swaption_vol_cube import _lookup_atm


class TestLookupAtmInterpolates:
    def test_at_pillar_returns_grid_value(self):
        """Querying exactly at a pillar returns that pillar's value.
        Grid indexing: grid[expiry_idx][tenor_idx]."""
        expiries = [1.0, 2.0, 5.0, 10.0]
        tenors = [1.0, 5.0, 10.0]
        grid = [
            [0.10, 0.12, 0.14],   # expiry = 1y
            [0.11, 0.13, 0.15],   # expiry = 2y
            [0.13, 0.15, 0.17],   # expiry = 5y
            [0.14, 0.16, 0.18],   # expiry = 10y
        ]
        # (expiry=5y, tenor=5y) → grid[2][1] = 0.15.
        assert _lookup_atm(expiries, tenors, grid, 5.0, 5.0) == pytest.approx(0.15)
        # (expiry=5y, tenor=10y) → grid[2][2] = 0.17.
        assert _lookup_atm(expiries, tenors, grid, 5.0, 10.0) == pytest.approx(0.17)
        # (expiry=1y, tenor=1y) → grid[0][0] = 0.10.
        assert _lookup_atm(expiries, tenors, grid, 1.0, 1.0) == pytest.approx(0.10)

    def test_midway_between_pillars_interpolates(self):
        """Midway between pillars returns the bilinear average."""
        expiries = [1.0, 5.0]
        tenors = [1.0, 10.0]
        grid = [[0.10, 0.14], [0.13, 0.17]]
        # Center of the rectangle: average of all four corners.
        center = _lookup_atm(expiries, tenors, grid, 3.0, 5.5)
        expected = 0.25 * (0.10 + 0.14 + 0.13 + 0.17)
        assert center == pytest.approx(expected, rel=1e-12), (
            f"center = {center:.4f}, expected = {expected:.4f} — "
            f"_lookup_atm should interpolate, not lookup"
        )

    def test_extrapolation_clamps(self):
        """Query below the lowest pillar returns the lowest pillar (no
        extrapolation)."""
        expiries = [1.0, 5.0]
        tenors = [1.0, 10.0]
        grid = [[0.10, 0.14], [0.13, 0.17]]
        assert _lookup_atm(expiries, tenors, grid, 0.5, 0.5) == pytest.approx(0.10)
        assert _lookup_atm(expiries, tenors, grid, 100.0, 100.0) == pytest.approx(0.17)

    def test_far_from_pillar_is_interpolation_not_rounding(self):
        """Query at expiry = 3.0, tenor = 5.0 — between pillars on the
        expiry axis only.  Pre-fix would round expiry up to 5.0.  Post-fix
        should interpolate."""
        expiries = [1.0, 5.0]
        tenors = [5.0]
        grid = [[0.10], [0.20]]
        v = _lookup_atm(expiries, tenors, grid, 3.0, 5.0)
        # Linear interp at fraction 2/4 = 0.5 → 0.5 * 0.10 + 0.5 * 0.20 = 0.15.
        assert v == pytest.approx(0.15, rel=1e-12), (
            f"v = {v:.4f}, expected 0.15 — _lookup_atm should interpolate"
        )
