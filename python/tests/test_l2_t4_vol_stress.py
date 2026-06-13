"""Regression for L2 phase-2 audit of `risk.vol_stress.twist_vol_bump`:

Pre-fix formula ``((T - T_mid) / W)² - 0.25`` evaluates to ``0`` at
both wings (x=±0.5: 0.25 - 0.25 = 0) and ``-0.25`` at the belly
(x=0).  That's "belly down, wings unchanged" — NOT the butterfly
shape promised by the docstring "wings up, belly down".

Fix: ``4·x² - 0.5`` gives wings = +0.5, belly = -0.5 — a true butterfly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.vol_stress import (
    parallel_vol_bump, tilt_vol_bump, twist_vol_bump,
)


class TestTwistButterfly:
    def test_wings_up_belly_down(self):
        """Butterfly: bumped vol at wings > base; at belly < base."""
        tenors = [1.0, 2.0, 5.0, 10.0, 20.0]  # 5 tenors symmetric around 10.5? No: midpoint = 10.5.
        # Use a symmetric grid for clarity.
        tenors = [1.0, 5.5, 10.0]  # T_mid = 5.5
        base_vols = [0.20, 0.20, 0.20]
        vega = [100.0, 100.0, 100.0]
        twist_bps = 100  # → wings +50bp, belly -50bp
        result = twist_vol_bump(tenors, base_vols, vega, twist_bps=twist_bps)
        bumped = result.bumped_vols
        # Wings (idx 0, 2) should be ABOVE base.
        assert bumped[0] > base_vols[0]
        assert bumped[2] > base_vols[2]
        # Belly (idx 1) should be BELOW base.
        assert bumped[1] < base_vols[1]
        # Symmetric: shift at idx 0 = shift at idx 2 (true butterfly).
        shift_left = bumped[0] - base_vols[0]
        shift_right = bumped[2] - base_vols[2]
        assert shift_left == pytest.approx(shift_right, rel=1e-9)

    def test_butterfly_magnitudes(self):
        """Wing shift = +0.5 × twist_bps/10000; belly = -0.5 × twist_bps/10000."""
        tenors = [1.0, 5.5, 10.0]
        base_vols = [0.20, 0.20, 0.20]
        vega = [100.0, 100.0, 100.0]
        twist_bps = 100
        result = twist_vol_bump(tenors, base_vols, vega, twist_bps=twist_bps)
        bumped = result.bumped_vols
        wing_shift = bumped[0] - base_vols[0]
        belly_shift = bumped[1] - base_vols[1]
        # Wings: +50bp = 0.005.  Belly: -50bp = -0.005.
        assert wing_shift == pytest.approx(0.005, abs=1e-9)
        assert belly_shift == pytest.approx(-0.005, abs=1e-9)


class TestParallelTilt:
    """Sanity checks for the other two — should still work after the fix."""

    def test_parallel_uniform_shift(self):
        result = parallel_vol_bump([0.20, 0.21, 0.22], [10, 20, 30], bump_bps=100)
        # All bumped by +0.01.
        np.testing.assert_allclose(result.bumped_vols, [0.21, 0.22, 0.23], atol=1e-12)

    def test_tilt_steepening(self):
        result = tilt_vol_bump([1.0, 5.5, 10.0], [0.20, 0.20, 0.20], [10, 20, 30], tilt_bps=100)
        # Short end down, long end up (steepening).
        assert result.bumped_vols[0] < 0.20
        assert result.bumped_vols[-1] > 0.20
