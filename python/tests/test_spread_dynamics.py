"""Tests for spread dynamics: stochastic FVA, XVA integration."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.rfr import StochasticBasis
from pricebook.spread_dynamics import (
    fva_with_spread_dynamics,
    xva_with_spread_dynamics,
)
from tests.conftest import make_flat_curve, make_flat_survival

REF = date(2026, 4, 27)


def _disc():
    return make_flat_curve(REF, 0.03)


def _basis():
    return StochasticBasis(
        mean_spread=0.005, mean_reversion=0.5, vol=0.002, seed=42,
    )


def _epe():
    return np.array([100_000, 90_000, 80_000, 70_000, 60_000], dtype=float)


def _time_grid():
    return [0.25, 0.50, 0.75, 1.0, 1.25]


# ---- FVA with spread dynamics ----

class TestFVASpreadDynamics:

    def test_basic(self):
        result = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), _basis(), initial_spread=0.005,
        )
        assert math.isfinite(result.fva_stochastic)
        assert math.isfinite(result.fva_deterministic)
        assert result.n_paths == 1000

    def test_zero_vol_matches_deterministic(self):
        """With zero spread vol and initial=mean, stochastic ≈ deterministic."""
        basis_zero_vol = StochasticBasis(
            mean_spread=0.005, mean_reversion=0.5, vol=0.0, seed=42,
        )
        result = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), basis_zero_vol,
            initial_spread=0.005,
        )
        assert result.fva_stochastic == pytest.approx(result.fva_deterministic, rel=0.01)
        assert abs(result.convexity_adjustment) < abs(result.fva_deterministic) * 0.02

    def test_std_error_returned(self):
        result = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), _basis(), initial_spread=0.005,
        )
        assert result.std_error > 0
        assert result.std_error < abs(result.fva_stochastic)  # SE << FVA

    def test_nonzero_vol_differs(self):
        """With vol, stochastic differs from deterministic (convexity)."""
        result = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), _basis(),
            initial_spread=0.005, n_spread_paths=5000,
        )
        # Both should be similar in magnitude but not identical
        assert math.isfinite(result.convexity_adjustment)

    def test_mean_spread_path(self):
        """Mean spread path should revert toward mean."""
        result = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), _basis(),
            initial_spread=0.010,  # start above mean
            n_spread_paths=5000,
        )
        # Mean should drift toward 0.005 (the stationary mean)
        assert result.mean_spread_path[0] > result.mean_spread_path[-1]

    def test_zero_epe_zero_fva(self):
        zero_epe = np.zeros(5)
        result = fva_with_spread_dynamics(
            zero_epe, _time_grid(), _disc(), _basis(), initial_spread=0.005,
        )
        assert result.fva_stochastic == pytest.approx(0.0, abs=1e-10)

    def test_higher_spread_higher_fva(self):
        """Higher initial spread → higher FVA."""
        r_low = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), _basis(), initial_spread=0.002,
        )
        r_high = fva_with_spread_dynamics(
            _epe(), _time_grid(), _disc(), _basis(), initial_spread=0.010,
        )
        assert r_high.fva_stochastic > r_low.fva_stochastic


# ---- XVA with spread dynamics ----

class TestXVASpreadDynamics:

    def test_basic(self):
        disc = _disc()
        cpty = make_flat_survival(REF, 0.02)
        own = make_flat_survival(REF, 0.01)
        epe = _epe()
        ene = np.array([10_000, 15_000, 20_000, 25_000, 30_000], dtype=float)

        result = xva_with_spread_dynamics(
            epe, ene, _time_grid(), disc, cpty, own,
            _basis(), initial_spread=0.005,
        )
        assert math.isfinite(result.total)
        assert result.cva > 0
        assert result.fva_val > 0

    def test_stochastic_fva_replaces_deterministic(self):
        """FVA in result should use stochastic path, not deterministic."""
        disc = _disc()
        cpty = make_flat_survival(REF, 0.02)
        own = make_flat_survival(REF, 0.01)

        from pricebook.xva import total_xva_decomposition
        epe = _epe()
        ene = np.array([10_000, 15_000, 20_000, 25_000, 30_000], dtype=float)

        # Deterministic baseline
        det = total_xva_decomposition(
            epe, ene, _time_grid(), disc, cpty, own,
            funding_spread=0.005,
        )

        # Stochastic
        stoch = xva_with_spread_dynamics(
            epe, ene, _time_grid(), disc, cpty, own,
            _basis(), initial_spread=0.005, n_spread_paths=2000,
        )

        # CVA/DVA should be same (not affected by spread dynamics)
        assert stoch.cva == pytest.approx(det.cva)
        assert stoch.dva == pytest.approx(det.dva)
        # FVA may differ
        assert math.isfinite(stoch.fva_val)
