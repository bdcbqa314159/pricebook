"""Tests for N-curve simultaneous global solver."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention
from pricebook.curves.ncurve_solver import (
    ncurve_solve, CurveSpec, NCurveResult,
    DepositPricer, OISSwapPricer, BasisSwapPricer,
)

REF = date(2024, 1, 15)


def _swap_dates(years):
    return [date(REF.year + y, REF.month, REF.day) for y in years]


class TestSingleCurve:
    def test_deposits_only(self):
        """Single curve from deposits → should recover exact DFs."""
        pillars = _swap_dates([1, 2, 3])
        spec = CurveSpec("ois", pillars)
        instruments = [
            DepositPricer("ois", pillars[0], 0.04, REF),
            DepositPricer("ois", pillars[1], 0.042, REF),
            DepositPricer("ois", pillars[2], 0.044, REF),
        ]
        result = ncurve_solve(REF, [spec], instruments)
        assert result.converged
        assert result.residual < 1e-8
        assert result.n_curves == 1

    def test_ois_swaps(self):
        """Single OIS curve from swap par rates."""
        pillars = _swap_dates([1, 2, 5, 10])
        spec = CurveSpec("ois", pillars)
        instruments = [
            DepositPricer("ois", pillars[0], 0.04, REF),
            OISSwapPricer("ois", pillars[1], 0.041, REF),
            OISSwapPricer("ois", pillars[2], 0.043, REF),
            OISSwapPricer("ois", pillars[3], 0.045, REF),
        ]
        result = ncurve_solve(REF, [spec], instruments)
        assert result.converged
        assert result.residual < 1e-6

    def test_dfs_positive(self):
        pillars = _swap_dates([1, 5, 10])
        spec = CurveSpec("ois", pillars)
        instruments = [
            DepositPricer("ois", pillars[0], 0.05, REF),
            OISSwapPricer("ois", pillars[1], 0.05, REF),
            OISSwapPricer("ois", pillars[2], 0.05, REF),
        ]
        result = ncurve_solve(REF, [spec], instruments)
        for d in pillars:
            assert result.curves["ois"].df(d) > 0


class TestDualCurve:
    def test_ois_plus_projection(self):
        """Two curves: OIS + projection (IBOR-style)."""
        ois_pillars = _swap_dates([1, 2, 5])
        proj_pillars = _swap_dates([1, 2, 5])

        specs = [
            CurveSpec("ois", ois_pillars),
            CurveSpec("proj", proj_pillars),
        ]
        instruments = [
            # OIS instruments
            DepositPricer("ois", ois_pillars[0], 0.04, REF),
            OISSwapPricer("ois", ois_pillars[1], 0.041, REF),
            OISSwapPricer("ois", ois_pillars[2], 0.043, REF),
            # Projection instruments (slightly higher)
            DepositPricer("proj", proj_pillars[0], 0.045, REF),
            OISSwapPricer("proj", proj_pillars[1], 0.046, REF),
            OISSwapPricer("proj", proj_pillars[2], 0.048, REF),
        ]
        result = ncurve_solve(REF, specs, instruments)
        assert result.converged
        assert result.n_curves == 2
        assert "ois" in result.curves
        assert "proj" in result.curves
        # Projection curve should have lower DFs (higher rates)
        assert result.curves["proj"].df(date(2029, 1, 15)) < result.curves["ois"].df(date(2029, 1, 15))

    def test_with_basis(self):
        """Two curves linked by a basis swap."""
        pillars = _swap_dates([1, 5])
        specs = [
            CurveSpec("disc", pillars),
            CurveSpec("proj", pillars),
        ]
        instruments = [
            DepositPricer("disc", pillars[0], 0.04, REF),
            OISSwapPricer("disc", pillars[1], 0.043, REF),
            DepositPricer("proj", pillars[0], 0.045, REF),
            BasisSwapPricer("disc", "proj", "disc", pillars[1], 0.005, REF),
        ]
        result = ncurve_solve(REF, specs, instruments)
        assert result.converged or result.residual < 1e-4


class TestThreeCurves:
    def test_ois_1m_3m(self):
        """Three curves: OIS + 1M projection + 3M projection."""
        pillars = _swap_dates([1, 3, 5])
        specs = [
            CurveSpec("ois", pillars),
            CurveSpec("proj_1m", pillars),
            CurveSpec("proj_3m", pillars),
        ]
        instruments = [
            DepositPricer("ois", pillars[0], 0.040, REF),
            OISSwapPricer("ois", pillars[1], 0.042, REF),
            OISSwapPricer("ois", pillars[2], 0.044, REF),
            DepositPricer("proj_1m", pillars[0], 0.042, REF),
            OISSwapPricer("proj_1m", pillars[1], 0.044, REF),
            OISSwapPricer("proj_1m", pillars[2], 0.046, REF),
            DepositPricer("proj_3m", pillars[0], 0.044, REF),
            OISSwapPricer("proj_3m", pillars[1], 0.046, REF),
            OISSwapPricer("proj_3m", pillars[2], 0.048, REF),
        ]
        result = ncurve_solve(REF, specs, instruments)
        assert result.n_curves == 3
        assert result.converged
        # Rate ordering: ois < 1m < 3m
        df_ois = result.curves["ois"].df(date(2029, 1, 15))
        df_1m = result.curves["proj_1m"].df(date(2029, 1, 15))
        df_3m = result.curves["proj_3m"].df(date(2029, 1, 15))
        assert df_ois > df_1m > df_3m


class TestEdgeCases:
    def test_to_dict(self):
        pillars = _swap_dates([1])
        spec = CurveSpec("ois", pillars)
        instruments = [DepositPricer("ois", pillars[0], 0.04, REF)]
        result = ncurve_solve(REF, [spec], instruments)
        d = result.to_dict()
        assert "n_curves" in d
        assert "converged" in d

    def test_single_instrument(self):
        pillars = _swap_dates([5])
        spec = CurveSpec("ois", pillars)
        instruments = [OISSwapPricer("ois", pillars[0], 0.04, REF)]
        result = ncurve_solve(REF, [spec], instruments)
        assert result.n_instruments == 1
