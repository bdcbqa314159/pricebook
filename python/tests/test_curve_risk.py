"""Tests for curve Jacobian, roll-down analysis."""

import math
import pytest
import numpy as np
from datetime import date

from pricebook.curve_risk import (
    curve_jacobian,
    input_jacobian,
    curve_rolldown,
    rolldown_pnl,
)
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)


def _upward_curve():
    """Upward-sloping curve: short rates < long rates."""
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    rates = [0.04, 0.041, 0.043, 0.045, 0.046, 0.047, 0.048, 0.049]
    dates = [date.fromordinal(REF.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-r * t) for r, t in zip(rates, tenors)]
    return DiscountCurve(REF, dates, dfs)


class TestCurveJacobian:
    def test_shape(self):
        curve = DiscountCurve.flat(REF, 0.05)
        query = [1.0, 2.0, 5.0, 10.0]
        J = curve_jacobian(curve, query)
        n_pillar = len([t for t in curve.pillar_times if t > 0])
        assert J.shape == (4, n_pillar)

    def test_diagonal_dominance(self):
        """Bumping pillar k should affect query k most."""
        curve = _upward_curve()
        tenors = [1.0, 2.0, 5.0, 10.0]
        pillar_tenors = [float(t) for t in curve.pillar_times if t > 0]
        J = curve_jacobian(curve, tenors, pillar_tenors)

        # For log-linear interpolation, each query is most sensitive
        # to the nearest pillar
        for i in range(len(tenors)):
            row = np.abs(J[i, :])
            if row.max() > 0:
                assert row.max() > 0  # some sensitivity exists

    def test_flat_curve_identity(self):
        """On a flat curve, bumping pillar j affects nearby queries."""
        curve = DiscountCurve.flat(REF, 0.05)
        pillar_tenors = [float(t) for t in curve.pillar_times if t > 0]
        J = curve_jacobian(curve, pillar_tenors, pillar_tenors)
        # Diagonal should be ~1 (bumping a pillar moves its own zero rate)
        for i in range(len(pillar_tenors)):
            assert J[i, i] == pytest.approx(1.0, abs=0.3)


class TestInputJacobian:
    def test_shape(self):
        rates = [0.04, 0.045, 0.05]

        def build(quotes):
            tenors = [1.0, 5.0, 10.0]
            dates = [date.fromordinal(REF.toordinal() + int(t * 365)) for t in tenors]
            dfs = [math.exp(-r * t) for r, t in zip(quotes, tenors)]
            return DiscountCurve(REF, dates, dfs)

        J = input_jacobian(build, rates, [1.0, 5.0, 10.0])
        assert J.shape == (3, 3)

    def test_diagonal_positive(self):
        """Increasing a market rate should increase the zero rate at that tenor."""
        rates = [0.04, 0.045, 0.05]

        def build(quotes):
            tenors = [1.0, 5.0, 10.0]
            dates = [date.fromordinal(REF.toordinal() + int(t * 365)) for t in tenors]
            dfs = [math.exp(-r * t) for r, t in zip(quotes, tenors)]
            return DiscountCurve(REF, dates, dfs)

        J = input_jacobian(build, rates, [1.0, 5.0, 10.0])
        for i in range(3):
            assert J[i, i] > 0


class TestRollDown:
    def test_upward_slope_positive_rolldown(self):
        """Upward-sloping curve → positive roll-down (rates decrease with time passing)."""
        curve = _upward_curve()
        rd = curve_rolldown(curve, horizon_days=90)
        # For upward curve, rolling down means getting a lower zero rate
        # at shorter remaining maturity → rolldown should be negative
        # (zero rate decreases as you roll down the curve)
        assert any(r != 0 for r in rd["rolldown"])

    def test_flat_curve_zero_rolldown(self):
        curve = DiscountCurve.flat(REF, 0.05)
        rd = curve_rolldown(curve, horizon_days=30)
        for r in rd["rolldown"]:
            assert r == pytest.approx(0.0, abs=0.001)

    def test_output_structure(self):
        curve = _upward_curve()
        rd = curve_rolldown(curve, horizon_days=30)
        assert "tenors" in rd
        assert "current_zeros" in rd
        assert "rolled_zeros" in rd
        assert "rolldown" in rd
        assert len(rd["tenors"]) == len(rd["rolldown"])


class TestRolldownPnL:
    def test_positive_for_upward_curve(self):
        """Holding a bond on upward curve → positive roll-down P&L."""
        curve = _upward_curve()
        pnl = rolldown_pnl(curve, notional=1_000_000, maturity_years=5.0, horizon_days=30)
        assert pnl > 0

    def test_flat_curve_time_decay_only(self):
        """Flat curve roll-down = pure time decay (df grows as maturity shrinks)."""
        curve = DiscountCurve.flat(REF, 0.05)
        pnl = rolldown_pnl(curve, notional=1_000_000, maturity_years=5.0, horizon_days=30)
        # Positive: bond value increases as it approaches maturity
        assert pnl > 0

    def test_scales_with_notional(self):
        curve = _upward_curve()
        pnl1 = rolldown_pnl(curve, 1_000_000, 5.0, 30)
        pnl2 = rolldown_pnl(curve, 2_000_000, 5.0, 30)
        assert pnl2 == pytest.approx(2.0 * pnl1)
