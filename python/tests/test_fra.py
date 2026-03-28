"""Tests for forward rate agreement."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fra import FRA
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.bootstrap import bootstrap


def _flat_curve(ref: date, rate: float = 0.05) -> DiscountCurve:
    tenors_years = [0.25, 0.5, 1.0, 2.0, 5.0]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors_years]
    dfs = [math.exp(-rate * t) for t in tenors_years]
    return DiscountCurve(ref, dates, dfs)


class TestForwardRate:

    def test_forward_rate_positive(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.0)
        assert fra.forward_rate(curve) > 0

    def test_forward_rate_on_flat_curve(self):
        """On a flat continuously compounded curve, the simply compounded
        forward rate should be close to the flat rate."""
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.0)
        fwd = fra.forward_rate(curve)
        assert abs(fwd - 0.05) < 0.003


class TestPV:

    def test_pv_zero_at_par(self):
        """FRA struck at the forward rate has PV = 0."""
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.0)
        par = fra.par_rate(curve)
        fra_at_par = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=par)
        assert fra_at_par.pv(curve) == pytest.approx(0.0, abs=0.01)

    def test_pv_positive_when_strike_below_forward(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.01)
        assert fra.pv(curve) > 0

    def test_pv_negative_when_strike_above_forward(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.10)
        assert fra.pv(curve) < 0

    def test_pv_scales_with_notional(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra1 = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.04, notional=1_000_000.0)
        fra2 = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.04, notional=2_000_000.0)
        assert fra2.pv(curve) == pytest.approx(2 * fra1.pv(curve), rel=1e-10)

    def test_par_rate_equals_forward(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.0)
        assert fra.par_rate(curve) == pytest.approx(fra.forward_rate(curve), rel=1e-10)


class TestWithBootstrappedCurve:

    def test_fra_consistent_with_swap_curve(self):
        """FRA forward rate should be consistent with a bootstrapped curve."""
        ref = date(2024, 1, 15)
        deposits = [
            (ref + relativedelta(months=3), 0.0515),
            (ref + relativedelta(months=6), 0.0500),
        ]
        swaps = [
            (ref + relativedelta(years=1), 0.0480),
        ]
        curve = bootstrap(ref, deposits, swaps)

        # 3x6 FRA: forward rate from 3M to 6M
        fra = FRA(
            ref + relativedelta(months=3),
            ref + relativedelta(months=6),
            strike=0.0,
        )
        fwd = fra.forward_rate(curve)
        # Should be positive and in a reasonable range
        assert 0.01 < fwd < 0.10


class TestDualCurve:

    def test_single_curve_equivalent(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.04)
        assert fra.pv(curve) == pytest.approx(fra.pv(curve, projection_curve=curve), rel=1e-10)

    def test_dual_curve_different_pv(self):
        ref = date(2024, 1, 15)
        discount = _flat_curve(ref, rate=0.03)
        projection = _flat_curve(ref, rate=0.06)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.04)
        pv_single = fra.pv(discount)
        pv_dual = fra.pv(discount, projection_curve=projection)
        assert pv_single != pytest.approx(pv_dual, rel=1e-3)

    def test_par_rate_from_projection_curve(self):
        ref = date(2024, 1, 15)
        discount = _flat_curve(ref, rate=0.03)
        projection = _flat_curve(ref, rate=0.06)
        fra = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=0.0)
        par = fra.par_rate(discount, projection_curve=projection)
        fra_at_par = FRA(date(2024, 7, 15), date(2025, 1, 15), strike=par)
        assert fra_at_par.pv(discount, projection_curve=projection) == pytest.approx(0.0, abs=0.01)


class TestValidation:

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            FRA(date(2025, 1, 1), date(2024, 1, 1), strike=0.05)

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError):
            FRA(date(2024, 6, 15), date(2024, 6, 15), strike=0.05)

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            FRA(date(2024, 1, 1), date(2024, 7, 1), strike=0.05, notional=-100.0)
