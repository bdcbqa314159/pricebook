"""Tests for floating leg."""

import math
import pytest
from datetime import date

from pricebook.floating_leg import FloatingLeg
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from tests.conftest import make_flat_curve


class TestCashflows:
    """Cashflow generation."""

    def test_quarterly_1y_generates_4_cashflows(self):
        leg = FloatingLeg(
            date(2024, 1, 15), date(2025, 1, 15),
            frequency=Frequency.QUARTERLY,
        )
        assert len(leg.cashflows) == 4

    def test_semi_annual_2y_generates_4_cashflows(self):
        leg = FloatingLeg(
            date(2024, 1, 15), date(2026, 1, 15),
            frequency=Frequency.SEMI_ANNUAL,
        )
        assert len(leg.cashflows) == 4

    def test_forward_rate_positive(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY)
        for cf in leg.cashflows:
            assert cf.forward_rate(curve) > 0

    def test_cashflow_amount_includes_spread(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg_no_spread = FloatingLeg(
            ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY, spread=0.0,
        )
        leg_with_spread = FloatingLeg(
            ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY, spread=0.01,
        )
        cf_no = leg_no_spread.cashflows[0]
        cf_yes = leg_with_spread.cashflows[0]
        assert cf_yes.amount(curve) > cf_no.amount(curve)


class TestTelescopingProperty:
    """Single-curve floating PV should telescope: PV = notional * (df_start - df_end)."""

    def test_pv_telescopes_flat_curve(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        notional = 1_000_000.0
        leg = FloatingLeg(
            ref, date(2025, 1, 15),
            frequency=Frequency.QUARTERLY,
            notional=notional,
            spread=0.0,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        pv = leg.pv(curve)
        expected = notional * (curve.df(ref) - curve.df(date(2025, 1, 15)))
        assert pv == pytest.approx(expected, rel=1e-4)

    def test_pv_telescopes_steep_curve(self):
        """Telescoping should hold for any shape of curve."""
        ref = date(2024, 1, 15)
        # Build a curve with increasing zero rates
        tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
        rates = [0.03, 0.035, 0.04, 0.045, 0.05]
        dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
        dfs = [math.exp(-r * t) for r, t in zip(rates, tenors)]
        curve = DiscountCurve(ref, dates, dfs)

        notional = 1_000_000.0
        end = date(2026, 1, 15)
        leg = FloatingLeg(
            ref, end, frequency=Frequency.QUARTERLY,
            notional=notional, spread=0.0,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        pv = leg.pv(curve)
        expected = notional * (curve.df(ref) - curve.df(end))
        assert pv == pytest.approx(expected, rel=1e-3)

    def test_pv_with_spread_exceeds_no_spread(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg_flat = FloatingLeg(
            ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY, spread=0.0,
        )
        leg_spread = FloatingLeg(
            ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY, spread=0.005,
        )
        assert leg_spread.pv(curve) > leg_flat.pv(curve)


class TestPresentValue:
    """General PV properties."""

    def test_pv_positive(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY)
        assert leg.pv(curve) > 0

    def test_pv_scales_with_notional(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg1 = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY, notional=1_000_000.0)
        leg2 = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY, notional=2_000_000.0)
        assert leg2.pv(curve) == pytest.approx(2 * leg1.pv(curve), rel=1e-10)

    def test_longer_maturity_higher_pv(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg_1y = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY)
        leg_3y = FloatingLeg(ref, date(2027, 1, 15), frequency=Frequency.QUARTERLY)
        assert leg_3y.pv(curve) > leg_1y.pv(curve)


class TestDualCurve:
    """Dual-curve pricing: separate projection and discount curves."""

    def test_single_curve_equivalent(self):
        """pv(curve) == pv(curve, projection_curve=curve)."""
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        leg = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY)
        assert leg.pv(curve) == pytest.approx(leg.pv(curve, projection_curve=curve), rel=1e-10)

    def test_higher_projection_rate_higher_pv(self):
        """Higher forward rates from projection curve -> higher floating PV."""
        ref = date(2024, 1, 15)
        discount = make_flat_curve(ref, rate=0.03)
        proj_low = make_flat_curve(ref, rate=0.04)
        proj_high = make_flat_curve(ref, rate=0.06)
        leg = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY)
        assert leg.pv(discount, proj_high) > leg.pv(discount, proj_low)

    def test_dual_curve_differs_from_single(self):
        """Dual-curve PV should differ from single-curve when curves differ."""
        ref = date(2024, 1, 15)
        discount = make_flat_curve(ref, rate=0.03)
        projection = make_flat_curve(ref, rate=0.05)
        leg = FloatingLeg(ref, date(2025, 1, 15), frequency=Frequency.QUARTERLY)
        pv_single = leg.pv(discount)
        pv_dual = leg.pv(discount, projection_curve=projection)
        assert pv_single != pytest.approx(pv_dual, rel=1e-3)


class TestValidation:
    """Input validation."""

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError):
            FloatingLeg(date(2024, 1, 1), date(2025, 1, 1),
                        frequency=Frequency.QUARTERLY, notional=0.0)

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            FloatingLeg(date(2024, 1, 1), date(2025, 1, 1),
                        frequency=Frequency.QUARTERLY, notional=-100.0)
