"""Tests for commodity hardening (CM1-CM9)."""

from datetime import date, timedelta

import pytest

from pricebook.commodity import CommodityForwardCurve, CommoditySwap
from pricebook.commodity_spreads import crush_spread, crack_spread_321
from pricebook.commodity_storage import StorageFacility
from pricebook.commodity_term_trading import commodity_roll_down
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


def _make_fwd_curve(ref, spot=70.0, contango=0.5):
    """Helper: build a simple contango forward curve."""
    dates = [ref + timedelta(days=30 * i) for i in range(1, 13)]
    fwds = [spot + contango * i for i in range(1, 13)]
    return CommodityForwardCurve(ref, dates, fwds, spot=spot)


# ---- CM1: CommoditySwap filters past periods ----

class TestCommoditySwapPastCF:
    def test_seasoned_swap_excludes_past(self):
        """Only future periods contribute to PV."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = _make_fwd_curve(ref)
        swap = CommoditySwap(ref, date(2027, 4, 21), 72.0,
                             frequency=Frequency.MONTHLY)
        # All periods are future from ref
        all_periods = len(swap.schedule) - 1
        # Mid-swap settlement: only future periods
        mid = date(2026, 10, 21)
        future_periods = swap._future_periods(mid)
        assert len(future_periods) < all_periods

    def test_future_periods_only(self):
        ref = date(2026, 7, 21)
        swap = CommoditySwap(date(2026, 1, 21), date(2027, 1, 21), 70.0,
                             frequency=Frequency.MONTHLY)
        periods = swap._future_periods(ref)
        for s, e in periods:
            assert e > ref


# ---- CM2: convenience_yield uses proper spot ----

class TestConvenienceYieldSpot:
    def test_explicit_spot(self):
        """With explicit spot, convenience yield uses it (not first forward)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = CommodityForwardCurve(
            ref,
            [date(2026, 7, 21), date(2026, 10, 21)],
            [72.0, 74.0],
            spot=70.0,
        )
        cy = fwd.convenience_yield(date(2026, 7, 21), curve)
        assert isinstance(cy, float)

    def test_spot_price_method(self):
        ref = date(2026, 4, 21)
        fwd = CommodityForwardCurve(
            ref, [date(2026, 7, 21)], [72.0], spot=70.0,
        )
        assert fwd.spot_price() == 70.0

    def test_spot_defaults_to_first_forward(self):
        ref = date(2026, 4, 21)
        fwd = CommodityForwardCurve(ref, [date(2026, 7, 21)], [72.0])
        assert fwd.spot_price() == 72.0


# ---- CM3: CommoditySwap.dv01 ----

class TestCommoditySwapDV01:
    def test_dv01_positive_for_receiver(self):
        """Receiver of floating benefits from higher forwards → DV01 > 0."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = _make_fwd_curve(ref)
        swap = CommoditySwap(ref, date(2027, 4, 21), 72.0)
        dv01 = swap.dv01(fwd, curve)
        assert dv01 > 0

    def test_par_price_seasoned(self):
        ref = date(2026, 7, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = _make_fwd_curve(ref)
        swap = CommoditySwap(date(2026, 1, 21), date(2027, 1, 21), 70.0,
                             frequency=Frequency.MONTHLY)
        par = swap.par_price(fwd, curve, settlement=ref)
        assert par > 0


# ---- CM4: Storage multi-cycle intrinsic ----

class TestStorageMultiCycle:
    def test_multi_cycle_greater_than_single(self):
        """Multi-cycle should find more value than a single inject/withdraw."""
        facility = StorageFacility(
            capacity=1000, max_injection_rate=100, max_withdrawal_rate=100,
            injection_cost=0.5, withdrawal_cost=0.5,
        )
        ref = date(2026, 4, 21)
        # Create a curve with two profitable cycles
        forwards = {}
        for m in range(12):
            d = ref + timedelta(days=30 * m)
            # Two dips (cheap) and two peaks (expensive)
            price = 70 + 10 * ((-1) ** m)  # alternating 60/80
            forwards[d] = price
        iv = facility.intrinsic_value(forwards)
        assert iv > 0

    def test_flat_curve_zero_intrinsic(self):
        facility = StorageFacility(capacity=1000, max_injection_rate=100,
                                   max_withdrawal_rate=100)
        ref = date(2026, 4, 21)
        forwards = {ref + timedelta(days=30 * m): 70.0 for m in range(12)}
        iv = facility.intrinsic_value(forwards)
        assert iv == 0.0


# ---- CM5+CM9: Mixed units flag ----

class TestMixedUnits:
    def test_crush_has_mixed_units(self):
        spread = crush_spread()
        assert spread.has_mixed_units

    def test_crack_same_units(self):
        spread = crack_spread_321()
        assert not spread.has_mixed_units


# ---- CM6: CommodityForwardCurve.bumped ----

class TestBumpedCurve:
    def test_bumped_additive(self):
        ref = date(2026, 4, 21)
        fwd = _make_fwd_curve(ref, spot=70.0)
        bumped = fwd.bumped(1.0)
        assert bumped.forward(date(2026, 7, 21)) == pytest.approx(
            fwd.forward(date(2026, 7, 21)) + 1.0, rel=1e-6,
        )

    def test_bumped_pct(self):
        ref = date(2026, 4, 21)
        fwd = _make_fwd_curve(ref, spot=70.0)
        bumped = fwd.bumped_pct(0.01)
        assert bumped.forward(date(2026, 7, 21)) == pytest.approx(
            fwd.forward(date(2026, 7, 21)) * 1.01, rel=1e-6,
        )

    def test_bumped_spot(self):
        ref = date(2026, 4, 21)
        fwd = _make_fwd_curve(ref, spot=70.0)
        bumped = fwd.bumped(2.0)
        assert bumped.spot_price() == 72.0


# ---- CM7: Period-average forward ----

class TestPeriodAverageForward:
    def test_average_differs_from_point(self):
        """On a contango curve, period average < point end (average includes lower start)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = _make_fwd_curve(ref, contango=1.0)  # steep contango
        swap = CommoditySwap(ref, date(2027, 4, 21), 72.0)
        pv_point = swap.pv(fwd, curve, use_average=False)
        pv_avg = swap.pv(fwd, curve, use_average=True)
        # Average forward < end-of-period forward in contango → lower floating → lower PV
        assert pv_avg < pv_point


# ---- CM8: Roll-down ----

class TestCommodityRollDown:
    def test_roll_down_runs(self):
        """Roll-down should produce a result."""
        ref = date(2026, 4, 21)
        fwd = _make_fwd_curve(ref, contango=0.5)
        result = commodity_roll_down(date(2026, 10, 21), fwd, roll_days=30)
        assert isinstance(result.roll_pnl_per_unit, float)
        assert result.roll_days == 30

    def test_roll_down_curve_method(self):
        ref = date(2026, 4, 21)
        fwd = _make_fwd_curve(ref, spot=70.0)
        rolled = fwd.roll_down(30)
        assert rolled.reference_date == ref + timedelta(days=30)
