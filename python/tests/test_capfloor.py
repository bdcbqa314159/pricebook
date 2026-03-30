"""Tests for IR caps and floors."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.capfloor import CapFloor
from pricebook.black76 import OptionType
from pricebook.vol_surface import FlatVol
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestCapPV:

    def test_cap_positive(self):
        curve = make_flat_curve(REF, 0.05)
        vol = FlatVol(0.20)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        assert cap.pv(curve, vol) > 0

    def test_floor_positive(self):
        curve = make_flat_curve(REF, 0.05)
        vol = FlatVol(0.20)
        floor = CapFloor(REF, REF + relativedelta(years=3), strike=0.05,
                         option_type=OptionType.PUT)
        assert floor.pv(curve, vol) > 0

    def test_higher_vol_higher_price(self):
        curve = make_flat_curve(REF, 0.05)
        cap_low = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        cap_high = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        assert cap_high.pv(curve, FlatVol(0.40)) > cap_low.pv(curve, FlatVol(0.10))

    def test_deep_otm_cap_near_zero(self):
        curve = make_flat_curve(REF, rate=0.03)
        vol = FlatVol(0.10)
        cap = CapFloor(REF, REF + relativedelta(years=2), strike=0.10)
        assert cap.pv(curve, vol) < 100.0  # very small relative to notional

    def test_longer_cap_higher_pv(self):
        curve = make_flat_curve(REF, 0.05)
        vol = FlatVol(0.20)
        cap_2y = CapFloor(REF, REF + relativedelta(years=2), strike=0.05)
        cap_5y = CapFloor(REF, REF + relativedelta(years=5), strike=0.05)
        assert cap_5y.pv(curve, vol) > cap_2y.pv(curve, vol)

    def test_pv_scales_with_notional(self):
        curve = make_flat_curve(REF, 0.05)
        vol = FlatVol(0.20)
        cap1 = CapFloor(REF, REF + relativedelta(years=3), strike=0.05, notional=1_000_000.0)
        cap2 = CapFloor(REF, REF + relativedelta(years=3), strike=0.05, notional=2_000_000.0)
        assert cap2.pv(curve, vol) == pytest.approx(2 * cap1.pv(curve, vol), rel=1e-10)


class TestCapFloorParity:
    """Cap - Floor = Swap (when strike = swap fixed rate, single curve)."""

    def test_cap_minus_floor_approx_swap(self):
        """cap(K) - floor(K) ≈ payer_swap(K) for ATM-ish strike."""
        curve = make_flat_curve(REF, rate=0.05)
        vol = FlatVol(0.20)
        end = REF + relativedelta(years=3)

        # Find approximately ATM strike
        swap = InterestRateSwap(
            REF, end, fixed_rate=0.0,
            fixed_frequency=Frequency.QUARTERLY,
            float_frequency=Frequency.QUARTERLY,
        )
        atm = swap.par_rate(curve)

        cap = CapFloor(REF, end, strike=atm, option_type=OptionType.CALL)
        floor = CapFloor(REF, end, strike=atm, option_type=OptionType.PUT)

        cap_pv = cap.pv(curve, vol)
        floor_pv = floor.pv(curve, vol)

        # Cap - Floor should be small at ATM (exact parity requires matched
        # day counts between swap and cap/floor; the residual comes from
        # the 30/360 vs ACT/360 day count mismatch in the swap par rate)
        assert abs(cap_pv - floor_pv) / 1_000_000 < 0.005  # < 50bp of notional

    def test_itm_cap_exceeds_floor(self):
        """When forward > strike, cap > floor."""
        curve = make_flat_curve(REF, rate=0.06)
        vol = FlatVol(0.20)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.04, option_type=OptionType.CALL)
        floor = CapFloor(REF, REF + relativedelta(years=3), strike=0.04, option_type=OptionType.PUT)
        assert cap.pv(curve, vol) > floor.pv(curve, vol)


class TestValidation:

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            CapFloor(REF, REF + relativedelta(years=1), strike=0.05, notional=-1.0)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            CapFloor(REF + relativedelta(years=1), REF, strike=0.05)
