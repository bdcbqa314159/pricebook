"""Tests for IR caps and floors."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.options.capfloor import CapFloor
from pricebook.models.black76 import OptionType
from pricebook.models.models import Black76Model
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


def _model(vol=0.20):
    return Black76Model(vol=vol)


class TestCapPV:

    def test_cap_positive(self):
        curve = make_flat_curve(REF, 0.05)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        assert cap.price(_model(), curve) > 0

    def test_floor_positive(self):
        curve = make_flat_curve(REF, 0.05)
        floor = CapFloor(REF, REF + relativedelta(years=3), strike=0.05,
                         option_type=OptionType.PUT)
        assert floor.price(_model(), curve) > 0

    def test_higher_vol_higher_price(self):
        curve = make_flat_curve(REF, 0.05)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        assert cap.price(_model(0.40), curve) > cap.price(_model(0.10), curve)

    def test_deep_otm_cap_near_zero(self):
        curve = make_flat_curve(REF, rate=0.03)
        cap = CapFloor(REF, REF + relativedelta(years=2), strike=0.10)
        assert cap.price(_model(0.10), curve) < 100.0

    def test_longer_cap_higher_pv(self):
        curve = make_flat_curve(REF, 0.05)
        cap_2y = CapFloor(REF, REF + relativedelta(years=2), strike=0.05)
        cap_5y = CapFloor(REF, REF + relativedelta(years=5), strike=0.05)
        assert cap_5y.price(_model(), curve) > cap_2y.price(_model(), curve)

    def test_pv_scales_with_notional(self):
        curve = make_flat_curve(REF, 0.05)
        cap1 = CapFloor(REF, REF + relativedelta(years=3), strike=0.05, notional=1_000_000.0)
        cap2 = CapFloor(REF, REF + relativedelta(years=3), strike=0.05, notional=2_000_000.0)
        assert cap2.price(_model(), curve) == pytest.approx(2 * cap1.price(_model(), curve), rel=1e-10)


class TestCapFloorParity:

    def test_cap_minus_floor_approx_swap(self):
        curve = make_flat_curve(REF, rate=0.05)
        end = REF + relativedelta(years=3)

        swap = InterestRateSwap(
            REF, end, fixed_rate=0.0,
            fixed_frequency=Frequency.QUARTERLY,
            float_frequency=Frequency.QUARTERLY,
        )
        atm = swap.par_rate(curve)

        cap = CapFloor(REF, end, strike=atm, option_type=OptionType.CALL)
        floor = CapFloor(REF, end, strike=atm, option_type=OptionType.PUT)

        cap_pv = cap.price(_model(), curve)
        floor_pv = floor.price(_model(), curve)

        assert abs(cap_pv - floor_pv) / 1_000_000 < 0.005

    def test_itm_cap_exceeds_floor(self):
        curve = make_flat_curve(REF, rate=0.06)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.04, option_type=OptionType.CALL)
        floor = CapFloor(REF, REF + relativedelta(years=3), strike=0.04, option_type=OptionType.PUT)
        assert cap.price(_model(), curve) > floor.price(_model(), curve)


class TestValidation:

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            CapFloor(REF, REF + relativedelta(years=1), strike=0.05, notional=-1.0)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            CapFloor(REF + relativedelta(years=1), REF, strike=0.05)
