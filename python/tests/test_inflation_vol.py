"""Tests for inflation caps/floors and vol surface."""

import math
import pytest
from datetime import date

from pricebook.inflation_vol import (
    zc_inflation_cap, yoy_inflation_cap, InflationVolSurface,
)
from pricebook.inflation import CPICurve, zc_inflation_swap_pv
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)
BASE_CPI = 300.0


def _dc(rate=0.03):
    return DiscountCurve.flat(REF, rate)


def _cpi(inflation_rate=0.025):
    """Flat CPI curve at given annual inflation."""
    dates = [date(2025, 1, 15), date(2029, 1, 15), date(2034, 1, 15)]
    levels = [BASE_CPI * (1 + inflation_rate) ** ((d - REF).days / 365.0) for d in dates]
    return CPICurve(REF, BASE_CPI, dates, levels)


# ---- ZC inflation cap ----

class TestZCInflationCap:
    def test_cap_positive(self):
        pv = zc_inflation_cap(REF, date(2029, 1, 15), 0.02, _cpi(), _dc(), 0.05)
        assert pv > 0

    def test_floor_positive(self):
        pv = zc_inflation_cap(
            REF, date(2029, 1, 15), 0.03, _cpi(), _dc(), 0.05,
            option_type=OptionType.PUT,
        )
        assert pv > 0

    def test_deep_itm_cap_approx_swap(self):
        """Deep ITM cap (low strike) ≈ inflation swap PV."""
        dc = _dc()
        cpi = _cpi(0.025)
        T = (date(2029, 1, 15) - REF).days / 365.0
        cap = zc_inflation_cap(REF, date(2029, 1, 15), 0.001, cpi, dc, 0.05)
        swap = zc_inflation_swap_pv(0.001, T, dc, cpi, date(2029, 1, 15))
        # Deep ITM cap ≈ swap + small time value
        assert cap > swap  # cap includes time value

    def test_higher_vol_higher_cap(self):
        dc = _dc()
        cpi = _cpi()
        low = zc_inflation_cap(REF, date(2029, 1, 15), 0.025, cpi, dc, 0.03)
        high = zc_inflation_cap(REF, date(2029, 1, 15), 0.025, cpi, dc, 0.10)
        assert high > low

    def test_higher_strike_lower_cap(self):
        dc = _dc()
        cpi = _cpi()
        low_k = zc_inflation_cap(REF, date(2029, 1, 15), 0.01, cpi, dc, 0.05)
        high_k = zc_inflation_cap(REF, date(2029, 1, 15), 0.05, cpi, dc, 0.05)
        assert high_k < low_k

    def test_expired_zero(self):
        pv = zc_inflation_cap(REF, REF, 0.02, _cpi(), _dc(), 0.05)
        assert pv == 0.0


# ---- YoY inflation cap ----

class TestYoYInflationCap:
    def test_cap_positive(self):
        pv = yoy_inflation_cap(REF, date(2029, 1, 15), 0.02, _cpi(), _dc(), 0.05)
        assert pv > 0

    def test_floor_positive(self):
        pv = yoy_inflation_cap(
            REF, date(2029, 1, 15), 0.03, _cpi(), _dc(), 0.05,
            option_type=OptionType.PUT,
        )
        assert pv > 0

    def test_higher_vol_higher_cap(self):
        dc = _dc()
        cpi = _cpi()
        low = yoy_inflation_cap(REF, date(2029, 1, 15), 0.025, cpi, dc, 0.03)
        high = yoy_inflation_cap(REF, date(2029, 1, 15), 0.025, cpi, dc, 0.10)
        assert high > low

    def test_longer_tenor_higher_cap(self):
        dc = _dc()
        cpi = _cpi()
        short = yoy_inflation_cap(REF, date(2027, 1, 15), 0.02, cpi, dc, 0.05)
        long = yoy_inflation_cap(REF, date(2034, 1, 15), 0.02, cpi, dc, 0.05)
        assert long > short


# ---- Inflation vol surface ----

class TestInflationVolSurface:
    def test_interpolation(self):
        surf = InflationVolSurface(
            REF,
            [date(2025, 1, 15), date(2029, 1, 15), date(2034, 1, 15)],
            [0.04, 0.05, 0.06],
        )
        v = surf.vol(date(2027, 1, 15))
        assert 0.04 < v < 0.06

    def test_flat_extrapolation(self):
        surf = InflationVolSurface(
            REF,
            [date(2025, 1, 15), date(2029, 1, 15)],
            [0.04, 0.05],
        )
        v_before = surf.vol(date(2024, 6, 15))
        v_after = surf.vol(date(2040, 1, 15))
        assert v_before == pytest.approx(0.04)
        assert v_after == pytest.approx(0.05)

    def test_bump(self):
        surf = InflationVolSurface(
            REF,
            [date(2029, 1, 15)],
            [0.05],
        )
        bumped = surf.bump(0.01)
        assert bumped.vol(date(2029, 1, 15)) == pytest.approx(0.06)

    def test_single_pillar(self):
        surf = InflationVolSurface(REF, [date(2029, 1, 15)], [0.05])
        assert surf.vol(date(2025, 1, 15)) == pytest.approx(0.05)
        assert surf.vol(date(2035, 1, 15)) == pytest.approx(0.05)
