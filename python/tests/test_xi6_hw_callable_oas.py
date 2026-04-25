"""XI6: Hull-White → Callable Bond → OAS → Bermudan integration chain.

Calibrate HW to curve → price callable ≤ straight → OAS ≈ 0 for model curve →
Bermudan ≥ European. Verify HW ZCB matches curve DF.

Bug hotspots:
- HW _forward_rate(0) must match curve.instantaneous_forward(0)
- Tree rounding of exercise dates
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bootstrap import bootstrap
from pricebook.bermudan_swaption import bermudan_swaption_tree
from pricebook.callable_bond import callable_bond_price, oas, _straight_bond_hw
from pricebook.discount_curve import DiscountCurve
from pricebook.hull_white import HullWhite


# ---- Helpers ----

REF = date(2026, 4, 25)


def _curve(ref: date) -> DiscountCurve:
    deposits = [
        (ref + timedelta(days=91), 0.040),
        (ref + timedelta(days=182), 0.039),
    ]
    swaps = [
        (ref + timedelta(days=365), 0.038),
        (ref + timedelta(days=730), 0.037),
        (ref + timedelta(days=1095), 0.036),
        (ref + timedelta(days=1825), 0.035),
        (ref + timedelta(days=3650), 0.034),
    ]
    return bootstrap(ref, deposits, swaps)


def _hw(ref: date) -> HullWhite:
    return HullWhite(a=0.05, sigma=0.01, curve=_curve(ref))


# ---- R1: HW calibration to curve ----

class TestXI6R1HWCurve:
    """HW ZCB price must match curve discount factor."""

    def test_zcb_matches_curve_df_5y(self):
        """HW analytical ZCB(0, 5, r0) ≈ curve.df(5Y)."""
        curve = _curve(REF)
        hw = HullWhite(a=0.05, sigma=0.01, curve=curve)

        d5 = REF + timedelta(days=1825)
        T = 5.0
        r0 = hw._forward_rate(0.0)

        zcb = hw.zcb_price(0.0, T, r0)
        market_df = curve.df(d5)
        assert zcb == pytest.approx(market_df, rel=0.02)

    def test_tree_zcb_matches_curve(self):
        """Tree-based ZCB should match curve DF."""
        curve = _curve(REF)
        hw = HullWhite(a=0.05, sigma=0.01, curve=curve)

        d5 = REF + timedelta(days=1825)
        tree_df = hw.tree_zcb(T=5.0, n_steps=100)
        market_df = curve.df(d5)
        assert tree_df == pytest.approx(market_df, rel=0.02)

    def test_zcb_multiple_tenors(self):
        """ZCB matches curve DF at multiple tenors."""
        curve = _curve(REF)
        hw = HullWhite(a=0.05, sigma=0.01, curve=curve)
        r0 = hw._forward_rate(0.0)

        for years, days in [(1, 365), (2, 730), (5, 1825)]:
            d = REF + timedelta(days=days)
            zcb = hw.zcb_price(0.0, float(years), r0)
            df = curve.df(d)
            assert zcb == pytest.approx(df, rel=0.02), f"Mismatch at {years}Y"


# ---- R2: Callable ≤ Straight ----

class TestXI6R2CallableStraight:
    """Callable bond price must be ≤ straight bond price."""

    def test_callable_leq_straight(self):
        """Callable ≤ straight (call option has non-negative value to issuer)."""
        hw = _hw(REF)
        straight = _straight_bond_hw(hw, coupon_rate=0.04, maturity_years=10.0,
                                      n_steps=100)
        call_dates = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        callable_p = callable_bond_price(hw, coupon_rate=0.04, maturity_years=10.0,
                                          call_dates_years=call_dates, n_steps=100)
        assert callable_p <= straight + 0.01  # small tolerance for numerics

    def test_callable_positive(self):
        """Callable bond price should be positive."""
        hw = _hw(REF)
        call_dates = [3.0, 4.0, 5.0]
        price = callable_bond_price(hw, coupon_rate=0.04, maturity_years=5.0,
                                     call_dates_years=call_dates, n_steps=100)
        assert price > 0

    def test_higher_coupon_higher_price(self):
        """Higher coupon → higher callable bond price."""
        hw = _hw(REF)
        call_dates = [3.0, 4.0, 5.0]
        low = callable_bond_price(hw, coupon_rate=0.02, maturity_years=5.0,
                                   call_dates_years=call_dates, n_steps=100)
        high = callable_bond_price(hw, coupon_rate=0.06, maturity_years=5.0,
                                    call_dates_years=call_dates, n_steps=100)
        assert high > low


# ---- R3: OAS ----

class TestXI6R3OAS:
    """OAS: spread that reprices callable to market."""

    def test_oas_near_zero_at_model_price(self):
        """OAS should be ≈ 0 when market price = model price."""
        hw = _hw(REF)
        call_dates = [3.0, 4.0, 5.0]
        model_price = callable_bond_price(hw, coupon_rate=0.04, maturity_years=5.0,
                                           call_dates_years=call_dates, n_steps=100)
        oas_val = oas(hw, market_price=model_price, coupon_rate=0.04,
                      maturity_years=5.0, call_put_dates=call_dates, n_steps=100)
        assert abs(oas_val) < 0.005  # < 50bp

    def test_oas_positive_for_cheap_bond(self):
        """Bond trading below model → positive OAS (cheap)."""
        hw = _hw(REF)
        call_dates = [3.0, 4.0, 5.0]
        model_price = callable_bond_price(hw, coupon_rate=0.04, maturity_years=5.0,
                                           call_dates_years=call_dates, n_steps=100)
        cheap_price = model_price - 2.0  # $2 below model
        oas_val = oas(hw, market_price=cheap_price, coupon_rate=0.04,
                      maturity_years=5.0, call_put_dates=call_dates, n_steps=100)
        assert oas_val > 0

    def test_oas_negative_for_rich_bond(self):
        """Bond trading above model → negative OAS (rich)."""
        hw = _hw(REF)
        call_dates = [3.0, 4.0, 5.0]
        model_price = callable_bond_price(hw, coupon_rate=0.04, maturity_years=5.0,
                                           call_dates_years=call_dates, n_steps=100)
        rich_price = model_price + 2.0
        oas_val = oas(hw, market_price=rich_price, coupon_rate=0.04,
                      maturity_years=5.0, call_put_dates=call_dates, n_steps=100)
        assert oas_val < 0


# ---- R4: Bermudan ≥ European ----

class TestXI6R4BermudanEuropean:
    """Bermudan swaption ≥ European swaption (more optionality)."""

    def test_bermudan_geq_european(self):
        """Bermudan with multiple exercise dates ≥ single European."""
        hw = _hw(REF)
        strike = 0.035

        european = bermudan_swaption_tree(
            hw, exercise_years=[2.0], swap_end_years=10.0,
            strike=strike, n_steps=100,
        )
        bermudan = bermudan_swaption_tree(
            hw, exercise_years=[2.0, 3.0, 4.0, 5.0], swap_end_years=10.0,
            strike=strike, n_steps=100,
        )
        assert bermudan >= european * 0.99  # allow small MC noise

    def test_swaption_positive(self):
        """ATM swaption should have positive value."""
        hw = _hw(REF)
        pv = bermudan_swaption_tree(
            hw, exercise_years=[2.0], swap_end_years=7.0,
            strike=0.035, n_steps=100,
        )
        assert pv > 0

    def test_higher_vol_higher_swaption(self):
        """Higher HW sigma → higher swaption price."""
        curve = _curve(REF)
        hw_low = HullWhite(a=0.05, sigma=0.005, curve=curve)
        hw_high = HullWhite(a=0.05, sigma=0.020, curve=curve)

        pv_low = bermudan_swaption_tree(
            hw_low, exercise_years=[2.0], swap_end_years=7.0,
            strike=0.035, n_steps=100,
        )
        pv_high = bermudan_swaption_tree(
            hw_high, exercise_years=[2.0], swap_end_years=7.0,
            strike=0.035, n_steps=100,
        )
        assert pv_high > pv_low
