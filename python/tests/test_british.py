"""Tests for UK fixed income: SONIA swap, Gilt, ILG, breakeven."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestSONIACurve:
    def test_strip_generation(self):
        from pricebook.fixed_income.british import synthetic_sonia_strip
        strip = synthetic_sonia_strip(REF)
        assert len(strip) == 10
        assert all(c["rate"] > 0 for c in strip)

    def test_curve_dfs_valid(self):
        from pricebook.fixed_income.british import build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        assert all(0 < curve.df(c["maturity"]) < 1 for c in synthetic_sonia_strip(REF))

    def test_curve_dfs_decreasing(self):
        from pricebook.fixed_income.british import build_sonia_curve, synthetic_sonia_strip
        strip = synthetic_sonia_strip(REF)
        curve = build_sonia_curve(REF, strip)
        dfs = [curve.df(c["maturity"]) for c in strip]
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i-1] + 0.001


class TestSONIASwap:
    def test_dv01_positive(self):
        from pricebook.fixed_income.british import SONIASwap, build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        swap = SONIASwap(REF, REF + relativedelta(years=5), 0.045)
        assert swap.price(curve).dv01 > 0

    def test_par_rate_positive(self):
        from pricebook.fixed_income.british import SONIASwap, build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        swap = SONIASwap(REF, REF + relativedelta(years=10), 0.045)
        assert swap.price(curve).par_rate > 0

    def test_at_par_pv_near_zero(self):
        from pricebook.fixed_income.british import SONIASwap, build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        swap = SONIASwap(REF, REF + relativedelta(years=5), 0.045)
        par = swap.price(curve).par_rate
        at_par = SONIASwap(REF, REF + relativedelta(years=5), par)
        assert abs(at_par.price(curve).pv) < 100  # near zero for 10M notional

    def test_direction_symmetry(self):
        from pricebook.fixed_income.british import SONIASwap, build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        pay = SONIASwap(REF, REF + relativedelta(years=5), 0.045, direction=1)
        rec = SONIASwap(REF, REF + relativedelta(years=5), 0.045, direction=-1)
        assert pay.price(curve).pv == pytest.approx(-rec.price(curve).pv)


class TestGilt:
    def test_dirty_price_range(self):
        from pricebook.fixed_income.british import GiltBond, build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        gilt = GiltBond(REF, REF + relativedelta(years=10), 0.0375)
        price = gilt.dirty_price(curve)
        assert 70 < price < 110

    def test_higher_coupon_higher_price(self):
        from pricebook.fixed_income.british import GiltBond, build_sonia_curve, synthetic_sonia_strip
        curve = build_sonia_curve(REF, synthetic_sonia_strip(REF))
        low = GiltBond(REF, REF + relativedelta(years=10), 0.02)
        high = GiltBond(REF, REF + relativedelta(years=10), 0.06)
        assert high.dirty_price(curve) > low.dirty_price(curve)

    def test_synthetic_strip(self):
        from pricebook.fixed_income.british import synthetic_gilt_strip
        strip = synthetic_gilt_strip(REF)
        assert len(strip) == 4
        assert all(s["price"] > 0 for s in strip)


class TestILG:
    def test_real_price_positive(self):
        from pricebook.fixed_income.british import ILGBond, build_sonia_curve, synthetic_sonia_strip
        real_curve = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.01))
        ilg = ILGBond(REF, REF + relativedelta(years=30), 0.0125, base_rpi=300)
        r = ilg.price(REF, real_curve, current_rpi=380)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price  # RPI has risen

    def test_no_deflation_floor(self):
        """UK ILGs have NO deflation floor — unlike TIPS."""
        from pricebook.fixed_income.british import ILGBond, build_sonia_curve, synthetic_sonia_strip
        real_curve = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.01))
        ilg = ILGBond(REF, REF + relativedelta(years=10), 0.0125, base_rpi=300)
        r = ilg.price(REF, real_curve, current_rpi=280)  # deflation
        assert r.rpi_ratio < 1.0  # ratio below 1 = deflation
        assert r.nominal_price < r.real_price  # nominal reduced by deflation

    def test_real_yield_positive(self):
        from pricebook.fixed_income.british import ILGBond, build_sonia_curve, synthetic_sonia_strip
        real_curve = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.01))
        ilg = ILGBond(REF, REF + relativedelta(years=20), 0.0125, base_rpi=300)
        r = ilg.price(REF, real_curve, current_rpi=380)
        assert r.real_yield > 0

    def test_rpi_ratio_scales_nominal(self):
        from pricebook.fixed_income.british import ILGBond, build_sonia_curve, synthetic_sonia_strip
        real_curve = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.01))
        ilg = ILGBond(REF, REF + relativedelta(years=10), 0.0125, base_rpi=300)
        r = ilg.price(REF, real_curve, current_rpi=360)
        assert r.nominal_price == pytest.approx(r.real_price * r.rpi_ratio)


class TestBreakevenUK:
    def test_bei_positive(self):
        from pricebook.fixed_income.british import (
            breakeven_inflation_uk, build_sonia_curve, synthetic_sonia_strip)
        nom = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.045))
        real = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.01))
        bei = breakeven_inflation_uk(nom, real)
        assert len(bei) == 6  # 2,5,10,20,30,50Y
        assert all(row["bei"] > 0 for row in bei)

    def test_bei_magnitude(self):
        """UK BEI (RPI-based) typically ~3-4%."""
        from pricebook.fixed_income.british import (
            breakeven_inflation_uk, build_sonia_curve, synthetic_sonia_strip)
        nom = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.045))
        real = build_sonia_curve(REF, synthetic_sonia_strip(REF, sonia=0.01))
        bei = breakeven_inflation_uk(nom, real, [5, 10])
        assert all(0.02 < row["bei"] < 0.06 for row in bei)
