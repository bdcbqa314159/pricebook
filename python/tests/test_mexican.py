"""Tests for Mexican fixed income derivatives."""

import pytest
import math
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from pricebook.fixed_income.mexican import (
    TIIESwap, CETESBill, UDIBond,
    build_tiie_curve, synthetic_tiie_strip, synthetic_cetes_quotes,
)
from pricebook.core.discount_curve import DiscountCurve


REF = date(2024, 11, 4)


class TestTIIECurve:
    @pytest.fixture
    def tiie_strip(self):
        return synthetic_tiie_strip(REF, tiie_28d=0.1125, n_contracts=10)

    def test_strip_generates(self, tiie_strip):
        assert len(tiie_strip) == 10
        assert all(c["rate"] > 0 for c in tiie_strip)

    def test_build_curve(self, tiie_strip):
        curve = build_tiie_curve(REF, tiie_strip)
        for c in tiie_strip:
            df = curve.df(c["maturity"])
            assert 0 < df < 1

    def test_curve_dfs_decreasing(self, tiie_strip):
        curve = build_tiie_curve(REF, tiie_strip)
        dfs = [curve.df(c["maturity"]) for c in tiie_strip]
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i-1] + 1e-6


class TestTIIESwap:
    @pytest.fixture
    def curve(self):
        strip = synthetic_tiie_strip(REF, 0.1125, 10)
        return build_tiie_curve(REF, strip)

    def test_par_swap_near_zero(self, curve):
        mat = REF + relativedelta(years=2)
        swap = TIIESwap(REF, mat, 0.1125, direction=1)
        r = swap.price(curve)
        # Near par rate → PV near zero
        assert abs(r.pv) < r.notional * 0.05

    def test_above_par_positive_pv(self, curve):
        mat = REF + relativedelta(years=1)
        swap = TIIESwap(REF, mat, 0.15, direction=1)
        r = swap.price(curve)
        assert r.pv > 0

    def test_dv01_positive(self, curve):
        mat = REF + relativedelta(years=2)
        r = TIIESwap(REF, mat, 0.11).price(curve)
        assert r.dv01 > 0

    def test_28_day_periods(self, curve):
        mat = REF + timedelta(days=28 * 13)  # 13 periods
        r = TIIESwap(REF, mat, 0.11).price(curve)
        assert r.n_periods == 13


class TestCETES:
    def test_price_positive(self):
        cetes = CETESBill(REF + timedelta(days=91), 0.1100)
        r = cetes.price(REF)
        assert 0 < r.price < 10

    def test_discount_at_maturity(self):
        cetes = CETESBill(REF, 0.10)
        r = cetes.price(REF)
        assert abs(r.price - 10.0) < 0.01  # at maturity → face

    def test_higher_rate_lower_price(self):
        c_low = CETESBill(REF + timedelta(days=182), 0.10).price(REF)
        c_high = CETESBill(REF + timedelta(days=182), 0.12).price(REF)
        assert c_high.price < c_low.price

    def test_standard_tenors(self):
        quotes = synthetic_cetes_quotes(REF)
        assert len(quotes) == 4
        assert all(q["price"] > 0 for q in quotes)


class TestUDIBond:
    @pytest.fixture
    def real_curve(self):
        """Flat 4% real rate curve."""
        dates = [REF + relativedelta(years=i) for i in range(1, 11)]
        dfs = [math.exp(-0.04 * i) for i in range(1, 11)]
        return DiscountCurve(REF, dates, dfs)

    def test_udi_price_positive(self, real_curve):
        udi = UDIBond(REF, REF + relativedelta(years=5), 0.04, udi_at_issue=7.50)
        r = udi.price(REF, real_curve, current_udi=7.80)
        assert r.real_price > 0
        assert r.nominal_price > 0

    def test_nominal_scales_with_udi(self, real_curve):
        udi = UDIBond(REF, REF + relativedelta(years=5), 0.04)
        r1 = udi.price(REF, real_curve, current_udi=7.50)
        r2 = udi.price(REF, real_curve, current_udi=8.00)
        assert r2.nominal_price > r1.nominal_price  # higher UDI → higher MXN price

    def test_real_yield_positive(self, real_curve):
        udi = UDIBond(REF, REF + relativedelta(years=10), 0.035)
        r = udi.price(REF, real_curve, current_udi=7.80)
        assert r.real_yield > 0


class TestMBONOPricing:
    def test_mbono_via_sovereign(self):
        from pricebook.fixed_income.sovereign_bonds import create_sovereign_bond
        strip = synthetic_tiie_strip(REF, 0.1125, 10)
        curve = build_tiie_curve(REF, strip)
        bond = create_sovereign_bond("MBONO", REF, REF + relativedelta(years=5), 0.08)
        price = bond.dirty_price(curve)
        assert 70 < price < 110
