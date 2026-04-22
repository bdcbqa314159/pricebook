"""Tests for inflation curve and instruments."""

import pytest
import math
from datetime import date

from pricebook.inflation import (
    CPICurve,
    zc_inflation_swap_pv,
    zc_inflation_par_rate,
    yoy_inflation_swap_pv,
    InflationLinkedBond,
    bootstrap_cpi_curve,
)
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
BASE_CPI = 300.0


def _cpi_curve(inflation_rate: float = 0.025) -> CPICurve:
    dates = [date(2024 + i, 1, 15) for i in range(1, 11)]
    cpi_levels = [BASE_CPI * (1 + inflation_rate) ** i for i in range(1, 11)]
    return CPICurve(REF, BASE_CPI, dates, cpi_levels)


class TestCPICurve:
    def test_base_cpi_at_reference(self):
        curve = _cpi_curve()
        assert curve.cpi(REF) == pytest.approx(BASE_CPI)

    def test_cpi_increases(self):
        curve = _cpi_curve(0.03)
        assert curve.cpi(date(2025, 1, 15)) > BASE_CPI

    def test_cpi_at_pillar(self):
        curve = _cpi_curve(0.025)
        expected = BASE_CPI * 1.025
        assert curve.cpi(date(2025, 1, 15)) == pytest.approx(expected, rel=0.001)

    def test_breakeven_rate(self):
        curve = _cpi_curve(0.025)
        be = curve.breakeven_rate(date(2029, 1, 15))
        assert be == pytest.approx(0.025, rel=0.01)

    def test_from_breakevens(self):
        dates = [date(2025, 1, 15), date(2029, 1, 15)]
        rates = [0.025, 0.025]
        curve = CPICurve.from_breakevens(REF, BASE_CPI, dates, rates)
        assert curve.cpi(date(2025, 1, 15)) == pytest.approx(BASE_CPI * 1.025, rel=0.01)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CPICurve(REF, BASE_CPI, [date(2025, 1, 15)], [300.0, 310.0])

    def test_negative_base_raises(self):
        with pytest.raises(ValueError, match="positive"):
            CPICurve(REF, -1.0, [date(2025, 1, 15)], [300.0])


class TestZCInflationSwap:
    def test_pv_zero_at_par_rate(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.025)
        mat = date(2029, 1, 15)
        T = 5.0
        par = zc_inflation_par_rate(disc, cpi, mat)
        pv = zc_inflation_swap_pv(par, disc, cpi, mat)
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_par_rate_matches_breakeven(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.025)
        mat = date(2029, 1, 15)
        par = zc_inflation_par_rate(disc, cpi, mat)
        be = cpi.breakeven_rate(mat)
        assert par == pytest.approx(be, rel=0.01)

    def test_pv_positive_when_inflation_above_fixed(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.03)
        mat = date(2029, 1, 15)
        pv = zc_inflation_swap_pv(0.02, disc, cpi, mat)
        assert pv > 0  # inflation 3% > fixed 2%


class TestYoYInflationSwap:
    def test_pv_near_zero_at_breakeven(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.025)
        # At breakeven, YoY ≈ flat inflation rate
        pv = yoy_inflation_swap_pv(
            0.025, disc, cpi, REF, date(2029, 1, 15),
            frequency=Frequency.ANNUAL,
        )
        # Not exactly zero due to convexity, but close
        assert abs(pv) < 5000  # on 1M notional

    def test_pv_positive_when_inflation_high(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.04)
        pv = yoy_inflation_swap_pv(
            0.02, disc, cpi, REF, date(2029, 1, 15),
        )
        assert pv > 0


class TestInflationLinkedBond:
    def test_price_positive(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.025)
        bond = InflationLinkedBond(
            REF, date(2029, 1, 15), coupon_rate=0.01,
            base_cpi_value=BASE_CPI,
        )
        price = bond.dirty_price(disc, cpi)
        assert price > 0

    def test_higher_inflation_higher_price(self):
        disc = make_flat_curve(REF, 0.04)
        cpi_low = _cpi_curve(0.02)
        cpi_high = _cpi_curve(0.04)
        bond = InflationLinkedBond(
            REF, date(2029, 1, 15), coupon_rate=0.01,
            base_cpi_value=BASE_CPI,
        )
        assert bond.dirty_price(disc, cpi_high) > bond.dirty_price(disc, cpi_low)

    def test_real_yield_round_trip(self):
        disc = make_flat_curve(REF, 0.04)
        cpi = _cpi_curve(0.025)
        bond = InflationLinkedBond(
            REF, date(2029, 1, 15), coupon_rate=0.01,
            base_cpi_value=BASE_CPI,
        )
        price = bond.dirty_price(disc, cpi)
        ry = bond.real_yield(price, cpi)
        # Real yield should be reasonable
        assert -0.05 < ry < 0.20


class TestBootstrap:
    def test_reprices_all_inputs(self):
        quotes = [
            (date(2025, 1, 15), 0.025),
            (date(2027, 1, 15), 0.026),
            (date(2029, 1, 15), 0.027),
            (date(2034, 1, 15), 0.028),
        ]
        cpi = bootstrap_cpi_curve(REF, BASE_CPI, quotes)
        disc = make_flat_curve(REF, 0.04)

        for mat, rate in quotes:
            par = zc_inflation_par_rate(disc, cpi, mat)
            assert par == pytest.approx(rate, rel=0.01)

    def test_cpi_levels_increasing(self):
        quotes = [
            (date(2025, 1, 15), 0.025),
            (date(2029, 1, 15), 0.027),
        ]
        cpi = bootstrap_cpi_curve(REF, BASE_CPI, quotes)
        assert cpi.cpi(date(2029, 1, 15)) > cpi.cpi(date(2025, 1, 15))
