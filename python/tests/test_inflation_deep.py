"""Deep tests for inflation — DD8 hardening.

Covers: ZC swap payoff, YoY swap, linker real yield, CPI curve, inflation cap.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.inflation import (
    zc_inflation_swap_pv, zc_inflation_par_rate,
    yoy_inflation_swap_pv, InflationLinkedBond, CPICurve,
)
from pricebook.inflation_vol import zc_inflation_cap
from pricebook.black76 import OptionType
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


def _make_cpi_curve(base_cpi=100.0, inflation_rate=0.02):
    dates = [REF + relativedelta(years=y) for y in range(1, 11)]
    cpis = [base_cpi * (1 + inflation_rate) ** y for y in range(1, 11)]
    return CPICurve(REF, base_cpi, dates, cpis)


class TestZCInflationSwap:

    def test_pv_positive_for_above_market_rate(self):
        """ZC swap paying below market inflation has positive PV (receiver infl)."""
        curve = make_flat_curve(REF, 0.03)
        cpi = _make_cpi_curve(inflation_rate=0.025)
        mat = REF + relativedelta(years=5)
        pv = zc_inflation_swap_pv(0.01, curve, cpi, mat)
        assert pv > 0  # receiving ~2.5% inflation, paying 1%

    def test_par_rate_positive(self):
        """Breakeven inflation should be positive."""
        curve = make_flat_curve(REF, 0.03)
        cpi = _make_cpi_curve(inflation_rate=0.02)
        mat = REF + relativedelta(years=5)
        par = zc_inflation_par_rate(curve, cpi, mat)
        assert par > 0


class TestYoYInflationSwap:

    def test_yoy_pv_positive(self):
        curve = make_flat_curve(REF, 0.03)
        cpi = _make_cpi_curve(inflation_rate=0.03)
        pv = yoy_inflation_swap_pv(0.01, curve, cpi,
                                    REF, REF + relativedelta(years=5))
        assert pv > 0


class TestInflationLinker:

    def test_linker_dirty_price_positive(self):
        curve = make_flat_curve(REF, 0.03)
        cpi = _make_cpi_curve()
        bond = InflationLinkedBond(REF, REF + relativedelta(years=10),
                                    coupon_rate=0.01, base_cpi_value=100.0)
        dp = bond.dirty_price(curve, cpi)
        assert dp > 0


class TestCPICurve:

    def test_cpi_at_base_is_base(self):
        cpi = _make_cpi_curve(base_cpi=100.0, inflation_rate=0.02)
        assert cpi.cpi(REF) == pytest.approx(100.0)

    def test_cpi_increases_with_positive_inflation(self):
        cpi = _make_cpi_curve(base_cpi=100.0, inflation_rate=0.02)
        assert cpi.cpi(REF + relativedelta(years=5)) > 100.0

    def test_cpi_round_trip(self):
        """CPI at pillar dates matches input."""
        base = 100.0
        rate = 0.025
        cpi = _make_cpi_curve(base_cpi=base, inflation_rate=rate)
        d5 = REF + relativedelta(years=5)
        expected = base * (1 + rate) ** 5
        assert cpi.cpi(d5) == pytest.approx(expected, rel=0.01)


class TestInflationCap:

    def test_zc_cap_positive(self):
        """ZC inflation cap has positive value."""
        curve = make_flat_curve(REF, 0.03)
        cpi = _make_cpi_curve(inflation_rate=0.02)
        price = zc_inflation_cap(
            REF, REF + relativedelta(years=5),
            strike_rate=0.02, cpi_curve=cpi,
            discount_curve=curve, vol=0.05,
        )
        assert price > 0

    def test_higher_vol_higher_cap(self):
        curve = make_flat_curve(REF, 0.03)
        cpi = _make_cpi_curve(inflation_rate=0.02)
        low = zc_inflation_cap(REF, REF + relativedelta(years=5),
                                strike_rate=0.02, cpi_curve=cpi,
                                discount_curve=curve, vol=0.02)
        high = zc_inflation_cap(REF, REF + relativedelta(years=5),
                                 strike_rate=0.02, cpi_curve=cpi,
                                 discount_curve=curve, vol=0.10)
        assert high > low
