"""Tests for Japanese fixed income: TONA swap, JGB, JGBi, BEI."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestTONACurve:
    def test_curve_valid(self):
        from pricebook.fixed_income.japanese import build_tona_curve, synthetic_tona_strip
        strip = synthetic_tona_strip(REF)
        curve = build_tona_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) <= 1.01 for c in strip)  # near-zero rates

    def test_near_zero_rates(self):
        """Japan rates near zero — DFs should be close to 1 at short end."""
        from pricebook.fixed_income.japanese import build_tona_curve, synthetic_tona_strip
        strip = synthetic_tona_strip(REF, tona=0.001)  # 0.1%
        curve = build_tona_curve(REF, strip)
        assert curve.df(strip[0]["maturity"]) > 0.99


class TestTONASwap:
    def test_dv01(self):
        from pricebook.fixed_income.japanese import TONASwap, build_tona_curve, synthetic_tona_strip
        curve = build_tona_curve(REF, synthetic_tona_strip(REF))
        swap = TONASwap(REF, REF + relativedelta(years=10), 0.005)
        assert swap.price(curve).dv01 > 0

    def test_par_rate(self):
        from pricebook.fixed_income.japanese import TONASwap, build_tona_curve, synthetic_tona_strip
        curve = build_tona_curve(REF, synthetic_tona_strip(REF))
        swap = TONASwap(REF, REF + relativedelta(years=5), 0.005)
        assert swap.price(curve).par_rate > 0


class TestJGB:
    def test_price_range(self):
        from pricebook.fixed_income.japanese import JGBBond, build_tona_curve, synthetic_tona_strip
        curve = build_tona_curve(REF, synthetic_tona_strip(REF))
        jgb = JGBBond(REF, REF + relativedelta(years=10), 0.005)
        assert 80 < jgb.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        tona = get_rate_index("TONA")
        assert tona.currency == "JPY"
        assert tona.is_overnight


class TestJGBi:
    def test_deflation_floor(self):
        """JGBi has deflation floor — CPI ratio ≥ 1.0."""
        from pricebook.fixed_income.japanese import JGBiLinker, build_tona_curve, synthetic_tona_strip
        curve = build_tona_curve(REF, synthetic_tona_strip(REF, tona=0.001))
        jgbi = JGBiLinker(REF, REF + relativedelta(years=10), 0.001, base_cpi=105)
        r = jgbi.price(REF, curve, current_cpi=100)  # deflation
        assert r.cpi_ratio >= 1.0  # floor

    def test_inflation_increases_nominal(self):
        from pricebook.fixed_income.japanese import JGBiLinker, build_tona_curve, synthetic_tona_strip
        curve = build_tona_curve(REF, synthetic_tona_strip(REF, tona=0.001))
        jgbi = JGBiLinker(REF, REF + relativedelta(years=10), 0.001, base_cpi=100)
        r = jgbi.price(REF, curve, current_cpi=110)
        assert r.nominal_price > r.real_price

    def test_real_price_positive(self):
        from pricebook.fixed_income.japanese import JGBiLinker, build_tona_curve, synthetic_tona_strip
        curve = build_tona_curve(REF, synthetic_tona_strip(REF, tona=0.001))
        jgbi = JGBiLinker(REF, REF + relativedelta(years=20), 0.001, base_cpi=100)
        r = jgbi.price(REF, curve, current_cpi=105)
        assert r.real_price > 0

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_JP")
        assert cpi.currency == "JPY"
        assert cpi.deflation_floor is True


class TestBEI:
    def test_bei_positive(self):
        from pricebook.fixed_income.japanese import (
            breakeven_inflation_jp, build_tona_curve, synthetic_tona_strip)
        nom = build_tona_curve(REF, synthetic_tona_strip(REF, tona=0.005))
        real = build_tona_curve(REF, synthetic_tona_strip(REF, tona=-0.002))
        bei = breakeven_inflation_jp(nom, real)
        assert all(row["bei"] > 0 for row in bei)
