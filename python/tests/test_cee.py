"""Tests for CEE markets: Poland, Czech Republic, Hungary."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestPoland:
    def test_pln_curve(self):
        from pricebook.fixed_income.polish import build_pln_curve, synthetic_wibor_strip
        strip = synthetic_wibor_strip(REF)
        curve = build_pln_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_wibor_swap(self):
        from pricebook.fixed_income.polish import WIBORSwap, build_pln_curve, synthetic_wibor_strip
        curve = build_pln_curve(REF, synthetic_wibor_strip(REF))
        swap = WIBORSwap(REF, REF + relativedelta(years=5), 0.0575)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_wiron_swap(self):
        from pricebook.fixed_income.polish import WIRONSwap, build_pln_curve, synthetic_wibor_strip
        curve = build_pln_curve(REF, synthetic_wibor_strip(REF))
        swap = WIRONSwap(REF, REF + relativedelta(years=5), 0.0575)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_polgb_bond(self):
        from pricebook.fixed_income.polish import POLGBBond, build_pln_curve, synthetic_wibor_strip
        curve = build_pln_curve(REF, synthetic_wibor_strip(REF))
        bond = POLGBBond(REF, REF + relativedelta(years=10), 0.055)
        assert 70 < bond.dirty_price(curve) < 110

    def test_polgb_linker(self):
        from pricebook.fixed_income.polish import POLGBLinker, build_pln_curve, synthetic_wibor_strip
        curve = build_pln_curve(REF, synthetic_wibor_strip(REF, rate=0.02))
        linker = POLGBLinker(REF, REF + relativedelta(years=10), 0.025)
        r = linker.price(REF, curve, current_cpi=115.0)
        assert r.real_price > 0
        assert r.nominal_price > 0

    def test_rate_indices(self):
        from pricebook.core.rate_index import get_rate_index
        wibor = get_rate_index("WIBOR_3M")
        assert wibor.currency == "PLN"
        wiron = get_rate_index("WIRON")
        assert wiron.is_overnight

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_PL")
        assert cpi.currency == "PLN"

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        polgb = get_conventions("POLGB")
        assert polgb.currency == "PLN"


class TestCzech:
    def test_czk_curve(self):
        from pricebook.fixed_income.czech import build_czk_curve, synthetic_pribor_strip
        strip = synthetic_pribor_strip(REF)
        curve = build_czk_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_pribor_swap(self):
        from pricebook.fixed_income.czech import PRIBORSwap, build_czk_curve, synthetic_pribor_strip
        curve = build_czk_curve(REF, synthetic_pribor_strip(REF))
        swap = PRIBORSwap(REF, REF + relativedelta(years=5), 0.045)
        assert swap.price(curve).dv01 > 0

    def test_czeonia_swap(self):
        from pricebook.fixed_income.czech import CZEONIASwap, build_czk_curve, synthetic_pribor_strip
        curve = build_czk_curve(REF, synthetic_pribor_strip(REF))
        swap = CZEONIASwap(REF, REF + relativedelta(years=5), 0.045)
        assert swap.price(curve).dv01 > 0

    def test_czgb_bond(self):
        from pricebook.fixed_income.czech import CZGBBond, build_czk_curve, synthetic_pribor_strip
        curve = build_czk_curve(REF, synthetic_pribor_strip(REF))
        bond = CZGBBond(REF, REF + relativedelta(years=10), 0.04)
        assert 70 < bond.dirty_price(curve) < 110

    def test_czgb_linker(self):
        from pricebook.fixed_income.czech import CZGBLinker, build_czk_curve, synthetic_pribor_strip
        curve = build_czk_curve(REF, synthetic_pribor_strip(REF, rate=0.015))
        linker = CZGBLinker(REF, REF + relativedelta(years=10), 0.02)
        r = linker.price(REF, curve, current_cpi=110.0)
        assert r.real_price > 0

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        czeonia = get_rate_index("CZEONIA")
        assert czeonia.currency == "CZK"
        assert czeonia.is_overnight

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_CZ")
        assert cpi.currency == "CZK"


class TestHungary:
    def test_huf_curve(self):
        from pricebook.fixed_income.hungarian import build_huf_curve, synthetic_bubor_strip
        strip = synthetic_bubor_strip(REF)
        curve = build_huf_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_bubor_swap(self):
        from pricebook.fixed_income.hungarian import BUBORSwap, build_huf_curve, synthetic_bubor_strip
        curve = build_huf_curve(REF, synthetic_bubor_strip(REF))
        swap = BUBORSwap(REF, REF + relativedelta(years=5), 0.065)
        assert swap.price(curve).dv01 > 0

    def test_hufonia_swap(self):
        from pricebook.fixed_income.hungarian import HUFONIASwap, build_huf_curve, synthetic_bubor_strip
        curve = build_huf_curve(REF, synthetic_bubor_strip(REF))
        swap = HUFONIASwap(REF, REF + relativedelta(years=5), 0.065)
        assert swap.price(curve).dv01 > 0

    def test_hgb_bond_act365(self):
        """HGB uses ACT/365F (unique among CEE — not ACT/ACT ICMA)."""
        from pricebook.fixed_income.hungarian import HGBBond, build_huf_curve, synthetic_bubor_strip
        curve = build_huf_curve(REF, synthetic_bubor_strip(REF))
        bond = HGBBond(REF, REF + relativedelta(years=10), 0.06)
        assert 70 < bond.dirty_price(curve) < 110

    def test_hgb_linker(self):
        from pricebook.fixed_income.hungarian import HGBLinker, build_huf_curve, synthetic_bubor_strip
        curve = build_huf_curve(REF, synthetic_bubor_strip(REF, rate=0.02))
        linker = HGBLinker(REF, REF + relativedelta(years=10), 0.025)
        r = linker.price(REF, curve, current_cpi=120.0)
        assert r.real_price > 0

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        hufonia = get_rate_index("HUFONIA")
        assert hufonia.currency == "HUF"
        assert hufonia.is_overnight

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_HU")
        assert cpi.currency == "HUF"
