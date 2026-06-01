"""Tests for Australia, South Africa, Israel fixed income."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestAustralia:
    def test_aud_curve(self):
        from pricebook.fixed_income.australian import build_aud_curve, synthetic_aonia_strip
        strip = synthetic_aonia_strip(REF)
        curve = build_aud_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_aonia_swap(self):
        from pricebook.fixed_income.australian import AONIASwap, build_aud_curve, synthetic_aonia_strip
        curve = build_aud_curve(REF, synthetic_aonia_strip(REF))
        swap = AONIASwap(REF, REF + relativedelta(years=5), 0.0435)
        assert swap.price(curve).dv01 > 0

    def test_acgb_bond(self):
        from pricebook.fixed_income.australian import ACGBBond, build_aud_curve, synthetic_aonia_strip
        curve = build_aud_curve(REF, synthetic_aonia_strip(REF))
        bond = ACGBBond(REF, REF + relativedelta(years=10), 0.04)
        assert 70 < bond.dirty_price(curve) < 110

    def test_tib_quarterly(self):
        """TIB is the only quarterly sovereign linker globally."""
        from pricebook.fixed_income.australian import TIBBond, build_aud_curve, synthetic_aonia_strip
        curve = build_aud_curve(REF, synthetic_aonia_strip(REF, aonia=0.015))
        tib = TIBBond(REF, REF + relativedelta(years=10), 0.01, base_cpi=120)
        r = tib.price(REF, curve, current_cpi=130)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price

    def test_tib_no_deflation_floor(self):
        """AUD TIBs have NO deflation floor (unlike TIPS)."""
        from pricebook.fixed_income.australian import TIBBond, build_aud_curve, synthetic_aonia_strip
        curve = build_aud_curve(REF, synthetic_aonia_strip(REF, aonia=0.015))
        tib = TIBBond(REF, REF + relativedelta(years=10), 0.01, base_cpi=130)
        r = tib.price(REF, curve, current_cpi=120)  # deflation
        assert r.cpi_ratio < 1.0  # no floor

    def test_bei_positive(self):
        from pricebook.fixed_income.australian import breakeven_inflation_au, build_aud_curve, synthetic_aonia_strip
        nom = build_aud_curve(REF, synthetic_aonia_strip(REF, aonia=0.04))
        real = build_aud_curve(REF, synthetic_aonia_strip(REF, aonia=0.015))
        bei = breakeven_inflation_au(nom, real)
        assert all(row["bei"] > 0 for row in bei)

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        aonia = get_rate_index("AONIA")
        assert aonia.currency == "AUD"

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_AU")
        assert cpi.currency == "AUD"


class TestSouthAfrica:
    def test_zar_curve(self):
        from pricebook.fixed_income.south_african import build_zar_curve, synthetic_jibar_strip
        strip = synthetic_jibar_strip(REF)
        curve = build_zar_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_jibar_swap(self):
        from pricebook.fixed_income.south_african import JIBARSwap, build_zar_curve, synthetic_jibar_strip
        curve = build_zar_curve(REF, synthetic_jibar_strip(REF))
        swap = JIBARSwap(REF, REF + relativedelta(years=5), 0.0825)
        assert swap.price(curve).dv01 > 0

    def test_sagb_bond(self):
        from pricebook.fixed_income.south_african import SAGBBond, build_zar_curve, synthetic_jibar_strip
        curve = build_zar_curve(REF, synthetic_jibar_strip(REF))
        bond = SAGBBond(REF, REF + relativedelta(years=10), 0.08)
        assert 70 < bond.dirty_price(curve) < 110

    def test_sa_ilb(self):
        from pricebook.fixed_income.south_african import SAILBBond, build_zar_curve, synthetic_jibar_strip
        curve = build_zar_curve(REF, synthetic_jibar_strip(REF, jibar=0.03))
        ilb = SAILBBond(REF, REF + relativedelta(years=10), 0.025, base_cpi=100)
        r = ilb.price(REF, curve, current_cpi=115)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price

    def test_bei_positive(self):
        from pricebook.fixed_income.south_african import breakeven_inflation_za, build_zar_curve, synthetic_jibar_strip
        nom = build_zar_curve(REF, synthetic_jibar_strip(REF, jibar=0.08))
        real = build_zar_curve(REF, synthetic_jibar_strip(REF, jibar=0.03))
        bei = breakeven_inflation_za(nom, real)
        assert all(row["bei"] > 0 for row in bei)

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_ZA")
        assert cpi.currency == "ZAR"


class TestIsrael:
    def test_ils_curve(self):
        from pricebook.fixed_income.israeli import build_ils_curve, synthetic_telbor_strip
        strip = synthetic_telbor_strip(REF)
        curve = build_ils_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_telbor_swap(self):
        from pricebook.fixed_income.israeli import TelborSwap, build_ils_curve, synthetic_telbor_strip
        curve = build_ils_curve(REF, synthetic_telbor_strip(REF))
        swap = TelborSwap(REF, REF + relativedelta(years=5), 0.045)
        assert swap.price(curve).dv01 > 0

    def test_shahar_bond(self):
        """Shahar = Israeli nominal government bond, annual coupon."""
        from pricebook.fixed_income.israeli import ShaharBond, build_ils_curve, synthetic_telbor_strip
        curve = build_ils_curve(REF, synthetic_telbor_strip(REF))
        bond = ShaharBond(REF, REF + relativedelta(years=10), 0.04)
        assert 70 < bond.dirty_price(curve) < 110

    def test_galil_annual(self):
        """Galil = Israeli linker, annual coupon, 1-month lag."""
        from pricebook.fixed_income.israeli import GalilBond, build_ils_curve, synthetic_telbor_strip
        curve = build_ils_curve(REF, synthetic_telbor_strip(REF, telbor=0.015))
        galil = GalilBond(REF, REF + relativedelta(years=10), 0.02, base_cpi=100)
        r = galil.price(REF, curve, current_cpi=110)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price

    def test_galil_no_floor(self):
        """Galil has no deflation floor."""
        from pricebook.fixed_income.israeli import GalilBond, build_ils_curve, synthetic_telbor_strip
        curve = build_ils_curve(REF, synthetic_telbor_strip(REF, telbor=0.015))
        galil = GalilBond(REF, REF + relativedelta(years=10), 0.02, base_cpi=110)
        r = galil.price(REF, curve, current_cpi=100)  # deflation
        assert r.cpi_ratio < 1.0

    def test_bei_positive(self):
        from pricebook.fixed_income.israeli import breakeven_inflation_il, build_ils_curve, synthetic_telbor_strip
        nom = build_ils_curve(REF, synthetic_telbor_strip(REF, telbor=0.045))
        real = build_ils_curve(REF, synthetic_telbor_strip(REF, telbor=0.01))
        bei = breakeven_inflation_il(nom, real)
        assert all(row["bei"] > 0 for row in bei)

    def test_inflation_index(self):
        from pricebook.fixed_income.inflation_indices import get_inflation_index
        cpi = get_inflation_index("CPI_IL")
        assert cpi.currency == "ILS"
