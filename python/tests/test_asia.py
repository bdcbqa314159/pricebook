"""Tests for Asian markets: China, Korea, Singapore, Hong Kong, Thailand."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestChina:
    def test_cny_curve(self):
        from pricebook.fixed_income.chinese import build_cny_curve, synthetic_dr007_strip
        strip = synthetic_dr007_strip(REF)
        curve = build_cny_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_dr007_swap(self):
        from pricebook.fixed_income.chinese import DR007Swap, build_cny_curve, synthetic_dr007_strip
        curve = build_cny_curve(REF, synthetic_dr007_strip(REF))
        swap = DR007Swap(REF, REF + relativedelta(years=5), 0.018)
        assert swap.price(curve).dv01 > 0

    def test_cgb_bond(self):
        from pricebook.fixed_income.chinese import CGBBond, build_cny_curve, synthetic_dr007_strip
        curve = build_cny_curve(REF, synthetic_dr007_strip(REF))
        bond = CGBBond(REF, REF + relativedelta(years=10), 0.025)
        assert 80 < bond.dirty_price(curve) < 115

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        dr007 = get_rate_index("DR007")
        assert dr007.currency == "CNY"

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        cgb = get_conventions("CGB")
        assert cgb.currency == "CNY"


class TestKorea:
    def test_krw_curve(self):
        from pricebook.fixed_income.korean import build_krw_curve, synthetic_kofr_strip
        strip = synthetic_kofr_strip(REF)
        curve = build_krw_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_kofr_swap(self):
        from pricebook.fixed_income.korean import KOFRSwap, build_krw_curve, synthetic_kofr_strip
        curve = build_krw_curve(REF, synthetic_kofr_strip(REF))
        swap = KOFRSwap(REF, REF + relativedelta(years=5), 0.035)
        assert swap.price(curve).dv01 > 0

    def test_ktb_bond(self):
        from pricebook.fixed_income.korean import KTBBond, build_krw_curve, synthetic_kofr_strip
        curve = build_krw_curve(REF, synthetic_kofr_strip(REF))
        bond = KTBBond(REF, REF + relativedelta(years=10), 0.03)
        assert 70 < bond.dirty_price(curve) < 110

    def test_ktbi_deflation_floor(self):
        """KTBi has deflation floor like TIPS."""
        from pricebook.fixed_income.korean import KTBiLinker, build_krw_curve, synthetic_kofr_strip
        curve = build_krw_curve(REF, synthetic_kofr_strip(REF, rate=0.01))
        ktbi = KTBiLinker(REF, REF + relativedelta(years=10), 0.01, base_cpi=105)
        r = ktbi.price(REF, curve, current_cpi=100)  # deflation
        assert r.cpi_ratio >= 1.0  # floor

    def test_ktbi_inflation(self):
        from pricebook.fixed_income.korean import KTBiLinker, build_krw_curve, synthetic_kofr_strip
        curve = build_krw_curve(REF, synthetic_kofr_strip(REF, rate=0.01))
        ktbi = KTBiLinker(REF, REF + relativedelta(years=10), 0.01, base_cpi=100)
        r = ktbi.price(REF, curve, current_cpi=110)
        assert r.nominal_price > r.real_price

    def test_bei(self):
        from pricebook.fixed_income.korean import breakeven_inflation_kr, build_krw_curve, synthetic_kofr_strip
        nom = build_krw_curve(REF, synthetic_kofr_strip(REF, rate=0.035))
        real = build_krw_curve(REF, synthetic_kofr_strip(REF, rate=0.01))
        bei = breakeven_inflation_kr(nom, real)
        assert all(row["bei"] > 0 for row in bei)

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        kofr = get_rate_index("KOFR")
        assert kofr.currency == "KRW"
        assert kofr.is_overnight


class TestSingapore:
    def test_sgd_curve(self):
        from pricebook.fixed_income.singaporean import build_sgd_curve, synthetic_sora_strip
        strip = synthetic_sora_strip(REF)
        curve = build_sgd_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_sora_swap(self):
        from pricebook.fixed_income.singaporean import SORASwap, build_sgd_curve, synthetic_sora_strip
        curve = build_sgd_curve(REF, synthetic_sora_strip(REF))
        swap = SORASwap(REF, REF + relativedelta(years=5), 0.035)
        assert swap.price(curve).dv01 > 0

    def test_sgs_bond(self):
        from pricebook.fixed_income.singaporean import SGSBond, build_sgd_curve, synthetic_sora_strip
        curve = build_sgd_curve(REF, synthetic_sora_strip(REF))
        bond = SGSBond(REF, REF + relativedelta(years=10), 0.03)
        assert 70 < bond.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        sora = get_rate_index("SORA")
        assert sora.currency == "SGD"


class TestHongKong:
    def test_hkd_curve(self):
        from pricebook.fixed_income.hong_kong import build_hkd_curve, synthetic_honia_strip
        strip = synthetic_honia_strip(REF)
        curve = build_hkd_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_honia_swap(self):
        from pricebook.fixed_income.hong_kong import HONIASwap, build_hkd_curve, synthetic_honia_strip
        curve = build_hkd_curve(REF, synthetic_honia_strip(REF))
        swap = HONIASwap(REF, REF + relativedelta(years=5), 0.04)
        assert swap.price(curve).dv01 > 0

    def test_hkgb_bond(self):
        from pricebook.fixed_income.hong_kong import HKGBBond, build_hkd_curve, synthetic_honia_strip
        curve = build_hkd_curve(REF, synthetic_honia_strip(REF))
        bond = HKGBBond(REF, REF + relativedelta(years=10), 0.035)
        assert 70 < bond.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        honia = get_rate_index("HONIA")
        assert honia.currency == "HKD"


class TestThailand:
    def test_thb_curve(self):
        from pricebook.fixed_income.thai import build_thb_curve, synthetic_thor_strip
        strip = synthetic_thor_strip(REF)
        curve = build_thb_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_thor_swap(self):
        from pricebook.fixed_income.thai import THORSwap, build_thb_curve, synthetic_thor_strip
        curve = build_thb_curve(REF, synthetic_thor_strip(REF))
        swap = THORSwap(REF, REF + relativedelta(years=5), 0.025)
        assert swap.price(curve).dv01 > 0

    def test_thaigb_bond(self):
        from pricebook.fixed_income.thai import THAIGBBond, build_thb_curve, synthetic_thor_strip
        curve = build_thb_curve(REF, synthetic_thor_strip(REF))
        bond = THAIGBBond(REF, REF + relativedelta(years=10), 0.03)
        assert 70 < bond.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        thor = get_rate_index("THOR")
        assert thor.currency == "THB"


# ═══════════════════════════════════════════════════════════════
# India
# ═══════════════════════════════════════════════════════════════

class TestIndia:
    def test_inr_curve(self):
        from pricebook.fixed_income.indian import build_inr_curve, synthetic_mibor_strip
        strip = synthetic_mibor_strip(REF)
        curve = build_inr_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_mibor_swap(self):
        from pricebook.fixed_income.indian import MIBORSwap, build_inr_curve, synthetic_mibor_strip
        curve = build_inr_curve(REF, synthetic_mibor_strip(REF))
        swap = MIBORSwap(REF, REF + relativedelta(years=5), 0.065)
        assert swap.price(curve).dv01 > 0

    def test_gsec_30_360(self):
        """GSEC is the ONLY sovereign globally using 30/360."""
        from pricebook.fixed_income.indian import GSECBond, build_inr_curve, synthetic_mibor_strip
        curve = build_inr_curve(REF, synthetic_mibor_strip(REF))
        bond = GSECBond(REF, REF + relativedelta(years=10), 0.07)
        assert 70 < bond.dirty_price(curve) < 115

    def test_iib_deflation_floor(self):
        from pricebook.fixed_income.indian import IIBBond, build_inr_curve, synthetic_mibor_strip
        curve = build_inr_curve(REF, synthetic_mibor_strip(REF, rate=0.02))
        iib = IIBBond(REF, REF + relativedelta(years=10), 0.015, base_cpi=110)
        r = iib.price(REF, curve, current_cpi=100)
        assert r.cpi_ratio >= 1.0  # deflation floor

    def test_bei(self):
        from pricebook.fixed_income.indian import breakeven_inflation_in, build_inr_curve, synthetic_mibor_strip
        nom = build_inr_curve(REF, synthetic_mibor_strip(REF, rate=0.065))
        real = build_inr_curve(REF, synthetic_mibor_strip(REF, rate=0.02))
        bei = breakeven_inflation_in(nom, real)
        assert all(row["bei"] > 0 for row in bei)

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        mibor = get_rate_index("MIBOR")
        assert mibor.currency == "INR"


# ═══════════════════════════════════════════════════════════════
# Indonesia
# ═══════════════════════════════════════════════════════════════

class TestIndonesia:
    def test_idr_curve(self):
        from pricebook.fixed_income.indonesian import build_idr_curve, synthetic_indonia_strip
        strip = synthetic_indonia_strip(REF)
        curve = build_idr_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_indonia_swap(self):
        from pricebook.fixed_income.indonesian import INDONIASwap, build_idr_curve, synthetic_indonia_strip
        curve = build_idr_curve(REF, synthetic_indonia_strip(REF))
        swap = INDONIASwap(REF, REF + relativedelta(years=5), 0.06)
        assert swap.price(curve).dv01 > 0

    def test_indogb_bond(self):
        from pricebook.fixed_income.indonesian import INDOGBBond, build_idr_curve, synthetic_indonia_strip
        curve = build_idr_curve(REF, synthetic_indonia_strip(REF))
        bond = INDOGBBond(REF, REF + relativedelta(years=10), 0.065)
        assert 70 < bond.dirty_price(curve) < 115

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        indonia = get_rate_index("INDONIA")
        assert indonia.currency == "IDR"


# ═══════════════════════════════════════════════════════════════
# Malaysia
# ═══════════════════════════════════════════════════════════════

class TestMalaysia:
    def test_myr_curve(self):
        from pricebook.fixed_income.malaysian import build_myr_curve, synthetic_myor_strip
        strip = synthetic_myor_strip(REF)
        curve = build_myr_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_myor_swap(self):
        from pricebook.fixed_income.malaysian import MYORSwap, build_myr_curve, synthetic_myor_strip
        curve = build_myr_curve(REF, synthetic_myor_strip(REF))
        swap = MYORSwap(REF, REF + relativedelta(years=5), 0.03)
        assert swap.price(curve).dv01 > 0

    def test_mgs_bond(self):
        from pricebook.fixed_income.malaysian import MGSBond, build_myr_curve, synthetic_myor_strip
        curve = build_myr_curve(REF, synthetic_myor_strip(REF))
        bond = MGSBond(REF, REF + relativedelta(years=10), 0.035)
        assert 70 < bond.dirty_price(curve) < 115

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        myor = get_rate_index("MYOR")
        assert myor.currency == "MYR"


# ═══════════════════════════════════════════════════════════════
# Philippines
# ═══════════════════════════════════════════════════════════════

class TestPhilippines:
    def test_php_curve(self):
        from pricebook.fixed_income.philippine import build_php_curve, synthetic_phiref_strip
        strip = synthetic_phiref_strip(REF)
        curve = build_php_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_phiref_swap(self):
        from pricebook.fixed_income.philippine import PHIREFSwap, build_php_curve, synthetic_phiref_strip
        curve = build_php_curve(REF, synthetic_phiref_strip(REF))
        swap = PHIREFSwap(REF, REF + relativedelta(years=5), 0.055)
        assert swap.price(curve).dv01 > 0

    def test_rpgb_quarterly(self):
        """RPGB is the ONLY quarterly sovereign bond globally."""
        from pricebook.fixed_income.philippine import RPGBBond, build_php_curve, synthetic_phiref_strip
        curve = build_php_curve(REF, synthetic_phiref_strip(REF))
        bond = RPGBBond(REF, REF + relativedelta(years=10), 0.06)
        assert 70 < bond.dirty_price(curve) < 115

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        phiref = get_rate_index("PHIREF")
        assert phiref.currency == "PHP"

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        rpgb = get_conventions("RPGB")
        assert rpgb.currency == "PHP"
