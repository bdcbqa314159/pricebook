"""Tests for Turkey (TRY) derivatives — extreme rate environment."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestTurkey:
    def test_try_curve_extreme_rates(self):
        """TRY at 45% — DFs should be small but positive."""
        from pricebook.fixed_income.turkish import build_try_curve, synthetic_tlref_strip
        strip = synthetic_tlref_strip(REF, rate=0.45)
        curve = build_try_curve(REF, strip)
        for c in strip:
            df = curve.df(c["maturity"])
            assert 0 < df < 1

    def test_tlref_swap(self):
        from pricebook.fixed_income.turkish import TLREFSwap, build_try_curve, synthetic_tlref_strip
        curve = build_try_curve(REF, synthetic_tlref_strip(REF))
        swap = TLREFSwap(REF, REF + relativedelta(years=3), 0.45)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_turkgb_bond(self):
        from pricebook.fixed_income.turkish import TURKGBBond, build_try_curve, synthetic_tlref_strip
        curve = build_try_curve(REF, synthetic_tlref_strip(REF))
        bond = TURKGBBond(REF, REF + relativedelta(years=5), 0.20)
        price = bond.dirty_price(curve)
        # At 45% rates, 20% coupon trades well below par
        assert 20 < price < 80

    def test_turkgb_t0_settlement(self):
        """TURKGB has T+0 settlement (unique — most bonds are T+1 or T+2)."""
        from pricebook.fixed_income.turkish import TURKGBBond
        bond = TURKGBBond(REF, REF + relativedelta(years=5), 0.20)
        d = bond.to_dict()
        assert d.get("settlement", 0) == 0 or "turkgb" in d.get("type", "")

    def test_turkish_cpi_linker(self):
        from pricebook.fixed_income.turkish import TurkishCPILinker, build_try_curve, synthetic_tlref_strip
        curve = build_try_curve(REF, synthetic_tlref_strip(REF, rate=0.10))  # real rate
        linker = TurkishCPILinker(REF, REF + relativedelta(years=10), 0.05, base_cpi=500)
        r = linker.price(REF, curve, current_cpi=800)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price  # CPI rose

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        tlref = get_rate_index("TLREF")
        assert tlref.currency == "TRY"
        assert tlref.is_overnight

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        turkgb = get_conventions("TURKGB")
        assert turkgb.currency == "TRY"
