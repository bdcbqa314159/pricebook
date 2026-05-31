"""Tests for Americas markets: Colombia, Peru, Argentina, Canada."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


# ═══════════════════════════════════════════════════════════════
# Colombia
# ═══════════════════════════════════════════════════════════════

class TestColombia:
    def test_ibr_curve(self):
        from pricebook.fixed_income.colombian import build_ibr_curve, synthetic_ibr_strip
        strip = synthetic_ibr_strip(REF, ibr=0.0975)
        curve = build_ibr_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_ibr_swap(self):
        from pricebook.fixed_income.colombian import IBRSwap, build_ibr_curve, synthetic_ibr_strip
        curve = build_ibr_curve(REF, synthetic_ibr_strip(REF))
        swap = IBRSwap(REF, REF + relativedelta(years=2), 0.0975)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_tes_bond(self):
        from pricebook.fixed_income.colombian import TESBond, build_ibr_curve, synthetic_ibr_strip
        curve = build_ibr_curve(REF, synthetic_ibr_strip(REF))
        tes = TESBond(REF, REF + relativedelta(years=5), 0.09)
        assert 80 < tes.dirty_price(curve) < 110

    def test_tes_uvr(self):
        from pricebook.fixed_income.colombian import TESUVRBond, build_ibr_curve, synthetic_ibr_strip
        curve = build_ibr_curve(REF, synthetic_ibr_strip(REF, ibr=0.03))  # real curve
        uvr = TESUVRBond(REF, REF + relativedelta(years=10), 0.035)
        r = uvr.price(REF, curve, current_uvr=350.0)
        assert r.real_price > 0
        assert r.nominal_price > 0


# ═══════════════════════════════════════════════════════════════
# Peru
# ═══════════════════════════════════════════════════════════════

class TestPeru:
    def test_calendar(self):
        from pricebook.core.calendar import get_calendar
        cal = get_calendar("PEN")
        assert not cal.is_business_day(date(2024, 7, 28))  # Independence Day

    def test_pen_curve(self):
        from pricebook.fixed_income.peruvian import build_pen_curve, synthetic_pen_strip
        strip = synthetic_pen_strip(REF)
        curve = build_pen_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_btp_peru(self):
        from pricebook.fixed_income.peruvian import BTPPeru, build_pen_curve, synthetic_pen_strip
        curve = build_pen_curve(REF, synthetic_pen_strip(REF))
        btp = BTPPeru(REF, REF + relativedelta(years=5), 0.06)
        assert btp.dirty_price(curve) > 0

    def test_vac_bond(self):
        from pricebook.fixed_income.peruvian import VACBond, build_pen_curve, synthetic_pen_strip
        curve = build_pen_curve(REF, synthetic_pen_strip(REF, rate=0.02))
        vac = VACBond(REF, REF + relativedelta(years=10), 0.03)
        r = vac.price(REF, curve, ipc=120.0)
        assert r.real_price > 0

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        conv = get_conventions("BTP_PE")
        assert conv.currency == "PEN"


# ═══════════════════════════════════════════════════════════════
# Argentina
# ═══════════════════════════════════════════════════════════════

class TestArgentina:
    def test_calendar(self):
        from pricebook.core.calendar import get_calendar
        cal = get_calendar("ARS")
        assert not cal.is_business_day(date(2024, 7, 9))  # Independence Day

    def test_ars_curve_extreme_rates(self):
        """ARS curve with 40% rate should not overflow."""
        from pricebook.fixed_income.argentine import build_ars_curve, synthetic_ars_strip
        strip = synthetic_ars_strip(REF, policy_rate=0.40)
        curve = build_ars_curve(REF, strip)
        # DFs should be very small but positive
        for c in strip:
            df = curve.df(c["maturity"])
            assert 0 < df < 1

    def test_lecap(self):
        from pricebook.fixed_income.argentine import LecapBond
        lecap = LecapBond(REF + relativedelta(months=6), 0.40)
        r = lecap.price(REF)
        # At 40% rate, 6-month Lecap trades at deep discount
        assert r.price < 1000  # below face
        assert r.price > 500   # but not absurdly low

    def test_lecer(self):
        from pricebook.fixed_income.argentine import LecerBond, build_ars_curve, synthetic_ars_strip
        curve = build_ars_curve(REF, synthetic_ars_strip(REF, policy_rate=0.05))  # real
        lecer = LecerBond(REF - relativedelta(months=3), REF + relativedelta(months=9))
        r = lecer.price(REF, curve, current_cer=1200.0, cer_at_issue=1100.0)
        assert r.nominal_price > r.real_price  # CER inflation adjustment

    def test_bonar(self):
        from pricebook.fixed_income.argentine import BONARBond, build_ars_curve, synthetic_ars_strip
        curve = build_ars_curve(REF, synthetic_ars_strip(REF, policy_rate=0.40))
        bonar = BONARBond(REF, REF + relativedelta(years=3), 0.20)
        price = bonar.dirty_price(curve)
        # At 40% rate, 20% coupon bond trades well below par
        assert 30 < price < 80

    def test_sovereign_conventions(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        bonar = get_conventions("BONAR")
        assert bonar.currency == "ARS"
        global_ar = get_conventions("GLOBAL_AR")
        assert global_ar.currency == "USD"


# ═══════════════════════════════════════════════════════════════
# Canada
# ═══════════════════════════════════════════════════════════════

class TestCanada:
    def test_corra_curve(self):
        from pricebook.fixed_income.canadian import build_corra_curve, synthetic_corra_strip
        strip = synthetic_corra_strip(REF)
        curve = build_corra_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_corra_swap(self):
        from pricebook.fixed_income.canadian import CORRASwap, build_corra_curve, synthetic_corra_strip
        curve = build_corra_curve(REF, synthetic_corra_strip(REF))
        swap = CORRASwap(REF, REF + relativedelta(years=5), 0.0425)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_rrb(self):
        from pricebook.fixed_income.canadian import RRBBond, build_corra_curve, synthetic_corra_strip
        curve = build_corra_curve(REF, synthetic_corra_strip(REF, corra=0.02))  # real curve
        rrb = RRBBond(REF, REF + relativedelta(years=30), 0.015, base_cpi=130, face=100)
        r = rrb.price(REF, curve, current_cpi=160)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price  # CPI > base → inflation adjustment
        assert r.cpi_ratio >= 1.0  # deflation floor

    def test_cgb_via_sovereign(self):
        from pricebook.fixed_income.sovereign_bonds import create_sovereign_bond
        from pricebook.fixed_income.canadian import build_corra_curve, synthetic_corra_strip
        curve = build_corra_curve(REF, synthetic_corra_strip(REF))
        cgb = create_sovereign_bond("CGB_CA", REF, REF + relativedelta(years=10), 0.035)
        price = cgb.dirty_price(curve)
        assert 80 < price < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        corra = get_rate_index("CORRA")
        assert corra.currency == "CAD"
        assert corra.is_overnight
