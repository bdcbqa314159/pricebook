"""Tests for European non-euro markets: Switzerland, Sweden, Norway, Denmark."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


# ═══════════════════════════════════════════════════════════════
# Switzerland
# ═══════════════════════════════════════════════════════════════

class TestSwitzerland:
    def test_saron_curve(self):
        from pricebook.fixed_income.swiss import build_saron_curve, synthetic_saron_strip
        strip = synthetic_saron_strip(REF)
        curve = build_saron_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_saron_swap(self):
        from pricebook.fixed_income.swiss import SARONSwap, build_saron_curve, synthetic_saron_strip
        curve = build_saron_curve(REF, synthetic_saron_strip(REF))
        swap = SARONSwap(REF, REF + relativedelta(years=5), 0.0125)
        r = swap.price(curve)
        assert r.dv01 > 0
        assert r.par_rate > 0

    def test_confed_bond(self):
        from pricebook.fixed_income.swiss import ConfedBond, build_saron_curve, synthetic_saron_strip
        curve = build_saron_curve(REF, synthetic_saron_strip(REF))
        bond = ConfedBond(REF, REF + relativedelta(years=10), 0.01)
        price = bond.dirty_price(curve)
        assert 80 < price < 110

    def test_negative_rates(self):
        """CHF can trade at negative rates — curve should handle DF > 1."""
        from pricebook.fixed_income.swiss import build_saron_curve, synthetic_saron_strip
        strip = synthetic_saron_strip(REF, saron=-0.005)  # negative
        curve = build_saron_curve(REF, strip)
        # Short-end DF > 1 when rates negative
        assert curve.df(strip[0]["maturity"]) > 1.0

    def test_confed_low_coupon(self):
        """Confed with very low coupon (Swiss style)."""
        from pricebook.fixed_income.swiss import ConfedBond, build_saron_curve, synthetic_saron_strip
        curve = build_saron_curve(REF, synthetic_saron_strip(REF))
        bond = ConfedBond(REF, REF + relativedelta(years=30), 0.005)  # 0.5% coupon
        price = bond.dirty_price(curve)
        assert price > 0

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        saron = get_rate_index("SARON")
        assert saron.currency == "CHF"
        assert saron.is_overnight


# ═══════════════════════════════════════════════════════════════
# Sweden
# ═══════════════════════════════════════════════════════════════

class TestSweden:
    def test_swestr_curve(self):
        from pricebook.fixed_income.swedish import build_sek_curve, synthetic_swestr_strip
        strip = synthetic_swestr_strip(REF)
        curve = build_sek_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_swestr_swap(self):
        from pricebook.fixed_income.swedish import SWESTRSwap, build_sek_curve, synthetic_swestr_strip
        curve = build_sek_curve(REF, synthetic_swestr_strip(REF))
        swap = SWESTRSwap(REF, REF + relativedelta(years=5), 0.035)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_sgb_bond(self):
        from pricebook.fixed_income.swedish import SGBBond, build_sek_curve, synthetic_swestr_strip
        curve = build_sek_curve(REF, synthetic_swestr_strip(REF))
        sgb = SGBBond(REF, REF + relativedelta(years=10), 0.025)
        assert 70 < sgb.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        swestr = get_rate_index("SWESTR")
        assert swestr.currency == "SEK"
        assert swestr.is_overnight

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        sgb = get_conventions("SGB")
        assert sgb.currency == "SEK"


# ═══════════════════════════════════════════════════════════════
# Norway
# ═══════════════════════════════════════════════════════════════

class TestNorway:
    def test_nowa_curve(self):
        from pricebook.fixed_income.norwegian import build_nok_curve, synthetic_nowa_strip
        strip = synthetic_nowa_strip(REF)
        curve = build_nok_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_nowa_swap(self):
        from pricebook.fixed_income.norwegian import NOWASwap, build_nok_curve, synthetic_nowa_strip
        curve = build_nok_curve(REF, synthetic_nowa_strip(REF))
        swap = NOWASwap(REF, REF + relativedelta(years=5), 0.045)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_ngb_bond(self):
        from pricebook.fixed_income.norwegian import NGBBond, build_nok_curve, synthetic_nowa_strip
        curve = build_nok_curve(REF, synthetic_nowa_strip(REF))
        ngb = NGBBond(REF, REF + relativedelta(years=10), 0.03)
        assert 70 < ngb.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        nowa = get_rate_index("NOWA")
        assert nowa.currency == "NOK"
        assert nowa.is_overnight

    def test_sovereign_convention(self):
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        ngb = get_conventions("NGB")
        assert ngb.currency == "NOK"


# ═══════════════════════════════════════════════════════════════
# Denmark
# ═══════════════════════════════════════════════════════════════

class TestDenmark:
    def test_destr_curve(self):
        from pricebook.fixed_income.danish import build_dkk_curve, synthetic_destr_strip
        strip = synthetic_destr_strip(REF)
        curve = build_dkk_curve(REF, strip)
        assert all(0 < curve.df(c["maturity"]) < 1 for c in strip)

    def test_destr_swap(self):
        from pricebook.fixed_income.danish import DESTRSwap, build_dkk_curve, synthetic_destr_strip
        curve = build_dkk_curve(REF, synthetic_destr_strip(REF))
        swap = DESTRSwap(REF, REF + relativedelta(years=5), 0.0335)
        r = swap.price(curve)
        assert r.dv01 > 0

    def test_dgb_bond(self):
        from pricebook.fixed_income.danish import DGBBond, build_dkk_curve, synthetic_destr_strip
        curve = build_dkk_curve(REF, synthetic_destr_strip(REF))
        dgb = DGBBond(REF, REF + relativedelta(years=10), 0.025)
        assert 70 < dgb.dirty_price(curve) < 110

    def test_rate_index(self):
        from pricebook.core.rate_index import get_rate_index
        destr = get_rate_index("DESTR")
        assert destr.currency == "DKK"
        assert destr.is_overnight

    def test_ois_convention(self):
        from pricebook.fixed_income.ois import get_ois_convention
        dkk = get_ois_convention("DKK")
        assert dkk.currency == "DKK"
