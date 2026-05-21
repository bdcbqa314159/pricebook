"""Tests for sovereign bond factory.

Covers: convention lookup, factory creation, correct conventions per market,
region grouping, market enumeration, pricing sanity.
"""

import pytest
from datetime import date

from pricebook.core.day_count import DayCountConvention
from pricebook.core.schedule import Frequency
from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.sovereign_bonds import (
    create_sovereign_bond, get_conventions, list_markets,
    markets_by_region, SovereignConventions,
)


REF = date(2024, 1, 15)
MAT_5Y = date(2029, 1, 15)
MAT_10Y = date(2034, 1, 15)


# ═══════════════════════════════════════════════════════════════
# Convention lookup
# ═══════════════════════════════════════════════════════════════


class TestConventionLookup:
    def test_ust(self):
        c = get_conventions("UST")
        assert c.currency == "USD"
        assert c.frequency == Frequency.SEMI_ANNUAL
        assert c.day_count == DayCountConvention.ACT_ACT_ICMA
        assert c.settlement_days == 1

    def test_bund(self):
        c = get_conventions("BUND")
        assert c.frequency == Frequency.ANNUAL
        assert c.day_count == DayCountConvention.ACT_ACT_ICMA

    def test_gilt_ex_div(self):
        c = get_conventions("GILT")
        assert c.ex_div_days == 7

    def test_jgb(self):
        c = get_conventions("JGB")
        assert c.day_count == DayCountConvention.ACT_365_FIXED

    def test_mbono_act_360(self):
        """MBONOs use ACT/360 — unusual for government bonds."""
        c = get_conventions("MBONO")
        assert c.day_count == DayCountConvention.ACT_360

    def test_gsec_30_360(self):
        """Indian GSECs use 30/360 — unusual for sovereign."""
        c = get_conventions("GSEC")
        assert c.day_count == DayCountConvention.THIRTY_360

    def test_ntn_f_bus_252(self):
        """Brazilian NTN-F uses BUS/252."""
        c = get_conventions("NTN_F")
        assert c.day_count == DayCountConvention.BUS_252

    def test_turkgb_t0(self):
        """Turkish bonds settle T+0."""
        c = get_conventions("TURKGB")
        assert c.settlement_days == 0

    def test_ggb_t3(self):
        """Greek bonds settle T+3."""
        c = get_conventions("GGB")
        assert c.settlement_days == 3

    def test_sagb_t3(self):
        """South African bonds settle T+3."""
        c = get_conventions("SAGB")
        assert c.settlement_days == 3

    def test_rpgb_quarterly(self):
        """Philippines pays quarterly."""
        c = get_conventions("RPGB")
        assert c.frequency == Frequency.QUARTERLY

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown market"):
            get_conventions("FAKE")

    def test_case_insensitive(self):
        c = get_conventions("ust")
        assert c.market_code == "UST"


# ═══════════════════════════════════════════════════════════════
# Market enumeration
# ═══════════════════════════════════════════════════════════════


class TestMarketEnumeration:
    def test_list_markets_count(self):
        markets = list_markets()
        assert len(markets) == 50

    def test_list_markets_sorted(self):
        markets = list_markets()
        assert markets == sorted(markets)

    def test_markets_by_region_complete(self):
        regions = markets_by_region()
        all_codes = []
        for codes in regions.values():
            all_codes.extend(codes)
        assert len(all_codes) == 50
        assert set(all_codes) == set(list_markets())

    def test_g10_core(self):
        regions = markets_by_region()
        assert "UST" in regions["G10_core"]
        assert "BUND" in regions["G10_core"]
        assert len(regions["G10_core"]) == 6


# ═══════════════════════════════════════════════════════════════
# Factory creation
# ═══════════════════════════════════════════════════════════════


class TestFactory:
    def test_create_ust(self):
        bond = create_sovereign_bond("UST", REF, MAT_10Y, 0.04)
        assert bond.frequency == Frequency.SEMI_ANNUAL
        assert bond.day_count == DayCountConvention.ACT_ACT_ICMA
        assert bond.settlement_days == 1
        assert bond.maturity == MAT_10Y

    def test_create_bund(self):
        bond = create_sovereign_bond("BUND", REF, MAT_10Y, 0.025)
        assert bond.frequency == Frequency.ANNUAL

    def test_create_mbono(self):
        bond = create_sovereign_bond("MBONO", REF, MAT_5Y, 0.08)
        assert bond.day_count == DayCountConvention.ACT_360

    def test_create_ntn_f(self):
        bond = create_sovereign_bond("NTN_F", REF, MAT_5Y, 0.10)
        assert bond.day_count == DayCountConvention.BUS_252

    def test_create_rpgb(self):
        bond = create_sovereign_bond("RPGB", REF, MAT_5Y, 0.06)
        assert bond.frequency == Frequency.QUARTERLY

    def test_create_gilt_ex_div(self):
        bond = create_sovereign_bond("GILT", REF, MAT_10Y, 0.035)
        assert bond.ex_div_days == 7

    def test_create_all_markets(self):
        """Every market should produce a valid bond."""
        for code in list_markets():
            bond = create_sovereign_bond(code, REF, MAT_5Y, 0.05)
            assert bond.maturity == MAT_5Y
            assert bond.coupon_rate == 0.05


# ═══════════════════════════════════════════════════════════════
# Pricing sanity
# ═══════════════════════════════════════════════════════════════


class TestPricingSanity:
    @pytest.fixture
    def flat_curve(self):
        return DiscountCurve.flat(REF, 0.04)

    def test_ust_near_par(self, flat_curve):
        """UST at coupon = discount rate should price near par."""
        bond = create_sovereign_bond("UST", REF, MAT_10Y, 0.04)
        dirty = bond.dirty_price(flat_curve)
        assert 98.0 < dirty < 102.0

    def test_bund_near_par(self, flat_curve):
        bond = create_sovereign_bond("BUND", REF, MAT_10Y, 0.04)
        dirty = bond.dirty_price(flat_curve)
        assert 98.0 < dirty < 102.0

    def test_high_coupon_above_par(self, flat_curve):
        """Bond with coupon > discount rate trades above par."""
        bond = create_sovereign_bond("UST", REF, MAT_10Y, 0.06)
        dirty = bond.dirty_price(flat_curve)
        assert dirty > 100.0

    def test_low_coupon_below_par(self, flat_curve):
        """Bond with coupon < discount rate trades below par."""
        bond = create_sovereign_bond("UST", REF, MAT_10Y, 0.02)
        dirty = bond.dirty_price(flat_curve)
        assert dirty < 100.0

    def test_gilt_prices(self, flat_curve):
        bond = create_sovereign_bond("GILT", REF, MAT_10Y, 0.04)
        dirty = bond.dirty_price(flat_curve)
        assert 95.0 < dirty < 105.0

    def test_oat_annual(self, flat_curve):
        """Annual OAT should price differently from semi-annual UST."""
        ust = create_sovereign_bond("UST", REF, MAT_10Y, 0.04)
        oat = create_sovereign_bond("OAT", REF, MAT_10Y, 0.04)
        # Same coupon/rate but different frequency → slightly different price
        p_ust = ust.dirty_price(flat_curve)
        p_oat = oat.dirty_price(flat_curve)
        # Should be close but not identical
        assert abs(p_ust - p_oat) < 2.0  # within 2 points
        assert abs(p_ust - p_oat) > 0.01  # but not identical


# ═══════════════════════════════════════════════════════════════
# Convention coverage checks
# ═══════════════════════════════════════════════════════════════


class TestConventionCoverage:
    def test_all_currencies_have_calendars(self):
        """Every sovereign market's calendar currency should be valid."""
        from pricebook.core.calendar import get_calendar
        for code in list_markets():
            conv = get_conventions(code)
            cal = get_calendar(conv.calendar_currency)
            assert cal is not None, f"{code} calendar {conv.calendar_currency} not found"

    def test_frequency_distribution(self):
        """Check that we have annual, semi-annual, and quarterly bonds."""
        freqs = {get_conventions(c).frequency for c in list_markets()}
        assert Frequency.ANNUAL in freqs
        assert Frequency.SEMI_ANNUAL in freqs
        assert Frequency.QUARTERLY in freqs

    def test_day_count_distribution(self):
        """Check variety of day count conventions."""
        dcs = {get_conventions(c).day_count for c in list_markets()}
        assert DayCountConvention.ACT_ACT_ICMA in dcs
        assert DayCountConvention.ACT_365_FIXED in dcs
        assert DayCountConvention.ACT_360 in dcs
        assert DayCountConvention.THIRTY_360 in dcs
        assert DayCountConvention.BUS_252 in dcs

    def test_settlement_range(self):
        """Settlement days should be 0-3."""
        for code in list_markets():
            conv = get_conventions(code)
            assert 0 <= conv.settlement_days <= 3, \
                f"{code}: settlement_days={conv.settlement_days}"

    def test_frozen_dataclass(self):
        """SovereignConventions should be immutable."""
        c = get_conventions("UST")
        with pytest.raises(Exception):
            c.currency = "GBP"
