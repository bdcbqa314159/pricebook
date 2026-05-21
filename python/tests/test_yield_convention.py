"""Tests for market-convention yield quotation.

Covers: yield↔price, convention mapping, yield conversion, per-market tests.
"""

import math
import pytest

from pricebook.fixed_income.yield_convention import (
    YieldConvention, yield_to_price, price_to_yield,
    convert_yield, get_yield_convention,
)


# ═══════════════════════════════════════════════════════════════
# Yield ↔ Price roundtrip
# ═══════════════════════════════════════════════════════════════


class TestYieldPriceRoundtrip:
    @pytest.mark.parametrize("conv", [
        YieldConvention.SEMI_ANNUAL,
        YieldConvention.ANNUAL,
        YieldConvention.QUARTERLY,
        YieldConvention.CONTINUOUS,
    ])
    def test_coupon_roundtrip(self, conv):
        """yield → price → yield should be identity."""
        ytm = 0.05
        price = yield_to_price(ytm, 0.04, 10.0, conv)
        ytm_back = price_to_yield(price, 0.04, 10.0, conv)
        assert abs(ytm_back - ytm) < 1e-8

    def test_simple_roundtrip(self):
        ytm = 0.05
        price = yield_to_price(ytm, 0.0, 0.5, YieldConvention.SIMPLE)
        ytm_back = price_to_yield(price, 0.0, 0.5, YieldConvention.SIMPLE)
        assert abs(ytm_back - ytm) < 1e-10

    def test_discount_roundtrip(self):
        dr = 0.04
        price = yield_to_price(dr, 0.0, 0.25, YieldConvention.DISCOUNT)
        dr_back = price_to_yield(price, 0.0, 0.25, YieldConvention.DISCOUNT)
        assert abs(dr_back - dr) < 1e-10


# ═══════════════════════════════════════════════════════════════
# Known values
# ═══════════════════════════════════════════════════════════════


class TestKnownValues:
    def test_par_bond_semi(self):
        """5% coupon, 5% semi-annual yield → price ≈ 100."""
        price = yield_to_price(0.05, 0.05, 10.0, YieldConvention.SEMI_ANNUAL)
        assert abs(price - 100.0) < 0.01

    def test_par_bond_annual(self):
        """5% coupon, 5% annual yield → price ≈ 100."""
        price = yield_to_price(0.05, 0.05, 10.0, YieldConvention.ANNUAL, frequency=1)
        assert abs(price - 100.0) < 0.01

    def test_zero_coupon_semi(self):
        """Zero coupon, 5% semi-annual, 10Y → price = 100/(1.025)^20."""
        price = yield_to_price(0.05, 0.0, 10.0, YieldConvention.SEMI_ANNUAL)
        expected = 100.0 / (1.025 ** 20)
        assert abs(price - expected) < 0.01

    def test_premium_bond(self):
        """6% coupon at 4% yield → above par."""
        price = yield_to_price(0.04, 0.06, 10.0, YieldConvention.SEMI_ANNUAL)
        assert price > 100.0

    def test_discount_bond(self):
        """3% coupon at 5% yield → below par."""
        price = yield_to_price(0.05, 0.03, 10.0, YieldConvention.SEMI_ANNUAL)
        assert price < 100.0

    def test_tbill_discount(self):
        """T-Bill at 4% discount, 3 months: P = 100 × (1 - 0.04 × 0.25) = 99."""
        price = yield_to_price(0.04, 0.0, 0.25, YieldConvention.DISCOUNT)
        assert abs(price - 99.0) < 0.01

    def test_semi_vs_annual_different(self):
        """Same yield number under different conventions → different prices."""
        p_semi = yield_to_price(0.05, 0.04, 10.0, YieldConvention.SEMI_ANNUAL)
        p_annual = yield_to_price(0.05, 0.04, 10.0, YieldConvention.ANNUAL, frequency=1)
        assert abs(p_semi - p_annual) > 0.01  # meaningfully different


# ═══════════════════════════════════════════════════════════════
# Yield conversion
# ═══════════════════════════════════════════════════════════════


class TestYieldConversion:
    def test_semi_to_annual_zero(self):
        """5% semi-annual = (1+0.025)^2 - 1 = 5.0625% annual."""
        y_a = convert_yield(0.05, YieldConvention.SEMI_ANNUAL, YieldConvention.ANNUAL)
        expected = (1 + 0.025) ** 2 - 1
        assert abs(y_a - expected) < 1e-10

    def test_annual_to_semi_zero(self):
        """Inverse of above."""
        y_s = convert_yield(0.050625, YieldConvention.ANNUAL, YieldConvention.SEMI_ANNUAL)
        assert abs(y_s - 0.05) < 1e-6

    def test_semi_to_continuous_zero(self):
        """5% semi-annual → continuous = 2 × ln(1.025)."""
        y_c = convert_yield(0.05, YieldConvention.SEMI_ANNUAL, YieldConvention.CONTINUOUS)
        expected = 2 * math.log(1.025)
        assert abs(y_c - expected) < 1e-10

    def test_identity_conversion(self):
        """Converting to same convention should be identity."""
        y = 0.05
        for conv in [YieldConvention.SEMI_ANNUAL, YieldConvention.ANNUAL, YieldConvention.CONTINUOUS]:
            y_out = convert_yield(y, conv, conv)
            assert abs(y_out - y) < 1e-10

    def test_coupon_bond_conversion(self):
        """Convert coupon bond yield between conventions via price roundtrip."""
        # 4% coupon semi-annual bond, 5% yield → what's the annual yield?
        y_a = convert_yield(
            0.05, YieldConvention.SEMI_ANNUAL, YieldConvention.ANNUAL,
            coupon_rate=0.04, maturity_years=10.0, frequency=2,
        )
        # Both conventions should give same price with same frequency
        p1 = yield_to_price(0.05, 0.04, 10.0, YieldConvention.SEMI_ANNUAL, frequency=2)
        p2 = yield_to_price(y_a, 0.04, 10.0, YieldConvention.ANNUAL, frequency=2)
        assert abs(p1 - p2) < 0.01


# ═══════════════════════════════════════════════════════════════
# Market convention mapping
# ═══════════════════════════════════════════════════════════════


class TestMarketMapping:
    def test_ust_semi(self):
        assert get_yield_convention("UST") == YieldConvention.SEMI_ANNUAL

    def test_bund_annual(self):
        assert get_yield_convention("BUND") == YieldConvention.ANNUAL

    def test_gilt_semi(self):
        assert get_yield_convention("GILT") == YieldConvention.SEMI_ANNUAL

    def test_oat_annual(self):
        assert get_yield_convention("OAT") == YieldConvention.ANNUAL

    def test_ntn_f_continuous(self):
        assert get_yield_convention("NTN_F") == YieldConvention.CONTINUOUS

    def test_ltn_continuous(self):
        assert get_yield_convention("LTN") == YieldConvention.CONTINUOUS

    def test_mbono_semi(self):
        assert get_yield_convention("MBONO") == YieldConvention.SEMI_ANNUAL

    def test_rpgb_quarterly(self):
        assert get_yield_convention("RPGB") == YieldConvention.QUARTERLY

    def test_ustbill_discount(self):
        assert get_yield_convention("USTBILL") == YieldConvention.DISCOUNT

    def test_cetes_discount(self):
        assert get_yield_convention("CETES") == YieldConvention.DISCOUNT

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_yield_convention("FAKE")

    def test_all_53_markets_covered(self):
        """Every sovereign market should have a yield convention."""
        from pricebook.fixed_income.sovereign_bonds import list_markets
        for code in list_markets():
            conv = get_yield_convention(code)
            assert isinstance(conv, YieldConvention), f"{code} missing"
