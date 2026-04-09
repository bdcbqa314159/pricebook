"""Tests for credit RWA: SA-CR, F-IRB, A-IRB, slotting."""

import pytest

from pricebook.regulatory.credit_rwa import (
    SA_SOVEREIGN_RW, SA_CORPORATE_RW, SA_BANK_RW,
    get_sa_sovereign_rw, get_sa_bank_rw, get_sa_corporate_rw, get_sa_retail_rw,
    get_sa_residential_re_rw, get_sa_commercial_re_rw,
    SAExposure, calculate_sa_rwa,
    calculate_correlation, calculate_maturity_adjustment,
    calculate_capital_requirement, calculate_irb_rwa, calculate_airb_rwa,
    sme_correlation_adjustment, slotting_risk_weight, calculate_slotting_rwa,
    compare_sa_vs_irb, PD_FLOORS, LGD_FLOORS,
)


# ---- SA risk weights ----

class TestSARiskWeights:
    def test_aaa_sovereign_zero(self):
        assert get_sa_sovereign_rw("AAA") == 0
        assert get_sa_sovereign_rw("AA") == 0

    def test_unrated_sovereign(self):
        assert get_sa_sovereign_rw("unrated") == 100

    def test_bank_rating_monotonic(self):
        assert get_sa_bank_rw("AAA") < get_sa_bank_rw("BBB") < get_sa_bank_rw("B")

    def test_short_term_lower(self):
        """Short-term bank exposures get lower RW."""
        assert get_sa_bank_rw("BBB", short_term=True) <= get_sa_bank_rw("BBB", short_term=False)

    def test_corporate(self):
        assert get_sa_corporate_rw("AAA") == 20
        assert get_sa_corporate_rw("BBB") == 75
        assert get_sa_corporate_rw("unrated") == 100

    def test_retail(self):
        assert get_sa_retail_rw("regulatory_retail") == 75
        assert get_sa_retail_rw("transactor") == 45


class TestRealEstate:
    def test_low_ltv_lower_rw(self):
        assert get_sa_residential_re_rw(0.4) < get_sa_residential_re_rw(0.95)

    def test_income_producing_higher(self):
        general = get_sa_residential_re_rw(0.7, income_producing=False)
        income = get_sa_residential_re_rw(0.7, income_producing=True)
        assert income > general

    def test_commercial_re(self):
        assert get_sa_commercial_re_rw(0.5) < get_sa_commercial_re_rw(0.9)


# ---- SA RWA calculation ----

class TestSARWA:
    def test_sovereign(self):
        exp = SAExposure(ead=10_000_000, asset_class="sovereign", rating="AAA")
        r = calculate_sa_rwa(exp)
        assert r["risk_weight_pct"] == 0
        assert r["rwa"] == 0

    def test_corporate_bbb(self):
        exp = SAExposure(ead=10_000_000, asset_class="corporate", rating="BBB")
        r = calculate_sa_rwa(exp)
        assert r["risk_weight_pct"] == 75
        assert r["rwa"] == pytest.approx(7_500_000)

    def test_capital_is_8pct(self):
        exp = SAExposure(ead=10_000_000, asset_class="corporate", rating="BBB")
        r = calculate_sa_rwa(exp)
        assert r["capital_requirement"] == pytest.approx(r["rwa"] * 0.08)


# ---- IRB correlation ----

class TestCorrelation:
    def test_corporate_range(self):
        """Corporate correlation between 12% and 24%."""
        for pd in [0.001, 0.01, 0.05, 0.20]:
            r = calculate_correlation(pd, "corporate")
            assert 0.12 <= r <= 0.24

    def test_retail_mortgage_fixed(self):
        assert calculate_correlation(0.01, "retail_mortgage") == 0.15

    def test_retail_revolving_fixed(self):
        assert calculate_correlation(0.01, "retail_revolving") == 0.04

    def test_high_pd_lower_correlation(self):
        """Correlation decreases as PD increases."""
        r_low = calculate_correlation(0.001, "corporate")
        r_high = calculate_correlation(0.20, "corporate")
        assert r_low > r_high

    def test_sme_adjustment(self):
        """SME with low sales gets correlation reduction."""
        r_no_sme = calculate_correlation(0.02, "corporate")
        r_sme = calculate_correlation(0.02, "corporate", sales_turnover=10)
        assert r_sme < r_no_sme


class TestSMEAdjustment:
    def test_no_adjustment_above_50m(self):
        assert sme_correlation_adjustment(60) == 0.0

    def test_max_adjustment_at_5m(self):
        assert sme_correlation_adjustment(5) == pytest.approx(0.04)

    def test_zero_at_50m(self):
        assert sme_correlation_adjustment(50) == pytest.approx(0.0)

    def test_none(self):
        assert sme_correlation_adjustment(None) == 0.0


class TestMaturityAdjustment:
    def test_decreases_with_pd(self):
        b1 = calculate_maturity_adjustment(0.001)
        b2 = calculate_maturity_adjustment(0.10)
        assert b1 > b2

    def test_positive(self):
        for pd in [0.001, 0.01, 0.05, 0.20]:
            assert calculate_maturity_adjustment(pd) > 0


# ---- IRB capital ----

class TestCapitalRequirement:
    def test_positive(self):
        k = calculate_capital_requirement(0.01, 0.45, 2.5, "corporate")
        assert k > 0

    def test_higher_pd_higher_k(self):
        k_low = calculate_capital_requirement(0.001, 0.45)
        k_high = calculate_capital_requirement(0.10, 0.45)
        assert k_high > k_low

    def test_higher_lgd_higher_k(self):
        k_low = calculate_capital_requirement(0.01, 0.20)
        k_high = calculate_capital_requirement(0.01, 0.80)
        assert k_high > k_low

    def test_pd_floor_applied(self):
        """Tiny PD gets floored."""
        k_below_floor = calculate_capital_requirement(0.0000001, 0.45)
        k_at_floor = calculate_capital_requirement(0.0005, 0.45)
        assert k_below_floor == pytest.approx(k_at_floor, rel=0.01)


# ---- IRB RWA ----

class TestIRBRWA:
    def test_basic(self):
        r = calculate_irb_rwa(10_000_000, pd=0.01, lgd=0.45, maturity=5)
        assert r["rwa"] > 0
        assert r["approach"] == "F-IRB"

    def test_default_lgd_45(self):
        """F-IRB default LGD = 45% for senior unsecured."""
        r = calculate_irb_rwa(10_000_000, pd=0.01)
        assert r["lgd"] == 0.45

    def test_scales_with_ead(self):
        r1 = calculate_irb_rwa(10_000_000, pd=0.01, lgd=0.45)
        r2 = calculate_irb_rwa(20_000_000, pd=0.01, lgd=0.45)
        assert r2["rwa"] == pytest.approx(2 * r1["rwa"])

    def test_expected_loss(self):
        r = calculate_irb_rwa(10_000_000, pd=0.01, lgd=0.45)
        assert r["expected_loss"] == pytest.approx(10_000_000 * 0.01 * 0.45)


# ---- A-IRB ----

class TestAIRB:
    def test_lgd_floor_applied(self):
        """LGD below floor → floored."""
        r = calculate_airb_rwa(10_000_000, pd=0.01, lgd=0.10)
        assert r["lgd_floored"] >= LGD_FLOORS["senior_unsecured"]

    def test_lgd_above_floor_unchanged(self):
        r = calculate_airb_rwa(10_000_000, pd=0.01, lgd=0.45)
        assert r["lgd_floored"] == 0.45

    def test_retail_mortgage_floor(self):
        r = calculate_airb_rwa(
            10_000_000, pd=0.01, lgd=0.02,
            asset_class="retail_mortgage", collateral_type="secured_rre",
        )
        assert r["lgd_floored"] >= 0.05


# ---- Specialised lending slotting ----

class TestSlotting:
    def test_strong_lower_than_weak(self):
        assert slotting_risk_weight("strong", "PF") < slotting_risk_weight("weak", "PF")

    def test_hvcre_higher_than_pf(self):
        """HVCRE has higher RW than other classes."""
        assert slotting_risk_weight("strong", "HVCRE") > slotting_risk_weight("strong", "PF")

    def test_calculate_rwa(self):
        r = calculate_slotting_rwa(10_000_000, "strong", "PF")
        assert r["rwa"] == pytest.approx(7_000_000)  # 10M × 70%

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError):
            slotting_risk_weight("excellent", "PF")


# ---- Compare SA vs IRB ----

class TestCompareSAvsIRB:
    def test_basic(self):
        r = compare_sa_vs_irb(
            ead=10_000_000, pd=0.005, lgd=0.45, maturity=5,
            asset_class="corporate", rating="A",
        )
        assert "sa_rwa" in r
        assert "irb_rwa" in r
        assert r["sa_rwa"] > 0
        assert r["irb_rwa"] > 0
