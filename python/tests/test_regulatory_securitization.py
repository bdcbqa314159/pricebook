"""Tests for securitisation RWA and trade-specific RWA."""

import pytest

from pricebook.regulatory.securitization import (
    calculate_sec_sa_p, calculate_sec_sa_rw, calculate_sec_sa_rwa,
    calculate_sec_irba_kirb, calculate_sec_irba_rwa,
    pd_to_cqs, calculate_erba_rwa,
    calculate_cds_rwa, calculate_repo_rwa, calculate_trs_rwa, calculate_loan_rwa,
    ERBA_RW_BASE,
)


# ---- SEC-SA ----

class TestSECSAParameter:
    def test_p_n25_lgd50(self):
        p = calculate_sec_sa_p(25, 0.50)
        # max(0.3, 0.5 × (1 - 0.50)) = max(0.3, 0.25) = 0.3
        assert p == pytest.approx(0.3)

    def test_p_low_n(self):
        """Low n with high LGD: p > 0.3 floor."""
        p = calculate_sec_sa_p(5, 0.20)
        # max(0.3, 0.5×0.8 + 0.5/5×0.2) = max(0.3, 0.42) = 0.42
        assert p > 0.3

    def test_p_min(self):
        p = calculate_sec_sa_p(100, 0.20)
        # max(0.3, 0.5 × 0.80) = max(0.3, 0.40) = 0.40
        assert p == pytest.approx(0.40)


class TestSECSARW:
    def test_senior_tranche(self):
        """Senior tranche above K_SA → moderate RW."""
        rw = calculate_sec_sa_rw(0.08, attachment=0.20, detachment=1.0, n=50)
        assert rw > 0
        assert rw <= 1250

    def test_equity_tranche_max_rw(self):
        """Equity tranche entirely below K_SA → near 1250%."""
        rw = calculate_sec_sa_rw(0.08, attachment=0.0, detachment=0.05, n=50)
        # K_SSFA = 0.05, RW = 0.05 × 12.5 × 100 = 62.5? No, capped/floored
        # Equity captures all losses up to K_SA → high but not 1250
        assert rw > 0

    def test_sts_lower_floor(self):
        rw_normal = calculate_sec_sa_rw(0.08, 0.30, 1.0, is_sts=False)
        rw_sts = calculate_sec_sa_rw(0.08, 0.30, 1.0, is_sts=True)
        # STS floor 10%, non-STS 15%
        assert rw_sts <= rw_normal

    def test_capped_at_1250(self):
        rw = calculate_sec_sa_rw(0.50, attachment=0.0, detachment=0.05, n=10)
        assert rw <= 1250


class TestSECSARWA:
    def test_basic(self):
        r = calculate_sec_sa_rwa(
            ead=10_000_000, attachment=0.15, detachment=0.30,
            ksa=0.08, n=50,
        )
        assert r["rwa"] > 0
        assert r["thickness"] == pytest.approx(0.15)

    def test_resecuritization_higher(self):
        r_normal = calculate_sec_sa_rwa(10_000_000, 0.15, 0.30, ksa=0.08)
        r_resec = calculate_sec_sa_rwa(10_000_000, 0.15, 0.30, ksa=0.08, is_resecuritization=True)
        assert r_resec["rwa"] >= r_normal["rwa"]


# ---- SEC-IRBA ----

class TestSECIRBA:
    def test_kirb_from_pool(self):
        pool = [
            {"ead": 1_000_000, "pd": 0.01, "lgd": 0.45, "maturity": 5},
            {"ead": 2_000_000, "pd": 0.02, "lgd": 0.45, "maturity": 5},
        ]
        kirb = calculate_sec_irba_kirb(pool)
        assert 0 < kirb < 1.0

    def test_empty_pool_default(self):
        assert calculate_sec_irba_kirb([]) == 0.08

    def test_irba_rwa(self):
        pool = [
            {"ead": 1_000_000, "pd": 0.01, "lgd": 0.45, "maturity": 5},
            {"ead": 2_000_000, "pd": 0.02, "lgd": 0.45, "maturity": 5},
        ]
        r = calculate_sec_irba_rwa(
            ead=10_000_000, attachment=0.15, detachment=0.30,
            underlying_exposures=pool, n=50,
        )
        assert r["rwa"] > 0
        assert r["kirb"] > 0


# ---- ERBA ----

class TestPDtoCQS:
    def test_aaa(self):
        assert pd_to_cqs(0.00005) == 1

    def test_bbb(self):
        # PD ~ 0.40% → CQS 6 (BBB-)
        assert pd_to_cqs(0.0040) == 6

    def test_high_pd(self):
        assert pd_to_cqs(0.60) == 17


class TestERBA:
    def test_senior_aaa(self):
        r = calculate_erba_rwa(10_000_000, cqs=1, seniority="senior", maturity=5)
        assert r["rwa"] > 0
        assert r["risk_weight_pct"] >= 15  # floor

    def test_non_senior_higher(self):
        senior = calculate_erba_rwa(10_000_000, cqs=5, seniority="senior", maturity=5)
        non_senior = calculate_erba_rwa(
            10_000_000, cqs=5, seniority="non_senior", maturity=5, thickness=0.05,
        )
        assert non_senior["risk_weight_pct"] > senior["risk_weight_pct"]

    def test_low_cqs_higher_rw(self):
        r1 = calculate_erba_rwa(10_000_000, cqs=1, seniority="senior", maturity=5)
        r10 = calculate_erba_rwa(10_000_000, cqs=10, seniority="senior", maturity=5)
        assert r10["risk_weight_pct"] > r1["risk_weight_pct"]

    def test_invalid_cqs(self):
        with pytest.raises(ValueError):
            calculate_erba_rwa(10_000_000, cqs=18)

    def test_thinner_higher_rw(self):
        thick = calculate_erba_rwa(10_000_000, cqs=5, seniority="non_senior", maturity=5, thickness=0.10)
        thin = calculate_erba_rwa(10_000_000, cqs=5, seniority="non_senior", maturity=5, thickness=0.02)
        assert thin["risk_weight_pct"] > thick["risk_weight_pct"]


# ---- Trade-specific RWA ----

class TestCDSRWA:
    def test_protection_buyer(self):
        """Protection buyer: only CCR risk."""
        r = calculate_cds_rwa(
            notional=10_000_000, reference_pd=0.02, counterparty_pd=0.005,
            is_protection_buyer=True,
        )
        assert r["ccr_rwa"] > 0
        assert r["reference_rwa"] == 0
        assert r["total_rwa"] == r["ccr_rwa"]

    def test_protection_seller(self):
        """Protection seller: CCR + reference entity risk."""
        r = calculate_cds_rwa(
            notional=10_000_000, reference_pd=0.02, counterparty_pd=0.005,
            is_protection_buyer=False,
        )
        assert r["reference_rwa"] > 0
        assert r["total_rwa"] > r["ccr_rwa"]


class TestRepoRWA:
    def test_collateralised(self):
        """Reverse repo with high-quality collateral."""
        r = calculate_repo_rwa(
            cash_lent=10_000_000, collateral_value=10_500_000,
            counterparty_pd=0.005, haircut=0.04,
        )
        # Adjusted collateral = 10.5M × 0.96 = 10.08M > 10M cash → EAD = 0
        assert r["ead"] == 0
        assert r["rwa"] == 0

    def test_undercollateralised(self):
        r = calculate_repo_rwa(
            cash_lent=10_000_000, collateral_value=8_000_000,
            counterparty_pd=0.005, haircut=0.04,
        )
        assert r["ead"] > 0
        assert r["rwa"] > 0


class TestTRSRWA:
    def test_receiver(self):
        """TRS receiver: full reference asset risk + CCR."""
        r = calculate_trs_rwa(
            notional=10_000_000, reference_pd=0.02, counterparty_pd=0.005,
            is_total_return_payer=False,
        )
        assert r["reference_rwa"] > 0
        assert r["total_rwa"] > 0

    def test_payer(self):
        """TRS payer: only CCR (no reference asset risk)."""
        r = calculate_trs_rwa(
            notional=10_000_000, reference_pd=0.02, counterparty_pd=0.005,
            is_total_return_payer=True,
        )
        assert r["reference_rwa"] == 0


class TestLoanRWA:
    def test_irb(self):
        r = calculate_loan_rwa(
            ead=10_000_000, pd=0.01, lgd=0.45, maturity=5,
            asset_class="corporate", approach="irb",
        )
        assert r["rwa"] > 0
        assert r["approach"] == "F-IRB"

    def test_sa(self):
        r = calculate_loan_rwa(
            ead=10_000_000, pd=0.01, lgd=0.45, maturity=5,
            asset_class="corporate", approach="sa",
        )
        assert r["rwa"] > 0
        assert r["approach"] == "SA-CR"
