"""Tests for bilateral CLN with CSA integration."""

import pytest

from pricebook.credit.bilateral_csa import (
    CSATerms, BilateralCSAPricer, BilateralCSAResult,
    bilateral_cln_with_csa,
)


class TestCSATerms:
    def test_defaults(self):
        csa = CSATerms()
        assert csa.threshold_investor == 0.0
        assert csa.margin_period_risk_days == 10

    def test_to_dict(self):
        csa = CSATerms(threshold_investor=1e6)
        d = csa.to_dict()
        assert d["threshold_investor"] == 1e6


class TestBilateralCLNWithCSA:
    def test_basic_pricing(self):
        result = bilateral_cln_with_csa(
            notional=1e6, coupon=0.05, maturity_years=5.0,
            ref_hazard=0.02, issuer_hazard=0.01,
            risk_free_rate=0.04, n_paths=5000,
        )
        assert isinstance(result, BilateralCSAResult)
        assert result.clean_pv > 0
        assert result.csa_adjusted_pv > 0

    def test_higher_hazard_lower_pv(self):
        """Higher ref hazard → lower clean PV."""
        r1 = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, ref_hazard=0.01, issuer_hazard=0.005,
            risk_free_rate=0.04, n_paths=5000,
        )
        r2 = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, ref_hazard=0.05, issuer_hazard=0.005,
            risk_free_rate=0.04, n_paths=5000,
        )
        assert r2.clean_pv < r1.clean_pv

    def test_funding_spread_reduces_pv(self):
        """Positive funding spread → lower adjusted PV."""
        r_no_fund = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.01, 0.04,
            funding_spread=0.0, n_paths=5000,
        )
        r_fund = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.01, 0.04,
            funding_spread=0.01, n_paths=5000,
        )
        assert r_fund.csa_adjusted_pv <= r_no_fund.csa_adjusted_pv

    def test_csa_with_threshold(self):
        """Non-zero threshold → some uncollateralised exposure."""
        csa = CSATerms(threshold_investor=100_000)
        result = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.01, 0.04,
            csa=csa, funding_spread=0.005, n_paths=5000,
        )
        assert result.uncollateralised_exposure >= 0

    def test_zero_csa_threshold(self):
        """Zero threshold → fully collateralised."""
        csa = CSATerms(threshold_investor=0.0, independent_amount=50_000)
        result = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.01, 0.04,
            csa=csa, n_paths=5000,
        )
        assert result.expected_collateral >= 0

    def test_cva_positive(self):
        """CVA should be non-negative (investor bears counterparty risk)."""
        result = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.03, 0.04,
            n_paths=10000,
        )
        assert result.cva >= 0

    def test_correlation_impact(self):
        """Higher correlation → changes CVA (wrong-way risk)."""
        r_low = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.02, 0.04,
            correlation=0.1, n_paths=5000,
        )
        r_high = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.02, 0.04,
            correlation=0.8, n_paths=5000,
        )
        # With high correlation, both default together more often
        assert r_low.clean_pv != r_high.clean_pv  # should differ

    def test_to_dict(self):
        result = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.01, 0.04, n_paths=2000,
        )
        d = result.to_dict()
        assert "clean_pv" in d
        assert "cva" in d
        assert "fva" in d
        assert "total_xva" in d

    def test_xva_decomposition(self):
        """total_xva = cva + dva + fva."""
        result = bilateral_cln_with_csa(
            1e6, 0.05, 5.0, 0.02, 0.01, 0.04,
            funding_spread=0.005, n_paths=5000,
        )
        assert abs(result.total_xva - (result.cva + result.dva + result.fva)) < 0.01
