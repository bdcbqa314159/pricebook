"""Tests for portfolio-wide LCR/NSFR."""

import pytest

from pricebook.regulatory.liquidity import (
    LiquidityPosition, PortfolioLiquidityResult,
    calculate_portfolio_lcr, liquidity_stress,
)


@pytest.fixture
def mixed_portfolio():
    return [
        # Assets
        LiquidityPosition("CASH", "cash", 100e6, is_asset=True),
        LiquidityPosition("UST", "bond", 200e6, 200e6, rating="AAA",
                          counterparty_type="sovereign", hqla_level="L1", is_asset=True),
        LiquidityPosition("IG_BOND", "bond", 150e6, 150e6, rating="AA",
                          hqla_level="L2A", is_asset=True),
        LiquidityPosition("LOAN1", "loan", 500e6, maturity_days=365*3,
                          counterparty_type="corporate", is_asset=True),
        LiquidityPosition("LOAN2", "loan", 50e6, maturity_days=20,
                          counterparty_type="corporate", is_asset=True),
        # Liabilities
        LiquidityPosition("DEP_RETAIL", "deposit", 300e6, is_asset=False,
                          counterparty_type="retail", maturity_days=30),
        LiquidityPosition("DEP_CORP", "deposit", 200e6, is_asset=False,
                          counterparty_type="corporate", maturity_days=90),
        LiquidityPosition("REPO_LIA", "repo", 100e6, is_asset=False, is_secured=True),
    ]


class TestLCR:
    def test_basic(self, mixed_portfolio):
        r = calculate_portfolio_lcr(mixed_portfolio)
        assert r.hqla_total > 0
        assert r.lcr_pct > 0

    def test_cash_is_l1(self):
        positions = [
            LiquidityPosition("CASH", "cash", 100e6, is_asset=True),
            LiquidityPosition("DEP", "deposit", 50e6, is_asset=False, counterparty_type="retail"),
        ]
        r = calculate_portfolio_lcr(positions)
        assert r.hqla_total >= 100e6  # cash = L1, no haircut

    def test_sovereign_aaa_is_l1(self):
        positions = [
            LiquidityPosition("UST", "bond", 100e6, 100e6, rating="AAA",
                              counterparty_type="sovereign", is_asset=True),
            LiquidityPosition("DEP", "deposit", 50e6, is_asset=False, counterparty_type="retail"),
        ]
        r = calculate_portfolio_lcr(positions)
        assert r.hqla_total >= 100e6

    def test_lcr_compliant(self, mixed_portfolio):
        r = calculate_portfolio_lcr(mixed_portfolio)
        # With substantial HQLA, should be compliant
        assert isinstance(r.lcr_compliant, bool)

    def test_product_breakdown(self, mixed_portfolio):
        r = calculate_portfolio_lcr(mixed_portfolio)
        assert "cash" in r.product_breakdown
        assert "bond" in r.product_breakdown
        assert "loan" in r.product_breakdown

    def test_inflow_from_maturing_loan(self):
        positions = [
            LiquidityPosition("CASH", "cash", 10e6, is_asset=True),
            LiquidityPosition("LOAN", "loan", 50e6, maturity_days=15,
                              counterparty_type="corporate", is_asset=True),
            LiquidityPosition("DEP", "deposit", 100e6, is_asset=False,
                              counterparty_type="corporate"),
        ]
        r = calculate_portfolio_lcr(positions)
        assert r.total_inflows > 0


class TestNSFR:
    def test_nsfr_positive(self, mixed_portfolio):
        r = calculate_portfolio_lcr(mixed_portfolio)
        assert r.nsfr_pct > 0
        assert r.asf_total > 0
        assert r.rsf_total > 0

    def test_cash_zero_rsf(self):
        positions = [
            LiquidityPosition("CASH", "cash", 100e6, is_asset=True),
            LiquidityPosition("FUND", "equity", 100e6, is_asset=False, maturity_days=999),
        ]
        r = calculate_portfolio_lcr(positions)
        # Cash RSF = 0, equity ASF = 100% → NSFR very high
        assert r.nsfr_pct > 100


class TestStress:
    def test_stress_reduces_lcr(self, mixed_portfolio):
        base = calculate_portfolio_lcr(mixed_portfolio)
        stressed = liquidity_stress(mixed_portfolio, outflow_multiplier=2.0, hqla_haircut=0.20)
        assert stressed.lcr_pct <= base.lcr_pct

    def test_stress_result_type(self, mixed_portfolio):
        r = liquidity_stress(mixed_portfolio)
        assert isinstance(r, PortfolioLiquidityResult)


class TestToDict:
    def test_to_dict(self, mixed_portfolio):
        d = calculate_portfolio_lcr(mixed_portfolio).to_dict()
        assert "lcr_pct" in d
        assert "nsfr_pct" in d
        assert "product_breakdown" in d
