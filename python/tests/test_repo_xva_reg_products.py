"""Tests for repo XVA, regulatory capital, and product types."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.repo_desk import RepoTrade
from pricebook.repo_xva import (
    repo_fva, repo_kva, repo_mva, repo_gap_cost,
    repo_total_cost, RepoAllInCost,
)
from pricebook.regulatory.repo_capital import (
    sft_ead, repo_rwa, repo_capital_requirement,
    repo_lcr_outflow, repo_nsfr_rsf,
    repo_capital_summary,
)

REF = date(2024, 7, 15)


def _trade(**kwargs):
    defaults = dict(
        counterparty="JPM", collateral_issuer="UST10Y",
        face_amount=100_000_000, bond_price=100.0, repo_rate=0.045,
        term_days=30, direction="repo", start_date=REF,
        haircut=0.02, settlement_days=0,
    )
    defaults.update(kwargs)
    return RepoTrade(**defaults)


# ═══ Phase 1: Repo XVA ═══

class TestRepoFVA:

    def test_zero_when_no_spread(self):
        t = _trade()
        assert repo_fva(t, funding_spread=0.0) == 0.0

    def test_positive_when_spread_positive(self):
        t = _trade()
        fva = repo_fva(t, funding_spread=0.002)
        assert fva > 0

    def test_proportional_to_spread(self):
        t = _trade()
        fva1 = repo_fva(t, 0.001)
        fva2 = repo_fva(t, 0.002)
        assert fva2 == pytest.approx(2 * fva1)

    def test_hand_calc(self):
        t = _trade(face_amount=100_000_000, bond_price=100.0, haircut=0.02, term_days=30)
        # cash = 100M × (1-0.02) = 98M. FVA = 98M × 0.002 × 30/360 = $16,333
        fva = repo_fva(t, 0.002)
        assert fva == pytest.approx(98_000_000 * 0.002 * 30 / 360, rel=1e-6)


class TestRepoKVA:

    def test_proportional(self):
        t = _trade(term_days=365)
        kva = repo_kva(t, capital_charge=100_000, hurdle_rate=0.10)
        assert kva == pytest.approx(10_000)  # 100K × 10% × 1Y

    def test_zero_capital_zero_kva(self):
        t = _trade()
        assert repo_kva(t, 0, 0.10) == 0.0


class TestRepoMVA:

    def test_uses_haircut_as_im(self):
        t = _trade(haircut=0.02)
        mva = repo_mva(t, funding_spread=0.002)
        # IM proxy = 100M × 0.02 = 2M. MVA = 2M × 0.002 × 30/360
        expected = 2_000_000 * 0.002 * 30 / 360
        assert mva == pytest.approx(expected, rel=0.01)


class TestRepoGapCost:

    def test_zero_at_full_coverage(self):
        t = _trade()
        assert repo_gap_cost(t, 0.05, 1.0) == 0.0

    def test_positive_at_partial(self):
        t = _trade()
        cost = repo_gap_cost(t, 0.06, 0.90)
        assert cost > 0


class TestRepoTotalCost:

    def test_all_in_greater_than_headline(self):
        t = _trade()
        result = repo_total_cost(t, funding_spread=0.002, capital_charge=50_000)
        assert result.all_in_rate > result.headline_rate

    def test_decomposition_sums(self):
        t = _trade()
        result = repo_total_cost(t, funding_spread=0.002, capital_charge=50_000, agent_fees=100)
        expected = result.interest + result.agent_fees + result.fva + result.kva + result.mva + result.gap_cost
        assert result.total_cost == pytest.approx(expected)

    def test_to_dict(self):
        t = _trade()
        result = repo_total_cost(t)
        d = result.to_dict()
        assert "hidden_cost_bps" in d
        assert "all_in_rate_pct" in d


# ═══ Phase 2: Regulatory Capital ═══

class TestSFTEAD:

    def test_over_collateralised_zero(self):
        """Over-collateralised: EAD = 0."""
        ead = sft_ead(cash_lent=100, collateral_value=105, supervisory_haircut=0.02)
        assert ead == 0.0

    def test_under_collateralised(self):
        ead = sft_ead(cash_lent=100, collateral_value=90, supervisory_haircut=0.02)
        # EAD = max(0, 100 - 90×0.98) = max(0, 100 - 88.2) = 11.8
        assert ead == pytest.approx(11.8)

    def test_no_haircut(self):
        ead = sft_ead(100, 100, 0.0)
        assert ead == 0.0  # exactly covered


class TestRepoRWA:

    def test_sovereign_zero(self):
        assert repo_rwa(100, "sovereign") == 0.0

    def test_bank_20pct(self):
        assert repo_rwa(100, "bank") == pytest.approx(20)

    def test_corporate_100pct(self):
        assert repo_rwa(100, "corporate") == pytest.approx(100)


class TestRepoCapReq:

    def test_8pct(self):
        assert repo_capital_requirement(100, 0.08) == pytest.approx(8.0)


class TestLCR:

    def test_l1_zero(self):
        assert repo_lcr_outflow(100_000_000, "GC", 30) == 0.0

    def test_ig_corp_25pct(self):
        assert repo_lcr_outflow(100_000_000, "ig_corp", 30) == pytest.approx(25_000_000)

    def test_beyond_horizon(self):
        assert repo_lcr_outflow(100_000_000, "ig_corp", 60) == 0.0


class TestNSFR:

    def test_l1_short_10pct(self):
        rsf = repo_nsfr_rsf(100_000_000, "GC", 30)
        assert rsf == pytest.approx(10_000_000)

    def test_l1_long_15pct(self):
        rsf = repo_nsfr_rsf(100_000_000, "GC", 365)
        assert rsf == pytest.approx(15_000_000)

    def test_other_long_100pct(self):
        rsf = repo_nsfr_rsf(100_000_000, "hy_corp", 365)
        assert rsf == pytest.approx(100_000_000)


class TestCapitalSummary:

    def test_summary(self):
        trades = [
            {"cash_lent": 100_000_000, "collateral_value": 105_000_000,
             "supervisory_haircut": 0.02, "counterparty_type": "bank",
             "collateral_type": "GC", "remaining_days": 30},
        ]
        summary = repo_capital_summary(trades)
        assert summary.n_trades == 1
        # 105M × 0.98 = 102.9M > 100M → EAD = 0
        assert summary.total_ead == 0.0
        assert summary.total_lcr_outflow == 0.0  # L1 → 0%
        assert summary.total_nsfr_rsf > 0


# ═══ Phase 3: Product Types ═══

class TestBuySellBack:

    def test_implied_rate(self):
        t = RepoTrade.buy_sell_back(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, spot_dirty_price=102.0,
            forward_dirty_price=102.5, term_days=30,
            start_date=REF, settlement_days=0,
        )
        # Implied rate = (102.5/102 - 1) × 360/30 = 5.88%
        expected = (102.5 / 102.0 - 1) * 360 / 30
        assert t.repo_rate == pytest.approx(expected, rel=1e-4)

    def test_direction_is_repo(self):
        t = RepoTrade.buy_sell_back(
            "JPM", "UST10Y", 50_000_000, 102.0, 102.5, 30, REF, settlement_days=0,
        )
        assert t.direction == "repo"


class TestRepoToMaturity:

    def test_term_equals_bond_life(self):
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond.treasury_note(date(2024,2,15), date(2026,2,15), 0.04)
        t = RepoTrade.repo_to_maturity(
            "JPM", bond, 0.045, REF,
            bond_price=100.0, settlement_days=0,
        )
        expected_term = (date(2026,2,15) - REF).days
        assert t.term_days == expected_term


class TestEquityRepo:

    def test_equity_haircut(self):
        t = RepoTrade.equity_repo(
            "GS", "AAPL", 10_000, 150.0, 0.05, 30, REF, settlement_days=0,
        )
        assert t.collateral_type == "equity"
        assert t.haircut >= 0.15  # Basel equity min

    def test_dividend_as_coupon(self):
        t = RepoTrade.equity_repo(
            "GS", "AAPL", 10_000, 150.0, 0.05, 30, REF,
            dividend_yield=0.006, settlement_days=0,
        )
        assert t.coupon_rate == 0.006
