"""Tests for repo upgrades: tri-party, SOFR lookback, netting, fail states, key-rate DV01, Basel haircuts."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.repo_triparty import (
    TriPartyAgent, EligibilitySchedule, EligibilityRule,
    TriPartyRepo, CollateralAllocation, allocate_collateral,
)
from pricebook.repo_desk import (
    RepoBook, RepoTradeEntry,
    sofr_compounded_with_lookback,
    netting_by_counterparty, NettingResult,
    FailState, auto_escalate_fails,
    repo_key_rate_dv01,
    regulatory_haircut, BASEL_HAIRCUT_FLOORS,
)
from pricebook.repo_term import RepoCurve, RepoRate
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


# ── Tri-party repo ──

class TestTriParty:

    def test_agent_fees(self):
        agent = TriPartyAgent("BNY")
        cost = agent.fees.period_cost(100_000_000, 30, n_substitutions=2)
        assert cost > 0

    def test_eligibility_schedule(self):
        schedule = EligibilitySchedule.standard_govt()
        assert schedule.is_eligible("govt", "AAA", 10.0)
        assert not schedule.is_eligible("hy_corp", "BB", 5.0)

    def test_haircut_for_class(self):
        schedule = EligibilitySchedule.broad_ig()
        assert schedule.haircut_for("govt") == 2.0
        assert schedule.haircut_for("ig_corp") == 5.0

    def test_tri_party_repo_basic(self):
        agent = TriPartyAgent("BNY")
        tp = TriPartyRepo(
            cash_lender="FundA", cash_borrower="DealerB",
            agent=agent, cash_amount=50_000_000, repo_rate=0.045,
            term_days=30, start_date=REF,
        )
        assert tp.interest > 0
        assert tp.agent_cost > 0
        assert tp.all_in_rate > tp.repo_rate  # agent fees increase cost

    def test_allocate_collateral(self):
        schedule = EligibilitySchedule.standard_govt()
        bonds = [
            {"bond_id": "UST10Y", "asset_class": "govt", "face": 60_000_000,
             "market_value": 61_000_000, "rating": "AAA", "maturity_years": 10},
            {"bond_id": "AGY5Y", "asset_class": "agency", "face": 30_000_000,
             "market_value": 30_500_000, "rating": "AA", "maturity_years": 5},
        ]
        allocs = allocate_collateral(50_000_000, bonds, schedule)
        total_value = sum(a.collateral_value for a in allocs)
        assert total_value >= 50_000_000  # covers the cash

    def test_substitution(self):
        agent = TriPartyAgent("BNY")
        tp = TriPartyRepo(
            cash_lender="A", cash_borrower="B", agent=agent,
            cash_amount=10_000_000, repo_rate=0.04, term_days=7, start_date=REF,
        )
        tp.allocate(CollateralAllocation("UST10Y", "govt", 12_000_000, 12_200_000, 2.0, 11_956_000))
        assert len(tp.allocations) == 1

        cost = tp.substitute("UST10Y", CollateralAllocation(
            "UST5Y", "govt", 12_000_000, 12_100_000, 2.0, 11_858_000,
        ))
        assert cost == agent.fees.transaction_fee
        assert len(tp.allocations) == 1
        assert tp.allocations[0].bond_id == "UST5Y"

    def test_pv(self):
        agent = TriPartyAgent("BNY")
        tp = TriPartyRepo(
            cash_lender="A", cash_borrower="B", agent=agent,
            cash_amount=50_000_000, repo_rate=0.045, term_days=30, start_date=REF,
        )
        curve = make_flat_curve(REF, 0.04)
        pv = tp.pv(curve, REF)
        assert math.isfinite(pv)

    def test_to_dict(self):
        agent = TriPartyAgent("BNY")
        tp = TriPartyRepo(
            cash_lender="A", cash_borrower="B", agent=agent,
            cash_amount=50_000_000, repo_rate=0.045, term_days=30, start_date=REF,
        )
        d = tp.to_dict()
        assert d["type"] == "tri_party_repo"
        assert d["params"]["agent"]["name"] == "BNY"


# ── SOFR lookback/lockout ──

class TestSOFRConventions:

    def test_no_lookback(self):
        rates = [0.045] * 30
        r = sofr_compounded_with_lookback(rates, lookback_days=0, lockout_days=0)
        assert r == pytest.approx(0.045, rel=0.01)

    def test_lookback_shifts(self):
        # Rates jump from 4% to 5% on day 15
        rates = [0.04] * 15 + [0.05] * 15
        r_no_lb = sofr_compounded_with_lookback(rates, lookback_days=0)
        r_with_lb = sofr_compounded_with_lookback(rates, lookback_days=2)
        # With lookback, the jump is delayed by 2 days
        assert r_no_lb != pytest.approx(r_with_lb, rel=0.001)

    def test_lockout_freezes(self):
        rates = [0.04] * 25 + [0.06] * 5  # spike in last 5 days
        r_no_lock = sofr_compounded_with_lookback(rates, lockout_days=0)
        r_with_lock = sofr_compounded_with_lookback(rates, lockout_days=5)
        # Lockout freezes last 5 days at day 25 rate (4%)
        assert r_with_lock < r_no_lock  # spike is excluded


# ── Netting ──

class TestNetting:

    def test_offsetting_positions(self):
        book = RepoBook("Test")
        book.add(RepoTradeEntry(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo",
        ))
        book.add(RepoTradeEntry(
            counterparty="JPM", collateral_issuer="UST5Y",
            face_amount=30_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="reverse",
        ))
        netting = netting_by_counterparty(book)
        jpm = [n for n in netting if n.counterparty == "JPM"][0]
        assert jpm.net_exposure == pytest.approx(20_000_000)
        assert jpm.gross_repo == pytest.approx(50_000_000)
        assert jpm.gross_reverse == pytest.approx(30_000_000)

    def test_no_netting_different_cp(self):
        book = RepoBook("Test")
        book.add(RepoTradeEntry(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo",
        ))
        book.add(RepoTradeEntry(
            counterparty="GS", collateral_issuer="UST5Y",
            face_amount=30_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="reverse",
        ))
        netting = netting_by_counterparty(book)
        assert len(netting) == 2  # no netting across CPs


# ── Fail state machine ──

class TestFailStates:

    def test_auto_escalate(self):
        fails = [
            FailState("A", "UST10Y", 10_000_000, REF, 1),
            FailState("B", "UST5Y", 20_000_000, REF, 3),
            FailState("C", "UST2Y", 5_000_000, REF, 7),
            FailState("D", "UST30Y", 15_000_000, REF, 12),
        ]
        auto_escalate_fails(fails)
        assert fails[0].state == "open"          # 1 day
        assert fails[1].state == "investigating"  # 3 days
        assert fails[2].state == "resolving"      # 7 days
        assert fails[3].state == "bought_in"      # 12 days

    def test_buy_in_cost(self):
        fail = FailState("A", "UST10Y", 10_000_000, REF, 12)
        auto_escalate_fails(
            [fail],
            current_prices={"UST10Y": 102.0},
            contract_prices={"UST10Y": 100.0},
        )
        assert fail.buy_in_cost == pytest.approx(200_000)

    def test_resolved_not_escalated(self):
        fail = FailState("A", "UST10Y", 10_000_000, REF, 15, state="resolved")
        auto_escalate_fails([fail])
        assert fail.state == "resolved"  # stays resolved


# ── Key-rate DV01 on repo curve ──

class TestRepoKeyRate:

    def test_per_tenor_sensitivity(self):
        rc = RepoCurve(REF, [
            RepoRate(1, 0.045), RepoRate(30, 0.046),
            RepoRate(90, 0.047), RepoRate(180, 0.048),
        ])
        book = RepoBook("Test")
        book.add(RepoTradeEntry(
            counterparty="A", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=100.0, repo_rate=0.045,
            term_days=1, coupon_rate=0.04, direction="repo",
        ))
        book.add(RepoTradeEntry(
            counterparty="B", collateral_issuer="UST5Y",
            face_amount=30_000_000, bond_price=100.0, repo_rate=0.047,
            term_days=90, coupon_rate=0.035, direction="repo",
        ))
        kr = repo_key_rate_dv01(book, rc)
        assert len(kr) == 4  # one per tenor
        # Some tenors should have non-zero sensitivity
        assert any(abs(v) > 0 for v in kr.values())


# ── Regulatory haircuts ──

class TestRegulatoryHaircuts:

    def test_sovereign_short(self):
        h = regulatory_haircut("sovereign", 0.5)
        assert h == pytest.approx(0.5)

    def test_sovereign_long(self):
        h = regulatory_haircut("sovereign", 10.0)
        assert h == pytest.approx(4.0)

    def test_ig_corp(self):
        h = regulatory_haircut("ig_corp", 3.0)
        assert h == pytest.approx(4.0)

    def test_xccy_add_on(self):
        h_same = regulatory_haircut("sovereign", 5.0, is_cross_currency=False)
        h_xccy = regulatory_haircut("sovereign", 5.0, is_cross_currency=True)
        assert h_xccy == h_same + 8.0  # FX add-on

    def test_hy_flat(self):
        h = regulatory_haircut("hy_corp", 3.0)
        assert h == pytest.approx(15.0)
