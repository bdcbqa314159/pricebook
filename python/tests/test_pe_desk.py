"""Tests for PE trading desk (9-component protocol)."""

import pytest
from datetime import date

from pricebook.fund_participation import FundParticipation, PEFundParticipation, WaterfallConfig
from pricebook.lbo import LBOModel
from pricebook.pe_desk import (
    PERiskMetrics, pe_risk_metrics,
    PEBook, PEBookEntry,
    PECarryDecomposition, pe_carry_decomposition,
    PEDailyPnL, pe_daily_pnl,
    PEDashboard, pe_dashboard,
    PEStressResult, pe_stress_suite,
    PECapitalResult, pe_capital,
    PEHedgeRecommendation, pe_hedge_recommendations,
    PELifecycle,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def fund_entry():
    fund = FundParticipation(100_000_000, vintage_year=2020, gross_return=0.12)
    return PEBookEntry(
        trade_id="PE001", instrument=fund, product_type="fund",
        fund_manager="Apollo", vintage_year=2020, sector="Buyout",
    )

@pytest.fixture
def lbo_entry():
    lbo = LBOModel(enterprise_value=500_000_000, entry_ebitda=100_000_000)
    return PEBookEntry(
        trade_id="PE002", instrument=lbo, product_type="lbo",
        fund_manager="KKR", vintage_year=2022, sector="Healthcare",
    )

@pytest.fixture
def pe_book(fund_entry, lbo_entry):
    book = PEBook("test_pe")
    book.add(fund_entry)
    book.add(lbo_entry)
    return book


# ═══════════════════════════════════════════════════════════════
# 1. Risk Metrics
# ═══════════════════════════════════════════════════════════════

class TestRiskMetrics:
    def test_fund_metrics(self, fund_entry):
        m = pe_risk_metrics(fund_entry)
        assert m.tvpi > 0
        assert m.irr > 0
        assert m.product_type == "fund"
        assert m.notional == 100_000_000

    def test_lbo_metrics(self, lbo_entry):
        m = pe_risk_metrics(lbo_entry)
        assert m.nav > 0
        assert m.irr > 0
        assert m.product_type == "lbo"

    def test_to_dict(self, fund_entry):
        d = pe_risk_metrics(fund_entry).to_dict()
        assert "nav" in d
        assert "irr" in d


# ═══════════════════════════════════════════════════════════════
# 2. Book
# ═══════════════════════════════════════════════════════════════

class TestBook:
    def test_add_and_len(self, pe_book):
        assert len(pe_book) == 2

    def test_total_commitment(self, pe_book):
        assert pe_book.total_commitment() > 0

    def test_by_vintage(self, pe_book):
        by_v = pe_book.by_vintage()
        assert 2020 in by_v
        assert 2022 in by_v

    def test_by_manager(self, pe_book):
        by_m = pe_book.by_manager()
        assert "Apollo" in by_m
        assert "KKR" in by_m

    def test_by_sector(self, pe_book):
        by_s = pe_book.by_sector()
        assert "Buyout" in by_s
        assert "Healthcare" in by_s


# ═══════════════════════════════════════════════════════════════
# 3. Carry Decomposition
# ═══════════════════════════════════════════════════════════════

class TestCarry:
    def test_fund_carry(self, fund_entry):
        c = pe_carry_decomposition(fund_entry)
        assert c.management_fee > 0
        assert c.distribution_income >= 0
        assert isinstance(c.net_carry, float)

    def test_lbo_carry(self, lbo_entry):
        c = pe_carry_decomposition(lbo_entry)
        assert c.management_fee == 0  # LBO has no periodic fees

    def test_to_dict(self, fund_entry):
        d = pe_carry_decomposition(fund_entry).to_dict()
        assert "net_carry" in d


# ═══════════════════════════════════════════════════════════════
# 4. Daily P&L
# ═══════════════════════════════════════════════════════════════

class TestDailyPnL:
    def test_pnl(self, fund_entry):
        pnl = pe_daily_pnl(fund_entry, 50_000_000, 51_000_000, date(2024, 6, 15))
        assert pnl.nav_change == 1_000_000
        assert pnl.total != 0
        assert pnl.date == date(2024, 6, 15)

    def test_to_dict(self, fund_entry):
        d = pe_daily_pnl(fund_entry, 50e6, 51e6, date(2024, 1, 1)).to_dict()
        assert "total" in d


# ═══════════════════════════════════════════════════════════════
# 5. Dashboard
# ═══════════════════════════════════════════════════════════════

class TestDashboard:
    def test_dashboard(self, pe_book):
        dash = pe_dashboard(pe_book, date(2024, 6, 15))
        assert dash.n_positions == 2
        assert dash.total_nav > 0
        assert dash.total_commitment > 0
        assert len(dash.by_vintage) >= 1
        assert len(dash.by_manager) >= 1

    def test_to_dict(self, pe_book):
        d = pe_dashboard(pe_book, date(2024, 1, 1)).to_dict()
        assert "weighted_irr" in d


# ═══════════════════════════════════════════════════════════════
# 6. Stress Suite
# ═══════════════════════════════════════════════════════════════

class TestStress:
    def test_stress_count(self, pe_book):
        results = pe_stress_suite(pe_book)
        assert len(results) == 5

    def test_stress_directions(self, pe_book):
        results = pe_stress_suite(pe_book)
        pnls = [r.pnl for r in results]
        assert any(p < 0 for p in pnls)
        assert any(p > 0 for p in pnls)

    def test_to_dict(self, pe_book):
        d = pe_stress_suite(pe_book)[0].to_dict()
        assert "scenario" in d


# ═══════════════════════════════════════════════════════════════
# 7. Capital
# ═══════════════════════════════════════════════════════════════

class TestCapital:
    def test_fund_capital(self):
        # Use a fund with unfunded commitment (only 2 years of drawdown)
        fund = FundParticipation(
            100_000_000, vintage_year=2024, fund_life_years=8,
            drawdown_schedule=[(1, 0.25), (2, 0.25)],
        )
        entry = PEBookEntry(trade_id="PE_CAP", instrument=fund, product_type="fund")
        cap = pe_capital(entry)
        assert cap.risk_weight == 2.50
        assert cap.total_exposure > 0
        assert cap.capital == cap.rwa * 0.08

    def test_to_dict(self, fund_entry):
        d = pe_capital(fund_entry).to_dict()
        assert "rwa" in d


# ═══════════════════════════════════════════════════════════════
# 8. Hedge Recommendations
# ═══════════════════════════════════════════════════════════════

class TestHedge:
    def test_no_breach(self, pe_book):
        recs = pe_hedge_recommendations(pe_book, concentration_limit=0.90)
        # With 90% limit and 2 managers, no breach
        mgr_recs = [r for r in recs if r.risk_type == "manager_concentration"]
        assert len(mgr_recs) == 0

    def test_concentration_breach(self):
        # Single manager with 100% concentration
        book = PEBook()
        for i in range(3):
            book.add(PEBookEntry(
                trade_id=f"PE{i}", instrument=FundParticipation(100_000_000),
                product_type="fund", fund_manager="Same GP",
            ))
        recs = pe_hedge_recommendations(book, concentration_limit=0.25)
        mgr_recs = [r for r in recs if r.risk_type == "manager_concentration"]
        assert len(mgr_recs) > 0

    def test_to_dict(self, pe_book):
        recs = pe_hedge_recommendations(pe_book)
        for r in recs:
            d = r.to_dict()
            assert "action" in d


# ═══════════════════════════════════════════════════════════════
# 9. Lifecycle
# ═══════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_creation(self, fund_entry):
        lc = PELifecycle(fund_entry, creation_date=date(2020, 1, 1))
        assert len(lc.history) == 1
        assert lc.history[0]["type"] == "creation"

    def test_record_event(self, fund_entry):
        lc = PELifecycle(fund_entry)
        ev = lc.record_event(PELifecycle.CAPITAL_CALL, date(2020, 6, 1), amount=25_000_000)
        assert ev["type"] == "capital_call"
        assert ev["amount"] == 25_000_000
        assert len(lc.history) == 1

    def test_maturity_alert(self, fund_entry):
        lc = PELifecycle(fund_entry)
        # Fund vintage 2020, 8-year life → matures 2028-12-31
        alert = lc.maturity_alert(date(2028, 6, 1), alert_days=365)
        assert alert is not None
        assert alert["days_remaining"] > 0

    def test_no_maturity_alert_early(self, fund_entry):
        lc = PELifecycle(fund_entry)
        alert = lc.maturity_alert(date(2022, 1, 1), alert_days=365)
        assert alert is None

    def test_upcoming_events(self, fund_entry):
        lc = PELifecycle(fund_entry)
        events = lc.upcoming_events(date(2028, 10, 1), horizon_days=365)
        assert isinstance(events, list)
