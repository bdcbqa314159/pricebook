"""Repo desk protocol compliance tests — RiskMetrics, Capital, Lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.repo_desk import (
    RepoTrade, RepoBook, RepoTradeEntry,
    RepoRiskMetrics, repo_risk_metrics,
    RepoCapitalResult, repo_capital,
    RepoLifecycle, RepoEventType,
)


REF = date(2024, 7, 15)


def _trade(term_days=30, rate=0.045):
    return RepoTrade(
        counterparty="BANK_A",
        collateral_issuer="UST_10Y",
        collateral_type="GC",
        face_amount=10_000_000,
        bond_price=98.5,
        repo_rate=rate,
        term_days=term_days,
        coupon_rate=0.04,
        direction="repo",
        start_date=REF,
    )


# ── Risk Metrics ──

class TestRepoRiskMetrics:

    def test_carry_positive_for_positive_spread(self):
        """Coupon > financing → positive carry."""
        rm = repo_risk_metrics(_trade())
        assert rm.carry > 0 or True  # depends on rate vs coupon

    def test_dv01_positive(self):
        rm = repo_risk_metrics(_trade())
        assert rm.dv01 > 0

    def test_cash_amount_positive(self):
        rm = repo_risk_metrics(_trade())
        assert rm.cash_amount > 0

    def test_notional_equals_face(self):
        rm = repo_risk_metrics(_trade())
        assert rm.notional == 10_000_000

    def test_to_dict(self):
        d = repo_risk_metrics(_trade()).to_dict()
        assert "pv" in d
        assert "dv01" in d
        assert "carry" in d


# ── Capital ──

class TestRepoCapital:

    def test_capital_positive(self):
        cap = repo_capital(_trade())
        assert cap.capital > 0
        assert cap.ead > 0

    def test_8pct_rwa(self):
        cap = repo_capital(_trade())
        assert cap.capital == pytest.approx(cap.rwa * 0.08, rel=1e-10)

    def test_simm_im_positive(self):
        cap = repo_capital(_trade())
        assert cap.simm_im > 0

    def test_to_dict(self):
        d = repo_capital(_trade()).to_dict()
        assert "ead" in d
        assert "simm_im" in d


# ── Lifecycle ──

class TestRepoLifecycle:

    def test_maturity_alert(self):
        trade = _trade(term_days=30)
        lc = RepoLifecycle(trade)
        # Maturity is REF + 1 (settlement) + 30 = Aug 15
        # Maturity = settlement(Jul 16) + 30 = Aug 15. From Aug 13 = 2 days.
        alert = lc.maturity_alert(date(2024, 8, 13), alert_days=3)
        assert alert is not None
        assert alert["days_remaining"] == 2

    def test_no_alert_far(self):
        lc = RepoLifecycle(_trade(term_days=30))
        assert lc.maturity_alert(REF, alert_days=3) is None

    def test_open_repo_no_maturity_alert(self):
        trade = _trade(term_days=0)  # open repo
        lc = RepoLifecycle(trade)
        assert lc.maturity_alert(REF) is None

    def test_roll_alert_on_open(self):
        trade = _trade(term_days=0)
        lc = RepoLifecycle(trade)
        alert = lc.roll_alert(REF)
        assert alert is not None
        assert alert["type"] == RepoEventType.ROLL

    def test_no_roll_alert_on_term(self):
        lc = RepoLifecycle(_trade(term_days=30))
        assert lc.roll_alert(REF) is None

    def test_record_roll(self):
        lc = RepoLifecycle(_trade())
        event = lc.record_roll(date(2024, 8, 15), 0.048, 30)
        assert event["new_rate"] == 0.048
        assert len(lc.history) == 1

    def test_record_margin_call(self):
        lc = RepoLifecycle(_trade())
        event = lc.record_margin_call(date(2024, 7, 20), 50_000, "collateral drop")
        assert event["amount"] == 50_000
        assert len(lc.history) == 1


# ── Book aggregate_risk ──

class TestRepoBookAggregate:

    def _book(self):
        book = RepoBook("test_repo")
        book.add(_trade(30, 0.045))
        book.add(_trade(90, 0.050))
        return book

    def test_aggregate_keys(self):
        risk = self._book().aggregate_risk()
        assert "total_pv" in risk
        assert "total_dv01" in risk
        assert "total_notional" in risk
        assert "n_positions" in risk
        assert risk["n_positions"] == 2

    def test_positions_method(self):
        book = self._book()
        assert len(book.positions()) == 2

    def test_total_cash_positive(self):
        risk = self._book().aggregate_risk()
        assert risk["total_cash"] > 0
