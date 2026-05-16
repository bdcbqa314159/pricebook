"""Structured credit desk tests — unified book, risk, carry, stress, capital."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.guaranteed_note import GuaranteedNote
from pricebook.credit.spv import SPV, SPVTranche
from pricebook.credit.fund_participation import FundParticipation
from pricebook.credit.illiquid_pricing import PrivatePlacementPricer
from pricebook.desks.structured_credit_desk import (
    sc_risk_metrics, SCRiskMetrics, SCBookEntry,
    StructuredCreditBook,
    sc_carry_decomposition, SCCarryDecomposition,
    sc_daily_pnl, SCDailyPnL,
    sc_dashboard, SCDashboard,
    sc_stress_suite, SCStressResult,
    sc_capital, SCCapitalResult,
    sc_hedge_recommendations,
    SCLifecycle,
)


REF = date(2024, 7, 15)
END = date(2029, 7, 15)


def _disc():
    return DiscountCurve.flat(REF, 0.04)


def _issuer_surv():
    return SurvivalCurve.flat(REF, 0.03)


def _guarantor_surv():
    return SurvivalCurve.flat(REF, 0.005)


def _gn_entry():
    gn = GuaranteedNote(REF, END, 0.05, 100.0, correlation=0.30)
    return SCBookEntry("GN1", gn, issuer="CORP_A", sector="IG",
                       issuer_surv=_issuer_surv(), guarantor_surv=_guarantor_surv())


def _pp_entry():
    pp = PrivatePlacementPricer(0.06, 5.0, 100.0,
                                 credit_spread_bp=200, illiquidity_premium_bp=75)
    return SCBookEntry("PP1", pp, issuer="PRIV_B", sector="HY")


def _spv_entry():
    tranches = [
        SPVTranche("AAA", 70e6, 0.012, 1),
        SPVTranche("Equity", 30e6, 0.0, 2),
    ]
    spv = SPV(100e6, 0.06, tranches, n_periods=10, cdr=0.02)
    return SCBookEntry("SPV1", spv, issuer="SPV_C", sector="ABS",
                       tranche_name="AAA")


def _fund_entry():
    fund = FundParticipation(50e6, gross_return=0.10)
    return SCBookEntry("FUND1", fund, issuer="FUND_D", sector="PE")


# ── Risk Metrics ──

class TestSCRiskMetrics:

    def test_guaranteed_note_metrics(self):
        rm = sc_risk_metrics(_gn_entry(), _disc())
        assert rm.product_type == "guaranteed_note"
        assert math.isfinite(rm.pv)
        assert rm.cs01 != 0
        assert rm.jtd < 0

    def test_private_placement_metrics(self):
        rm = sc_risk_metrics(_pp_entry(), _disc())
        assert rm.product_type == "private_placement"
        assert math.isfinite(rm.pv)
        assert rm.cs01 < 0  # wider spread → lower PV

    def test_spv_metrics(self):
        rm = sc_risk_metrics(_spv_entry(), _disc())
        assert rm.product_type == "spv_tranche"
        assert rm.pv > 0

    def test_fund_metrics(self):
        rm = sc_risk_metrics(_fund_entry(), _disc())
        assert rm.product_type == "fund"
        assert rm.notional == 50e6

    def test_to_dict(self):
        rm = sc_risk_metrics(_gn_entry(), _disc())
        d = rm.to_dict()
        assert "cs01" in d
        assert "product_type" in d


# ── Book ──

class TestSCBook:

    def _book(self):
        book = StructuredCreditBook("test_sc")
        book.add(_gn_entry())
        book.add(_pp_entry())
        book.add(_fund_entry())
        return book

    def test_len(self):
        assert len(self._book()) == 3

    def test_by_type(self):
        by = self._book().by_type()
        assert "GuaranteedNote" in by
        assert "PrivatePlacementPricer" in by

    def test_by_sector(self):
        by = self._book().by_sector()
        assert "IG" in by
        assert "HY" in by

    def test_aggregate_risk_keys(self):
        risk = self._book().aggregate_risk(_disc())
        assert "total_cs01" in risk
        assert "total_jtd" in risk
        assert risk["n_positions"] == 3

    def test_total_notional(self):
        assert self._book().total_notional() > 0


# ── Carry ──

class TestSCCarry:

    def test_guaranteed_note_carry(self):
        carry = sc_carry_decomposition(_gn_entry(), _disc())
        assert carry.coupon_income > 0
        assert carry.credit_cost < 0
        assert math.isfinite(carry.net_carry)

    def test_private_placement_carry(self):
        carry = sc_carry_decomposition(_pp_entry(), _disc())
        assert carry.coupon_income > 0

    def test_fund_carry(self):
        carry = sc_carry_decomposition(_fund_entry(), _disc())
        assert carry.coupon_income > 0  # gross return
        assert carry.funding_cost < 0   # mgmt fee

    def test_to_dict(self):
        d = sc_carry_decomposition(_gn_entry(), _disc()).to_dict()
        assert "coupon" in d
        assert "net" in d


# ── Daily P&L ──

class TestSCDailyPnL:

    def test_unchanged_curves(self):
        c = _disc()
        pnl = sc_daily_pnl(_pp_entry(), c, c, REF)
        assert abs(pnl.total) < 1e-6

    def test_to_dict(self):
        c = _disc()
        d = sc_daily_pnl(_pp_entry(), c, c, REF).to_dict()
        assert "total" in d
        assert "spread" in d


# ── Dashboard ──

class TestSCDashboard:

    def test_dashboard(self):
        book = StructuredCreditBook("test")
        book.add(_gn_entry())
        book.add(_pp_entry())
        db = sc_dashboard(book, REF, _disc())
        assert db.n_positions == 2
        assert db.total_notional > 0
        assert len(db.by_type) >= 2


# ── Stress ──

class TestSCStress:

    def test_stress_suite(self):
        book = StructuredCreditBook("test")
        book.add(_pp_entry())
        stress = sc_stress_suite(book, _disc())
        assert len(stress) >= 5
        assert any(s.scenario == "spread_up_300" for s in stress)


# ── Capital ──

class TestSCCapital:

    def test_capital_positive(self):
        cap = sc_capital(_gn_entry(), _disc())
        assert cap.capital > 0
        assert cap.ead > 0

    def test_8pct_rwa(self):
        cap = sc_capital(_pp_entry(), _disc())
        assert cap.capital == pytest.approx(cap.rwa * 0.08, rel=1e-10)


# ── Hedge ──

class TestSCHedge:

    def test_no_breach(self):
        book = StructuredCreditBook("test")
        book.add(_pp_entry())
        recs = sc_hedge_recommendations(book, _disc(),
                                         cs01_limit=1e9, jtd_limit=1e12)
        # No concentration breach (single issuer = 100% but only 1 position)
        concentration = [r for r in recs if r.risk_type == "concentration"]
        # With one issuer it's 100% > 25%, so there IS a breach
        assert len(concentration) == 1


# ── Lifecycle ──

class TestSCLifecycle:

    def test_maturity_alert(self):
        lc = SCLifecycle(_gn_entry())
        alert = lc.maturity_alert(date(2029, 7, 1))
        assert alert is not None
        assert alert["days_remaining"] == 14

    def test_no_alert_far(self):
        lc = SCLifecycle(_gn_entry())
        assert lc.maturity_alert(REF) is None

    def test_record_event(self):
        lc = SCLifecycle(_gn_entry())
        event = lc.record_event("coupon", date(2025, 1, 15), amount=2.5)
        assert event["amount"] == 2.5
        assert len(lc.history) == 1
