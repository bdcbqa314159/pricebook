"""Risk participation instrument + desk tests."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.risk_participation import (
    RiskParticipation, SubParticipation, risk_participation_capital_relief,
)
from pricebook.desks.risk_participation_desk import (
    rp_risk_metrics, RPRiskMetrics,
    RPBook, RPBookEntry,
    rp_carry_decomposition, RPCarryDecomposition,
    rp_daily_pnl, RPDailyPnL,
    rp_dashboard, RPDashboard,
    rp_stress_suite, RPStressResult,
    rp_capital, RPCapitalResult,
    rp_hedge_recommendations,
    RPLifecycle,
)


REF = date(2024, 7, 15)
END = date(2029, 7, 15)


def _disc():
    return DiscountCurve.flat(REF, 0.04)


def _surv():
    return SurvivalCurve.flat(REF, 0.02)


def _rp(spread=0.015):
    return RiskParticipation(
        REF, END, loan_notional=100_000_000,
        participation_rate=0.20, spread=spread, recovery=0.40,
    )


# ── Instrument ──

class TestRiskParticipation:

    def test_pv_positive_when_spread_above_par(self):
        """Participant earns when spread > par spread."""
        result = _rp(0.015).price(_disc(), _surv())
        par = result.par_spread
        assert result.pv > 0  # 150bp > par ~119bp
        assert par < 0.015

    def test_pv_zero_at_par_spread(self):
        result = _rp().price(_disc(), _surv())
        par = result.par_spread
        at_par = RiskParticipation(REF, END, 100e6, 0.20, par, 0.40)
        assert at_par.price(_disc(), _surv()).pv == pytest.approx(0.0, abs=100)

    def test_notional_equals_rate_times_loan(self):
        rp = _rp()
        assert rp.notional == 100_000_000 * 0.20

    def test_cs01_negative_for_participant(self):
        """Wider spreads → more defaults → participant loses."""
        cs01 = _rp().cs01(_disc(), _surv())
        assert cs01 < 0

    def test_jtd_large_negative(self):
        """On default, participant pays (1-R) × notional."""
        jtd = _rp().jtd(_disc(), _surv())
        assert jtd < -10_000_000  # ~12M loss on 20M notional

    def test_validation(self):
        with pytest.raises(ValueError):
            RiskParticipation(REF, END, 100e6, 0.0, 0.01)  # rate = 0
        with pytest.raises(ValueError):
            RiskParticipation(REF, END, -100e6, 0.2, 0.01)  # negative notional


class TestSubParticipation:

    def test_sub_pv_positive(self):
        sub = SubParticipation(_rp(), sub_rate=0.50, sub_spread=0.012)
        result = sub.price_sub(_disc(), _surv())
        assert math.isfinite(result.pv)

    def test_intermediary_spread_pickup(self):
        sub = SubParticipation(_rp(0.015), sub_rate=0.50, sub_spread=0.012)
        pnl = sub.intermediary_pnl(_disc(), _surv())
        assert pnl["spread_pickup_bp"] == pytest.approx(30.0)

    def test_intermediary_total_pv_positive(self):
        sub = SubParticipation(_rp(0.015), sub_rate=0.50, sub_spread=0.012)
        pnl = sub.intermediary_pnl(_disc(), _surv())
        assert pnl["total_pv"] > 0


class TestCapitalRelief:

    def test_relief_positive(self):
        relief = risk_participation_capital_relief(100e6, 0.20)
        assert relief > 0

    def test_relief_proportional(self):
        r1 = risk_participation_capital_relief(100e6, 0.10)
        r2 = risk_participation_capital_relief(100e6, 0.20)
        assert r2 == pytest.approx(2 * r1, rel=0.01)


# ── Desk ──

class TestRPRiskMetrics:

    def test_all_fields_finite(self):
        rm = rp_risk_metrics(_rp(), _disc(), _surv())
        assert math.isfinite(rm.pv)
        assert math.isfinite(rm.cs01)
        assert math.isfinite(rm.dv01)
        assert math.isfinite(rm.jtd)

    def test_to_dict(self):
        rm = rp_risk_metrics(_rp(), _disc(), _surv())
        d = rm.to_dict()
        assert "cs01" in d
        assert "jtd" in d


class TestRPBook:

    def _book(self):
        book = RPBook("test")
        book.add(RPBookEntry("T1", _rp(0.015), _surv(), borrower="CORP_A", sector="IG"))
        book.add(RPBookEntry("T2", _rp(0.020), _surv(), borrower="CORP_B", sector="HY"))
        return book

    def test_len(self):
        assert len(self._book()) == 2

    def test_aggregate_risk_keys(self):
        risk = self._book().aggregate_risk(_disc())
        assert "total_cs01" in risk
        assert "total_jtd" in risk
        assert risk["n_positions"] == 2

    def test_by_sector(self):
        by = self._book().by_sector()
        assert "IG" in by
        assert "HY" in by


class TestRPCarry:

    def test_fee_income_positive(self):
        carry = rp_carry_decomposition(_rp(), _disc(), _surv())
        assert carry.fee_income > 0

    def test_default_drag_negative(self):
        carry = rp_carry_decomposition(_rp(), _disc(), _surv())
        assert carry.default_drag < 0

    def test_net_carry_finite(self):
        carry = rp_carry_decomposition(_rp(), _disc(), _surv())
        assert math.isfinite(carry.net_carry)


class TestRPDailyPnL:

    def test_unchanged_curves(self):
        d, s = _disc(), _surv()
        pnl = rp_daily_pnl(_rp(), d, s, d, s, REF)
        assert abs(pnl.total) < 1e-6

    def test_spread_shift(self):
        d, s = _disc(), _surv()
        pnl = rp_daily_pnl(_rp(), d, s, d, s.bumped(0.001), REF)
        assert pnl.spread_pnl != 0


class TestRPDashboard:

    def test_dashboard(self):
        book = RPBook("test")
        book.add(RPBookEntry("T1", _rp(), _surv(), sector="IG"))
        db = rp_dashboard(book, REF, _disc())
        assert db.n_positions == 1
        assert db.total_notional > 0


class TestRPStress:

    def test_stress_suite(self):
        book = RPBook("test")
        book.add(RPBookEntry("T1", _rp(), _surv()))
        stress = rp_stress_suite(book, _disc())
        assert len(stress) >= 5


class TestRPCapital:

    def test_capital_positive(self):
        cap = rp_capital(_rp(), _disc(), _surv())
        assert cap.capital > 0

    def test_8pct_rwa(self):
        cap = rp_capital(_rp(), _disc(), _surv())
        assert cap.capital == pytest.approx(cap.rwa * 0.08, rel=1e-10)


class TestRPLifecycle:

    def test_maturity_alert(self):
        lc = RPLifecycle(_rp())
        alert = lc.maturity_alert(date(2029, 7, 1))
        assert alert is not None
        assert alert["days_remaining"] == 14

    def test_record_default(self):
        lc = RPLifecycle(_rp())
        event = lc.record_default(date(2026, 3, 1), 12_000_000, 4_800_000)
        assert event["net_payout"] == 7_200_000
        assert len(lc.history) == 1
