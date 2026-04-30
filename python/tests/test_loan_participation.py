"""Tests for LoanParticipation and PartialFundedParticipation."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.loan import TermLoan
from pricebook.survival_curve import SurvivalCurve
from pricebook.loan_participation import (
    LoanParticipation, PartialFundedParticipation, ParticipationResult,
)
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
END = REF + timedelta(days=1825)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _surv(h=0.03):
    return SurvivalCurve.flat(REF, h)


def _loan():
    return TermLoan(REF, END, spread=0.03, notional=10_000_000)


# ---- LoanParticipation ----

class TestLoanParticipation:

    def test_basic_pv(self):
        part = LoanParticipation(_loan(), participation_rate=0.10)
        r = part.pv(_disc())
        assert math.isfinite(r.pv) and r.pv > 0
        assert r.funded_amount == 1_000_000

    def test_credit_risk_lowers_pv(self):
        part = LoanParticipation(_loan(), participation_rate=0.10)
        pv_rf = part.pv(_disc()).pv
        pv_risky = part.pv(_disc(), survival_curve=_surv(0.05)).pv
        assert pv_risky < pv_rf

    def test_higher_participation_higher_pv(self):
        low = LoanParticipation(_loan(), participation_rate=0.05)
        high = LoanParticipation(_loan(), participation_rate=0.20)
        assert high.pv(_disc()).pv > low.pv(_disc()).pv

    def test_higher_recovery_higher_pv(self):
        low_r = LoanParticipation(_loan(), recovery=0.2)
        high_r = LoanParticipation(_loan(), recovery=0.8)
        sc = _surv(0.05)
        assert high_r.pv(_disc(), survival_curve=sc).pv > \
               low_r.pv(_disc(), survival_curve=sc).pv

    def test_assignment_premium(self):
        part = LoanParticipation(_loan(), trade_type="participation")
        prem = part.assignment_premium(counterparty_spread=0.001)
        assert prem > 0

    def test_assignment_no_premium(self):
        part = LoanParticipation(_loan(), trade_type="assignment")
        assert part.assignment_premium() == 0.0

    def test_invalid_participation_raises(self):
        with pytest.raises(ValueError, match="participation_rate"):
            LoanParticipation(_loan(), participation_rate=0.0)

    def test_result_dict(self):
        part = LoanParticipation(_loan())
        r = part.pv(_disc())
        d = r.to_dict()
        assert "funded_amount" in d
        assert "expected_loss" in d


class TestLoanParticipationSerialisation:

    def test_round_trip(self):
        part = LoanParticipation(_loan(), participation_rate=0.15,
                                  funding_cost=0.025, recovery=0.5,
                                  trade_type="assignment")
        d = part.to_dict()
        assert d["type"] == "loan_participation"
        part2 = from_dict(d)
        assert part2.participation_rate == 0.15
        assert part2.trade_type == "assignment"

    def test_json(self):
        part = LoanParticipation(_loan())
        s = json.dumps(part.to_dict())
        part2 = from_dict(json.loads(s))
        assert part2.funded_amount == part.funded_amount


# ---- PartialFundedParticipation ----

class TestPartialFunded:

    def test_basic(self):
        pfp = PartialFundedParticipation(_loan(), funded_rate=0.6)
        pv = pfp.total_pv(_disc())
        assert math.isfinite(pv)

    def test_full_funded_equals_participation(self):
        """100% funded ≈ LoanParticipation."""
        pfp = PartialFundedParticipation(_loan(), funded_rate=1.0, unfunded_spread=0.0)
        part = LoanParticipation(_loan(), participation_rate=1.0)
        pv_pfp = pfp.pv_funded(_disc())
        pv_part = part.pv(_disc()).pv
        assert pv_pfp == pytest.approx(pv_part, rel=0.01)

    def test_leverage(self):
        pfp = PartialFundedParticipation(_loan(), funded_rate=0.5)
        assert pfp.leverage == pytest.approx(2.0)

    def test_cash_outlay(self):
        pfp = PartialFundedParticipation(_loan(), funded_rate=0.3)
        assert pfp.cash_outlay == pytest.approx(3_000_000)

    def test_total_equals_funded_plus_unfunded(self):
        pfp = PartialFundedParticipation(_loan(), funded_rate=0.6, unfunded_spread=0.02)
        sc = _surv(0.03)
        funded = pfp.pv_funded(_disc(), survival_curve=sc)
        unfunded = pfp.pv_unfunded(_disc(), sc)
        total = pfp.total_pv(_disc(), survival_curve=sc)
        assert total == pytest.approx(funded + unfunded, abs=1.0)

    def test_higher_funded_lower_leverage(self):
        low = PartialFundedParticipation(_loan(), funded_rate=0.3)
        high = PartialFundedParticipation(_loan(), funded_rate=0.8)
        assert high.leverage < low.leverage


class TestPartialFundedSerialisation:

    def test_round_trip(self):
        pfp = PartialFundedParticipation(_loan(), funded_rate=0.7,
                                          unfunded_spread=0.015, recovery=0.5)
        d = pfp.to_dict()
        assert d["type"] == "partial_funded"
        pfp2 = from_dict(d)
        assert pfp2.funded_rate == 0.7
        assert pfp2.unfunded_spread == 0.015

    def test_json(self):
        pfp = PartialFundedParticipation(_loan())
        s = json.dumps(pfp.to_dict())
        pfp2 = from_dict(json.loads(s))
        assert pfp2.funded_rate == pfp.funded_rate
