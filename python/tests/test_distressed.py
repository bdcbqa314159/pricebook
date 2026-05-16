"""Tests for distressed debt: DIP, fulcrum, exchange, recovery waterfall, Chapter 11."""

import pytest
from datetime import date

from pricebook.credit.distressed import (
    CapitalStructureLayer, DIPResult, FulcrumResult, RecoveryDistribution,
    ExchangeResult, Chapter11Milestone, Chapter11Result,
    DIPLoan, RecoveryWaterfall, FulcrumAnalysis,
    ExchangeOffer, Chapter11Timeline,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def capital_structure():
    return [
        CapitalStructureLayer("DIP", 50_000_000, 0, secured=True),
        CapitalStructureLayer("1L TL", 200_000_000, 1, secured=True),
        CapitalStructureLayer("2L Notes", 100_000_000, 2, secured=True),
        CapitalStructureLayer("Senior Unsecured", 150_000_000, 3),
        CapitalStructureLayer("Sub Notes", 80_000_000, 4),
        CapitalStructureLayer("Equity", 50_000_000, 5),
    ]


# ═══════════════════════════════════════════════════════════════
# DIP Financing
# ═══════════════════════════════════════════════════════════════

class TestDIPLoan:
    def test_pv_positive(self):
        dip = DIPLoan(50_000_000, spread=0.08)
        assert dip.pv() > 0

    def test_roll_up(self):
        dip = DIPLoan(50_000_000, roll_up_amount=100_000_000)
        econ = dip.dip_economics()
        assert econ.total_super_priority == 150_000_000
        assert econ.expected_recovery_pct == 1.0

    def test_all_in_cost(self):
        dip = DIPLoan(50_000_000, spread=0.08, upfront_fee=0.02, maturity_months=18)
        econ = dip.dip_economics()
        # all_in = 0.08 + 0.02 / 1.5 ≈ 0.0933
        assert econ.dip_all_in_cost > econ.dip_spread

    def test_invalid_notional(self):
        with pytest.raises(ValueError):
            DIPLoan(-1)

    def test_to_dict(self):
        d = DIPLoan(50_000_000).to_dict()
        assert "notional" in d
        assert "spread" in d


# ═══════════════════════════════════════════════════════════════
# Recovery Waterfall
# ═══════════════════════════════════════════════════════════════

class TestRecoveryWaterfall:
    def test_full_recovery(self, capital_structure):
        wf = RecoveryWaterfall(capital_structure)
        total_claims = sum(l.notional for l in capital_structure)
        rd = wf.distribute(total_claims * 2)  # EV > all claims
        for name, pct in rd.recoveries.items():
            assert pct == 1.0  # all classes fully recovered

    def test_partial_recovery(self, capital_structure):
        wf = RecoveryWaterfall(capital_structure)
        # EV = 300M: DIP (50M) + 1L (200M) + 50M of 2L (100M)
        rd = wf.distribute(300_000_000)
        assert rd.recoveries["DIP"] == 1.0
        assert rd.recoveries["1L TL"] == 1.0
        assert abs(rd.recoveries["2L Notes"] - 0.50) < 1e-10
        assert rd.recoveries["Senior Unsecured"] == 0.0
        assert rd.recoveries["Equity"] == 0.0

    def test_zero_ev(self, capital_structure):
        wf = RecoveryWaterfall(capital_structure)
        rd = wf.distribute(0)
        for pct in rd.recoveries.values():
            assert pct == 0.0

    def test_total_claims(self, capital_structure):
        wf = RecoveryWaterfall(capital_structure)
        rd = wf.distribute(100_000_000)
        assert rd.total_claims == sum(l.notional for l in capital_structure)

    def test_empty_structure(self):
        with pytest.raises(ValueError):
            RecoveryWaterfall([])


# ═══════════════════════════════════════════════════════════════
# Fulcrum Analysis
# ═══════════════════════════════════════════════════════════════

class TestFulcrumAnalysis:
    def test_fulcrum_in_middle(self, capital_structure):
        fa = FulcrumAnalysis(capital_structure)
        # EV = 300M: DIP + 1L fully paid, 2L is fulcrum at 50%
        result = fa.identify_fulcrum(300_000_000)
        assert result.fulcrum_class == "2L Notes"
        assert abs(result.fulcrum_recovery_pct - 0.50) < 1e-10
        assert "DIP" in result.classes_above
        assert "1L TL" in result.classes_above
        assert "Senior Unsecured" in result.classes_below

    def test_fulcrum_at_top(self, capital_structure):
        # EV = 25M: DIP is fulcrum at 50%
        fa = FulcrumAnalysis(capital_structure)
        result = fa.identify_fulcrum(25_000_000)
        assert result.fulcrum_class == "DIP"
        assert abs(result.fulcrum_recovery_pct - 0.50) < 1e-10
        assert len(result.classes_above) == 0

    def test_ev_covers_all(self, capital_structure):
        fa = FulcrumAnalysis(capital_structure)
        total = sum(l.notional for l in capital_structure)
        result = fa.identify_fulcrum(total + 100)
        assert result.fulcrum_recovery_pct == 1.0

    def test_sensitivity(self, capital_structure):
        fa = FulcrumAnalysis(capital_structure)
        sens = fa.sensitivity((0, 500_000_000), n_points=10)
        assert len(sens) == len(capital_structure)
        for name, points in sens.items():
            assert len(points) == 10
            # Recovery should be monotonically increasing in EV
            for i in range(len(points) - 1):
                assert points[i + 1][1] >= points[i][1]

    def test_to_dict(self, capital_structure):
        d = FulcrumAnalysis(capital_structure).identify_fulcrum(300e6).to_dict()
        assert "fulcrum_class" in d
        assert "classes_below" in d


# ═══════════════════════════════════════════════════════════════
# Exchange Offer
# ═══════════════════════════════════════════════════════════════

class TestExchangeOffer:
    def test_exchange_premium(self):
        ex = ExchangeOffer(1_000_000, old_price=40, new_price=55, consent_fee=2)
        result = ex.exchange_value()
        # old = 400k, new = 550k, fee = 20k, premium = 550k + 20k - 400k = 170k
        assert abs(result.exchange_premium - 170_000) < 1.0

    def test_holdout_value(self):
        ex = ExchangeOffer(1_000_000, old_price=40, new_price=55)
        hv = ex.holdout_value(35)  # post-exchange price drops to 35
        assert hv == 350_000

    def test_prisoners_dilemma(self):
        ex = ExchangeOffer(1_000_000, old_price=40, new_price=55, consent_fee=2)
        pd = ex.prisoners_dilemma()
        assert "cooperate_payoff" in pd
        assert "defect_if_exchange_succeeds" in pd
        # Cooperation should pay more than defection if exchange succeeds
        assert pd["cooperate_payoff"] > pd["defect_if_exchange_succeeds"]

    def test_to_dict(self):
        d = ExchangeOffer(1e6, 40, 55).exchange_value().to_dict()
        assert "exchange_premium" in d


# ═══════════════════════════════════════════════════════════════
# Chapter 11 Timeline
# ═══════════════════════════════════════════════════════════════

class TestChapter11:
    def test_standard_timeline(self):
        ch11 = Chapter11Timeline("standard")
        milestones = ch11.timeline()
        assert len(milestones) == 7
        assert milestones[-1].event == "Emergence"
        assert milestones[-1].cumulative_months == 16

    def test_prepack(self):
        ch11 = Chapter11Timeline("simple")
        milestones = ch11.timeline()
        assert milestones[-1].cumulative_months == 3  # much shorter

    def test_complex(self):
        ch11 = Chapter11Timeline("complex")
        milestones = ch11.timeline()
        assert milestones[-1].cumulative_months == 32

    def test_estimate_recovery(self, capital_structure):
        ch11 = Chapter11Timeline("standard")
        result = ch11.estimate_recovery(
            ev_range=(200_000_000, 400_000_000),
            capital_structure=capital_structure,
            administrative_cost_pct=0.05,
        )
        assert len(result.recovery_by_class) == len(capital_structure)
        # Low EV → lower recoveries than high EV
        for name, (low_r, high_r) in result.recovery_by_class.items():
            assert high_r >= low_r

    def test_admin_costs_reduce_recovery(self, capital_structure):
        ch11 = Chapter11Timeline()
        no_admin = ch11.estimate_recovery((300e6, 300e6), capital_structure, 0.0)
        with_admin = ch11.estimate_recovery((300e6, 300e6), capital_structure, 0.10)
        # Admin costs reduce recoveries for junior classes
        for name in ["2L Notes", "Senior Unsecured"]:
            assert with_admin.recovery_by_class[name][0] <= no_admin.recovery_by_class[name][0]

    def test_to_dict(self, capital_structure):
        ch11 = Chapter11Timeline()
        d = ch11.estimate_recovery((200e6, 400e6), capital_structure).to_dict()
        assert "milestones" in d
        assert "recovery_by_class" in d
