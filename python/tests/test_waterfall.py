"""Tests for waterfall engine, triggers, and structured notes."""

import pytest

from pricebook.waterfall import (
    Tranche,
    Trigger,
    WaterfallEngine,
    AutocallObservation,
    autocall_payoff,
)


class TestWaterfall:
    def _three_tranches(self):
        return [
            Tranche("senior", target_notional=60, coupon_rate=0.03, seniority=0),
            Tranche("mezz", target_notional=30, coupon_rate=0.06, seniority=1),
            Tranche("equity", target_notional=10, coupon_rate=0.12, seniority=2),
        ]

    def test_allocates_interest_senior_first(self):
        engine = WaterfallEngine(self._three_tranches())
        result = engine.allocate(total_cashflow=5.0)
        alloc = result["allocations"]
        # Senior interest = 60 * 0.03 = 1.8
        assert alloc["senior"]["interest"] == pytest.approx(1.8)
        # Mezz interest = 30 * 0.06 = 1.8
        assert alloc["mezz"]["interest"] == pytest.approx(1.8)
        # Equity interest = 10 * 0.12 = 1.2
        assert alloc["equity"]["interest"] == pytest.approx(1.2)

    def test_insufficient_cashflow(self):
        engine = WaterfallEngine(self._three_tranches())
        result = engine.allocate(total_cashflow=2.0)
        alloc = result["allocations"]
        # Senior gets full 1.8, mezz gets remaining 0.2
        assert alloc["senior"]["interest"] == pytest.approx(1.8)
        assert alloc["mezz"]["interest"] == pytest.approx(0.2)

    def test_principal_after_interest(self):
        engine = WaterfallEngine(self._three_tranches())
        # 10 = enough for all interest (4.8) + some principal
        result = engine.allocate(total_cashflow=10.0)
        alloc = result["allocations"]
        total_principal = sum(a["principal"] for a in alloc.values())
        assert total_principal > 0

    def test_principal_senior_first(self):
        engine = WaterfallEngine(self._three_tranches())
        result = engine.allocate(total_cashflow=70.0)
        alloc = result["allocations"]
        # Senior should get principal before mezz
        assert alloc["senior"]["principal"] >= alloc["mezz"]["principal"]

    def test_remaining_after_full_payout(self):
        engine = WaterfallEngine(self._three_tranches())
        result = engine.allocate(total_cashflow=200.0)
        # More than enough for everything
        assert result["remaining"] > 0


class TestTriggers:
    def test_oc_trigger_fires(self):
        tranches = [
            Tranche("senior", 100, 0.03, 0),
            Tranche("equity", 10, 0.12, 1),
        ]
        trigger = Trigger("oc_test", "oc_ratio", threshold=1.2)
        engine = WaterfallEngine(tranches, [trigger])

        # Collateral = 100, outstanding = 110 → OC = 0.91 < 1.2 → breach
        result = engine.allocate(5.0, collateral_balance=100)
        assert result["diverted"]
        assert result["trigger_status"]["oc_test"]["breached"]

    def test_oc_trigger_no_fire(self):
        tranches = [
            Tranche("senior", 100, 0.03, 0),
            Tranche("equity", 10, 0.12, 1),
        ]
        trigger = Trigger("oc_test", "oc_ratio", threshold=0.5)
        engine = WaterfallEngine(tranches, [trigger])

        result = engine.allocate(5.0, collateral_balance=100)
        assert not result["diverted"]

    def test_ic_trigger(self):
        tranches = [
            Tranche("senior", 100, 0.05, 0),
            Tranche("equity", 10, 0.10, 1),
        ]
        trigger = Trigger("ic_test", "ic_ratio", threshold=1.5)
        engine = WaterfallEngine(tranches, [trigger])

        # Interest = 100*0.05 + 10*0.10 = 6.0. Cash = 5 → IC = 0.83 < 1.5
        result = engine.allocate(5.0)
        assert result["trigger_status"]["ic_test"]["breached"]

    def test_diversion_redirects_to_senior(self):
        tranches = [
            Tranche("senior", 100, 0.03, 0),
            Tranche("equity", 10, 0.12, 1),
        ]
        trigger = Trigger("oc_test", "oc_ratio", threshold=2.0)  # will breach
        engine = WaterfallEngine(tranches, [trigger])

        result = engine.allocate(50.0, collateral_balance=100)
        # Senior should get principal, equity should get nothing
        assert result["allocations"]["senior"]["principal"] > 0


class TestReset:
    def test_reset(self):
        t = Tranche("senior", 100, 0.03, 0)
        t.principal_paid = 50
        engine = WaterfallEngine([t])
        engine.reset()
        assert t.principal_paid == 0
        assert t.outstanding == 100


class TestAutocall:
    def test_autocall_triggered(self):
        path = [100, 105, 110, 115, 120]
        obs = [
            AutocallObservation(1.0, 110),
            AutocallObservation(2.0, 110),
        ]
        result = autocall_payoff(path, obs, notional=100, coupon_rate=0.05)
        assert result["called"]
        assert result["payout"] > 100  # notional + coupon

    def test_autocall_not_triggered(self):
        path = [100, 99, 98, 97, 96]
        obs = [
            AutocallObservation(1.0, 110),
            AutocallObservation(2.0, 110),
        ]
        result = autocall_payoff(path, obs, notional=100)
        assert not result["called"]
        assert result["payout"] == 100  # just notional

    def test_early_call(self):
        path = [100, 115, 120, 125, 130]
        obs = [
            AutocallObservation(0.5, 110),
            AutocallObservation(1.0, 110),
        ]
        result = autocall_payoff(path, obs, notional=100, coupon_rate=0.08)
        assert result["called"]
        # Early call: coupon for 0.5 years
        assert result["call_date"] == pytest.approx(0.5)
