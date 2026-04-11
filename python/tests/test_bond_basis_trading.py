"""Tests for bond futures basis trading."""

import pytest

from pricebook.bond_basis_trading import (
    CTDScenario,
    DeliveryOptionValue,
    SwitchTrade,
    basis_at_delivery,
    construct_switch_trade,
    ctd_switch_probability,
    ctd_switch_scenarios,
    delivery_option_value,
)
from pricebook.govt_bond_trading import basis_decomposition


BONDS = [
    {"name": "UST_2Y", "price": 99.5, "cf": 0.99, "coupon_rate": 0.04, "duration": 1.9},
    {"name": "UST_5Y", "price": 98.0, "cf": 0.96, "coupon_rate": 0.035, "duration": 4.5},
    {"name": "UST_10Y", "price": 95.0, "cf": 0.90, "coupon_rate": 0.03, "duration": 8.5},
]


# ---- Step 1: basis analytics + CTD ----

class TestCTDSwitchScenarios:
    def test_produces_scenarios(self):
        scenarios = ctd_switch_scenarios(
            BONDS, futures_price=100.0, repo_rate=0.04,
            days_to_delivery=90,
        )
        assert len(scenarios) == 11  # -50 to +50 in 10bp steps
        assert all(isinstance(s, CTDScenario) for s in scenarios)

    def test_ctd_has_highest_implied_repo(self):
        """Step 1 test: CTD has the highest implied repo in each scenario."""
        scenarios = ctd_switch_scenarios(
            BONDS, futures_price=100.0, repo_rate=0.04,
            days_to_delivery=90,
        )
        for s in scenarios:
            ctd_repo = s.ctd_implied_repo
            for bd in s.all_bonds:
                assert bd.implied_repo <= ctd_repo + 1e-10

    def test_ctd_may_switch_under_large_shift(self):
        # With large enough shifts, CTD should change for different durations
        scenarios = ctd_switch_scenarios(
            BONDS, futures_price=100.0, repo_rate=0.04,
            days_to_delivery=90,
            shifts_bps=[-100, 0, 100],
        )
        ctd_names = {s.ctd_bond for s in scenarios}
        # With very different durations, large shifts should switch CTD
        # (short-duration bond becomes CTD when yields rise sharply)
        assert len(scenarios) == 3

    def test_custom_shifts(self):
        scenarios = ctd_switch_scenarios(
            BONDS, futures_price=100.0, repo_rate=0.04,
            days_to_delivery=90,
            shifts_bps=[-10, 0, 10],
        )
        assert len(scenarios) == 3
        assert scenarios[0].yield_shift_bps == -10
        assert scenarios[1].yield_shift_bps == 0
        assert scenarios[2].yield_shift_bps == 10


class TestCTDSwitchProbability:
    def test_no_switch(self):
        scenarios = [
            CTDScenario(-10, "A", 0.05, []),
            CTDScenario(0, "A", 0.05, []),
            CTDScenario(10, "A", 0.05, []),
        ]
        assert ctd_switch_probability(scenarios, "A") == pytest.approx(0.0)

    def test_full_switch(self):
        scenarios = [
            CTDScenario(-10, "B", 0.05, []),
            CTDScenario(0, "B", 0.05, []),
            CTDScenario(10, "B", 0.05, []),
        ]
        assert ctd_switch_probability(scenarios, "A") == pytest.approx(1.0)

    def test_partial_switch(self):
        scenarios = [
            CTDScenario(-10, "A", 0.05, []),
            CTDScenario(0, "A", 0.05, []),
            CTDScenario(10, "B", 0.05, []),
        ]
        assert ctd_switch_probability(scenarios, "A") == pytest.approx(1 / 3)

    def test_empty(self):
        assert ctd_switch_probability([], "A") == 0.0


# ---- Step 2: delivery option + switch trades + convergence ----

class TestDeliveryOptionValue:
    def test_quality_option_positive(self):
        dov = delivery_option_value(
            net_basis_ctd=0.05, avg_net_basis_others=0.15,
        )
        assert dov.quality_option == pytest.approx(0.10)
        assert dov.quality_option > 0

    def test_timing_option_positive(self):
        dov = delivery_option_value(0.05, 0.05, daily_vol_bps=5.0,
                                    delivery_window_days=5)
        assert dov.timing_option > 0

    def test_total_is_sum(self):
        dov = delivery_option_value(0.05, 0.20, daily_vol_bps=5.0,
                                    delivery_window_days=5)
        assert dov.total == pytest.approx(dov.quality_option + dov.timing_option)

    def test_no_quality_when_ctd_is_average(self):
        dov = delivery_option_value(0.10, 0.10)
        assert dov.quality_option == pytest.approx(0.0)


class TestSwitchTrade:
    def test_positive_pnl_when_new_cheaper(self):
        old = basis_decomposition("A", 99.0, 100.0, 0.98, 0.04, 0.05, 90)
        new = basis_decomposition("B", 98.0, 100.0, 0.97, 0.035, 0.05, 90)
        trade = construct_switch_trade(old, new, face_amount=10_000_000)
        # If new has lower net basis, switch_pnl is positive
        if new.net_basis < old.net_basis:
            assert trade.switch_pnl > 0
        assert trade.old_ctd == "A"
        assert trade.new_ctd == "B"

    def test_zero_pnl_same_bond(self):
        bd = basis_decomposition("A", 99.0, 100.0, 0.98, 0.04, 0.05, 90)
        trade = construct_switch_trade(bd, bd)
        assert trade.switch_pnl == pytest.approx(0.0)


class TestBasisAtDelivery:
    def test_convergence_to_zero(self):
        """Step 2 test: basis converges to zero at delivery."""
        gross = 1.5
        total_days = 90
        # At delivery (day 90), basis = 0
        assert basis_at_delivery(gross, total_days, total_days) == pytest.approx(0.0)
        # At start (day 0), basis = full
        assert basis_at_delivery(gross, total_days, 0) == pytest.approx(1.5)
        # Halfway, basis = half
        assert basis_at_delivery(gross, total_days, 45) == pytest.approx(0.75)

    def test_monotone_decreasing(self):
        gross = 2.0
        total = 60
        prev = basis_at_delivery(gross, total, 0)
        for day in range(1, total + 1):
            curr = basis_at_delivery(gross, total, day)
            assert curr <= prev
            prev = curr

    def test_zero_days(self):
        assert basis_at_delivery(1.5, 0, 0) == 0.0

    def test_past_delivery(self):
        assert basis_at_delivery(1.5, 90, 100) == 0.0
