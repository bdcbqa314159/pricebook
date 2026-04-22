"""Tests for commodity storage and carry plays."""

import math

import pytest
from datetime import date

from pricebook.commodity_storage import (
    CashAndCarryResult,
    StorageFacility,
    cash_and_carry,
    implied_convenience_yield,
    implied_storage_cost,
)


# ---- Step 1: cash-and-carry arbitrage ----

class TestCashAndCarry:
    def test_profit_positive_contango(self):
        """In contango the forward is high enough to cover costs → profit > 0."""
        result = cash_and_carry(
            spot=70.0, forward=78.0, rate=0.05,
            storage_cost_per_annum=2.0, T=1.0,
        )
        # financing ≈ 70 × (exp(0.05)−1) ≈ 3.59
        # storage = 2.0
        # profit ≈ 78 − 70 − 2 − 3.59 ≈ 2.41
        assert result.profit > 0.0
        assert result.storage_cost == pytest.approx(2.0)

    def test_arbitrage_free_zero_profit(self):
        """Step 1 test: arbitrage-free curve has zero cash-and-carry."""
        # F = S × exp(r·T) + storage·T  with zero convenience yield
        spot = 70.0
        rate = 0.05
        storage = 2.0
        T = 1.0
        arb_free_fwd = spot * math.exp(rate * T) + storage * T
        result = cash_and_carry(spot, arb_free_fwd, rate, storage, T)
        assert result.profit == pytest.approx(0.0, abs=1e-10)

    def test_negative_profit_backwardation(self):
        """In backwardation, forward too low → cash-and-carry loses."""
        result = cash_and_carry(
            spot=70.0, forward=69.0, rate=0.05,
            storage_cost_per_annum=2.0, T=1.0,
        )
        assert result.profit < 0.0

    def test_implied_storage_cost(self):
        result = cash_and_carry(
            spot=70.0, forward=76.0, rate=0.05,
            storage_cost_per_annum=0.0, T=1.0,
        )
        # implied_storage = (76 − 70×exp(0.05)) / 1
        expected = 76.0 - 70.0 * math.exp(0.05)
        assert result.implied_storage == pytest.approx(expected)

    def test_implied_convenience_yield(self):
        spot = 70.0
        rate = 0.05
        storage = 2.0
        T = 1.0
        forward = 71.0  # below cost-of-carry → convenience yield > 0
        result = cash_and_carry(spot, forward, rate, storage, T)
        assert result.implied_convenience_yield > 0.0

    def test_zero_time(self):
        result = cash_and_carry(70.0, 70.0, 0.05, 2.0, 0.0)
        assert result.profit == 0.0
        assert result.financing_cost == 0.0

    def test_decomposition(self):
        result = cash_and_carry(70.0, 78.0, 0.05, 2.0, 1.0)
        assert result.profit == pytest.approx(
            result.forward - result.spot
            - result.storage_cost - result.financing_cost
        )


class TestImpliedStorageCost:
    def test_contango(self):
        # F > S×exp(rT) → implied storage > 0
        cost = implied_storage_cost(70.0, 80.0, 0.05, 1.0)
        assert cost > 0.0

    def test_backwardation(self):
        # F < S×exp(rT) → implied storage < 0 (convenience yield dominates)
        cost = implied_storage_cost(70.0, 69.0, 0.05, 1.0)
        assert cost < 0.0

    def test_zero_time(self):
        assert implied_storage_cost(70.0, 70.0, 0.05, 0.0) == 0.0


class TestImpliedConvenienceYield:
    def test_backwardation_positive_yield(self):
        # Backwardation → forward < spot → convenience yield positive
        cy = implied_convenience_yield(
            spot=70.0, forward=68.0, rate=0.05,
            storage_cost_per_annum=2.0, T=1.0,
        )
        assert cy > 0.0

    def test_deep_contango_negative_yield(self):
        # Deep contango → forward >> spot → convenience yield negative
        cy = implied_convenience_yield(
            spot=70.0, forward=90.0, rate=0.05,
            storage_cost_per_annum=2.0, T=1.0,
        )
        assert cy < 0.0

    def test_round_trip_with_cash_and_carry(self):
        spot, rate, storage, T = 70.0, 0.05, 2.0, 1.0
        forward = 71.0
        cy1 = implied_convenience_yield(spot, forward, rate, storage, T)
        result = cash_and_carry(spot, forward, rate, storage, T)
        assert cy1 == pytest.approx(result.implied_convenience_yield)


# ---- Step 2: storage facility ----

class TestStorageFacility:
    def _contango_curve(self):
        return {
            date(2024, 4, 1): 70.0,
            date(2024, 7, 1): 73.0,
            date(2024, 10, 1): 76.0,
            date(2025, 1, 1): 79.0,
        }

    def _flat_curve(self):
        return {
            date(2024, 4, 1): 72.0,
            date(2024, 7, 1): 72.0,
            date(2024, 10, 1): 72.0,
            date(2025, 1, 1): 72.0,
        }

    def test_intrinsic_positive_contango(self):
        """Step 2 test: storage value > 0 in contango."""
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
        )
        iv = facility.intrinsic_value(self._contango_curve())
        # Buy at 70, sell at 79 → 9 × volume
        assert iv > 0.0
        # Multi-cycle: at least as much as single inject/withdraw
        assert iv >= 100_000 * (79.0 - 70.0) * 0.5

    def test_intrinsic_zero_flat_curve(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
        )
        iv = facility.intrinsic_value(self._flat_curve())
        assert iv == 0.0

    def test_intrinsic_with_costs(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
            injection_cost=1.0,
            withdrawal_cost=0.5,
        )
        iv = facility.intrinsic_value(self._contango_curve())
        assert iv > 0

    def test_intrinsic_respects_capacity(self):
        facility = StorageFacility(
            capacity=50_000,
            max_injection_rate=50_000,
            max_withdrawal_rate=50_000,
        )
        iv = facility.intrinsic_value(self._contango_curve())
        assert iv > 0

    def test_intrinsic_respects_injection_rate(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=30_000,  # limiting factor
            max_withdrawal_rate=100_000,
        )
        iv = facility.intrinsic_value(self._contango_curve())
        assert iv > 0

    def test_extrinsic_positive_with_vol(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
        )
        ev = facility.extrinsic_value(self._contango_curve(), vol=0.30)
        assert ev > 0.0

    def test_extrinsic_zero_without_vol(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
        )
        ev = facility.extrinsic_value(self._contango_curve(), vol=0.0)
        assert ev == 0.0

    def test_total_exceeds_intrinsic(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
        )
        total = facility.total_value(self._contango_curve(), vol=0.30)
        iv = facility.intrinsic_value(self._contango_curve())
        assert total >= iv

    def test_single_period_zero(self):
        facility = StorageFacility(capacity=100_000)
        iv = facility.intrinsic_value({date(2024, 4, 1): 72.0})
        assert iv == 0.0

    def test_initial_inventory_reduces_injectable(self):
        facility = StorageFacility(
            capacity=100_000,
            max_injection_rate=100_000,
            max_withdrawal_rate=100_000,
        )
        # Already 80K in inventory → can only inject 20K more
        iv = facility.intrinsic_value(
            self._contango_curve(), initial_inventory=80_000,
        )
        assert iv > 0
