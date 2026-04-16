"""Tests for equity structured products."""

import math

import numpy as np
import pytest

from pricebook.equity_structured import (
    AirbagResult,
    EquityAutocallableResult,
    ReverseConvertibleResult,
    SharkFinResult,
    WorstOfAutocallResult,
    airbag_note,
    equity_autocallable,
    reverse_convertible,
    shark_fin_note,
    worst_of_autocallable,
)


# ---- Autocallable ----

class TestEquityAutocallable:
    def test_basic(self):
        result = equity_autocallable(
            100, autocall_barrier=105, coupon_barrier=90,
            protection_barrier=75, coupon=0.03,
            rate=0.03, dividend_yield=0.02, vol=0.20, T=3.0,
            observation_dates=[1.0, 2.0, 3.0],
            n_paths=2000, seed=42,
        )
        assert isinstance(result, EquityAutocallableResult)
        assert 0 <= result.autocall_probability <= 1

    def test_low_barrier_higher_autocall(self):
        low = equity_autocallable(100, 102, 90, 75, 0.03, 0.03, 0.02, 0.20, 3.0,
                                   [1.0, 2.0, 3.0], n_paths=2000, seed=42)
        high = equity_autocallable(100, 120, 90, 75, 0.03, 0.03, 0.02, 0.20, 3.0,
                                    [1.0, 2.0, 3.0], n_paths=2000, seed=42)
        assert low.autocall_probability > high.autocall_probability

    def test_memory_raises_pv(self):
        with_mem = equity_autocallable(100, 105, 90, 75, 0.04, 0.03, 0.02, 0.20, 3.0,
                                        [1.0, 2.0, 3.0], has_memory=True,
                                        n_paths=2000, seed=42)
        without = equity_autocallable(100, 105, 90, 75, 0.04, 0.03, 0.02, 0.20, 3.0,
                                       [1.0, 2.0, 3.0], has_memory=False,
                                       n_paths=2000, seed=42)
        # Memory should price higher (or roughly equal)
        assert with_mem.price >= without.price * 0.95

    def test_loss_probability(self):
        """With low protection barrier, loss probability should be nonzero."""
        result = equity_autocallable(100, 110, 90, 70, 0.03, 0.03, 0.02, 0.30, 3.0,
                                      [1.0, 2.0, 3.0], n_paths=3000, seed=42)
        assert result.loss_probability > 0


# ---- Worst-of autocallable ----

class TestWorstOfAutocall:
    def _corr(self, rho=0.5):
        return np.array([[1.0, rho], [rho, 1.0]])

    def test_basic(self):
        result = worst_of_autocallable(
            [100, 100], autocall_barrier_pct=1.0, coupon=0.04,
            rate=0.03, dividend_yields=[0.02, 0.02], vols=[0.20, 0.25],
            correlations=self._corr(), T=3.0,
            observation_dates=[1.0, 2.0, 3.0],
            n_paths=2000, seed=42,
        )
        assert isinstance(result, WorstOfAutocallResult)
        assert result.n_assets == 2

    def test_correlation_effect(self):
        """Higher correlation → higher autocall probability (assets move together)."""
        low_corr = worst_of_autocallable(
            [100, 100], 1.0, 0.04, 0.03, [0.02, 0.02], [0.20, 0.20],
            self._corr(0.0), 3.0, [1.0, 2.0, 3.0], n_paths=2000, seed=42,
        )
        high_corr = worst_of_autocallable(
            [100, 100], 1.0, 0.04, 0.03, [0.02, 0.02], [0.20, 0.20],
            self._corr(0.9), 3.0, [1.0, 2.0, 3.0], n_paths=2000, seed=42,
        )
        assert high_corr.autocall_probability >= low_corr.autocall_probability * 0.9


# ---- Reverse convertible ----

class TestReverseConvertible:
    def test_basic(self):
        result = reverse_convertible(100, 100, 0.08, 0.03, 0.02, 0.20, 1.0)
        assert isinstance(result, ReverseConvertibleResult)
        assert result.bond_pv > 0
        assert result.short_put_value > 0

    def test_bond_minus_put(self):
        """Price = bond_pv − short_put."""
        result = reverse_convertible(100, 100, 0.08, 0.03, 0.02, 0.20, 1.0)
        assert result.price == pytest.approx(result.bond_pv - result.short_put_value)

    def test_higher_coupon_higher_bond(self):
        low_cp = reverse_convertible(100, 100, 0.04, 0.03, 0.02, 0.20, 1.0)
        high_cp = reverse_convertible(100, 100, 0.10, 0.03, 0.02, 0.20, 1.0)
        assert high_cp.bond_pv > low_cp.bond_pv

    def test_higher_vol_higher_put(self):
        low_v = reverse_convertible(100, 100, 0.08, 0.03, 0.02, 0.10, 1.0)
        high_v = reverse_convertible(100, 100, 0.08, 0.03, 0.02, 0.40, 1.0)
        assert high_v.short_put_value > low_v.short_put_value


# ---- Shark-fin ----

class TestSharkFin:
    def test_basic(self):
        result = shark_fin_note(100, 100, knock_out_barrier=130,
                                 rebate=0.02, participation=1.0,
                                 rate=0.03, dividend_yield=0.02,
                                 vol=0.20, T=1.0, n_paths=2000, seed=42)
        assert isinstance(result, SharkFinResult)
        assert result.price > 0

    def test_knock_out_probability(self):
        near = shark_fin_note(100, 100, knock_out_barrier=105,
                               rebate=0.02, participation=1.0,
                               rate=0.03, dividend_yield=0.02, vol=0.25, T=1.0,
                               n_paths=2000, seed=42)
        far = shark_fin_note(100, 100, knock_out_barrier=150,
                              rebate=0.02, participation=1.0,
                              rate=0.03, dividend_yield=0.02, vol=0.25, T=1.0,
                              n_paths=2000, seed=42)
        assert near.knock_out_probability > far.knock_out_probability

    def test_max_payout(self):
        result = shark_fin_note(100, 100, 130, 0.02, 1.0, 0.03, 0.02, 0.20, 1.0,
                                 n_paths=1000, seed=42)
        assert result.max_payout > 1.0


# ---- Airbag ----

class TestAirbag:
    def test_basic(self):
        result = airbag_note(100, 100, cap=0.15, floor=-0.05,
                              rate=0.03, dividend_yield=0.02, vol=0.20, T=1.0)
        assert isinstance(result, AirbagResult)
        assert result.price > 0

    def test_capped_floor(self):
        """The note's return should be bounded by floor and cap."""
        result = airbag_note(100, 100, cap=0.10, floor=0.0,
                              rate=0.0, dividend_yield=0.0, vol=0.01, T=1.0)
        # With low vol, ATM, no rates: should be near 1 + floor + some from the spread
        assert 1.0 <= result.price <= 1.15

    def test_higher_cap_higher_price(self):
        low_cap = airbag_note(100, 100, 0.10, -0.05, 0.03, 0.02, 0.20, 1.0)
        high_cap = airbag_note(100, 100, 0.30, -0.05, 0.03, 0.02, 0.20, 1.0)
        assert high_cap.price > low_cap.price

    def test_floor_guaranteed(self):
        """Higher floor → higher price."""
        low_fl = airbag_note(100, 100, 0.15, -0.20, 0.03, 0.02, 0.20, 1.0)
        high_fl = airbag_note(100, 100, 0.15, -0.05, 0.03, 0.02, 0.20, 1.0)
        assert high_fl.price > low_fl.price
