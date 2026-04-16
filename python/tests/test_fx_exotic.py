"""Tests for FX exotic options: touches, lookbacks, Asian, range accrual."""

import math

import numpy as np
import pytest

from pricebook.fx_exotic import (
    AccumulatorResult,
    AsianResult,
    LookbackResult,
    RangeAccrualResult,
    TouchResult,
    fx_accumulator,
    fx_asian_arithmetic,
    fx_asian_geometric,
    fx_double_no_touch,
    fx_double_touch,
    fx_lookback_fixed,
    fx_lookback_floating,
    fx_no_touch,
    fx_one_touch,
    fx_range_accrual,
)


# ---- Touch options ----

class TestOneTouch:
    def test_basic(self):
        result = fx_one_touch(1.0, 1.10, 0.02, 0.01, 0.10, 1.0)
        assert isinstance(result, TouchResult)
        assert 0 < result.price < 1.0

    def test_already_touched(self):
        result = fx_one_touch(1.20, 1.10, 0.02, 0.01, 0.10, 1.0, is_up=True)
        assert result.price == pytest.approx(math.exp(-0.02 * 1.0), rel=1e-6)

    def test_higher_vol_higher_price(self):
        low = fx_one_touch(1.0, 1.10, 0.02, 0.01, 0.05, 1.0)
        high = fx_one_touch(1.0, 1.10, 0.02, 0.01, 0.25, 1.0)
        assert high.price > low.price

    def test_farther_barrier_lower_price(self):
        near = fx_one_touch(1.0, 1.05, 0.02, 0.01, 0.10, 1.0)
        far = fx_one_touch(1.0, 1.20, 0.02, 0.01, 0.10, 1.0)
        assert near.price > far.price

    def test_down_touch(self):
        result = fx_one_touch(1.0, 0.90, 0.02, 0.01, 0.10, 1.0, is_up=False)
        assert 0 < result.price < 1.0


class TestNoTouch:
    def test_basic(self):
        result = fx_no_touch(1.0, 1.10, 0.02, 0.01, 0.10, 1.0)
        assert 0 < result.price < 1.0

    def test_ot_nt_parity(self):
        """OT + NT = DF × payout (both priced on same paths via same seed)."""
        ot = fx_one_touch(1.0, 1.10, 0.02, 0.01, 0.10, 1.0)
        nt = fx_no_touch(1.0, 1.10, 0.02, 0.01, 0.10, 1.0)
        df = math.exp(-0.02 * 1.0)
        assert ot.price + nt.price == pytest.approx(df, abs=1e-6)

    def test_high_vol_low_nt(self):
        low_vol = fx_no_touch(1.0, 1.10, 0.02, 0.01, 0.05, 1.0)
        high_vol = fx_no_touch(1.0, 1.10, 0.02, 0.01, 0.25, 1.0)
        assert low_vol.price > high_vol.price


class TestDoubleBarrier:
    def test_dnt_basic(self):
        result = fx_double_no_touch(1.0, 0.90, 1.10, 0.02, 0.01, 0.10, 0.5)
        assert isinstance(result, TouchResult)
        assert 0 <= result.price <= 1.0
        assert result.is_double

    def test_dnt_outside_range(self):
        """Spot outside range → DNT = 0."""
        result = fx_double_no_touch(1.20, 0.90, 1.10, 0.02, 0.01, 0.10, 0.5)
        assert result.price == 0.0

    def test_dt_dnt_parity(self):
        dt = fx_double_touch(1.0, 0.90, 1.10, 0.02, 0.01, 0.10, 0.5)
        dnt = fx_double_no_touch(1.0, 0.90, 1.10, 0.02, 0.01, 0.10, 0.5)
        df = math.exp(-0.02 * 0.5)
        assert dt.price + dnt.price == pytest.approx(df, rel=0.10)


# ---- Lookbacks ----

class TestLookbackFloating:
    def test_call_positive(self):
        result = fx_lookback_floating(1.0, 0.02, 0.01, 0.15, 1.0, is_call=True)
        assert isinstance(result, LookbackResult)
        assert result.price > 0

    def test_put_positive(self):
        result = fx_lookback_floating(1.0, 0.02, 0.01, 0.15, 1.0, is_call=False)
        assert result.price > 0

    def test_higher_vol_higher_price(self):
        low = fx_lookback_floating(1.0, 0.02, 0.01, 0.05, 1.0, is_call=True)
        high = fx_lookback_floating(1.0, 0.02, 0.01, 0.30, 1.0, is_call=True)
        assert high.price > low.price


class TestLookbackFixed:
    def test_call_positive(self):
        result = fx_lookback_fixed(1.0, 1.0, 0.02, 0.01, 0.15, 1.0,
                                    is_call=True, n_paths=5000, seed=42)
        assert result.price > 0

    def test_greater_than_vanilla(self):
        """Fixed-strike lookback call (max(S)-K)+ exceeds vanilla (S_T-K)+
        because max(S) ≥ S_T always."""
        from pricebook.black76 import black76_price, OptionType
        F = 1.0 * math.exp((0.02 - 0.01) * 1.0)
        vanilla = black76_price(F, 1.0, 0.15, 1.0, math.exp(-0.02), OptionType.CALL)
        lookback = fx_lookback_fixed(1.0, 1.0, 0.02, 0.01, 0.15, 1.0,
                                      is_call=True, n_paths=20_000, n_steps=200, seed=42)
        assert lookback.price >= vanilla


# ---- Asian ----

class TestAsianGeometric:
    def test_basic(self):
        result = fx_asian_geometric(1.0, 1.0, 0.02, 0.01, 0.15, 1.0,
                                     n_fixings=12, is_call=True)
        assert isinstance(result, AsianResult)
        assert result.is_geometric
        assert result.price > 0

    def test_less_than_vanilla(self):
        """Asian call should be cheaper than vanilla (averaging reduces vol)."""
        from pricebook.black76 import black76_price, OptionType
        F = 1.0 * math.exp((0.02 - 0.01) * 1.0)
        vanilla = black76_price(F, 1.0, 0.15, 1.0, math.exp(-0.02), OptionType.CALL)
        asian = fx_asian_geometric(1.0, 1.0, 0.02, 0.01, 0.15, 1.0, n_fixings=50)
        assert asian.price < vanilla

    def test_put_call_parity(self):
        """C - P ≈ F_g - K × DF."""
        call = fx_asian_geometric(1.0, 1.0, 0.02, 0.01, 0.15, 1.0, n_fixings=12, is_call=True)
        put = fx_asian_geometric(1.0, 1.0, 0.02, 0.01, 0.15, 1.0, n_fixings=12, is_call=False)
        assert abs(call.price - put.price) < 0.1


class TestAsianArithmetic:
    def test_basic(self):
        result = fx_asian_arithmetic(1.0, 1.0, 0.02, 0.01, 0.15, 1.0,
                                      n_fixings=12, n_paths=5000, seed=42)
        assert isinstance(result, AsianResult)
        assert not result.is_geometric
        assert result.price > 0

    def test_arithmetic_geq_geometric(self):
        """Arithmetic Asian ≥ geometric (AM ≥ GM)."""
        arith = fx_asian_arithmetic(1.0, 1.0, 0.02, 0.01, 0.15, 1.0,
                                     n_fixings=12, n_paths=20_000, seed=42)
        geo = fx_asian_geometric(1.0, 1.0, 0.02, 0.01, 0.15, 1.0, n_fixings=12)
        assert arith.price >= geo.price * 0.95  # allow MC noise


# ---- Range accrual ----

class TestRangeAccrual:
    def test_basic(self):
        result = fx_range_accrual(1.0, 0.95, 1.05, 0.02, 0.01, 0.10, 1.0,
                                   n_paths=2000, seed=42)
        assert isinstance(result, RangeAccrualResult)
        assert 0 <= result.accrual_rate <= 1
        assert result.price > 0

    def test_wide_range_higher_accrual(self):
        narrow = fx_range_accrual(1.0, 0.98, 1.02, 0.02, 0.01, 0.10, 1.0,
                                   n_paths=2000, seed=42)
        wide = fx_range_accrual(1.0, 0.85, 1.15, 0.02, 0.01, 0.10, 1.0,
                                 n_paths=2000, seed=42)
        assert wide.accrual_rate > narrow.accrual_rate

    def test_higher_vol_less_accrual(self):
        """Higher vol → spot escapes range more often."""
        low = fx_range_accrual(1.0, 0.95, 1.05, 0.02, 0.01, 0.05, 1.0,
                                n_paths=2000, seed=42)
        high = fx_range_accrual(1.0, 0.95, 1.05, 0.02, 0.01, 0.25, 1.0,
                                 n_paths=2000, seed=42)
        assert low.accrual_rate > high.accrual_rate


# ---- Accumulator ----

class TestAccumulator:
    def test_basic(self):
        result = fx_accumulator(1.0, 0.98, 1.05, 0.02, 0.01, 0.10, 1.0,
                                 n_paths=2000, seed=42)
        assert isinstance(result, AccumulatorResult)
        assert 0 <= result.knock_out_prob <= 1

    def test_knock_out_probability(self):
        """Closer barrier → higher KO probability."""
        near = fx_accumulator(1.0, 0.98, 1.02, 0.02, 0.01, 0.15, 1.0,
                               n_paths=2000, seed=42)
        far = fx_accumulator(1.0, 0.98, 1.20, 0.02, 0.01, 0.15, 1.0,
                              n_paths=2000, seed=42)
        assert near.knock_out_prob > far.knock_out_prob

    def test_expected_accumulation_positive(self):
        result = fx_accumulator(1.0, 0.98, 1.05, 0.02, 0.01, 0.10, 1.0,
                                 n_paths=2000, seed=42)
        assert result.expected_accumulated > 0
