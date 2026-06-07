"""Tests for pricebook.structured.callable_structured."""

import pytest

from pricebook.structured.callable_structured import (
    callable_steepener,
    callable_inverse_floater,
)

N_PATHS = 3000
SEED = 42
MATURITY = 5.0
CALL_DATES = [1.0, 2.0, 3.0, 4.0]
RATE = 0.04


class TestCallableSteepener:
    def test_price_positive(self):
        res = callable_steepener(
            long_rate=0.04,
            short_rate=0.02,
            fixed_coupon=0.0,
            leverage=3.0,
            floor=0.0,
            cap=0.10,
            call_dates_years=CALL_DATES,
            maturity_years=MATURITY,
            rate=RATE,
            vol_long=0.005,
            vol_short=0.004,
            rho=-0.3,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.price > 0

    def test_call_option_value_nonneg(self):
        res = callable_steepener(
            long_rate=0.04,
            short_rate=0.02,
            fixed_coupon=0.0,
            leverage=3.0,
            floor=0.0,
            cap=0.10,
            call_dates_years=CALL_DATES,
            maturity_years=MATURITY,
            rate=RATE,
            vol_long=0.005,
            vol_short=0.004,
            rho=-0.3,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.call_option_value >= 0

    def test_call_probability_in_range(self):
        res = callable_steepener(
            long_rate=0.04,
            short_rate=0.02,
            fixed_coupon=0.0,
            leverage=3.0,
            floor=0.0,
            cap=0.10,
            call_dates_years=CALL_DATES,
            maturity_years=MATURITY,
            rate=RATE,
            vol_long=0.005,
            vol_short=0.004,
            rho=-0.3,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert 0.0 <= res.call_probability <= 1.0


class TestCallableInverseFloater:
    def test_price_positive(self):
        res = callable_inverse_floater(
            fixed_rate=0.10,
            floating_rate=0.04,
            leverage=1.0,
            floor=0.0,
            maturity_years=MATURITY,
            call_dates_years=CALL_DATES,
            rate=RATE,
            vol=0.01,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.price > 0

    def test_call_option_value_nonneg(self):
        res = callable_inverse_floater(
            fixed_rate=0.10,
            floating_rate=0.04,
            leverage=1.0,
            floor=0.0,
            maturity_years=MATURITY,
            call_dates_years=CALL_DATES,
            rate=RATE,
            vol=0.01,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.call_option_value >= 0

    def test_call_probability_in_range(self):
        res = callable_inverse_floater(
            fixed_rate=0.10,
            floating_rate=0.04,
            leverage=1.0,
            floor=0.0,
            maturity_years=MATURITY,
            call_dates_years=CALL_DATES,
            rate=RATE,
            vol=0.01,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert 0.0 <= res.call_probability <= 1.0
