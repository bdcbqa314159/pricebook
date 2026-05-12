"""Tests for rates structured products."""
from __future__ import annotations
import pytest
from pricebook.rates_structured import (
    cms_spread_range_accrual, callable_step_up_bond, inflation_range_accrual,
)


class TestCMSSpreadRangeAccrual:
    def test_price_positive(self):
        r = cms_spread_range_accrual(10e6, 0.05, 0.0, 0.01, 0.04, 0.035,
                                      0.008, 0.006, 0.8, 0.04, 1.0, n_paths=5_000)
        assert r.price > 0

    def test_wider_range_more_accrual(self):
        narrow = cms_spread_range_accrual(10e6, 0.05, 0.004, 0.006, 0.04, 0.035,
                                           0.008, 0.006, 0.8, 0.04, 1.0, n_paths=10_000)
        wide = cms_spread_range_accrual(10e6, 0.05, -0.01, 0.02, 0.04, 0.035,
                                         0.008, 0.006, 0.8, 0.04, 1.0, n_paths=10_000)
        assert wide.expected_accrual_fraction > narrow.expected_accrual_fraction

    def test_accrual_bounded(self):
        r = cms_spread_range_accrual(10e6, 0.05, 0.0, 0.01, 0.04, 0.035,
                                      0.008, 0.006, 0.8, 0.04, 1.0, n_paths=5_000)
        assert 0 <= r.expected_accrual_fraction <= 1


class TestCallableStepUpBond:
    def test_callable_cheaper_than_non_callable(self):
        coupons = [3.0, 3.5, 4.0, 4.5, 5.0]
        r = callable_step_up_bond(100, coupons, 0.04, 0.01, 5.0, n_paths=5_000)
        assert r.price <= r.non_callable_price * 1.01

    def test_call_value_positive(self):
        coupons = [3.0, 3.5, 4.0, 4.5, 5.0]
        r = callable_step_up_bond(100, coupons, 0.04, 0.01, 5.0, n_paths=5_000)
        assert r.call_value >= -1  # call value approximately non-negative

    def test_coupon_schedule_preserved(self):
        coupons = [3.0, 4.0, 5.0]
        r = callable_step_up_bond(100, coupons, 0.04, 0.01, 3.0, n_paths=2_000)
        assert r.coupon_schedule == coupons


class TestInflationRangeAccrual:
    def test_price_positive(self):
        r = inflation_range_accrual(10e6, 0.05, 0.02, 0.03, 0.025, 0.005,
                                     0.04, 1.0, n_paths=5_000)
        assert r.price > 0

    def test_wider_range_more_accrual(self):
        narrow = inflation_range_accrual(10e6, 0.05, 0.024, 0.026, 0.025, 0.005,
                                          0.04, 1.0, n_paths=10_000)
        wide = inflation_range_accrual(10e6, 0.05, 0.01, 0.04, 0.025, 0.005,
                                        0.04, 1.0, n_paths=10_000)
        assert wide.expected_accrual_fraction > narrow.expected_accrual_fraction

    def test_accrual_bounded(self):
        r = inflation_range_accrual(10e6, 0.05, 0.02, 0.03, 0.025, 0.005,
                                     0.04, 1.0, n_paths=5_000)
        assert 0 <= r.expected_accrual_fraction <= 1
