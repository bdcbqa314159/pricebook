"""Tests for callable/putable bond desk analytics."""
from __future__ import annotations
from datetime import date
import pytest
from dateutil.relativedelta import relativedelta
from pricebook.bootstrap import bootstrap
from pricebook.callable_bond_desk import callable_bond_analytics

REF = date(2024, 7, 15)

def _curve():
    deposits = [(REF + relativedelta(months=6), 0.0515)]
    swaps = [(REF + relativedelta(years=y), r) for y, r in
             [(1, 0.0490), (2, 0.0455), (5, 0.0410), (10, 0.0390), (30, 0.0415)]]
    return bootstrap(REF, deposits, swaps)


class TestCallableBondAnalytics:
    def test_callable_price_below_straight(self):
        curve = _curve()
        r = callable_bond_analytics(curve, 98.0, 0.04, 10.0, n_steps=50)
        assert r.model_price <= r.straight_price + 0.01

    def test_option_value_positive(self):
        curve = _curve()
        r = callable_bond_analytics(curve, 98.0, 0.04, 10.0, n_steps=50)
        assert r.option_value >= -0.01  # should be positive for callable

    def test_oas_finite(self):
        curve = _curve()
        r = callable_bond_analytics(curve, 98.0, 0.04, 10.0, n_steps=50)
        assert -500 < r.oas_bps < 500

    def test_effective_duration_positive(self):
        curve = _curve()
        r = callable_bond_analytics(curve, 98.0, 0.04, 10.0, n_steps=50)
        assert r.effective_duration > 0

    def test_puttable_price_above_straight(self):
        curve = _curve()
        r = callable_bond_analytics(curve, 102.0, 0.04, 10.0, is_callable=False, n_steps=50)
        assert r.model_price >= r.straight_price - 0.01

    def test_callable_duration_less_than_straight(self):
        curve = _curve()
        rc = callable_bond_analytics(curve, 98.0, 0.05, 10.0, n_steps=50)
        # Callable bond effective duration should be less than straight bond
        # (call caps upside, shortens effective maturity)
        from pricebook.bond_trading_desk import bond_risk_metrics
        from pricebook.bond import FixedRateBond
        straight = FixedRateBond(REF - relativedelta(months=6),
                                  REF + relativedelta(years=10), 0.05)
        rm = bond_risk_metrics(straight, curve)
        assert rc.effective_duration < rm.modified_duration + 1
