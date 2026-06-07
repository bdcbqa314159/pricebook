"""Tests for pricebook.fixed_income.callable_floater."""

import pytest
from datetime import date

from pricebook.fixed_income.callable_floater import (
    callable_frn,
    puttable_frn,
)

REF = date(2024, 1, 15)

HW_A = 0.05
HW_SIGMA = 0.01
R0 = 0.04
SPREAD = 0.005       # 50bp spread over floating
MATURITY = 5.0
CALL_DATES = [2.0, 3.0, 4.0]
NOTIONAL = 100.0
N_STEPS = 100


class TestCallableFRN:
    def test_price_positive(self):
        res = callable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        assert res.price > 0

    def test_callable_le_straight(self):
        """Call option reduces value to investor: callable price <= straight FRN."""
        res = callable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        # Allow a small numerical tolerance
        assert res.price <= res.straight_frn_price + 1e-6

    def test_option_value_nonneg(self):
        res = callable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        assert res.option_value >= -1e-10

    def test_straight_frn_price_near_par(self):
        """A flat-curve straight FRN with zero spread should price near par."""
        res = callable_frn(REF, MATURITY, 0.0, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        assert abs(res.straight_frn_price - NOTIONAL) < NOTIONAL * 0.15

    def test_to_dict_keys(self):
        res = callable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        d = res.to_dict()
        for k in ("price", "straight_frn_price", "option_value"):
            assert k in d


class TestPuttableFRN:
    def test_price_positive(self):
        res = puttable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        assert res.price > 0

    def test_puttable_ge_straight(self):
        """Put option adds value to investor: puttable price >= straight FRN."""
        res = puttable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        assert res.price >= res.straight_frn_price - 1e-6

    def test_option_value_nonneg(self):
        res = puttable_frn(REF, MATURITY, SPREAD, HW_A, HW_SIGMA, R0,
                           CALL_DATES, notional=NOTIONAL, n_steps=N_STEPS)
        assert res.option_value >= -1e-10
