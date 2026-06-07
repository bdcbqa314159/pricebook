"""Tests for pricebook.options.bermudan_capfloor."""

import pytest
from datetime import date

from pricebook.options.bermudan_capfloor import (
    bermudan_cap,
    bermudan_floor,
    bermudan_collar,
)

REF = date(2024, 1, 15)

HW_A = 0.05
HW_SIGMA = 0.01
R0 = 0.04
STRIKE = 0.05
MATURITY = 5.0
EXERCISE_DATES = [1.0, 2.0, 3.0, 4.0]
N_STEPS = 100


class TestBermudanCap:
    def test_price_positive(self):
        res = bermudan_cap(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                           EXERCISE_DATES, n_steps=N_STEPS)
        assert res.price > 0

    def test_price_ge_european(self):
        """Bermudan cap >= European cap (early exercise can only add value)."""
        res = bermudan_cap(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                           EXERCISE_DATES, n_steps=N_STEPS)
        assert res.price >= res.european_price - 1e-6

    def test_early_exercise_premium_nonneg(self):
        res = bermudan_cap(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                           EXERCISE_DATES, n_steps=N_STEPS)
        assert res.early_exercise_premium >= -1e-10

    def test_n_exercise_dates(self):
        res = bermudan_cap(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                           EXERCISE_DATES, n_steps=N_STEPS)
        assert res.n_exercise_dates == len(EXERCISE_DATES)

    def test_exercise_probabilities_in_range(self):
        res = bermudan_cap(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                           EXERCISE_DATES, n_steps=N_STEPS)
        for prob in res.exercise_probabilities:
            assert 0.0 <= prob <= 1.0 + 1e-9


class TestBermudanFloor:
    def test_price_positive(self):
        res = bermudan_floor(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                             EXERCISE_DATES, n_steps=N_STEPS)
        assert res.price > 0

    def test_price_ge_european(self):
        res = bermudan_floor(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                             EXERCISE_DATES, n_steps=N_STEPS)
        assert res.price >= res.european_price - 1e-6

    def test_early_exercise_premium_nonneg(self):
        res = bermudan_floor(REF, MATURITY, STRIKE, HW_A, HW_SIGMA, R0,
                             EXERCISE_DATES, n_steps=N_STEPS)
        assert res.early_exercise_premium >= -1e-10


class TestBermudanCollar:
    def test_returns_dict_with_expected_keys(self):
        result = bermudan_collar(
            REF, MATURITY,
            cap_strike=0.06, floor_strike=0.03,
            hw_a=HW_A, hw_sigma=HW_SIGMA, r0=R0,
            exercise_dates_years=EXERCISE_DATES,
            n_steps=N_STEPS,
        )
        assert "collar_price" in result
        assert "cap_result" in result
        assert "floor_result" in result

    def test_collar_price_is_cap_minus_floor(self):
        result = bermudan_collar(
            REF, MATURITY,
            cap_strike=0.06, floor_strike=0.03,
            hw_a=HW_A, hw_sigma=HW_SIGMA, r0=R0,
            exercise_dates_years=EXERCISE_DATES,
            n_steps=N_STEPS,
        )
        expected = result["cap_result"].price - result["floor_result"].price
        assert result["collar_price"] == pytest.approx(expected, abs=1e-8)
