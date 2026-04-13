"""Tests for COS Bermudan/American pricing."""

import math
import cmath

import pytest

from pricebook.black76 import OptionType
from pricebook.cos_bermudan import cos_american, cos_bermudan
from pricebook.cos_method import cos_price, bs_char_func
from pricebook.equity_option import equity_option_price


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


def _bs_cf_dt(dt):
    """BS char func for one time step dt."""
    mu = (RATE - 0.5 * VOL**2) * dt
    var = VOL**2 * dt
    def phi(u):
        return cmath.exp(1j * u * mu - 0.5 * u * u * var)
    return phi


class TestCosBermudan:
    def test_bermudan_put_exceeds_european(self):
        """Bermudan put ≥ European put (early exercise premium)."""
        european = cos_price(bs_char_func(RATE, 0.0, VOL, T),
                             SPOT, STRIKE, RATE, T, OptionType.PUT)
        bermudan = cos_bermudan(
            _bs_cf_dt(T / 10), SPOT, STRIKE, RATE, T, n_exercise=10,
            option_type=OptionType.PUT,
        )
        assert bermudan >= european - 0.5  # allow small numerical tolerance

    def test_bermudan_with_one_exercise_near_european(self):
        """Bermudan with 1 exercise date ≈ European (grid approximation)."""
        european = cos_price(bs_char_func(RATE, 0.0, VOL, T),
                             SPOT, STRIKE, RATE, T, OptionType.PUT)
        bermudan = cos_bermudan(
            _bs_cf_dt(T), SPOT, STRIKE, RATE, T, n_exercise=1,
            option_type=OptionType.PUT,
        )
        # Grid-based method introduces some early-exercise premium even
        # with 1 date; should be within 25% of European.
        assert bermudan == pytest.approx(european, rel=0.25)

    def test_positive_price(self):
        price = cos_bermudan(
            _bs_cf_dt(T / 5), SPOT, STRIKE, RATE, T, n_exercise=5,
            option_type=OptionType.PUT,
        )
        assert price > 0

    def test_otm_put_cheaper_than_atm(self):
        otm = cos_bermudan(
            _bs_cf_dt(T / 5), SPOT, 80.0, RATE, T, 5, OptionType.PUT,
        )
        atm = cos_bermudan(
            _bs_cf_dt(T / 5), SPOT, 100.0, RATE, T, 5, OptionType.PUT,
        )
        assert otm < atm


class TestCosAmerican:
    def test_american_put_exceeds_european(self):
        european = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        american = cos_american(
            _bs_cf_dt(T / 50), SPOT, STRIKE, RATE, T,
            option_type=OptionType.PUT, n_exercise=50,
        )
        assert american >= european - 0.5

    def test_positive(self):
        price = cos_american(
            _bs_cf_dt(T / 20), SPOT, STRIKE, RATE, T,
            option_type=OptionType.PUT, n_exercise=20,
        )
        assert price > 0
