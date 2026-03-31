"""Tests for barrier option pricing via PDE."""

import pytest
import math

from pricebook.finite_difference import (
    fd_european,
    fd_barrier_knockout,
    fd_barrier_knockin,
)
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N_S, N_T = 400, 400


class TestDownAndOutCall:
    def test_less_than_vanilla(self):
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                  barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        van = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                          n_spot=N_S, n_time=N_T, scheme="cn")
        assert ko < van

    def test_positive(self):
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                  barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        assert ko > 0

    def test_barrier_at_zero_equals_vanilla(self):
        """Barrier far below spot → effectively no barrier."""
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                  barrier_lower=1.0, n_spot=N_S, n_time=N_T)
        van = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                          n_spot=N_S, n_time=N_T, scheme="cn")
        assert ko == pytest.approx(van, rel=0.01)

    def test_higher_barrier_lower_price(self):
        """Higher lower barrier → more likely to knock out → lower price."""
        ko_80 = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                     barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        ko_90 = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                     barrier_lower=90.0, n_spot=N_S, n_time=N_T)
        assert ko_90 < ko_80


class TestUpAndOutPut:
    def test_less_than_vanilla(self):
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                                  barrier_upper=120.0, n_spot=N_S, n_time=N_T)
        van = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                          n_spot=N_S, n_time=N_T, scheme="cn")
        assert ko < van

    def test_positive(self):
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                                  barrier_upper=120.0, n_spot=N_S, n_time=N_T)
        assert ko > 0


class TestDoubleBarrier:
    def test_less_than_single(self):
        """Double barrier ≤ single barrier."""
        single = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                      barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        double = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                      barrier_lower=80.0, barrier_upper=130.0,
                                      n_spot=N_S, n_time=N_T)
        assert double <= single + 0.01

    def test_positive(self):
        double = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                      barrier_lower=80.0, barrier_upper=130.0,
                                      n_spot=N_S, n_time=N_T)
        assert double > 0


class TestInOutParity:
    def test_down_and_in_call(self):
        """knock-in + knock-out = vanilla."""
        ki = fd_barrier_knockin(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                 barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                  barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        van = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                          n_spot=N_S, n_time=N_T, scheme="cn")
        assert ki + ko == pytest.approx(van, rel=0.001)

    def test_up_and_in_put(self):
        ki = fd_barrier_knockin(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                                 barrier_upper=120.0, n_spot=N_S, n_time=N_T)
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                                  barrier_upper=120.0, n_spot=N_S, n_time=N_T)
        van = fd_european(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                          n_spot=N_S, n_time=N_T, scheme="cn")
        assert ki + ko == pytest.approx(van, rel=0.001)

    def test_knockin_positive(self):
        ki = fd_barrier_knockin(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                 barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        assert ki > 0


class TestRannacher:
    def test_rannacher_gives_reasonable_price(self):
        ko = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                  barrier_lower=80.0, n_spot=N_S, n_time=N_T,
                                  rannacher_steps=4)
        assert ko > 0

    def test_rannacher_close_to_standard(self):
        """Rannacher should give similar price to plain CN."""
        ko_plain = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                        barrier_lower=80.0, n_spot=N_S, n_time=N_T)
        ko_rann = fd_barrier_knockout(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL,
                                       barrier_lower=80.0, n_spot=N_S, n_time=N_T,
                                       rannacher_steps=4)
        assert ko_rann == pytest.approx(ko_plain, rel=0.02)
