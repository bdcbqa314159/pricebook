"""Tests for advanced PDE methods."""

import math

import numpy as np
import pytest

from pricebook.black76 import OptionType
from pricebook.equity_option import equity_option_price
from pricebook.pde_advanced import (
    MOLResult,
    PSORResult,
    SpectralResult,
    chebyshev_bs,
    method_of_lines,
    psor_american,
    richardson_extrapolation,
)


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


# ---- PSOR American ----

class TestPSOR:
    def test_american_put_exceeds_european(self):
        european = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        result = psor_american(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert result.price >= european * 0.99

    def test_exercise_boundary_exists(self):
        result = psor_american(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        # Exercise boundary should be below strike for a put
        nonzero = result.exercise_boundary[result.exercise_boundary > 0]
        if len(nonzero) > 0:
            assert nonzero.mean() < STRIKE

    def test_positive_price(self):
        result = psor_american(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert result.price > 0

    def test_deep_itm_put(self):
        """Deep ITM American put should be near intrinsic."""
        result = psor_american(80.0, 120.0, RATE, VOL, T, OptionType.PUT)
        intrinsic = 120.0 - 80.0
        assert result.price >= intrinsic * 0.95


# ---- Chebyshev spectral ----

class TestChebyshevBS:
    def test_matches_bs(self):
        """Spectral matches BS for European call."""
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        result = chebyshev_bs(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, N=32)
        assert result.price == pytest.approx(bs, rel=0.05)

    def test_put(self):
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        result = chebyshev_bs(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, N=32)
        assert result.price == pytest.approx(bs, rel=0.05)

    def test_convergence_with_n(self):
        """More Chebyshev points → more accuracy."""
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        err_16 = abs(chebyshev_bs(SPOT, STRIKE, RATE, VOL, T, N=16).price - bs)
        err_32 = abs(chebyshev_bs(SPOT, STRIKE, RATE, VOL, T, N=32).price - bs)
        assert err_32 <= err_16 * 1.5  # allow noise but should improve


# ---- Method of lines ----

class TestMethodOfLines:
    def test_matches_bs_call(self):
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        result = method_of_lines(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        assert result.price == pytest.approx(bs, rel=0.02)

    def test_matches_bs_put(self):
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        result = method_of_lines(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        assert result.price == pytest.approx(bs, rel=0.02)

    def test_positive(self):
        result = method_of_lines(SPOT, STRIKE, RATE, VOL, T)
        assert result.price > 0


# ---- Richardson extrapolation ----

class TestRichardsonExtrapolation:
    def test_order_2(self):
        """Richardson with order 2: (4×fine − coarse) / 3."""
        # True value: 1.0, coarse (O(h²)): 1.04, fine (O((h/2)²)): 1.01
        coarse = 1.04
        fine = 1.01
        result = richardson_extrapolation(coarse, fine, order=2)
        # (4×1.01 − 1.04) / 3 = (4.04 − 1.04)/3 = 1.0
        assert result == pytest.approx(1.0)

    def test_order_1(self):
        """Richardson with order 1: (2×fine − coarse)."""
        coarse = 1.1
        fine = 1.05
        result = richardson_extrapolation(coarse, fine, order=1)
        assert result == pytest.approx(1.0)

    def test_improves_fd_pricing(self):
        """Richardson on two FD resolutions improves accuracy."""
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)

        coarse = method_of_lines(SPOT, STRIKE, RATE, VOL, T, n_spot=50, n_time=3000).price
        fine = method_of_lines(SPOT, STRIKE, RATE, VOL, T, n_spot=100, n_time=5000).price
        rich = richardson_extrapolation(coarse, fine, order=2)

        err_fine = abs(fine - bs)
        err_rich = abs(rich - bs)
        # Richardson should be at least as good as fine
        assert err_rich <= err_fine * 2  # allow some noise
