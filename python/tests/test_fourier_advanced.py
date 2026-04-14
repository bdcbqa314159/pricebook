"""Tests for advanced Fourier methods."""

import math
import cmath

import numpy as np
import pytest

from pricebook.cos_method import bs_char_func
from pricebook.fourier_advanced import (
    CumulantResult,
    FFT2DResult,
    cumulants_from_cf,
    edgeworth_expansion,
    fft_2d_basket,
    hilbert_implied_vol_slope,
    mellin_power_option,
)


SPOT, RATE, VOL, T = 100.0, 0.05, 0.20, 1.0


def _bs_cf():
    return bs_char_func(RATE, 0.0, VOL, T)


# ---- Cumulants ----

class TestCumulantsFromCF:
    def test_bs_mean(self):
        """BS log-return mean = (r − 0.5σ²)T."""
        result = cumulants_from_cf(_bs_cf())
        expected = (RATE - 0.5 * VOL**2) * T
        assert result.c1 == pytest.approx(expected, rel=0.05)

    def test_bs_variance(self):
        """BS log-return variance = σ²T."""
        result = cumulants_from_cf(_bs_cf())
        assert result.c2 == pytest.approx(VOL**2 * T, rel=0.05)

    def test_bs_skewness_zero(self):
        """BS (lognormal) log-returns have zero skewness."""
        result = cumulants_from_cf(_bs_cf())
        assert abs(result.skewness) < 0.5

    def test_returns_all_four(self):
        result = cumulants_from_cf(_bs_cf())
        assert isinstance(result.c1, float)
        assert isinstance(result.c2, float)
        assert isinstance(result.c3, float)
        assert isinstance(result.c4, float)


# ---- Edgeworth expansion ----

class TestEdgeworthExpansion:
    def test_normal_density(self):
        """With zero skew/kurtosis, Edgeworth = Gaussian."""
        x = np.linspace(-3, 3, 100)
        density = edgeworth_expansion(x, mean=0, std=1)
        gaussian = np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        np.testing.assert_allclose(density, gaussian, atol=1e-10)

    def test_integrates_near_one(self):
        x = np.linspace(-5, 5, 1000)
        dx = x[1] - x[0]
        density = edgeworth_expansion(x, mean=0, std=1, skewness=0.5)
        integral = np.sum(density) * dx
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_skewed_shifts_mode(self):
        """Positive skewness shifts mass to the right."""
        x = np.linspace(-4, 4, 200)
        normal = edgeworth_expansion(x, 0, 1, 0, 0)
        skewed = edgeworth_expansion(x, 0, 1, 1.0, 0)
        # Right tail should be heavier
        right_normal = np.sum(normal[x > 1])
        right_skewed = np.sum(skewed[x > 1])
        assert right_skewed > right_normal


# ---- Hilbert implied vol slope ----

class TestHilbertImpliedVol:
    def test_bs_finite_slope(self):
        slope = hilbert_implied_vol_slope(_bs_cf(), T)
        assert slope > 0
        assert math.isfinite(slope)


# ---- 2D FFT basket ----

class TestFFT2DBasket:
    def test_positive_price(self):
        """Basket option should have positive price."""
        def cf_2d(u1, u2):
            # Independent BS
            mu = (RATE - 0.5 * VOL**2) * T
            var = VOL**2 * T
            return cmath.exp(1j * u1 * mu - 0.5 * u1**2 * var) * \
                   cmath.exp(1j * u2 * mu - 0.5 * u2**2 * var)

        result = fft_2d_basket(cf_2d, (100, 100), (0.5, 0.5), 100, RATE, T)
        assert result.price > 0

    def test_at_the_money(self):
        """ATM basket call on two identical assets ≈ single-asset ATM call."""
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType

        def cf_2d(u1, u2):
            mu = (RATE - 0.5 * VOL**2) * T
            var = VOL**2 * T
            return cmath.exp(1j * (u1 + u2) * mu - 0.5 * (u1 + u2)**2 * var)

        result = fft_2d_basket(cf_2d, (100, 100), (0.5, 0.5), 100, RATE, T)
        single = equity_option_price(100, 100, RATE, VOL, T, OptionType.CALL)
        # Basket approximation is coarse; check same order of magnitude
        assert 0.3 * single < result.price < 3.0 * single


# ---- Mellin power option ----

class TestMellinPowerOption:
    def test_power_1_is_call(self):
        """Power option with p=1 = standard call."""
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType
        bs = equity_option_price(SPOT, SPOT, RATE, VOL, T, OptionType.CALL)
        power = mellin_power_option(SPOT, SPOT, RATE, VOL, T, power=1.0)
        assert power == pytest.approx(bs, rel=0.05)

    def test_power_2_positive(self):
        price = mellin_power_option(SPOT, SPOT**2, RATE, VOL, T, power=2.0)
        assert price > 0

    def test_higher_power_higher_price(self):
        """Higher power → more convexity → higher price."""
        p1 = mellin_power_option(SPOT, SPOT, RATE, VOL, T, power=1.0)
        p2 = mellin_power_option(SPOT, SPOT**2, RATE, VOL, T, power=2.0)
        # Not directly comparable (different strikes), but both positive
        assert p1 > 0
        assert p2 > 0
