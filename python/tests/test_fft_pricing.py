"""Tests for FFT-based option pricing and density recovery."""

import math

import numpy as np
import pytest

from pricebook.black76 import OptionType
from pricebook.cos_method import bs_char_func, cos_price, heston_char_func_cos
from pricebook.equity_option import equity_option_price
from pricebook.fft_pricing import (
    DensityResult,
    FFTResult,
    carr_madan_fft,
    density_from_calls,
    density_from_cf,
    lewis_price,
)


SPOT, RATE, VOL, T = 100.0, 0.05, 0.20, 1.0


def _bs_cf():
    return bs_char_func(RATE, 0.0, VOL, T)


# ---- Carr-Madan FFT ----

class TestCarrMadanFFT:
    def test_atm_matches_bs(self):
        """FFT ATM call matches Black-Scholes."""
        result = carr_madan_fft(_bs_cf(), SPOT, RATE, T)
        # Find strike closest to ATM
        idx = np.argmin(np.abs(result.strikes - SPOT))
        fft_price = result.prices[idx]
        bs_price = equity_option_price(SPOT, SPOT, RATE, VOL, T, OptionType.CALL)
        assert fft_price == pytest.approx(bs_price, rel=0.01)

    def test_prices_across_strikes(self):
        """FFT prices should match BS at multiple strikes."""
        result = carr_madan_fft(_bs_cf(), SPOT, RATE, T)
        for K in [90, 95, 100, 105, 110]:
            idx = np.argmin(np.abs(result.strikes - K))
            fft_price = result.prices[idx]
            bs_price = equity_option_price(SPOT, K, RATE, VOL, T, OptionType.CALL)
            assert fft_price == pytest.approx(bs_price, rel=0.02), f"K={K}"

    def test_n_points(self):
        result = carr_madan_fft(_bs_cf(), SPOT, RATE, T, N=2048)
        assert result.n_points == 2048
        assert len(result.strikes) == 2048

    def test_fft_matches_cos(self):
        """FFT and COS should agree for the same model."""
        fft = carr_madan_fft(_bs_cf(), SPOT, RATE, T)
        K = 105.0
        idx = np.argmin(np.abs(fft.strikes - K))
        fft_price = fft.prices[idx]
        cos_price_val = cos_price(_bs_cf(), SPOT, K, RATE, T, OptionType.CALL)
        assert fft_price == pytest.approx(cos_price_val, rel=0.02)


# ---- Lewis contour integral ----
# NOTE: Lewis formula needs careful derivation for our CF convention
# (log(S_T/S_0) vs log(S_T)). Deferred to a future slice.

class TestLewisPrice:
    def test_produces_positive_price(self):
        """Lewis produces a positive call price matching Black-76."""
        import cmath as _cmath

        def cf(u):
            # CF of log(S_T) — absolute, not centred
            x = math.log(SPOT)
            drift = (RATE - 0.5 * VOL**2) * T
            return _cmath.exp(1j * u * (x + drift) - 0.5 * VOL**2 * u**2 * T)

        price = lewis_price(cf, SPOT, SPOT, RATE, T, N=1024)
        assert price > 0
        # Should be close to Black-76
        from pricebook.black76 import black76_price, OptionType
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)
        bs = black76_price(fwd, SPOT, VOL, T, df, OptionType.CALL)
        assert price == pytest.approx(bs, rel=0.05)


# ---- Breeden-Litzenberger density ----

class TestDensityFromCalls:
    def test_bs_density_non_negative(self):
        """Density from BS call prices should be non-negative."""
        strikes = np.linspace(60, 150, 91)
        calls = np.array([
            equity_option_price(SPOT, K, RATE, VOL, T, OptionType.CALL)
            for K in strikes
        ])
        result = density_from_calls(strikes, calls, RATE, T)
        assert result.is_non_negative

    def test_density_integrates_near_one(self):
        """Density should integrate to approximately 1."""
        strikes = np.linspace(50, 200, 151)
        calls = np.array([
            equity_option_price(SPOT, K, RATE, VOL, T, OptionType.CALL)
            for K in strikes
        ])
        result = density_from_calls(strikes, calls, RATE, T)
        dk = strikes[1] - strikes[0]
        integral = np.sum(result.density[1:-1]) * dk
        assert integral == pytest.approx(1.0, abs=0.15)

    def test_too_few_strikes(self):
        result = density_from_calls([100, 110], [5.0, 3.0], RATE, T)
        assert len(result.density) == 2


# ---- Density from characteristic function ----

class TestDensityFromCF:
    def test_bs_density_shape(self):
        """Recovered density from BS CF should be lognormal-shaped."""
        cf = _bs_cf()
        x_grid = np.linspace(-1.0, 1.0, 101)
        density = density_from_cf(cf, x_grid)
        # Should be positive and peaked near the mean
        assert np.all(density > -0.01)
        # Peak should be near x = (r - 0.5σ²)T
        peak_idx = np.argmax(density)
        expected_peak = (RATE - 0.5 * VOL**2) * T
        assert x_grid[peak_idx] == pytest.approx(expected_peak, abs=0.1)

    def test_density_integrates(self):
        cf = _bs_cf()
        x_grid = np.linspace(-2.0, 2.0, 201)
        density = density_from_cf(cf, x_grid)
        dx = x_grid[1] - x_grid[0]
        integral = np.sum(density) * dx
        assert integral == pytest.approx(1.0, abs=0.1)
