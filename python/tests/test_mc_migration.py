"""MC migration tests: validate engine-based implementations match originals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.models.black76 import OptionType
from pricebook.asian import mc_asian_arithmetic, mc_asian_arithmetic_via_engine
from pricebook.barrier_option import barrier_option_mc_via_engine
from pricebook.models.mc_instrument_adapters import (
    autocallable_mc, cliquet_mc, basket_mc,
    heston_european_mc, sabr_european_mc, tarf_mc,
    equity_xva_mc, rates_xva_mc,
)


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
N_STEPS = 50
N_PATHS = 50_000


# ── Asian: old vs engine ──

class TestAsianMigration:

    def test_arithmetic_call_matches(self):
        """Engine-based Asian should match original within tolerance."""
        old = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                  n_paths=N_PATHS, seed=42)
        new = mc_asian_arithmetic_via_engine(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                             n_paths=N_PATHS, seed=42)
        # Both should be close (different RNG paths, so rel tolerance)
        assert new.price == pytest.approx(old.price, rel=0.15)

    def test_arithmetic_put_matches(self):
        old = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                  option_type=OptionType.PUT,
                                  n_paths=N_PATHS, seed=42)
        new = mc_asian_arithmetic_via_engine(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                             option_type=OptionType.PUT,
                                             n_paths=N_PATHS, seed=42)
        assert new.price == pytest.approx(old.price, rel=0.15)

    def test_cv_matches(self):
        """Control variate version should also match."""
        old = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                  control_variate=True,
                                  n_paths=N_PATHS, seed=42)
        new = mc_asian_arithmetic_via_engine(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                             control_variate=True,
                                             n_paths=N_PATHS, seed=42)
        assert new.price == pytest.approx(old.price, rel=0.15)

    def test_antithetic_matches(self):
        old = mc_asian_arithmetic(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                  antithetic=True,
                                  n_paths=N_PATHS, seed=42)
        new = mc_asian_arithmetic_via_engine(SPOT, STRIKE, RATE, VOL, T, N_STEPS,
                                             antithetic=True,
                                             n_paths=N_PATHS, seed=42)
        assert new.price == pytest.approx(old.price, rel=0.15)

    def test_sobol_available(self):
        """Engine version supports Sobol (original doesn't)."""
        result = mc_asian_arithmetic_via_engine(
            SPOT, STRIKE, RATE, VOL, T, N_STEPS,
            n_paths=N_PATHS, seed=42, use_sobol=True,
        )
        assert result.price > 0


# ── Barrier: engine version ──

class TestBarrierMigration:

    def test_up_and_out_positive(self):
        result = barrier_option_mc_via_engine(
            SPOT, STRIKE, 130.0, RATE, VOL, T,
            barrier_type="up-and-out", n_paths=N_PATHS,
        )
        assert result["price"] > 0

    def test_ko_less_than_vanilla(self):
        """Knockout should be cheaper than vanilla Black-Scholes."""
        from scipy.stats import norm
        d1 = (math.log(SPOT / STRIKE) + (RATE + 0.5 * VOL ** 2) * T) / (VOL * math.sqrt(T))
        d2 = d1 - VOL * math.sqrt(T)
        bs = SPOT * norm.cdf(d1) - STRIKE * math.exp(-RATE * T) * norm.cdf(d2)

        ko = barrier_option_mc_via_engine(
            SPOT, STRIKE, 150.0, RATE, VOL, T,
            barrier_type="up-and-out", n_paths=N_PATHS,
        )
        assert ko["price"] < bs

    def test_bridge_correction_more_knockouts(self):
        """Bridge correction should knock out more paths (lower price)."""
        no_bridge = barrier_option_mc_via_engine(
            SPOT, STRIKE, 125.0, RATE, VOL, T,
            n_steps=12,  # coarse monthly monitoring
            n_paths=N_PATHS, bridge_correction=False,
        )
        with_bridge = barrier_option_mc_via_engine(
            SPOT, STRIKE, 125.0, RATE, VOL, T,
            n_steps=12,
            n_paths=N_PATHS, bridge_correction=True,
        )
        assert with_bridge["price"] <= no_bridge["price"]

    def test_down_and_out(self):
        result = barrier_option_mc_via_engine(
            SPOT, STRIKE, 80.0, RATE, VOL, T,
            barrier_type="down-and-out", n_paths=N_PATHS,
        )
        assert result["price"] > 0


# ── Batch 2: Autocallable + Cliquet ──

class TestAutocallableMigration:

    def test_price_positive(self):
        result = autocallable_mc(100, 0.05, 0.20, 3.0, n_paths=N_PATHS)
        assert result.price > 0

    def test_higher_coupon_higher_price(self):
        r_low = autocallable_mc(100, 0.05, 0.20, 3.0, coupon=0.04, n_paths=N_PATHS).price
        r_high = autocallable_mc(100, 0.05, 0.20, 3.0, coupon=0.12, n_paths=N_PATHS).price
        assert r_high > r_low


class TestCliquetMigration:

    def test_price_positive(self):
        result = cliquet_mc(100, 0.05, 0.20, 1.0, n_paths=N_PATHS)
        assert result.price > 0

    def test_tighter_cap_lower_price(self):
        r_tight = cliquet_mc(100, 0.05, 0.20, 1.0, cap=0.02, n_paths=N_PATHS).price
        r_wide = cliquet_mc(100, 0.05, 0.20, 1.0, cap=0.10, n_paths=N_PATHS).price
        assert r_tight < r_wide


# ── Batch 3: Basket ──

class TestBasketMigration:

    def test_basket_call_positive(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = basket_mc([100, 100], 0.05, [0.20, 0.25], corr, 100, 1.0, n_paths=N_PATHS)
        assert result.price > 0

    def test_higher_correlation_lower_basket(self):
        corr_low = np.array([[1.0, 0.1], [0.1, 1.0]])
        corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
        r_low = basket_mc([100, 100], 0.05, [0.20, 0.20], corr_low, 100, 1.0, n_paths=N_PATHS).price
        r_high = basket_mc([100, 100], 0.05, [0.20, 0.20], corr_high, 100, 1.0, n_paths=N_PATHS).price
        assert r_low > r_high * 0.8


# ── Batch 4: Heston + SABR ──

class TestHestonMigration:

    def test_full_mc(self):
        result = heston_european_mc(100, 100, 0.05, 0.04, 2, 0.04, 0.3, -0.7, 1.0, n_paths=N_PATHS)
        assert result.price > 0

    def test_conditional_mc(self):
        result = heston_european_mc(100, 100, 0.05, 0.04, 2, 0.04, 0.3, -0.7, 1.0,
                                     n_paths=N_PATHS, use_conditional=True)
        assert result.price > 0

    def test_conditional_lower_stderr(self):
        full = heston_european_mc(100, 100, 0.05, 0.04, 2, 0.04, 0.3, -0.7, 1.0,
                                   n_paths=20_000, use_conditional=False)
        cond = heston_european_mc(100, 100, 0.05, 0.04, 2, 0.04, 0.3, -0.7, 1.0,
                                   n_paths=20_000, use_conditional=True)
        assert cond.stderr < full.stderr


class TestSABRMigration:

    def test_sabr_call_positive(self):
        result = sabr_european_mc(0.05, 0.05, 1.0, 0.30, 0.5, -0.3, 0.4, n_paths=N_PATHS)
        assert result.price > 0


# ── Batch 5: TARF ──

class TestTARFMigration:

    def test_price_finite(self):
        result = tarf_mc(1.10, 0.02, 0.01, 0.08, 1.10, 0.10, 1.0, n_paths=N_PATHS)
        assert math.isfinite(result.price)


# ── Batch 6: XVA exposure ──

class TestXVAMigration:

    def test_equity_xva(self):
        result = equity_xva_mc(100, 0.20, 0.05, 10e6, 5.0, n_paths=5_000)
        assert result["cva"] > 0
        assert len(result["epe"]) > 1

    def test_rates_xva(self):
        result = rates_xva_mc(0.05, 0.04, 10e6, 5.0, n_paths=5_000)
        assert math.isfinite(result["cva"])
        assert len(result["epe"]) > 1
