"""MC migration tests: validate engine-based implementations match originals."""

from __future__ import annotations

import math

import pytest

from pricebook.black76 import OptionType
from pricebook.asian import mc_asian_arithmetic, mc_asian_arithmetic_via_engine
from pricebook.barrier_option import barrier_option_mc_via_engine


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
