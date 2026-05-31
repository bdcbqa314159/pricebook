"""Tests for continuous barrier monitoring via Brownian bridge."""

import pytest
import math
import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid
from pricebook.models.mc_processes import GBMProcess
from pricebook.models.mc_payoffs import barrier_knockout, barrier_knockin, european_call


class TestBarrierBridgeKnockout:
    def test_discrete_unchanged(self):
        """Discrete monitoring (default) should work as before."""
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)
        e = MCEngine(proc, tg, n_paths=10_000, seed=42)

        result = e.price(barrier_knockout(100, 130, "up-and-out"))
        assert result.price > 0

    def test_continuous_less_than_discrete(self):
        """Continuous up-and-out should be <= discrete (more knockouts)."""
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 20)  # coarse grid amplifies difference
        e = MCEngine(proc, tg, n_paths=5_000, seed=42)

        discrete = e.price(barrier_knockout(100, 120, "up-and-out"))

        e2 = MCEngine(proc, tg, n_paths=5_000, seed=42)
        continuous = e2.price(barrier_knockout(100, 120, "up-and-out",
                                                continuous=True, sigma=0.20, seed=42))

        assert continuous.price <= discrete.price + 0.5

    def test_down_and_out(self):
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)
        e = MCEngine(proc, tg, n_paths=5_000, seed=42)

        result = e.price(barrier_knockout(100, 80, "down-and-out"))
        assert result.price > 0

    def test_far_barrier_equals_vanilla(self):
        """Very far barrier should produce price ≈ vanilla."""
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)

        e1 = MCEngine(proc, tg, n_paths=10_000, seed=42)
        vanilla = e1.price(european_call(100))

        e2 = MCEngine(proc, tg, n_paths=10_000, seed=42)
        ko = e2.price(barrier_knockout(100, 300, "up-and-out"))  # barrier very far

        assert ko.price == pytest.approx(vanilla.price, rel=0.05)


class TestBarrierBridgeKnockin:
    def test_discrete_unchanged(self):
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)
        e = MCEngine(proc, tg, n_paths=10_000, seed=42)

        result = e.price(barrier_knockin(100, 120, "up-and-in"))
        assert result.price > 0

    def test_knockin_plus_knockout_equals_vanilla(self):
        """Knockin + knockout should approximately equal vanilla."""
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)

        e1 = MCEngine(proc, tg, n_paths=20_000, seed=42)
        vanilla = e1.price(european_call(100))

        e2 = MCEngine(proc, tg, n_paths=20_000, seed=42)
        ko = e2.price(barrier_knockout(100, 120, "up-and-out"))

        e3 = MCEngine(proc, tg, n_paths=20_000, seed=42)
        ki = e3.price(barrier_knockin(100, 120, "up-and-in"))

        # Parity: KI + KO ≈ Vanilla
        assert (ko.price + ki.price) == pytest.approx(vanilla.price, rel=0.10)
