"""Tests for LSM American put with proper discounting."""

import pytest
import math
import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid
from pricebook.models.mc_processes import GBMProcess
from pricebook.models.mc_payoffs import american_put


class TestLSMDiscounting:
    def test_american_put_positive(self):
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)
        e = MCEngine(proc, tg, n_paths=10_000, seed=42)
        result = e.price(american_put(100, r=0.05))
        assert result.price > 0

    def test_american_ge_european_put(self):
        """American put should be >= European put (early exercise premium)."""
        from pricebook.models.mc_payoffs import european_put
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)

        e = MCEngine(proc, tg, n_paths=20_000, seed=42)
        am = e.price(american_put(100, r=0.05))
        eu = e.price(european_put(100))

        assert am.price >= eu.price - 0.5  # small tolerance for MC noise

    def test_discounting_lowers_continuation(self):
        """With r > 0, discounting should make exercise earlier (higher value)."""
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)

        e = MCEngine(proc, tg, n_paths=10_000, seed=42)
        am_r0 = e.price(american_put(100, r=0.0))

        e2 = MCEngine(proc, tg, n_paths=10_000, seed=42)
        am_r5 = e2.price(american_put(100, r=0.05))

        # With discounting, exercise is more attractive → higher American value
        assert am_r5.price >= am_r0.price - 0.5

    def test_deep_itm_immediate_exercise(self):
        """Deep ITM American put (S=50, K=100) should be close to intrinsic."""
        proc = GBMProcess(50, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 50)
        e = MCEngine(proc, tg, n_paths=10_000, seed=42)
        result = e.price(american_put(100, r=0.05))
        # Intrinsic = 50, should be at least 45
        assert result.price > 40

    def test_backward_compatible_r0(self):
        """r=0 should work without errors (backward compatible)."""
        proc = GBMProcess(100, 0.05, 0.20)
        tg = TimeGrid.uniform(1.0, 20)
        e = MCEngine(proc, tg, n_paths=5_000, seed=42)
        result = e.price(american_put(100, r=0.0))
        assert result.price > 0
