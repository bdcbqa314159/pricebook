"""Tests for MC process RNG seeding — jump processes must be reproducible."""

import pytest
import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid
from pricebook.models.mc_processes import (
    JumpDiffusionProcess, BatesProcess, VarianceGammaProcess,
)


class TestJumpDiffusionSeeding:
    def test_same_seed_same_paths(self):
        """Two engines with same seed must produce identical paths."""
        proc1 = JumpDiffusionProcess(100, 0.05, 0.2, 1.0, -0.1, 0.15, seed=42)
        proc2 = JumpDiffusionProcess(100, 0.05, 0.2, 1.0, -0.1, 0.15, seed=42)
        tg = TimeGrid.uniform(1.0, 10)

        e1 = MCEngine(proc1, tg, n_paths=1000, seed=99)
        e2 = MCEngine(proc2, tg, n_paths=1000, seed=99)

        p1 = e1.generate_paths()
        p2 = e2.generate_paths()

        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_paths(self):
        proc1 = JumpDiffusionProcess(100, 0.05, 0.2, 1.0, -0.1, 0.15, seed=42)
        proc2 = JumpDiffusionProcess(100, 0.05, 0.2, 1.0, -0.1, 0.15, seed=99)
        tg = TimeGrid.uniform(1.0, 10)

        e1 = MCEngine(proc1, tg, n_paths=1000, seed=99)
        e2 = MCEngine(proc2, tg, n_paths=1000, seed=99)

        p1 = e1.generate_paths()
        p2 = e2.generate_paths()

        # Paths should differ because jump RNG seeds differ
        assert not np.allclose(p1, p2)

    def test_positive_terminal(self):
        proc = JumpDiffusionProcess(100, 0.05, 0.2, 1.0, -0.1, 0.15, seed=42)
        tg = TimeGrid.uniform(1.0, 50)
        e = MCEngine(proc, tg, n_paths=5000, seed=42)
        paths = e.generate_paths()
        terminals = np.exp(paths[:, -1])  # log-space → spot
        assert np.all(terminals > 0)


class TestBatesSeeding:
    def test_same_seed_same_paths(self):
        proc1 = BatesProcess(100, 0.04, 0.05, 1.5, 0.04, 0.3, -0.7,
                              0.5, -0.05, 0.1, seed=42)
        proc2 = BatesProcess(100, 0.04, 0.05, 1.5, 0.04, 0.3, -0.7,
                              0.5, -0.05, 0.1, seed=42)
        tg = TimeGrid.uniform(1.0, 10)

        e1 = MCEngine(proc1, tg, n_paths=500, seed=99)
        e2 = MCEngine(proc2, tg, n_paths=500, seed=99)

        p1 = e1.generate_paths()
        p2 = e2.generate_paths()

        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_paths(self):
        proc1 = BatesProcess(100, 0.04, 0.05, 1.5, 0.04, 0.3, -0.7,
                              0.5, -0.05, 0.1, seed=42)
        proc2 = BatesProcess(100, 0.04, 0.05, 1.5, 0.04, 0.3, -0.7,
                              0.5, -0.05, 0.1, seed=99)
        tg = TimeGrid.uniform(1.0, 10)

        e1 = MCEngine(proc1, tg, n_paths=500, seed=99)
        e2 = MCEngine(proc2, tg, n_paths=500, seed=99)

        p1 = e1.generate_paths()
        p2 = e2.generate_paths()

        assert not np.allclose(p1, p2)


class TestVarianceGammaSeeding:
    def test_same_seed_same_paths(self):
        proc1 = VarianceGammaProcess(100, 0.05, 0.2, 0.25, -0.14, seed=42)
        proc2 = VarianceGammaProcess(100, 0.05, 0.2, 0.25, -0.14, seed=42)
        tg = TimeGrid.uniform(1.0, 10)

        e1 = MCEngine(proc1, tg, n_paths=1000, seed=99)
        e2 = MCEngine(proc2, tg, n_paths=1000, seed=99)

        p1 = e1.generate_paths()
        p2 = e2.generate_paths()

        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_paths(self):
        proc1 = VarianceGammaProcess(100, 0.05, 0.2, 0.25, -0.14, seed=42)
        proc2 = VarianceGammaProcess(100, 0.05, 0.2, 0.25, -0.14, seed=99)
        tg = TimeGrid.uniform(1.0, 10)

        e1 = MCEngine(proc1, tg, n_paths=1000, seed=99)
        e2 = MCEngine(proc2, tg, n_paths=1000, seed=99)

        p1 = e1.generate_paths()
        p2 = e2.generate_paths()

        assert not np.allclose(p1, p2)
