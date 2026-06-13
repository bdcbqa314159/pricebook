"""Regression for L2 Wave-2 audit — `MCEngine(n_paths=1)` produced
bizarre downstream behaviour.

Pre-fix:
- ``MCEngine(..., n_paths=1, antithetic=True)``: ``n_half = 1 // 2 = 0``
  → zero paths generated → ``np.std(..., ddof=1)`` of zero samples →
  silent NaN stderr propagated into ``MCResult.stderr`` and
  ``confidence_95``.
- ``MCEngine(..., n_paths=1, antithetic=False)``: single-path "MC" with
  ``np.std(..., ddof=1)`` of one sample is also NaN.

Both modes are useless for Monte Carlo but pre-fix the engine ran them
silently and reported NaN downstream.

Post-fix:
- `n_paths < 2` raises `ValueError` upfront.
- `antithetic=True` with `n_paths < 4` raises `ValueError` (the engine
  uses ``n_half = n_paths // 2`` antithetic pairs, so 2 paths give only
  1 pair which is also degenerate for ddof=1 stderr).
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.models.mc_engine import MCEngine, ProcessSpec, TimeGrid


def _spec() -> ProcessSpec:
    return ProcessSpec(
        x0=np.array([100.0]),
        drift=lambda x, t: 0.0 * x,
        diffusion=lambda x, t: 0.2 * x,
        n_factors=1,
    )


class TestNPathsValidation:
    def test_n_paths_one_raises(self):
        with pytest.raises(ValueError, match="n_paths must be >= 2"):
            MCEngine(_spec(), TimeGrid.uniform(1.0, 10), n_paths=1)

    def test_n_paths_zero_raises(self):
        with pytest.raises(ValueError, match="n_paths must be >= 2"):
            MCEngine(_spec(), TimeGrid.uniform(1.0, 10), n_paths=0)

    def test_n_paths_negative_raises(self):
        with pytest.raises(ValueError, match="n_paths must be >= 2"):
            MCEngine(_spec(), TimeGrid.uniform(1.0, 10), n_paths=-100)


class TestAntitheticMinimum:
    def test_antithetic_with_two_paths_raises(self):
        with pytest.raises(ValueError, match="antithetic=True requires n_paths >= 4"):
            MCEngine(_spec(), TimeGrid.uniform(1.0, 10),
                     n_paths=2, antithetic=True)

    def test_antithetic_with_three_paths_raises(self):
        with pytest.raises(ValueError, match="antithetic=True requires n_paths >= 4"):
            MCEngine(_spec(), TimeGrid.uniform(1.0, 10),
                     n_paths=3, antithetic=True)

    def test_antithetic_with_four_paths_works(self):
        eng = MCEngine(_spec(), TimeGrid.uniform(1.0, 10),
                       n_paths=4, antithetic=True)
        # Generate paths and check shape — must NOT crash.
        paths = eng.generate_paths()
        assert paths.shape[0] == 4


class TestHealthyPathUnchanged:
    def test_large_n_paths_works(self):
        eng = MCEngine(_spec(), TimeGrid.uniform(1.0, 10), n_paths=1000)
        paths = eng.generate_paths()
        assert paths.shape[0] == 1000
