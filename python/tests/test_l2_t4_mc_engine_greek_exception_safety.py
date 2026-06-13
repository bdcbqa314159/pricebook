"""Regression for L2 Wave-2 audit — `MCEngine.greek` left process.x0 (or a
bumped attribute) in the bumped state when `price()` raised during the
up/down evaluation.

Pre-fix code:

    self.process.x0[i] = original + bump
    price_up = self.price(...)            # <-- if this raises, x0 stays bumped
    self.process.x0[i] = original - bump
    price_dn = self.price(...)
    self.process.x0 = original_x0         # <-- only runs on happy path

A subsequent unrelated call on the same engine sees a SILENTLY corrupted
process spec and produces wrong prices, with no exception or warning to
suggest the engine's state has been tampered with.  Same pattern for the
attribute branch.

Post-fix wraps the bump sequence in `try ... finally:` so the original
state is always restored, even when up/down prices fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.models.mc_engine import MCEngine, ProcessSpec, TimeGrid


def _gbm_spec() -> ProcessSpec:
    """A trivial 1-factor GBM-like spec.  x0 = [100]."""
    return ProcessSpec(
        x0=np.array([100.0]),
        drift=lambda x, t: 0.0 * x,
        diffusion=lambda x, t: 0.2 * x,
        n_factors=1,
    )


def _payoff_raises(_paths, _times):
    raise RuntimeError("synthetic payoff failure")


def _payoff_ok(paths, _times):
    """Vanilla expected terminal value (handles 1-factor 2D paths)."""
    if paths.ndim == 3:
        return paths[:, -1, 0]
    return paths[:, -1]


class TestGreekExceptionSafety:
    def test_x0_restored_when_price_raises(self):
        spec = _gbm_spec()
        original_x0 = spec.x0.copy()
        eng = MCEngine(
            spec, time_grid=TimeGrid([0.0, 0.5, 1.0]),
            n_paths=100, seed=42,
        )
        with pytest.raises(RuntimeError):
            eng.greek(_payoff_raises, param_name=0, bump=0.01)
        # After the failed greek call, x0 MUST be back to its original.
        np.testing.assert_array_equal(spec.x0, original_x0), \
            "x0 was not restored after price() raised inside greek()"

    def test_attribute_restored_when_price_raises(self):
        """Attribute branch: bump a process attribute, then raise."""
        spec = _gbm_spec()
        spec.my_sigma = 0.2  # custom scalar attribute the test will bump
        original_sigma = spec.my_sigma

        eng = MCEngine(
            spec, time_grid=TimeGrid([0.0, 0.5, 1.0]),
            n_paths=100, seed=42,
        )
        with pytest.raises(RuntimeError):
            eng.greek(_payoff_raises, param_name="my_sigma", bump=0.01)
        assert spec.my_sigma == original_sigma, \
            f"my_sigma was not restored (now {spec.my_sigma}, expected {original_sigma})"

    def test_subsequent_price_unaffected_after_greek_raises(self):
        """Most important: a healthy `price()` call AFTER a failed greek()
        must see the unbumped state."""
        spec = _gbm_spec()
        eng = MCEngine(
            spec, time_grid=TimeGrid([0.0, 0.5, 1.0]),
            n_paths=2000, seed=42,
        )

        # Reference price BEFORE the bad greek call.
        ref = eng.price(_payoff_ok).price

        # Greek call that raises during the up-bump.
        with pytest.raises(RuntimeError):
            eng.greek(_payoff_raises, param_name=0, bump=10.0)

        # Re-price.  Must match the reference (engine state must be untouched).
        after = eng.price(_payoff_ok).price
        assert after == pytest.approx(ref, rel=1e-9), \
            f"price corrupted by failed greek: ref={ref}, after={after}"


class TestGreekHappyPath:
    def test_greek_returns_finite_and_restores(self):
        """Sanity: happy-path greek still works after the try/finally rework."""
        spec = _gbm_spec()
        original_x0 = spec.x0.copy()
        eng = MCEngine(
            spec, time_grid=TimeGrid([0.0, 0.5, 1.0]),
            n_paths=2000, seed=42,
        )
        delta = eng.greek(_payoff_ok, param_name=0, bump=0.5)
        assert np.isfinite(delta)
        # Drift = 0 → forward = spot → ∂E[S_T]/∂S_0 should be ≈ 1.0.
        assert delta == pytest.approx(1.0, abs=0.05)
        # Original x0 restored.
        np.testing.assert_array_equal(spec.x0, original_x0)
