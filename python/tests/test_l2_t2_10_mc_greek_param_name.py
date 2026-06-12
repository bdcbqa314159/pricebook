"""Regression for L2 Tier-2 T2.10 — `MCEngine.greek()` honours `param_name`.

Pre-fix `MCEngine.greek()` accepted `param_name` but never used it: the
implementation always bumped the entire `process.x0` vector via
`self.process.x0 = original_x0 + bump` (scalar broadcasts to every
component).  For a multi-factor process (Heston, G2++, basket models, etc.)
this produced a meaningless "delta" that simultaneously moved every state
variable.

Post-fix:
* int or "x0[i]" → bump only x0[i].
* string attribute name → bump that scalar attribute on the ProcessSpec.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.models.mc_engine import MCEngine, ProcessSpec, TimeGrid


def _gbm_process(s0: float = 100.0, mu: float = 0.05, sigma: float = 0.20):
    """Plain 1D GBM."""
    return ProcessSpec(
        x0=s0,
        drift=lambda x, t: mu * x,
        diffusion=lambda x, t: sigma * x,
        n_factors=1,
    )


class _AttrProcess(ProcessSpec):
    """GBM with a mutable `sigma` attribute — lets us test attribute-bump
    via greek(param_name='sigma', ...)."""

    def __init__(self, s0: float = 100.0, mu: float = 0.05, sigma: float = 0.20):
        self.sigma = sigma  # set BEFORE the super().__init__ so closures see it
        proc_self = self
        super().__init__(
            x0=s0, n_factors=1,
            drift=lambda x, t: mu * x,
            diffusion=lambda x, t: proc_self.sigma * x,
        )


class TestGreekHonoursParamName:
    def test_x0_index_zero_only_bumps_spot(self):
        """For a 1D GBM, greek("x0[0]") should compute ∂PV/∂S₀."""
        proc = _gbm_process(s0=100.0)
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 50),
                          n_paths=5_000, seed=42)
        payoff = lambda paths, t: np.maximum(paths[:, -1] - 100.0, 0.0)
        delta = engine.greek(payoff, "x0[0]", bump=0.5, discount_factor=math.exp(-0.05))
        # For an ATM call, BS delta ≈ N(d1) ≈ 0.64 for these params.  MC noise
        # gives wide tolerance.
        assert 0.4 < delta < 0.85, f"delta = {delta:.3f} outside plausible range"

    def test_int_param_name_works(self):
        """Bare integer also works as index."""
        proc = _gbm_process(s0=100.0)
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 50),
                          n_paths=5_000, seed=42)
        payoff = lambda paths, t: np.maximum(paths[:, -1] - 100.0, 0.0)
        delta = engine.greek(payoff, 0, bump=0.5, discount_factor=math.exp(-0.05))
        assert 0.4 < delta < 0.85

    def test_x0_restored_after_greek(self):
        """The greek() call must leave x0 untouched after returning."""
        proc = _gbm_process(s0=100.0)
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 10),
                          n_paths=500, seed=42)
        payoff = lambda paths, t: np.maximum(paths[:, -1] - 100.0, 0.0)
        x0_before = proc.x0.copy()
        engine.greek(payoff, "x0[0]", bump=1.0)
        assert np.allclose(proc.x0, x0_before), (
            f"x0 not restored: before={x0_before}, after={proc.x0}"
        )

    def test_attribute_bump_vega_via_sigma(self):
        """Pre-fix `param_name` was ignored.  Post-fix, attribute-name path
        lets the user bump a stored process attribute (e.g. `sigma`) and
        compute its sensitivity."""
        proc = _AttrProcess(s0=100.0, sigma=0.20)
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 50),
                          n_paths=5_000, seed=42)
        payoff = lambda paths, t: np.maximum(paths[:, -1] - 100.0, 0.0)
        vega = engine.greek(payoff, "sigma", bump=0.01,
                            discount_factor=math.exp(-0.05))
        # ATM Black-Scholes vega ≈ S φ(d1) sqrt(T) ≈ 100 × 0.39 × 1 = 39.
        # Per unit vol move (we bumped 0.01, so divided by 0.02 in centred FD).
        # The greek() returns dPV/d(sigma).
        assert 20 < vega < 60, f"vega = {vega:.2f} outside plausible range"
        # And sigma was restored.
        assert proc.sigma == pytest.approx(0.20)

    def test_unknown_param_name_raises(self):
        proc = _gbm_process()
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 10), n_paths=100)
        payoff = lambda paths, t: np.maximum(paths[:, -1] - 100.0, 0.0)
        with pytest.raises(ValueError, match="param_name"):
            engine.greek(payoff, "doesnt_exist", bump=0.01)

    def test_index_out_of_range_raises(self):
        proc = _gbm_process()
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 10), n_paths=100)
        payoff = lambda paths, t: np.maximum(paths[:, -1] - 100.0, 0.0)
        with pytest.raises(IndexError):
            engine.greek(payoff, "x0[5]", bump=0.01)
