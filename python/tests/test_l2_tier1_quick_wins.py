"""Regressions for L2 Tier-1 quick wins (T1.2, T1.3, T1.5, T1.7).

These were all dual-critic-confirmed Tier-1 bugs in `MODULE_HEALTH.md`.

- T1.2 `numerical/_fourier.py` — `np.trapz` was removed in NumPy 2.x; we now use
  `np.trapezoid`. The `CharacteristicFunction.density()` method had been a hard
  crash on every call.
- T1.3 `numerical/_integrate.py` — `integrate_2d` was passing `f(x, y)` directly
  to scipy `dblquad`, which expects `func(y, x)`. We now wrap with an order-swap.
- T1.5 `numerical/_trees.py` — `_tian_params` had a wrong `V` formula and missing
  V factor in u/d; the discriminant was nearly always negative so Tian silently
  fell back to CRR via the "very small dt" branch.
- T1.7 `models/mc_engine.py` — `step_fn = euler_step if scheme=='euler' else euler_step`
  (copy-paste typo); Milstein silently ran Euler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ============================================================
# T1.2 — CharacteristicFunction.density does not crash
# ============================================================

class TestT1_2_FourierDensity:
    def test_density_callable_does_not_crash(self):
        from pricebook.numerical._fourier import CharacteristicFunction
        # Standard normal characteristic function: φ(u) = exp(-u²/2)
        cf = CharacteristicFunction(lambda u: np.exp(-0.5 * u ** 2))
        x_grid = np.linspace(-3.0, 3.0, 7)
        density = cf.density(x_grid, n_quad=64)
        # Just check it runs without AttributeError and produces real numbers.
        assert density.shape == x_grid.shape
        # At x=0 the density should be the largest.
        assert density[3] == max(density)

    def test_density_recovers_standard_normal_approximately(self):
        from pricebook.numerical._fourier import CharacteristicFunction
        cf = CharacteristicFunction(lambda u: np.exp(-0.5 * u ** 2))
        x = np.array([0.0])
        d = cf.density(x, n_quad=200)
        # True N(0,1) density at 0 = 1/√(2π) ≈ 0.3989.
        assert d[0] == pytest.approx(1 / math.sqrt(2 * math.pi), abs=5e-3)


# ============================================================
# T1.3 — integrate_2d uses (x, y) order, not (y, x)
# ============================================================

class TestT1_3_Integrate2D:
    def test_non_symmetric_integrand(self):
        """∫₀³ ∫₀¹ x dy dx = 3 × 1 × (3/2) = 4.5
        Pre-fix returned 1.5 because dblquad called f(y, x) → integrated y."""
        from pricebook.numerical._integrate import integrate_2d
        result = integrate_2d(lambda x, y: x, (0.0, 3.0), (0.0, 1.0))
        assert result.value == pytest.approx(4.5, abs=1e-8)

    def test_symmetric_unaffected(self):
        """∫₀¹ ∫₀¹ (x+y) dy dx = 1 — symmetric integrand was correct pre-fix."""
        from pricebook.numerical._integrate import integrate_2d
        result = integrate_2d(lambda x, y: x + y, (0.0, 1.0), (0.0, 1.0))
        assert result.value == pytest.approx(1.0, abs=1e-8)

    def test_callable_y_range_uses_x_correctly(self):
        """y_range as callable depends on x. ∫₀¹ ∫₀ˣ x dy dx = ∫₀¹ x² dx = 1/3."""
        from pricebook.numerical._integrate import integrate_2d
        result = integrate_2d(
            lambda x, y: x,
            x_range=(0.0, 1.0),
            y_range=lambda x: (0.0, x),
        )
        assert result.value == pytest.approx(1.0 / 3.0, abs=1e-8)


# ============================================================
# T1.5 — Tian binomial method actually runs Tian (not CRR)
# ============================================================

class TestT1_5_TianParams:
    def test_tian_params_match_tian_1993_formulas(self):
        """Recompute the Tian (1993) JFQA formulas and verify the impl matches."""
        from pricebook.numerical._trees import _tian_params
        r, q, vol, dt = 0.05, 0.0, 0.2, 0.01
        u, d, p, disc = _tian_params(r, q, vol, dt)

        M = math.exp((r - q) * dt)
        V = math.exp(vol ** 2 * dt)
        sqrt_disc = math.sqrt(V ** 2 + 2 * V - 3)
        u_expected = 0.5 * M * V * (V + 1 + sqrt_disc)
        d_expected = 0.5 * M * V * (V + 1 - sqrt_disc)

        assert u == pytest.approx(u_expected, rel=1e-12)
        assert d == pytest.approx(d_expected, rel=1e-12)
        assert 0 < p < 1
        assert 0 < d < 1 < u   # tree must straddle 1

    def test_tian_no_longer_falls_back_to_crr_for_typical_params(self):
        """Pre-fix: V² + 2V − 3 was almost always negative for normal r/q/σ/dt,
        triggering the CRR fallback. Post-fix: Tian gives its own (u, d)."""
        from pricebook.numerical._trees import _tian_params, _crr_params
        u_tian, d_tian, _, _ = _tian_params(0.05, 0.0, 0.2, 0.01)
        u_crr, d_crr, _, _ = _crr_params(0.05, 0.0, 0.2, 0.01)
        # Should NOT be equal (Tian and CRR give different lattice params).
        assert u_tian != pytest.approx(u_crr, rel=1e-6)


# ============================================================
# T1.7 — MCEngine actually uses Milstein when scheme="milstein"
# ============================================================

class TestT1_7_MilsteinScheme:
    def test_milstein_correction_term_actually_applied(self):
        """End-to-end: with scheme='milstein' AND `diffusion_deriv` supplied,
        the path-evolution result must differ from Euler. Pre-fix it didn't
        because (a) both branches dispatched euler_step (copy-paste typo) and
        (b) `diffusion_deriv` was never plumbed through ProcessSpec — so even
        once the dispatch is corrected, the Milstein correction never fired.
        Fix wires both."""
        from pricebook.models.mc_engine import ProcessSpec, MCEngine, TimeGrid

        # GBM: σ(x, t) = σ·x  →  σ'(x) = σ.
        sigma = 0.5    # large vol so Milstein correction is visible.
        proc = ProcessSpec(
            x0=np.array([100.0]),
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: sigma * x,
            diffusion_deriv=lambda x, t: sigma * np.ones_like(x),
            n_factors=1,
        )
        grid = TimeGrid(times=np.linspace(0.0, 1.0, 11))   # 10 steps

        # Euler path
        eng_e = MCEngine(process=proc, time_grid=grid, n_paths=200, seed=42, scheme="euler")
        paths_e = eng_e.generate_paths()

        # Milstein path — same seed, same process, same grid → should differ.
        eng_m = MCEngine(process=proc, time_grid=grid, n_paths=200, seed=42, scheme="milstein")
        paths_m = eng_m.generate_paths()

        # Mean terminal value differs by the Milstein correction.
        # Pre-fix the two paths would be IDENTICAL.
        diff = float(np.mean(paths_m[:, -1]) - np.mean(paths_e[:, -1]))
        assert abs(diff) > 1e-6, (
            f"Milstein and Euler produced identical paths "
            f"(Δmean(S_T) = {diff:.3e}). Either dispatch or correction-term "
            f"is broken."
        )

    def test_milstein_without_diffusion_deriv_warns(self):
        """If the user requests Milstein without `diffusion_deriv`, the engine
        must warn explicitly (not silently run Euler)."""
        import warnings
        from pricebook.models.mc_engine import ProcessSpec, MCEngine, TimeGrid

        proc = ProcessSpec(
            x0=np.array([100.0]),
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            # No diffusion_deriv — Milstein cannot fire correctly.
            n_factors=1,
        )
        grid = TimeGrid(times=np.linspace(0.0, 1.0, 5))
        engine = MCEngine(process=proc, time_grid=grid, n_paths=10, seed=42, scheme="milstein")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = engine.generate_paths()

        assert any("Milstein" in str(w.message) and "Euler" in str(w.message)
                   for w in caught), (
            "Expected RuntimeWarning about Milstein degrading to Euler"
        )
