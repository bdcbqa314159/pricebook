"""Phase 0 — solver-primitive layer: SolveReport is captured, not invented.

These tests pin the one guarantee the layer exists for: convergence / iterations
/ seed come from the optimiser that actually ran.
"""

import numpy as np
import pytest

from pricebook.calibration import (
    SolveReport,
    brentq_solve,
    global_local_solve,
    least_squares_solve,
    minimize_solve,
    particle_solve,
)


def test_analytic_report_is_honest():
    r = SolveReport.analytic()
    assert r.algorithm == "analytic"
    assert r.converged and r.iterations == 0 and r.tolerance is None and r.seed is None


def test_external_passes_through_metadata():
    r = SolveReport.external(algorithm="pb_minimize", converged=False, iterations=7, seed=3)
    assert (r.algorithm, r.converged, r.iterations, r.seed) == ("pb_minimize", False, 7, 3)


def test_minimize_solve_captures_success_and_iterations():
    x, r = minimize_solve(lambda v: (v[0] - 1.0) ** 2 + (v[1] + 2.0) ** 2,
                          [0.0, 0.0], method="Nelder-Mead", tol=1e-10)
    assert np.allclose(x, [1.0, -2.0], atol=1e-3)
    assert r.algorithm == "Nelder-Mead"
    assert r.converged is True
    assert r.iterations > 0          # real count from the optimiser, not a stub 0
    assert r.tolerance == 1e-10


def test_least_squares_solve_captures_report():
    x, r = least_squares_solve(lambda v: np.array([v[0] - 3.0, v[1] - 4.0]), [0.0, 0.0])
    assert np.allclose(x, [3.0, 4.0], atol=1e-6)
    assert r.algorithm == "least_squares" and r.converged and r.iterations > 0


def test_brentq_solve_captures_convergence_and_iterations():
    x, r = brentq_solve(lambda z: z ** 3 - 8.0, 0.0, 5.0)
    assert x == pytest.approx(2.0, abs=1e-9)
    assert r.algorithm == "brentq" and r.converged and r.iterations > 0


def test_global_local_solve_captures_seed():
    x, r = global_local_solve(lambda v: (v[0] - 0.5) ** 2, [(-2.0, 2.0)], seed=7)
    assert x[0] == pytest.approx(0.5, abs=1e-4)
    assert r.algorithm == "differential_evolution+L-BFGS-B"
    assert r.seed == 7               # stochastic → seed captured for reproducibility
    assert r.iterations > 0


def test_particle_solve_captures_seed_and_is_reproducible():
    def run(rng):
        return float(rng.standard_normal(1000).mean())

    a, ra = particle_solve(run, seed=42, n_steps=1000)
    b, rb = particle_solve(run, seed=42, n_steps=1000)
    assert a == b                    # same seed → same draw
    assert ra.seed == 42 and ra.iterations == 1000 and ra.algorithm == "particle"
