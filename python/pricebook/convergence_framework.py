"""Convergence testing framework for SDE discretisation schemes.

Given an SDE and a discretisation scheme, systematically computes
strong and weak errors at multiple step sizes, estimates the empirical
convergence order, and compares against the theoretical expectation.

* :func:`strong_convergence_study` — run a scheme at multiple dt, compute strong errors.
* :func:`weak_convergence_study` — run a scheme at multiple dt, compute weak errors.
* :func:`scheme_comparison` — compare multiple schemes on the same SDE.
* :class:`ConvergenceStudyResult` — full study output with order estimate.

References:
    Kloeden & Platen, *Numerical Solution of SDEs*, Ch. 9-10.
    Higham, *An Algorithmic Introduction to Numerical Simulation of SDEs*, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pricebook.numerical_safety import convergence_rate, ConvergenceResult


@dataclass
class ConvergenceStudyResult:
    """Result of a convergence study."""
    scheme_name: str
    step_sizes: list[float]
    errors: list[float]
    estimated_order: float
    expected_order: float
    is_consistent: bool
    error_type: str  # "strong" or "weak"


# ---- Strong convergence study ----

def strong_convergence_study(
    scheme_name: str,
    simulate: Callable[[int, int, int | None], np.ndarray],
    reference: Callable[[int, int | None], np.ndarray],
    T: float,
    steps_list: list[int],
    n_paths: int = 10_000,
    expected_order: float = 0.5,
    seed: int = 42,
) -> ConvergenceStudyResult:
    """Run a strong convergence study for an SDE scheme.

    Computes E[|X_scheme(T) − X_ref(T)|] at multiple step sizes
    and estimates the convergence order via log-log regression.

    Args:
        scheme_name: label for the scheme.
        simulate: function(n_steps, n_paths, seed) → (n_paths,) terminal values.
        reference: function(n_paths, seed) → (n_paths,) reference terminal values.
            Must use the SAME Brownian draws as simulate for strong error.
        T: time horizon.
        steps_list: list of step counts (increasing).
        n_paths: number of MC paths.
        expected_order: theoretical strong order.
        seed: random seed (both simulate and reference must respect it).

    Returns:
        :class:`ConvergenceStudyResult`.
    """
    errors = []
    dts = []

    for n_steps in steps_list:
        dt = T / n_steps
        dts.append(dt)

        sim_terminal = simulate(n_steps, n_paths, seed)
        ref_terminal = reference(n_paths, seed)

        err = float(np.mean(np.abs(sim_terminal - ref_terminal)))
        errors.append(err)

    cr = convergence_rate(dts, errors, expected_order, order_tol=0.4)

    return ConvergenceStudyResult(
        scheme_name=scheme_name,
        step_sizes=dts,
        errors=errors,
        estimated_order=cr.estimated_order,
        expected_order=expected_order,
        is_consistent=cr.is_consistent,
        error_type="strong",
    )


# ---- Weak convergence study ----

def weak_convergence_study(
    scheme_name: str,
    simulate: Callable[[int, int, int | None], np.ndarray],
    reference_value: float,
    T: float,
    steps_list: list[int],
    n_paths: int = 100_000,
    expected_order: float = 1.0,
    seed: int = 42,
) -> ConvergenceStudyResult:
    """Run a weak convergence study for an SDE scheme.

    Computes |E[f(X_scheme(T))] − E[f(X_ref(T))]| at multiple step sizes.
    The default test function is f(x) = x (terminal mean).

    Args:
        scheme_name: label for the scheme.
        simulate: function(n_steps, n_paths, seed) → (n_paths,) terminal values.
        reference_value: analytical or fine-grid E[X(T)].
        T: time horizon.
        steps_list: list of step counts (increasing).
        n_paths: number of MC paths.
        expected_order: theoretical weak order.
        seed: random seed.

    Returns:
        :class:`ConvergenceStudyResult`.
    """
    errors = []
    dts = []

    for n_steps in steps_list:
        dt = T / n_steps
        dts.append(dt)

        sim_terminal = simulate(n_steps, n_paths, seed)
        err = abs(float(np.mean(sim_terminal)) - reference_value)
        errors.append(max(err, 1e-15))

    cr = convergence_rate(dts, errors, expected_order, order_tol=0.5)

    return ConvergenceStudyResult(
        scheme_name=scheme_name,
        step_sizes=dts,
        errors=errors,
        estimated_order=cr.estimated_order,
        expected_order=expected_order,
        is_consistent=cr.is_consistent,
        error_type="weak",
    )


# ---- Scheme comparison ----

@dataclass
class SchemeComparisonResult:
    """Comparison of multiple schemes on the same SDE."""
    studies: list[ConvergenceStudyResult]
    best_scheme: str
    best_order: float


def scheme_comparison(
    studies: list[ConvergenceStudyResult],
) -> SchemeComparisonResult:
    """Compare multiple convergence studies and identify the best scheme.

    The best scheme is the one with the highest estimated convergence order.
    """
    if not studies:
        return SchemeComparisonResult([], "", 0.0)

    best = max(studies, key=lambda s: s.estimated_order)
    return SchemeComparisonResult(
        studies=studies,
        best_scheme=best.scheme_name,
        best_order=best.estimated_order,
    )
