"""Solver-primitive layer — the single origin of `SolveReport`.

Phase 0 of the calibration "capture-not-reconstruct" migration (OPEN.md §0c).

A calibration's **convergence verdict, iteration count, tolerance and RNG seed**
must be *captured from the optimiser that actually ran* — never re-derived from
the fitted result after the fact. Re-derivation is how a record ends up claiming
a fit converged when it did not (e.g. ``converged = rmse < 0.01`` with an
invented threshold). These primitives wrap the handful of solve idioms the
calibrators use and return, alongside the solution, a `SolveReport` carrying
exactly those captured facts.

The Phase-1 builder (`model_calibration_record`) takes a `SolveReport` as a
*required* argument, so a calibrator can neither omit nor fabricate it.
`SolveReport` is the **only** type that records ``converged``; a Phase-4 grep
gate keeps ``converged=`` literals out of calibrator modules.

Each primitive returns ``(solution, SolveReport)``. The ``algorithm`` string is
left human-readable here (``"nelder_mead"``, ``"differential_evolution+L-BFGS-B"``);
it is canonicalised to the snake_case audit key when it reaches `OptimiserSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import scipy.optimize as _opt


@dataclass(frozen=True, slots=True)
class SolveReport:
    """The optimiser facts captured at the moment of solving.

    Built only by the primitives below (or the two honest classmethods). It maps
    directly onto the record's ``OptimiserSpec`` / ``OptimiserRun`` in Phase 1.
    """

    algorithm: str
    converged: bool
    iterations: int                 # iterations/evals the optimiser actually took
    tolerance: float | None = None
    seed: int | None = None
    max_iterations: int = 0         # the configured cap (0 = unknown → builder uses `iterations`)

    @classmethod
    def analytic(cls) -> "SolveReport":
        """A closed-form evaluation — no iteration, no optimiser.

        Honest about itself: the reprice error of the formula lives in the
        record's residuals; this does not fake an iterative convergence verdict.
        """
        return cls(algorithm="analytic", converged=True, iterations=0, tolerance=None)

    @classmethod
    def external(
        cls,
        *,
        algorithm: str,
        converged: bool,
        iterations: int = 0,
        tolerance: float | None = None,
        seed: int | None = None,
        max_iterations: int = 0,
    ) -> "SolveReport":
        """Escape hatch for a black-box optimiser this layer does not wrap
        (e.g. the pricebook ``statistics.optimization.minimize`` wrapper). The
        caller passes through whatever metadata the optimiser exposed — still a
        single, explicit capture point, not a reconstruction."""
        return cls(
            algorithm=str(algorithm), converged=bool(converged),
            iterations=int(iterations), tolerance=tolerance,
            seed=None if seed is None else int(seed),
            max_iterations=int(max_iterations),
        )


def _iters(res: Any) -> int:
    """Best available iteration/eval count from a scipy result."""
    nit = getattr(res, "nit", None)
    if nit is not None:
        return int(nit)
    return int(getattr(res, "nfev", 0))


def minimize_solve(
    objective: Callable[[np.ndarray], float],
    x0: Sequence[float],
    *,
    method: str,
    tol: float = 1e-8,
    bounds: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, SolveReport]:
    """`scipy.optimize.minimize`, capturing ``success`` / ``nit``.

    Covers SABR, Hull-White, joint equity/credit, dividend, dispersion.
    """
    res = _opt.minimize(objective, np.asarray(x0, float), method=method,
                        tol=tol, bounds=bounds, **kwargs)
    return res.x, SolveReport(
        algorithm=str(method), converged=bool(res.success),
        iterations=_iters(res), tolerance=tol,
    )


def least_squares_solve(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    *,
    tol: float = 1e-10,
    bounds: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, SolveReport]:
    """`scipy.optimize.least_squares` (TRF), capturing ``success`` / ``nfev``."""
    kw: dict[str, Any] = {"ftol": tol, "xtol": tol, **kwargs}
    if bounds is not None:
        kw["bounds"] = bounds
    res = _opt.least_squares(residual_fn, np.asarray(x0, float), **kw)
    return res.x, SolveReport(
        algorithm="least_squares", converged=bool(res.success),
        iterations=int(res.nfev), tolerance=tol,
    )


def global_local_solve(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int | None = None,
    tol: float = 1e-8,
    local_method: str = "L-BFGS-B",
    de_maxiter: int = 100,
    **kwargs: Any,
) -> tuple[np.ndarray, SolveReport]:
    """Global `differential_evolution` then a local polish — the two-stage
    idiom used by G2++ and the jump models. Captures the ``seed`` (these are
    stochastic) and the combined iteration count.
    """
    de = _opt.differential_evolution(objective, bounds, seed=seed, tol=tol,
                                     maxiter=de_maxiter, **kwargs)
    local = _opt.minimize(objective, de.x, method=local_method, bounds=bounds, tol=tol)
    best = local.x if local.fun <= de.fun else de.x
    return best, SolveReport(
        algorithm=f"differential_evolution+{local_method}",
        converged=bool(de.success or local.success),
        iterations=_iters(de) + _iters(local),
        tolerance=tol,
        seed=seed,
    )


def brentq_solve(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> tuple[float, SolveReport]:
    """`scipy.optimize.brentq` with ``full_output``, capturing convergence and
    iteration count. The scalar-root idiom (e.g. Jarrow-Yildirim)."""
    x, r = _opt.brentq(f, a, b, xtol=tol, maxiter=max_iter, full_output=True)
    return x, SolveReport(
        algorithm="brentq", converged=bool(r.converged),
        iterations=int(r.iterations), tolerance=tol,
    )


def particle_solve(
    run: Callable[[np.random.Generator], Any],
    *,
    seed: int,
    n_steps: int,
    algorithm: str = "particle",
    converged: bool = True,
) -> tuple[Any, SolveReport]:
    """Run a Monte-Carlo / particle loop under a seeded RNG and capture the
    seed — so a stochastic calibration (FX-SLV leverage) is reproducible from
    its record. ``run(rng)`` executes the loop and returns its result.

    Convergence for an MC method is "it ran to completion"; the *fit quality*
    lives in the record's residuals, not in this flag.
    """
    rng = np.random.default_rng(seed)
    result = run(rng)
    return result, SolveReport(
        algorithm=algorithm, converged=bool(converged),
        iterations=int(n_steps), tolerance=None, seed=int(seed),
    )
