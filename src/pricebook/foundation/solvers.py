"""Least squares, behind our own API (L0).

**A thin adapter over scipy, on the seam `distributions.py` established.** The justification is
architectural, not rule-of-two (which is about domain abstractions, not meta-patterns): scipy stays
behind our own API so models and calibrators are scipy-free, oracles test the finance rather than
scipy's numerics, and a port has one place to swap. Cowork ruled this explicitly in thread 0035
after I offered the weaker "the adapter shape has two instances" argument.

**The signature is exactly its one caller's need and no more** -- a residual callable, a start,
optional bounds. None of `least_squares`' wider surface (method, loss, jac, x_scale, tr_solver) is
exposed, because nothing here needs it. If a knob appears that the caller does not use, that is the
smell to stop and reconsider the seam.

**Non-convergence is a reported value, not an exception and not a silent number.** A solver that
returns a plausible-looking `x` after failing is how a calibration reports success on garbage; the
`converged` flag is what lets a calibrator return a `PricingFailure` instead.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py used `scipy.optimize.minimize`
          (Nelder-Mead) inline, with no seam -- purified to a bounded least-squares behind our API
  source: scipy.optimize.least_squares (Trust Region Reflective), pinned in pyproject since the
          pinned version is part of the reproducibility contract
  oracle: tests/L0/test_solvers.py -- a hand-solvable minimum (3, -1); the overdetermined answer
          is the mean; bounds bind; a NaN residual reports converged=False
  slice:  tranche 6 -- calibrating Jarrow-Yildirim (its first consumer)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

Residuals = Callable[[Sequence[float]], Sequence[float]]


@dataclass(frozen=True)
class SolveOutcome:
    """What a least-squares solve reports back. `cost` is half the sum of squared residuals
    (scipy's convention, kept rather than rescaled so the number means what its source says)."""

    x: tuple[float, ...]
    cost: float
    iterations: int
    converged: bool


def least_squares_fit(
    residuals: Residuals,
    start: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> SolveOutcome:
    """Minimise the sum of squared `residuals` from `start`, optionally within `bounds`."""
    lo, hi = bounds if bounds is not None else (-np.inf, np.inf)

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.asarray(residuals(x), dtype=float)

    x0 = np.asarray(start, dtype=float)
    try:
        out = least_squares(evaluate, x0, bounds=(lo, hi))
    except ValueError:
        # scipy *raises* when the residuals are non-finite at the start point rather than
        # reporting it. Converting that to a value is precisely this adapter's job: the caller
        # asked for a fit outcome, and "could not start" is one. Narrow on purpose -- only the
        # non-finite-start ValueError is expected here; anything else is a real defect and must
        # keep propagating.
        return SolveOutcome(
            x=tuple(float(v) for v in x0), cost=math.inf, iterations=0, converged=False
        )
    x = tuple(float(v) for v in out.x)
    cost = float(out.cost)
    # `status > 0` is scipy's "a termination criterion was satisfied"; 0 means the iteration limit
    # was hit and negative means the step failed. A non-finite cost is non-convergence whatever
    # the status says -- scipy can report success having walked into a NaN.
    converged = bool(out.status > 0) and math.isfinite(cost) and all(math.isfinite(v) for v in x)
    return SolveOutcome(x=x, cost=cost, iterations=int(out.nfev), converged=converged)
