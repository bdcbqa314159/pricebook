"""`SolveReport` — the captured optimiser facts of a calibration.

A calibration's **convergence verdict, iteration count, tolerance and RNG seed**
must be *captured from the optimiser that actually ran* — never re-derived from
the fitted result after the fact (e.g. ``converged = rmse < 0.01`` with an
invented threshold, which is how a record ends up claiming a fit converged when
it did not).

This layer's job is to **record** a solve, not to run it — optimiser setups are
irreducibly bespoke (scipy, the pricebook `minimize` wrapper, two-stage
differential-evolution, hand-written particle loops), so the calibrator owns its
solve and hands the result here. Two classmethods are the intended entry points
(the bare constructor stays available for tests / deserialisation, but coerces
nothing — prefer these, which validate and normalise their inputs):

* ``SolveReport.external(...)`` — wrap an already-run optimiser's result; pass
  ``converged=None`` when no optimiser ran (a reconstructed, hand-built record).
* ``SolveReport.analytic()`` — a closed-form evaluation, no iteration.

The builder `model_calibration_record` takes a `SolveReport` as a *required*
argument, so a calibrator can neither omit nor fabricate it. `SolveReport` is the
**only** type that records ``converged``. The ``algorithm`` string is left
human-readable here; `OptimiserSpec` canonicalises it to the snake_case audit key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolveReport:
    """The optimiser facts captured at the moment of solving.

    Maps directly onto the record's ``OptimiserSpec`` / ``OptimiserRun``.
    """

    algorithm: str
    converged: bool | None          # None = not captured (reconstructed records)
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
        converged: bool | None,
        iterations: int = 0,
        tolerance: float | None = None,
        seed: int | None = None,
        max_iterations: int = 0,
    ) -> "SolveReport":
        """Capture an already-run optimiser's result. The caller passes through
        whatever metadata the optimiser exposed — including ``converged=None``
        when no optimiser ran (a reconstructed record), so convergence is never
        guessed from a threshold."""
        return cls(
            algorithm=str(algorithm),
            converged=None if converged is None else bool(converged),
            iterations=int(iterations), tolerance=tolerance,
            seed=None if seed is None else int(seed),
            max_iterations=int(max_iterations),
        )
