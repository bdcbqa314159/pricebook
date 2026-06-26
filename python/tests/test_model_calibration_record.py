"""Phase 1 — the single model-calibrator builder.

Pins that `model_calibration_record` reads optimiser facts straight off the
`SolveReport` (no re-derivation), and demonstrates the end-to-end clean pattern
a future calibrator follows: primitive → SolveReport → builder → record → DB.
"""

import numpy as np
import pytest

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    SolveReport,
    model_calibration_record,
)
from pricebook.db.db import PricebookDB


def test_builder_reads_optimiser_facts_off_the_report():
    solve = SolveReport(algorithm="Nelder-Mead", converged=True, iterations=42,
                        tolerance=1e-10, seed=7)
    rec = model_calibration_record(
        model_class="demo_model",
        parameters={"a": 1.0, "b": 2.0},
        residuals=[0.001, -0.002],
        quotes_fitted=["k_90", "k_100"],
        solve=solve,
    )
    assert isinstance(rec, CalibrationResult)
    # captured, not invented:
    assert rec.optimiser_run.converged is True
    assert rec.optimiser_run.iterations == 42
    assert rec.optimiser_run.spec.seed == 7
    assert rec.optimiser_run.spec.tolerance == 1e-10
    # algorithm canonicalised to the audit vocabulary:
    assert rec.optimiser_run.spec.algorithm == "nelder_mead"
    assert rec.fit.model_class == "demo_model"
    assert len(rec.fit.residuals) == len(rec.fit.quotes_fitted) == 2


def test_non_convergence_warning_is_added_centrally():
    solve = SolveReport(algorithm="least_squares", converged=False, iterations=500, tolerance=1e-8)
    rec = model_calibration_record(
        model_class="stalled", parameters={"x": 0.0}, residuals=[1.0], quotes_fitted=["q"],
        solve=solve,
    )
    assert rec.optimiser_run.converged is False
    assert any("did not converge" in w for w in rec.diagnostics.warnings)


def test_caller_diagnostics_preserved_and_merged():
    solve = SolveReport(algorithm="brentq", converged=False, iterations=3)
    diag = CalibrationDiagnostics(extra={"rmse": 0.05}, warnings=("custom note",))
    rec = model_calibration_record(
        model_class="m", parameters={"x": 1.0}, residuals=[0.05], quotes_fitted=["q"],
        solve=solve, diagnostics=diag, objective=ObjectiveKind.SSE,
    )
    assert rec.diagnostics.extra["rmse"] == 0.05
    assert "custom note" in rec.diagnostics.warnings
    assert any("did not converge" in w for w in rec.diagnostics.warnings)  # auto-appended too


def test_unlabelled_residuals_rejected():
    solve = SolveReport.analytic()
    with pytest.raises(ValueError, match="quotes_fitted is required"):
        model_calibration_record(model_class="m", parameters={}, residuals=[1.0],
                                 quotes_fitted=[], solve=solve)


def test_end_to_end_clean_pattern_persists():
    """The shape every new calibrator follows — run an optimiser, CAPTURE its
    verdict in SolveReport.external, build the record, persist. No eager/lazy
    fork, no hand-rolled skeleton, no invented convergence."""
    import scipy.optimize as opt
    # 1. run the optimiser (toy 2-target least-squares-ish objective)
    target = np.array([1.5, -0.5])
    result = opt.minimize(lambda v: float(np.sum((v - target) ** 2)),
                          [0.0, 0.0], method="Nelder-Mead", tol=1e-10)
    # 2. CAPTURE its verdict, then build the record straight from the report
    solve = SolveReport.external(algorithm="nelder_mead", converged=bool(result.success),
                                 iterations=int(result.nit), tolerance=1e-10)
    residuals = (result.x - target).tolist()
    rec = model_calibration_record(
        model_class="toy_smile",
        parameters={"p0": float(result.x[0]), "p1": float(result.x[1])},
        residuals=residuals,
        quotes_fitted=["t0", "t1"],
        solve=solve,
        objective=ObjectiveKind.SSE,
    )
    assert rec.optimiser_run.spec.algorithm == "nelder_mead"
    assert rec.optimiser_run.converged is True
    assert rec.optimiser_run.iterations > 0  # real, from the optimiser
    # 3. persist + round-trip through the one sink
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(rec)
        assert db.load_calibration(cid) == rec
