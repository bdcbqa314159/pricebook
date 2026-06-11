"""Tests for `CalibrationResult` integration on curve bootstrap (G1 P1 Slice 5).

Covers:
- `DiscountCurve.calibration_result` defaults to None on direct construction.
- `bootstrap()` populates calibration_result on the returned curve.
- `global_bootstrap()` populates calibration_result on the returned curve.
- `multicurve_newton()` populates calibration_result on both OIS and projection
  curves AND on the MultiCurveResult.
- Backward compatibility: hand-constructed MultiCurveResult without
  calibration_result still works (default = None); `to_calibration_result()`
  builds on demand.
- `to_dict()` exposes `calibration_id`.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.calibration import CalibrationResult, ObjectiveKind
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention
from pricebook.curves.bootstrap import bootstrap
from pricebook.curves.global_solver import global_bootstrap
from pricebook.curves.multicurve_solver import MultiCurveResult, multicurve_newton


REF = date(2026, 6, 11)


def _deposits_and_swaps():
    deposits = [(REF + timedelta(days=91), 0.0525)]
    swaps = [
        (REF + timedelta(days=365 * 1), 0.048),
        (REF + timedelta(days=365 * 5), 0.041),
        (REF + timedelta(days=365 * 10), 0.0395),
    ]
    return deposits, swaps


# ============================================================
# DiscountCurve default
# ============================================================

class TestDiscountCurveDefault:
    def test_calibration_result_defaults_to_none(self):
        """Curves built directly (not by bootstrap) have no calibration provenance."""
        c = DiscountCurve.flat(REF, 0.04)
        assert c.calibration_result is None

    def test_calibration_result_can_be_set(self):
        c = DiscountCurve.flat(REF, 0.04)
        # Manual attachment — simulates what bootstrap does
        c.calibration_result = "marker"
        assert c.calibration_result == "marker"


# ============================================================
# bootstrap() — sequential bootstrap
# ============================================================

class TestBootstrapCalibrationResult:
    def test_populated(self):
        deposits, swaps = _deposits_and_swaps()
        c = bootstrap(REF, deposits, swaps)
        assert c.calibration_result is not None
        assert isinstance(c.calibration_result, CalibrationResult)

    def test_model_class_and_optimiser(self):
        deposits, swaps = _deposits_and_swaps()
        c = bootstrap(REF, deposits, swaps)
        cr = c.calibration_result
        assert cr.model_class == "discount_curve_bootstrap"
        assert cr.optimiser.algorithm == "brentq-sequential"
        assert cr.converged is True

    def test_residuals_essentially_zero(self):
        """Sequential bootstrap is exact-fit by construction — residuals must be ~0."""
        deposits, swaps = _deposits_and_swaps()
        c = bootstrap(REF, deposits, swaps)
        cr = c.calibration_result
        # Each residual at machine precision
        for r in cr.residuals:
            assert abs(r) < 1e-10

    def test_parameters_are_pillar_dfs(self):
        deposits, swaps = _deposits_and_swaps()
        c = bootstrap(REF, deposits, swaps)
        cr = c.calibration_result
        # Number of parameters = number of pillar dates
        assert len(cr.parameters) == len(c.pillar_dates)
        # Values match the curve's df at each pillar
        for key, val in cr.parameters.items():
            # Key is f"df(YYYY-MM-DD)"
            iso = key[len("df("):-1]
            d = date.fromisoformat(iso)
            assert val == pytest.approx(c.df(d), abs=1e-12)

    def test_quotes_fitted_named_by_instrument(self):
        deposits, swaps = _deposits_and_swaps()
        c = bootstrap(REF, deposits, swaps)
        cr = c.calibration_result
        # 1 deposit + 3 swaps
        assert len(cr.quotes_fitted) == 4
        assert any(q.startswith("deposit_") for q in cr.quotes_fitted)
        assert sum(1 for q in cr.quotes_fitted if q.startswith("swap_")) == 3

    def test_diagnostics_record_input_counts(self):
        deposits, swaps = _deposits_and_swaps()
        c = bootstrap(REF, deposits, swaps)
        cr = c.calibration_result
        assert cr.diagnostics.extra["n_deposits"] == len(deposits)
        assert cr.diagnostics.extra["n_swaps"] == len(swaps)

    def test_unique_id_per_call(self):
        deposits, swaps = _deposits_and_swaps()
        c1 = bootstrap(REF, deposits, swaps)
        c2 = bootstrap(REF, deposits, swaps)
        assert c1.calibration_result.id != c2.calibration_result.id


# ============================================================
# global_bootstrap()
# ============================================================

class TestGlobalBootstrapCalibrationResult:
    def test_populated(self):
        deposits, swaps = _deposits_and_swaps()
        c = global_bootstrap(REF, deposits, swaps)
        assert c.calibration_result is not None
        cr = c.calibration_result
        assert cr.model_class == "discount_curve_global"
        assert cr.optimiser.algorithm == "newton-global"
        assert cr.converged is True

    def test_residuals_below_solver_tol(self):
        deposits, swaps = _deposits_and_swaps()
        c = global_bootstrap(REF, deposits, swaps, tol=1e-10)
        cr = c.calibration_result
        # Newton should converge well below tol
        assert max(abs(r) for r in cr.residuals) < 1e-9


# ============================================================
# multicurve_newton() — both curves + result carry the calibration
# ============================================================

class TestMulticurveNewtonCalibrationResult:
    def _setup(self):
        # Build a small two-curve calibration problem.
        # Mimic a single deposit on OIS and a single deposit on projection.
        ref = REF
        ois_pillars = [ref + timedelta(days=365)]
        proj_pillars = [ref + timedelta(days=365)]
        ois_instruments = [{"type": "deposit", "maturity": ois_pillars[0], "rate": 0.04}]
        proj_instruments = [{"type": "deposit", "maturity": proj_pillars[0], "rate": 0.05}]
        return ref, ois_instruments, proj_instruments, ois_pillars, proj_pillars

    def test_populated_on_both_curves_and_result(self):
        ref, oi, pi, op, pp = self._setup()
        r = multicurve_newton(ref, oi, pi, op, pp,
                              day_count=DayCountConvention.ACT_360,
                              tol=1e-10, max_iter=50)
        # All three locations carry the same calibration result
        assert r.calibration_result is not None
        assert r.ois_curve.calibration_result is r.calibration_result
        assert r.projection_curve.calibration_result is r.calibration_result

    def test_parameters_partition(self):
        ref, oi, pi, op, pp = self._setup()
        r = multicurve_newton(ref, oi, pi, op, pp,
                              day_count=DayCountConvention.ACT_360,
                              tol=1e-10, max_iter=50)
        cr = r.calibration_result
        assert cr.model_class == "multicurve"
        ois_keys = [k for k in cr.parameters if k.startswith("ois_df(")]
        proj_keys = [k for k in cr.parameters if k.startswith("proj_df(")]
        assert len(ois_keys) == len(op)
        assert len(proj_keys) == len(pp)

    def test_to_dict_has_calibration_id(self):
        ref, oi, pi, op, pp = self._setup()
        r = multicurve_newton(ref, oi, pi, op, pp,
                              day_count=DayCountConvention.ACT_360,
                              tol=1e-10, max_iter=50)
        d = r.to_dict()
        assert d["calibration_id"] == str(r.calibration_result.id)


# ============================================================
# MultiCurveResult back-compat
# ============================================================

class TestMultiCurveResultBackCompat:
    def test_hand_constructed_no_calibration_result(self):
        c = DiscountCurve.flat(REF, 0.04)
        r = MultiCurveResult(
            ois_curve=c, projection_curve=c,
            residual=0.0, n_iterations=1, jacobian=None,
        )
        assert r.calibration_result is None

    def test_on_demand_to_calibration_result(self):
        c = DiscountCurve.flat(REF, 0.04)
        r = MultiCurveResult(
            ois_curve=c, projection_curve=c,
            residual=1e-12, n_iterations=3, jacobian=None,
        )
        cr = r.to_calibration_result()
        assert isinstance(cr, CalibrationResult)
        assert cr.model_class == "multicurve"
        assert cr.iterations == 3
        assert cr.converged is True

    def test_to_dict_has_none_when_unpopulated(self):
        c = DiscountCurve.flat(REF, 0.04)
        r = MultiCurveResult(
            ois_curve=c, projection_curve=c,
            residual=0.0, n_iterations=1, jacobian=None,
        )
        d = r.to_dict()
        assert d["calibration_id"] is None
