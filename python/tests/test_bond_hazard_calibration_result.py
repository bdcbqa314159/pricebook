"""Tests for the `CalibrationResult` integration on `HazardBootstrapResult` (G1 P1 Slice 2).

Covers:
- Sequential and global bootstrap entry points both populate `.calibration_result`.
- The populated `CalibrationResult` has the expected shape (model_class, parameters,
  residuals, weights, optimiser, iterations, converged).
- Global + Tikhonov stores `lam` in `optimiser.extra` and `roughness` in
  `diagnostics.extra`.
- Each invocation produces a fresh `id`.
- `to_calibration_result()` returns the stored instance when populated.
- `to_calibration_result()` builds on-demand when the result was hand-constructed
  (back-compat path).
- `to_dict` includes the `calibration_id` key.
- Existing API (rmse_bp, max_error_bp, pillar_hazards) is unchanged.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.bond_hazard_bootstrap import (
    BondInput,
    HazardBootstrapResult,
    bootstrap_hazard_from_bonds,
    _price_risky_bond,
)


REF = date(2026, 6, 11)


def _flat_rf():
    return DiscountCurve.flat(REF, 0.04)


def _truth_survival():
    dates = [REF + timedelta(days=365 * y) for y in [3, 7, 15]]
    survs = []
    cum = 1.0
    prev_t = 0.0
    for t_y, h in zip([3.0, 7.0, 15.0], [0.02, 0.03, 0.04]):
        cum *= math.exp(-h * (t_y - prev_t))
        survs.append(cum)
        prev_t = t_y
    return SurvivalCurve(REF, dates, survs)


def _well_spaced_bonds() -> list[BondInput]:
    rf = _flat_rf()
    truth = _truth_survival()
    specs = [(1.0, 0.040), (3.0, 0.045), (5.0, 0.050), (10.0, 0.055)]
    out = []
    for y, c in specs:
        bm = REF + timedelta(days=int(round(365 * y)))
        pr = _price_risky_bond(REF, bm, c, 2, 0.40, rf, truth)
        out.append(BondInput(maturity=bm, coupon=c, market_price=pr, frequency=2, recovery=0.40))
    return out


class TestSequentialPopulatesCalibrationResult:
    def test_calibration_result_attached(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        assert result.calibration_result is not None
        assert isinstance(result.calibration_result, CalibrationResult)

    def test_model_class_and_optimiser(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        cr = result.calibration_result
        assert cr.model_class == "bond_hazard_pwc"
        assert cr.objective is ObjectiveKind.SSE
        assert cr.optimiser.algorithm == "brentq-per-bond"
        assert cr.iterations == len(_well_spaced_bonds())  # one root-find per bond
        assert cr.converged is True

    def test_parameters_match_pillar_hazards(self):
        rf = _flat_rf()
        bonds = _well_spaced_bonds()
        result = bootstrap_hazard_from_bonds(REF, bonds, rf, method="sequential")
        cr = result.calibration_result
        # Same number of parameters as pillar hazards
        assert len(cr.parameters) == len(result.pillar_hazards)
        # Values match (parameters dict iterates in pillar order)
        assert list(cr.parameters.values()) == [float(h) for h in result.pillar_hazards]

    def test_weights_match_bond_weights(self):
        rf = _flat_rf()
        bonds = _well_spaced_bonds()
        # Customise weights so the test catches anything mis-wired
        bonds[1].weight = 2.0
        bonds[3].weight = 0.5
        result = bootstrap_hazard_from_bonds(REF, bonds, rf, method="sequential")
        cr = result.calibration_result
        assert list(cr.weights) == [b.weight for b in sorted(bonds, key=lambda b: b.maturity)]

    def test_residuals_match_residuals_bp(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        cr = result.calibration_result
        assert list(cr.residuals) == result.residuals_bp


class TestGlobalPopulatesCalibrationResult:
    def test_calibration_result_attached_unregularised(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="global", n_pillars=4)
        cr = result.calibration_result
        assert cr is not None
        assert cr.model_class == "bond_hazard_pwc"
        assert cr.objective is ObjectiveKind.WEIGHTED_SSE
        assert cr.optimiser.algorithm == "L-BFGS-B"
        assert cr.optimiser.extra["lam"] == 0.0
        # roughness in diagnostics
        assert "roughness" in cr.diagnostics.extra

    def test_calibration_result_with_tikhonov(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(
            REF, _well_spaced_bonds(), rf, method="global", n_pillars=4, lam=1e6,
        )
        cr = result.calibration_result
        assert cr is not None
        assert "tikhonov" in cr.optimiser.algorithm
        assert cr.optimiser.extra["lam"] == 1e6
        assert cr.diagnostics.extra["roughness"] == pytest.approx(result.roughness)

    def test_each_invocation_has_unique_id(self):
        rf = _flat_rf()
        r1 = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="global", n_pillars=4)
        r2 = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="global", n_pillars=4)
        assert r1.calibration_result.id != r2.calibration_result.id


class TestToCalibrationResult:
    def test_returns_stored_instance_when_populated(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        # Identity, not just equality: the stored result must be returned as-is
        # so callers can rely on the id being stable across reads.
        assert result.to_calibration_result() is result.calibration_result

    def test_builds_on_demand_when_hand_constructed(self):
        """A `HazardBootstrapResult` built by hand (e.g., legacy code path)
        with no `calibration_result` should still produce one via
        `to_calibration_result()`. The on-demand version has a fresh id and
        a stub optimiser, but the parameters and residuals match."""
        sc = SurvivalCurve(REF, [REF + timedelta(days=365)], [0.95])
        hand = HazardBootstrapResult(
            survival_curve=sc,
            pillar_dates=[REF + timedelta(days=365)],
            pillar_hazards=[0.05],
            fitted_prices=[99.5],
            market_prices=[100.0],
            residuals_bp=[-50.0],
            rmse_bp=50.0,
            max_error_bp=50.0,
            n_bonds=1,
            method="hand_constructed",
            converged=True,
        )
        assert hand.calibration_result is None
        cr = hand.to_calibration_result()
        assert isinstance(cr, CalibrationResult)
        assert cr.model_class == "bond_hazard_pwc"
        assert list(cr.residuals) == [-50.0]
        assert list(cr.parameters.values()) == [0.05]


class TestToDictIncludesCalibrationId:
    def test_calibration_id_in_dict(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        d = result.to_dict()
        assert "calibration_id" in d
        assert d["calibration_id"] == str(result.calibration_result.id)

    def test_calibration_id_none_when_unpopulated(self):
        sc = SurvivalCurve(REF, [REF + timedelta(days=365)], [0.95])
        hand = HazardBootstrapResult(
            survival_curve=sc,
            pillar_dates=[REF + timedelta(days=365)],
            pillar_hazards=[0.05],
            fitted_prices=[99.5],
            market_prices=[100.0],
            residuals_bp=[-50.0],
            rmse_bp=50.0,
            max_error_bp=50.0,
            n_bonds=1,
            method="hand_constructed",
            converged=True,
        )
        assert hand.to_dict()["calibration_id"] is None


class TestBackwardCompatibility:
    """The existing API surface must not change. These tests are guards."""

    def test_rmse_bp_unchanged(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        # Sequential reproduces input prices to machine precision
        assert result.rmse_bp < 1e-6

    def test_pillar_hazards_unchanged(self):
        rf = _flat_rf()
        result = bootstrap_hazard_from_bonds(REF, _well_spaced_bonds(), rf, method="sequential")
        # Truth flat-2% on [0,3]; bonds at 1y and 3y should recover ~2%
        assert result.pillar_hazards[0] == pytest.approx(0.02, abs=1e-6)
        assert result.pillar_hazards[1] == pytest.approx(0.02, abs=1e-6)

    def test_hand_constructed_result_still_works(self):
        """The new field has a default — pre-existing hand-constructions
        without `calibration_result=...` must still construct."""
        sc = SurvivalCurve(REF, [REF + timedelta(days=365)], [0.95])
        # No `calibration_result` argument — uses default None
        r = HazardBootstrapResult(
            survival_curve=sc,
            pillar_dates=[REF + timedelta(days=365)],
            pillar_hazards=[0.05],
            fitted_prices=[99.5],
            market_prices=[100.0],
            residuals_bp=[-50.0],
            rmse_bp=50.0,
            max_error_bp=50.0,
            n_bonds=1,
            method="x",
            converged=True,
        )
        assert r.calibration_result is None
        assert r.lam == 0.0
        assert r.roughness == 0.0
