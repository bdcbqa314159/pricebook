"""Tests for calibration robustness utilities."""

import math
import pytest
import numpy as np

from pricebook.calibration_utils import (
    tikhonov_regularise,
    enforce_bounds,
    calibration_quality,
    multi_start_calibrate,
    perturbation_stability,
)


class TestTikhonov:
    def test_penalises_deviation(self):
        obj = lambda x: sum(xi**2 for xi in x)
        reg = tikhonov_regularise(obj, prior=[1.0, 1.0], lambda_reg=1.0)
        # At prior: obj=2, penalty=0 → total=2
        assert reg([1.0, 1.0]) == pytest.approx(2.0)
        # Away from prior: obj=0, penalty=2 → total=2
        assert reg([0.0, 0.0]) == pytest.approx(2.0)

    def test_stronger_regularisation(self):
        obj = lambda x: x[0]**2
        reg_weak = tikhonov_regularise(obj, [1.0], lambda_reg=0.01)
        reg_strong = tikhonov_regularise(obj, [1.0], lambda_reg=10.0)
        # At x=0: obj=0, penalty proportional to lambda
        assert reg_strong([0.0]) > reg_weak([0.0])


class TestEnforceBounds:
    def test_within_bounds(self):
        result = enforce_bounds([0.5, 1.5], [(0, 1), (1, 2)])
        assert result == [0.5, 1.5]

    def test_clip_lower(self):
        result = enforce_bounds([-1.0], [(0, 1)])
        assert result == [0.0]

    def test_clip_upper(self):
        result = enforce_bounds([5.0], [(0, 1)])
        assert result == [1.0]


class TestCalibrationQuality:
    def test_perfect_fit(self):
        model = lambda p: [p[0], p[0]**2]
        q = calibration_quality(model, [2.0], [2.0, 4.0])
        assert q["rmse"] == pytest.approx(0.0)
        assert q["max_error"] == pytest.approx(0.0)

    def test_imperfect_fit(self):
        model = lambda p: [1.0, 2.0]
        q = calibration_quality(model, [0.0], [1.1, 2.1])
        assert q["rmse"] > 0
        assert len(q["residuals"]) == 2


class TestMultiStart:
    def test_finds_minimum(self):
        """Rosenbrock function: minimum at (1, 1)."""
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        result = multi_start_calibrate(
            rosenbrock,
            bounds=[(-5, 5), (-5, 5)],
            n_starts=10,
            method="nelder_mead",
            maxiter=2000,
        )
        assert result.converged
        assert result.params[0] == pytest.approx(1.0, abs=0.1)
        assert result.params[1] == pytest.approx(1.0, abs=0.1)

    def test_better_than_single_start(self):
        """Multi-start should find better minimum than bad single start."""
        def multimodal(x):
            return math.sin(5 * x[0])**2 + (x[0] - 1)**2

        # Single start from bad point
        from pricebook.optimization import minimize
        single = minimize(multimodal, x0=[4.0], method="nelder_mead")

        multi = multi_start_calibrate(
            multimodal, bounds=[(-5, 5)], n_starts=20,
        )
        assert multi.objective <= single.fun + 0.01

    def test_respects_bounds(self):
        result = multi_start_calibrate(
            lambda x: sum(xi**2 for xi in x),
            bounds=[(0.5, 1.5), (0.5, 1.5)],
            n_starts=5,
        )
        for p, (lo, hi) in zip(result.params, [(0.5, 1.5), (0.5, 1.5)]):
            assert lo <= p <= hi


class TestPerturbationStability:
    def test_stable_calibration(self):
        """Linear model should be perfectly stable."""
        def calibrate(inputs):
            return [2 * inputs[0], 3 * inputs[1]]

        result = perturbation_stability(
            calibrate, [1.0, 1.0], perturbation=0.01, n_trials=5,
        )
        assert result["n_successful"] == 6  # base + 5 trials
        assert all(s < 0.1 for s in result["param_std"])

    def test_detects_instability(self):
        """Highly nonlinear model may show parameter variance."""
        def calibrate(inputs):
            # Sensitive: small input change → large param change
            return [inputs[0]**10]

        result = perturbation_stability(
            calibrate, [1.0], perturbation=0.01, n_trials=10,
        )
        assert result["max_deviation"] > 0
