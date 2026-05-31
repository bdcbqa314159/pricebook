"""Tests for parameter uncertainty, model reserves, P&L attribution, and model selection."""

import pytest
import math
import numpy as np

from pricebook.risk.parameter_uncertainty import (
    ParameterBand, calibration_uncertainty, sensitivity_ladder,
    joint_parameter_surface,
)
from pricebook.risk.model_reserve import (
    compute_model_reserve, reserve_by_risk_factor, model_risk_reserve_ava,
)
from pricebook.risk.model_selection import (
    ModelCandidate, model_committee_price, bayesian_model_average,
    model_risk_matrix,
)
from pricebook.risk.pnl_explain import (
    NonLinearPnLResult, surface_pnl, gamma_pnl_decompose,
)


# ═══════════════════════════════════════════════════════════════
# C1: Parameter uncertainty
# ═══════════════════════════════════════════════════════════════

class TestParameterUncertainty:
    def _simple_pricer(self, params):
        """BS-like: price = sigma * sqrt(T) * 100."""
        return params["sigma"] * math.sqrt(params.get("T", 1.0)) * 100

    def test_sensitivity_ladder(self):
        bands = [
            ParameterBand("sigma", 0.20, 0.18, 0.22, 0.95),
            ParameterBand("T", 1.0, 0.9, 1.1, 0.95),
        ]
        ladder = sensitivity_ladder(self._simple_pricer, {"sigma": 0.20, "T": 1.0}, bands)
        assert len(ladder) == 2
        assert all(e.impact >= 0 for e in ladder)
        # Sorted by impact
        assert ladder[0].impact >= ladder[1].impact

    def test_joint_surface(self):
        bands_s = ParameterBand("sigma", 0.20, 0.15, 0.25, 0.95)
        bands_t = ParameterBand("T", 1.0, 0.5, 1.5, 0.95)
        result = joint_parameter_surface(
            self._simple_pricer, {"sigma": 0.20, "T": 1.0},
            bands_s, bands_t, n_grid=5,
        )
        assert len(result["pv_surface"]) == 5
        assert len(result["pv_surface"][0]) == 5
        assert result["base_pv"] == pytest.approx(20.0)

    def test_band_width(self):
        b = ParameterBand("sigma", 0.20, 0.18, 0.22, 0.95)
        assert b.width() == pytest.approx(0.04)

    def test_bootstrap_ci(self):
        """Bootstrap should produce confidence intervals around base params."""
        def mock_calibrator(base_params, data):
            # Noisy calibration: sigma = mean(data) + noise
            return {"sigma": float(np.mean(data)) + 0.001}

        data = np.random.default_rng(42).normal(0.20, 0.01, size=100)
        bands = calibration_uncertainty(
            mock_calibrator, {"sigma": 0.20}, data, n_bootstrap=50, seed=42)
        assert len(bands) == 1
        assert bands[0].low <= bands[0].base <= bands[0].high


# ═══════════════════════════════════════════════════════════════
# C2: Model reserve
# ═══════════════════════════════════════════════════════════════

class TestModelReserve:
    def _pricer(self, params):
        return params["sigma"] * 100 + params.get("rho", 0) * 50

    def test_worst_case(self):
        bands = [
            ParameterBand("sigma", 0.20, 0.15, 0.25, 0.95),
            ParameterBand("rho", 0.0, -0.5, 0.5, 0.95),
        ]
        result = compute_model_reserve(self._pricer, {"sigma": 0.20, "rho": 0.0},
                                        bands, method="worst_case")
        assert result.reserve > 0
        assert len(result.components) == 2

    def test_quadrature(self):
        bands = [ParameterBand("sigma", 0.20, 0.15, 0.25, 0.95)]
        result = compute_model_reserve(self._pricer, {"sigma": 0.20},
                                        bands, method="quadrature")
        assert result.reserve > 0

    def test_zero_uncertainty_zero_reserve(self):
        bands = [ParameterBand("sigma", 0.20, 0.20, 0.20, 0.95)]
        result = compute_model_reserve(self._pricer, {"sigma": 0.20}, bands)
        assert result.reserve == pytest.approx(0.0)

    def test_reserve_by_factor(self):
        bands = [
            ParameterBand("sigma", 0.20, 0.15, 0.25, 0.95),
            ParameterBand("rho", 0.0, -0.5, 0.5, 0.95),
        ]
        factors = reserve_by_risk_factor(self._pricer, {"sigma": 0.20, "rho": 0.0}, bands)
        assert len(factors) == 2

    def test_ava_format(self):
        bands = [ParameterBand("sigma", 0.20, 0.15, 0.25, 0.95)]
        ava = model_risk_reserve_ava(self._pricer, {"sigma": 0.20}, bands)
        assert ava["ava_category"] == "model_risk"
        assert ava["ava_model_risk"] > 0

    def test_to_dict(self):
        bands = [ParameterBand("sigma", 0.20, 0.15, 0.25, 0.95)]
        result = compute_model_reserve(self._pricer, {"sigma": 0.20}, bands)
        d = result.to_dict()
        assert "reserve" in d


# ═══════════════════════════════════════════════════════════════
# C3: Non-linear P&L attribution
# ═══════════════════════════════════════════════════════════════

class TestNonLinearPnL:
    def test_surface_pnl(self):
        def vol_pricer(surface, **params):
            return surface["atm"] * 100 + surface.get("skew", 0) * 20

        base = {"atm": 0.20, "skew": 0.01, "smile": 0.005, "term": 0.0}
        current = {"atm": 0.22, "skew": 0.02, "smile": 0.005, "term": 0.0}

        result = surface_pnl(vol_pricer, base, current, {})
        assert result.total_pnl == pytest.approx(2.2, abs=0.1)
        assert result.atm_vol_pnl > 0  # ATM went up
        assert result.skew_pnl > 0     # skew steepened

    def test_gamma_decompose(self):
        result = gamma_pnl_decompose(
            delta=0.5, gamma=0.03, spot_change=2.0,
            realised_vol=0.25, implied_vol=0.20, dt=1/252, spot=100)

        assert result["delta_pnl"] == pytest.approx(1.0)
        assert result["gamma_actual"] == pytest.approx(0.5 * 0.03 * 4.0)
        # Realised > implied → positive gamma P&L
        assert result["gamma_pnl"] > 0

    def test_gamma_pnl_negative_when_implied_gt_realised(self):
        result = gamma_pnl_decompose(
            delta=0.5, gamma=0.03, spot_change=1.0,
            realised_vol=0.15, implied_vol=0.25, dt=1/252, spot=100)
        assert result["gamma_pnl"] < 0

    def test_surface_pnl_to_dict(self):
        def pricer(s, **kw): return s.get("atm", 0.2) * 100
        result = surface_pnl(pricer, {"atm": 0.2}, {"atm": 0.22}, {})
        d = result.to_dict()
        assert "atm_vol_pnl" in d


# ═══════════════════════════════════════════════════════════════
# C4: Model selection
# ═══════════════════════════════════════════════════════════════

class TestModelSelection:
    def test_committee_price(self):
        models = [
            ModelCandidate("BS", lambda: 10.0),
            ModelCandidate("Heston", lambda: 11.0),
            ModelCandidate("SABR", lambda: 10.5),
        ]
        result = model_committee_price(models)
        assert result.price == pytest.approx(10.5, abs=0.01)
        assert result.price_std > 0
        assert result.model_uncertainty_reserve == pytest.approx(0.5)

    def test_single_model(self):
        models = [ModelCandidate("BS", lambda: 10.0)]
        result = model_committee_price(models)
        assert result.price == pytest.approx(10.0)
        assert result.price_std == 0
        assert result.model_uncertainty_reserve == 0

    def test_weighted_committee(self):
        models = [
            ModelCandidate("BS", lambda: 10.0, weight=3.0),
            ModelCandidate("Heston", lambda: 12.0, weight=1.0),
        ]
        result = model_committee_price(models)
        # 3/4 × 10 + 1/4 × 12 = 10.5
        assert result.price == pytest.approx(10.5)

    def test_bma_weights(self):
        models = [
            ModelCandidate("A", lambda: 10.0, bic=-100),
            ModelCandidate("B", lambda: 11.0, bic=-90),  # worse fit
        ]
        updated = bayesian_model_average(models, use_bic=True)
        # Lower BIC → higher weight
        assert updated[0].weight > updated[1].weight
        assert sum(c.weight for c in updated) == pytest.approx(1.0)

    def test_model_risk_matrix(self):
        models = [
            ModelCandidate("BS", lambda spot=100: spot * 0.10),
            ModelCandidate("Heston", lambda spot=100: spot * 0.11),
        ]
        scenarios = [{"spot": 90}, {"spot": 100}, {"spot": 110}]
        result = model_risk_matrix(models, scenarios)
        assert len(result["model_names"]) == 2
        assert result["n_scenarios"] == 3
        assert len(result["prices"]) == 2
        assert len(result["range_per_scenario"]) == 3

    def test_committee_to_dict(self):
        models = [ModelCandidate("BS", lambda: 10.0)]
        result = model_committee_price(models)
        d = result.to_dict()
        assert "model_uncertainty_reserve" in d

    def test_candidate_to_dict(self):
        c = ModelCandidate("BS", lambda: 0, aic=-50, bic=-45)
        d = c.to_dict()
        assert d["name"] == "BS"
        assert d["aic"] == -50
