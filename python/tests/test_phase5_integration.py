"""Tests for Phase 5 advanced theory integration.

5.1: Regime-aware pricing engine
5.2: Information-theoretic calibration quality
5.3: Network-enhanced XVA
"""

import pytest
import math
import numpy as np

from pricebook.models.black76 import OptionType


# ═══════════════════════════════════════════════════════════════
# 5.1: Regime-Aware Pricing Engine
# ═══════════════════════════════════════════════════════════════

from pricebook.models.regime_pricing import (
    RegimePricingEngine, RegimePricingResult, RegimeGreeksResult,
    regime_option_price, regime_greeks,
)


class TestRegimeOptionPrice:
    def test_two_regime_basic(self):
        """Price with two explicit regimes."""
        r = regime_option_price(100, 100, 0.05, 1.0,
                                 regime_vols=[0.15, 0.35],
                                 regime_probs=[0.7, 0.3])
        assert isinstance(r, RegimePricingResult)
        assert r.blended_price > 0
        assert r.n_regimes == 2
        # Blended price between low-vol and high-vol prices
        assert r.regime_prices[0] < r.blended_price < r.regime_prices[1]

    def test_single_regime_equals_bs(self):
        """Single regime should equal BS."""
        from pricebook.options.equity_option import equity_option_price
        r = regime_option_price(100, 100, 0.05, 1.0,
                                 regime_vols=[0.20],
                                 regime_probs=[1.0])
        bs = equity_option_price(100, 100, 0.05, 0.20, 1.0)
        assert r.blended_price == pytest.approx(bs, rel=1e-10)

    def test_put(self):
        r = regime_option_price(100, 100, 0.05, 1.0,
                                 regime_vols=[0.15, 0.30],
                                 regime_probs=[0.5, 0.5],
                                 option_type=OptionType.PUT)
        assert r.blended_price > 0

    def test_higher_vol_regime_higher_price(self):
        r = regime_option_price(100, 100, 0.05, 1.0,
                                 regime_vols=[0.10, 0.40],
                                 regime_probs=[0.5, 0.5])
        assert r.regime_prices[1] > r.regime_prices[0]

    def test_blended_vol(self):
        r = regime_option_price(100, 100, 0.05, 1.0,
                                 regime_vols=[0.10, 0.30],
                                 regime_probs=[0.5, 0.5])
        # Variance blend: sqrt(0.5*0.01 + 0.5*0.09) = sqrt(0.05)
        expected_vol = math.sqrt(0.5 * 0.01 + 0.5 * 0.09)
        assert r.blended_vol == pytest.approx(expected_vol, rel=1e-10)

    def test_to_dict(self):
        r = regime_option_price(100, 100, 0.05, 1.0,
                                 regime_vols=[0.15, 0.30],
                                 regime_probs=[0.6, 0.4])
        d = r.to_dict()
        assert "blended_price" in d
        assert "regime_spread" in d


class TestRegimeGreeks:
    def test_delta_between_regimes(self):
        r = regime_greeks(100, 100, 0.05, 1.0,
                           regime_vols=[0.15, 0.35],
                           regime_probs=[0.5, 0.5])
        assert isinstance(r, RegimeGreeksResult)
        # Blended delta between regime deltas
        assert 0 < r.blended_delta < 1

    def test_vega_positive(self):
        r = regime_greeks(100, 100, 0.05, 1.0,
                           regime_vols=[0.15, 0.30],
                           regime_probs=[0.6, 0.4])
        assert r.blended_vega > 0

    def test_gamma_positive(self):
        r = regime_greeks(100, 100, 0.05, 1.0,
                           regime_vols=[0.20, 0.40],
                           regime_probs=[0.5, 0.5])
        assert r.blended_gamma > 0


class TestRegimePricingEngine:
    @pytest.fixture
    def engine_data(self):
        """Generate synthetic regime-switching returns."""
        rng = np.random.default_rng(42)
        n = 500
        # Low-vol regime (70%), high-vol regime (30%)
        regime = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        returns = np.where(regime == 0,
                           rng.normal(0.0003, 0.008, n),
                           rng.normal(-0.0001, 0.022, n))
        return returns

    def test_fit(self, engine_data):
        engine = RegimePricingEngine(n_regimes=2)
        result = engine.fit(engine_data)
        assert "regime_vols" in result
        assert len(result["regime_vols"]) == 2
        # Low vol < high vol after sorting
        assert result["regime_vols"][0] < result["regime_vols"][1]

    def test_price_after_fit(self, engine_data):
        engine = RegimePricingEngine(n_regimes=2)
        engine.fit(engine_data)
        r = engine.price(100, 100, 0.05, 1.0)
        assert r.blended_price > 0
        assert r.n_regimes == 2

    def test_greeks_after_fit(self, engine_data):
        engine = RegimePricingEngine(n_regimes=2)
        engine.fit(engine_data)
        g = engine.greeks(100, 100, 0.05, 1.0)
        assert 0 < g.blended_delta < 1
        assert g.blended_vega > 0

    def test_risk_decomposition(self, engine_data):
        engine = RegimePricingEngine(n_regimes=2)
        engine.fit(engine_data)
        d = engine.regime_risk_decomposition(100, 100, 0.05, 1.0)
        assert "contributions" in d
        assert len(d["contributions"]) == 2
        # Contributions sum to blended price
        total = sum(c["price_contribution"] for c in d["contributions"])
        assert total == pytest.approx(d["blended_price"], rel=1e-8)

    def test_price_before_fit_raises(self):
        engine = RegimePricingEngine()
        with pytest.raises(ValueError):
            engine.price(100, 100, 0.05, 1.0)

    def test_to_dict(self, engine_data):
        engine = RegimePricingEngine(n_regimes=2)
        engine.fit(engine_data)
        d = engine.to_dict()
        assert d["n_regimes"] == 2


# ═══════════════════════════════════════════════════════════════
# 5.2: Information-Theoretic Calibration Quality
# ═══════════════════════════════════════════════════════════════

from pricebook.statistics.calibration_quality import (
    calibration_entropy, calibration_kl, parameter_stability,
    model_comparison, CalibrationQualityResult, ModelComparisonResult,
    ParameterStabilityResult,
)


class TestCalibrationEntropy:
    def test_perfect_calibration(self):
        """Perfect fit should have near-zero RMSE."""
        prices = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        r = calibration_entropy(prices, prices)
        assert isinstance(r, CalibrationQualityResult)
        assert r.rmse < 1e-10
        assert r.r_squared > 0.999

    def test_bad_calibration(self):
        """Bad fit should have high RMSE, low R²."""
        market = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        model = np.array([12.0, 18.0, 25.0, 20.0, 35.0])
        r = calibration_entropy(market, model)
        assert r.rmse > 1.0
        assert r.r_squared < 0.9

    def test_to_dict(self):
        r = calibration_entropy(np.ones(5), np.ones(5) * 1.01)
        d = r.to_dict()
        assert "rmse" in d
        assert "information_ratio" in d


class TestCalibrationKL:
    def test_better_model_lower_kl(self):
        """Model closer to market should have lower KL."""
        market = np.array([10.0, 15.0, 20.0])
        good = np.array([10.1, 14.9, 20.1])
        bad = np.array([12.0, 18.0, 25.0])
        # negative = model A (good) is better
        diff = calibration_kl(market, good, bad)
        assert diff < 0


class TestParameterStability:
    def test_stable_params(self):
        """Low-variance parameters should have high stability score."""
        history = np.array([
            [0.20, 0.50, 1.0],
            [0.21, 0.49, 1.01],
            [0.19, 0.51, 0.99],
            [0.20, 0.50, 1.00],
        ])
        r = parameter_stability(history, ["vol", "rho", "kappa"])
        assert isinstance(r, ParameterStabilityResult)
        assert r.stability_score > 0.8

    def test_unstable_params(self):
        """High-variance parameters should have lower stability score than stable."""
        rng = np.random.default_rng(42)
        unstable = rng.uniform(0, 10, (10, 3))
        stable = np.array([[1.0, 2.0, 3.0]] * 10) + rng.normal(0, 0.01, (10, 3))
        r_unstable = parameter_stability(unstable)
        r_stable = parameter_stability(stable)
        assert r_unstable.stability_score < r_stable.stability_score

    def test_to_dict(self):
        history = np.array([[1, 2], [1.1, 2.1]])
        r = parameter_stability(history, ["a", "b"])
        d = r.to_dict()
        assert "stability_score" in d


class TestModelComparison:
    def test_better_model_preferred(self):
        """Model with lower error should be preferred by BIC."""
        market = np.array([10, 15, 20, 25, 30], dtype=float)
        good = np.array([10.1, 14.9, 20.1, 24.9, 30.1])
        bad = np.array([12, 18, 23, 27, 35], dtype=float)
        r = model_comparison(market, good, bad, 3, 3, "Good", "Bad")
        assert isinstance(r, ModelComparisonResult)
        assert r.preferred == "Good"

    def test_parsimony_penalty(self):
        """Same fit but more params → should penalise in BIC."""
        market = np.array([10, 15, 20, 25, 30], dtype=float)
        model = np.array([10.1, 14.9, 20.1, 24.9, 30.1])
        r = model_comparison(market, model, model, 3, 10, "Simple", "Complex")
        # BIC penalises more parameters
        assert r.bic_a < r.bic_b

    def test_to_dict(self):
        market = np.ones(5)
        r = model_comparison(market, market * 1.01, market * 1.02, 2, 3)
        d = r.to_dict()
        assert "js_divergence" in d


# ═══════════════════════════════════════════════════════════════
# 5.3: Network-Enhanced XVA
# ═══════════════════════════════════════════════════════════════

from pricebook.risk.network_xva import (
    NetworkXVAEngine, NetworkCVAResult, SystemicStressResult,
    systemic_cva_adjustment, contagion_cva_stress,
)


class TestNetworkCVA:
    @pytest.fixture
    def engine(self):
        nodes = ["BankA", "BankB", "BankC", "BankD"]
        exposures = np.array([
            [0, 100, 50, 0],
            [80, 0, 0, 60],
            [0, 30, 0, 40],
            [20, 0, 10, 0],
        ], dtype=float)
        buffers = np.array([50, 40, 30, 20], dtype=float)
        return NetworkXVAEngine(nodes, exposures, buffers)

    def test_network_cva_exceeds_standalone(self, engine):
        """Network CVA should be >= standalone CVA."""
        r = engine.compute_network_cva("BankA", 1_000_000)
        assert isinstance(r, NetworkCVAResult)
        assert r.network_cva >= r.standalone_cva

    def test_adjustment_ratio_geq_1(self, engine):
        r = engine.compute_network_cva("BankB", 500_000)
        assert r.adjustment_ratio >= 1.0

    def test_zero_alpha_no_adjustment(self, engine):
        """With alpha=0, network CVA = standalone CVA."""
        r = engine.compute_network_cva("BankA", 1_000_000, alpha=0.0)
        assert r.network_cva == pytest.approx(r.standalone_cva, rel=1e-10)

    def test_centrality_score_positive(self, engine):
        r = engine.compute_network_cva("BankA", 1_000_000)
        assert r.centrality_score > 0

    def test_to_dict(self, engine):
        r = engine.compute_network_cva("BankA", 1_000_000)
        d = r.to_dict()
        assert "systemic_adjustment" in d
        assert "contagion_multiplier" in d


class TestStressTest:
    @pytest.fixture
    def engine(self):
        nodes = ["A", "B", "C"]
        exp = np.array([[0, 50, 30], [40, 0, 20], [10, 15, 0]], dtype=float)
        buf = np.array([30, 25, 20], dtype=float)
        return NetworkXVAEngine(nodes, exp, buf)

    def test_stress_test_basic(self, engine):
        cvas = {"A": 100_000, "B": 200_000, "C": 50_000}
        r = engine.stress_test(cvas)
        assert isinstance(r, SystemicStressResult)
        assert r.total_network_cva >= r.total_standalone_cva
        assert r.systemic_surcharge >= 0

    def test_stress_test_to_dict(self, engine):
        cvas = {"A": 100_000, "B": 200_000, "C": 50_000}
        d = engine.stress_test(cvas).to_dict()
        assert "systemic_surcharge" in d


class TestSystemicRanking:
    def test_ranking(self):
        nodes = ["A", "B", "C"]
        exp = np.array([[0, 100, 10], [5, 0, 5], [5, 5, 0]], dtype=float)
        buf = np.array([20, 15, 10], dtype=float)
        engine = NetworkXVAEngine(nodes, exp, buf)
        ranking = engine.systemic_ranking()
        assert len(ranking) == 3
        assert ranking[0]["rank"] == 1
        # Most connected node should rank highest
        assert ranking[0]["systemic_score"] >= ranking[1]["systemic_score"]


class TestConvenienceFunctions:
    def test_systemic_cva_adjustment(self):
        adj = systemic_cva_adjustment(1_000_000, centrality=0.5, contagion_multiplier=2.0)
        assert adj > 1_000_000

    def test_zero_centrality_no_adjustment(self):
        adj = systemic_cva_adjustment(1_000_000, centrality=0.0, contagion_multiplier=5.0)
        assert adj == 1_000_000

    def test_contagion_cva_stress(self):
        nodes = ["A", "B"]
        exp = np.array([[0, 50], [30, 0]], dtype=float)
        buf = np.array([20, 15], dtype=float)
        r = contagion_cva_stress(nodes, exp, buf, {"A": 100_000, "B": 200_000})
        assert isinstance(r, SystemicStressResult)
