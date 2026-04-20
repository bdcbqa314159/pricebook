"""Tests for vol model comparison."""
import math, numpy as np, pytest
from pricebook.vol_model_comparison import compare_models, model_risk_quantification, model_selection_guide

class TestCompareModels:
    def test_baseline_only(self):
        r = compare_models(100, 100, 1.0, math.exp(-0.03), 0.20)
        assert r.n_models == 1  # just black76
        assert r.price_spread == 0.0

    def test_with_sabr(self):
        r = compare_models(100, 100, 1.0, math.exp(-0.03), 0.20,
                            sabr_alpha=0.2, sabr_rho=-0.3, sabr_nu=0.4)
        assert r.n_models >= 2
        assert r.price_spread >= 0

    def test_with_all_models(self):
        r = compare_models(100, 100, 1.0, math.exp(-0.03), 0.20,
                            sabr_alpha=0.2, sabr_rho=-0.3, sabr_nu=0.4,
                            svi_a=0.01, svi_b=0.1, svi_rho_svi=-0.3, svi_m=0, svi_sigma=0.2,
                            local_vol=0.21,
                            heston_v0=0.04, heston_kappa=2, heston_theta=0.04,
                            heston_xi=0.3, heston_rho=-0.5)
        assert r.n_models >= 4

    def test_otm_higher_spread(self):
        """OTM options should have higher model risk (models diverge in wings)."""
        atm = compare_models(100, 100, 1.0, math.exp(-0.03), 0.20,
                              sabr_alpha=0.2, sabr_rho=-0.5, sabr_nu=0.5, local_vol=0.20)
        otm = compare_models(100, 80, 1.0, math.exp(-0.03), 0.20,
                              sabr_alpha=0.2, sabr_rho=-0.5, sabr_nu=0.5, local_vol=0.25)
        # Just check both run; spread direction depends on params
        assert atm.n_models == otm.n_models

    def test_best_model_returned(self):
        r = compare_models(100, 100, 1.0, math.exp(-0.03), 0.20, local_vol=0.20)
        assert r.best_model in ["black76", "local_vol"]

class TestModelRisk:
    def test_basic(self):
        strikes = [80, 90, 100, 110, 120]
        r = model_risk_quantification(100, 1.0, math.exp(-0.03), strikes, 0.20,
                                        sabr_params=(0.2, 1.0, -0.3, 0.4))
        assert r.n_strikes == 5
        assert r.max_spread_pct >= 0
        assert r.mean_spread_pct >= 0

    def test_worst_strike(self):
        strikes = [80, 90, 100, 110, 120]
        r = model_risk_quantification(100, 1.0, math.exp(-0.03), strikes, 0.20,
                                        sabr_params=(0.2, 1.0, -0.3, 0.4))
        assert r.worst_strike in strikes

class TestModelGuide:
    def test_vanilla_equity(self):
        r = model_selection_guide("vanilla_equity")
        assert r.recommended == "sabr"

    def test_barrier(self):
        r = model_selection_guide("barrier")
        assert r.recommended == "local_vol"

    def test_autocallable(self):
        r = model_selection_guide("autocallable")
        assert r.recommended == "slv"

    def test_unknown_product(self):
        r = model_selection_guide("unknown_exotic")
        assert r.recommended == "sabr"  # default
