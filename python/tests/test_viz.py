"""Tests for pricebook.viz — all chart functions produce valid Figure objects."""
import pytest
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class TestSeabornCharts:
    def test_greeks_profile(self):
        from pricebook.viz import greeks_profile
        fig = greeks_profile([90,95,100,105,110], {"Delta": [0.3,0.4,0.5,0.6,0.7]})
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_pnl_distribution(self):
        from pricebook.viz import pnl_distribution
        fig = pnl_distribution(np.random.default_rng(42).normal(0, 100, 500).tolist())
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_sensitivity_grid(self):
        from pricebook.viz import sensitivity_grid
        fig = sensitivity_grid([1,2,3], [4,5,6], [[10,20,30],[40,50,60],[70,80,90]])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_exposure_profile(self):
        from pricebook.viz import exposure_profile
        fig = exposure_profile([0.5,1,2,3,5], [10,20,30,25,15])
        assert isinstance(fig, Figure)
        plt.close("all")


class TestRiskCharts:
    def test_pnl_waterfall(self):
        from pricebook.viz import pnl_waterfall
        fig = pnl_waterfall({"Carry": 100, "Rate": -50, "Vol": 30})
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_risk_decomposition(self):
        from pricebook.viz import risk_decomposition
        fig = risk_decomposition(["2Y","5Y","10Y"], [100,-200,300])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_stress_comparison(self):
        from pricebook.viz import stress_comparison
        fig = stress_comparison([{"name": "+100bp", "total": -500}, {"name": "-100bp", "total": 400}])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_tenor_bucketing(self):
        from pricebook.viz import tenor_bucketing
        fig = tenor_bucketing(["1Y","5Y","10Y"], [50,300,200])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_vega_ladder(self):
        from pricebook.viz import vega_ladder
        fig = vega_ladder(["0-3M","3-6M","6-12M"], [500,300,-100])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_pnl_table(self):
        from pricebook.viz import pnl_table
        fig = pnl_table([{"Factor": "Rates", "PnL": -22.5}, {"Factor": "Total", "PnL": 1.5}])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_greeks_surface(self):
        from pricebook.viz import greeks_surface
        fig = greeks_surface([80,100,120], [0.5,1.0], np.ones((2,3))*0.5)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_greeks_evolution(self):
        from pricebook.viz import greeks_evolution
        fig = greeks_evolution(list(range(30)), {"Delta": [0.5]*30})
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_hedge_pnl_tracking(self):
        from pricebook.viz import hedge_pnl_tracking
        fig = hedge_pnl_tracking(["D1","D2","D3"], [100,-50,30], [-80,40,-20])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_rolling_correlation(self):
        from pricebook.viz import rolling_correlation
        fig = rolling_correlation(["W1","W2","W3"], {"A/B": [0.3,0.5,0.7]})
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_football_field(self):
        from pricebook.viz import football_field
        fig = football_field(["DCF","Comps"], [400,350], [500,450], [600,550])
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_j_curve(self):
        from pricebook.viz import j_curve
        fig = j_curve([1,2,3,4,5], [0.85,0.78,0.95,1.1,1.3])
        assert isinstance(fig, Figure)
        plt.close("all")


class TestTheme:
    def test_configure_theme(self):
        from pricebook.viz import configure_theme, LIGHT, DARK
        configure_theme(dark=False)
        configure_theme(dark=True)
        configure_theme(theme=LIGHT)
        configure_theme(theme=DARK)

    def test_theme_properties(self):
        from pricebook.viz import LIGHT
        assert hasattr(LIGHT, "colors")
        assert hasattr(LIGHT, "background")
        assert hasattr(LIGHT, "foreground")
        assert len(LIGHT.colors) >= 5


class TestBuilder:
    def test_plot_builder_exists(self):
        from pricebook.viz import PlotBuilder
        assert PlotBuilder is not None


class TestEmptyGuards:
    def test_waterfall_empty(self):
        from pricebook.viz import pnl_waterfall
        with pytest.raises(ValueError):
            pnl_waterfall({})

    def test_risk_decomp_empty(self):
        from pricebook.viz import risk_decomposition
        with pytest.raises(ValueError):
            risk_decomposition([], [])

    def test_stress_empty(self):
        from pricebook.viz import stress_comparison
        with pytest.raises(ValueError):
            stress_comparison([])
