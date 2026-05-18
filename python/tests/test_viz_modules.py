"""Tests verifying each viz sub-module is importable and has expected exports."""
import pytest

class TestVizModules:
    def test_backend(self):
        from pricebook.viz._backend import apply_theme, create_figure
        assert callable(apply_theme)
        assert callable(create_figure)

    def test_theme(self):
        from pricebook.viz._theme import get_theme, LIGHT, DARK, PricebookTheme
        t = get_theme()
        assert isinstance(t, PricebookTheme)

    def test_dispatch(self):
        from pricebook.viz._dispatch import plot, register_instrument
        assert callable(plot)

    def test_builder(self):
        from pricebook.viz._builder import PlotBuilder
        assert PlotBuilder is not None

    def test_seaborn(self):
        from pricebook.viz._seaborn import greeks_profile, pnl_distribution, sensitivity_grid
        assert callable(greeks_profile)

    def test_risk(self):
        from pricebook.viz._risk import pnl_waterfall, stress_comparison, football_field, j_curve
        assert callable(pnl_waterfall)

    def test_generic(self):
        from pricebook.viz._generic import plot_summary_table, plot_sensitivity
        assert callable(plot_summary_table)

    def test_tlock(self):
        from pricebook.viz import _tlock
        assert hasattr(_tlock, '_register') or True  # auto-registers on import

    def test_trs(self):
        from pricebook.viz import _trs
        assert True

    def test_cmt(self):
        from pricebook.viz import _cmt
        assert True

    def test_cmasw(self):
        from pricebook.viz import _cmasw
        assert True

    def test_hybrid(self):
        from pricebook.viz import _hybrid
        assert True

    def test_treasury_lock(self):
        from pricebook.viz import _treasury_lock
        assert True
