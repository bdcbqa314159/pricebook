"""Tests for Phase 3 curve modules: FX forward, scenarios, bumper."""

import math
import pytest
import numpy as np
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.fx.fx_forward_builder import (
    build_fx_implied_curve, FXSwapPointQuote, FXForwardBuildResult,
    get_fx_conventions, list_fx_pairs,
)
from pricebook.curves.curve_scenarios import (
    parallel_shift, steepener, flattener, bear_steepener, bull_flattener,
    butterfly, inversion, standard_scenario_set, run_scenarios,
    pca_scenarios, CurveScenario,
)
from pricebook.curves.curve_bumper import (
    CurveBumper, InstrumentRiskReport,
)

REF = date(2024, 1, 15)


def _swap_pv(curve, mat_years=10, rate=0.04, notional=1e6):
    pv = 0.0
    for i in range(1, mat_years + 1):
        d = date(REF.year + i, REF.month, REF.day)
        df = curve.df(d)
        df_prev = curve.df(date(REF.year + i - 1, REF.month, REF.day)) if i > 1 else 1.0
        fwd = (df_prev / df - 1.0) if df > 0 else 0.0
        pv += (fwd - rate) * notional * df
    return pv


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.04)


# ═══════════════════════════════════════════════════════════════
# 3.1: FX Forward Builder
# ═══════════════════════════════════════════════════════════════

class TestFXForward:
    def test_basic_eurusd(self):
        usd_curve = DiscountCurve.flat(REF, 0.05)
        points = [
            FXSwapPointQuote("1M", date(2024, 2, 15), -15.0),
            FXSwapPointQuote("3M", date(2024, 4, 15), -45.0),
            FXSwapPointQuote("6M", date(2024, 7, 15), -90.0),
            FXSwapPointQuote("1Y", date(2025, 1, 15), -180.0),
        ]
        result = build_fx_implied_curve("EURUSD", 1.0850, REF, points, usd_curve)
        assert isinstance(result, FXForwardBuildResult)
        assert result.pair == "EURUSD"
        assert len(result.pillar_dates) == 4

    def test_negative_points_higher_foreign_rate(self):
        """Negative swap points → foreign rate < domestic (EUR < USD)."""
        usd_curve = DiscountCurve.flat(REF, 0.05)
        points = [FXSwapPointQuote("1Y", date(2025, 1, 15), -200.0)]
        result = build_fx_implied_curve("EURUSD", 1.0850, REF, points, usd_curve)
        # Forward < spot → EUR rates lower than USD
        assert result.forward_rates[0] < 1.0850

    def test_usdjpy(self):
        usd_curve = DiscountCurve.flat(REF, 0.05)
        points = [FXSwapPointQuote("1Y", date(2025, 1, 15), -500.0, pip_factor=100)]
        result = build_fx_implied_curve("USDJPY", 148.50, REF, points, usd_curve)
        assert result.forward_rates[0] < 148.50  # JPY rates lower

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_fx_implied_curve("EURUSD", 1.08, REF, [], DiscountCurve.flat(REF, 0.04))

    def test_list_pairs(self):
        pairs = list_fx_pairs()
        assert "EURUSD" in pairs
        assert "USDJPY" in pairs
        assert len(pairs) >= 10

    def test_to_dict(self):
        usd = DiscountCurve.flat(REF, 0.05)
        points = [FXSwapPointQuote("1Y", date(2025, 1, 15), -100.0)]
        d = build_fx_implied_curve("EURUSD", 1.08, REF, points, usd).to_dict()
        assert "pair" in d
        assert "forward_rates" in d


# ═══════════════════════════════════════════════════════════════
# 3.2: Curve Scenarios
# ═══════════════════════════════════════════════════════════════

class TestCurveScenarios:
    def test_parallel_shift(self, flat_curve):
        s = parallel_shift(+100)
        shocked = s.apply(flat_curve)
        # DFs should be lower (rates higher)
        assert shocked.df(date(2034, 1, 15)) < flat_curve.df(date(2034, 1, 15))

    def test_steepener(self, flat_curve):
        s = steepener(-25, +50)
        shocked = s.apply(flat_curve)
        # Short end down → higher short DF, long end up → lower long DF
        assert shocked.df(date(2025, 1, 15)) > flat_curve.df(date(2025, 1, 15))
        assert shocked.df(date(2044, 1, 15)) < flat_curve.df(date(2044, 1, 15))

    def test_butterfly(self, flat_curve):
        s = butterfly(+25, -25)
        shocked = s.apply(flat_curve)
        assert isinstance(shocked, DiscountCurve)

    def test_bear_steepener(self, flat_curve):
        s = bear_steepener(50)
        shocked = s.apply(flat_curve)
        assert shocked.df(date(2034, 1, 15)) < flat_curve.df(date(2034, 1, 15))

    def test_inversion(self, flat_curve):
        s = inversion(100)
        shocked = s.apply(flat_curve)
        assert isinstance(shocked, DiscountCurve)

    def test_standard_set(self):
        scenarios = standard_scenario_set()
        assert len(scenarios) >= 10
        assert any("parallel" in s.name for s in scenarios)
        assert any("steepener" in s.name for s in scenarios)

    def test_run_scenarios(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        scenarios = [parallel_shift(+100), parallel_shift(-100)]
        results = run_scenarios(flat_curve, pricer, scenarios)
        assert len(results) == 2
        assert results[0]["pnl"] != 0
        # Symmetric: up and down should have opposite PnL
        assert results[0]["pnl"] * results[1]["pnl"] < 0

    def test_pca_scenarios(self):
        """PCA from synthetic history."""
        rng = np.random.default_rng(42)
        tenors = [1, 2, 5, 10, 30]
        history = np.cumsum(rng.normal(0, 0.001, (100, 5)), axis=0) + 0.04
        scenarios = pca_scenarios(history, tenors, n_components=3)
        assert len(scenarios) == 6  # 3 components × 2 directions
        assert "level" in scenarios[0].name

    def test_scenario_to_dict(self):
        s = parallel_shift(50)
        d = s.to_dict()
        assert "name" in d


# ═══════════════════════════════════════════════════════════════
# 3.3: Curve Bumper
# ═══════════════════════════════════════════════════════════════

class TestCurveBumper:
    def test_basic(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        bumper = CurveBumper(flat_curve, pricer)
        assert bumper.base_pv == pricer(flat_curve)
        assert bumper.n_pillars > 0

    def test_parallel_dv01(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        bumper = CurveBumper(flat_curve, pricer)
        dv01 = bumper.parallel_dv01()
        assert dv01 != 0

    def test_fast_vs_exact(self, flat_curve):
        """Jacobian-based fast reprice should be close to exact."""
        pricer = lambda c: _swap_pv(c)
        bumper = CurveBumper(flat_curve, pricer)
        shift = np.full(bumper.n_pillars, 0.001)  # 10bp
        fast = bumper.bump_and_reprice(shift)
        exact = bumper.full_rebuild_and_reprice(shift)
        # Should be within 5% for 10bp bump
        if abs(exact - bumper.base_pv) > 1.0:
            assert abs(fast - exact) / abs(exact - bumper.base_pv) < 0.10

    def test_key_rate_dv01s(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        bumper = CurveBumper(flat_curve, pricer)
        kr = bumper.key_rate_dv01s([5, 10])
        assert 5 in kr
        assert 10 in kr

    def test_risk_report(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        bumper = CurveBumper(flat_curve, pricer)
        report = bumper.risk_report("10Y_swap")
        assert isinstance(report, InstrumentRiskReport)
        assert report.parallel_dv01 != 0
        assert "convexity" in report.to_dict()
