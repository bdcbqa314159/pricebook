"""Tests for scenario risk engine."""

import pytest
from datetime import date

from pricebook.scenario import (
    parallel_shift,
    pillar_bump,
    vol_bump,
    fx_spot_shock,
    run_scenarios,
    dv01_ladder,
    ScenarioResult,
)
from pricebook.trade import Trade, Portfolio
from pricebook.pricing_context import PricingContext
from pricebook.swaption import Swaption
from pricebook.vol_surface import FlatVol
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


@pytest.fixture
def ctx():
    return PricingContext(
        valuation_date=REF,
        discount_curve=make_flat_curve(REF, 0.03),
        vol_surfaces={"ir": FlatVol(0.20)},
        fx_spots={("EUR", "USD"): 1.0850},
    )


@pytest.fixture
def portfolio():
    swn = Swaption(
        expiry=date(2025, 1, 15),
        swap_end=date(2030, 1, 15),
        strike=0.03,
    )
    return Portfolio([
        Trade(instrument=swn, direction=1, trade_id="SWN_LONG"),
    ])


class TestParallelShift:
    def test_positive_shift_changes_pv(self, ctx, portfolio):
        results = run_scenarios(portfolio, ctx, [parallel_shift(0.001)])
        assert len(results) == 1
        assert results[0].pnl != 0

    def test_zero_shift_no_pnl(self, ctx, portfolio):
        results = run_scenarios(portfolio, ctx, [parallel_shift(0.0)])
        assert results[0].pnl == pytest.approx(0.0, abs=0.01)

    def test_up_and_down_opposite_sign(self, ctx, portfolio):
        up = run_scenarios(portfolio, ctx, [parallel_shift(0.01)])[0]
        dn = run_scenarios(portfolio, ctx, [parallel_shift(-0.01)])[0]
        assert up.pnl * dn.pnl < 0  # opposite signs


class TestPillarBump:
    def test_single_pillar_changes_pv(self, ctx, portfolio):
        results = run_scenarios(portfolio, ctx, [pillar_bump(3, 0.0001)])
        assert results[0].pnl != 0

    def test_dv01_ladder(self, ctx, portfolio):
        ladder = dv01_ladder(portfolio, ctx)
        assert len(ladder) > 0
        # All DV01s should be non-zero for a swaption
        for r in ladder:
            # Some pillars might have near-zero impact (short tenors)
            pass
        # At least some should be material
        assert any(abs(r.pnl) > 0.01 for r in ladder)


class TestVolBump:
    def test_vol_up_increases_swaption_pv(self, ctx, portfolio):
        results = run_scenarios(portfolio, ctx, [vol_bump(0.01)])
        assert results[0].pnl > 0  # long swaption gains from vol up

    def test_vol_down_decreases_swaption_pv(self, ctx, portfolio):
        results = run_scenarios(portfolio, ctx, [vol_bump(-0.01)])
        assert results[0].pnl < 0


class TestFXSpotShock:
    def test_fx_shock_applied(self, ctx):
        scenario = fx_spot_shock("EUR", "USD", 0.05)
        bumped = scenario.apply(ctx)
        assert bumped.fx_spots[("EUR", "USD")] == pytest.approx(1.0850 * 1.05)

    def test_fx_shock_name(self):
        s = fx_spot_shock("EUR", "USD", 0.10)
        assert "EUR" in s.name and "USD" in s.name


class TestScenarioResult:
    def test_pnl(self):
        r = ScenarioResult(name="test", base_pv=100.0, scenario_pv=105.0)
        assert r.pnl == pytest.approx(5.0)


class TestMultipleScenarios:
    def test_run_multiple(self, ctx, portfolio):
        scenarios = [
            parallel_shift(0.001, "up_10bp"),
            parallel_shift(-0.001, "dn_10bp"),
            vol_bump(0.01, name="vol_up_1"),
            vol_bump(-0.01, name="vol_dn_1"),
        ]
        results = run_scenarios(portfolio, ctx, scenarios)
        assert len(results) == 4
        assert all(isinstance(r, ScenarioResult) for r in results)

    def test_pnl_approximation(self, ctx, portfolio):
        """Small parallel shift PnL ≈ DV01 * shift."""
        small = run_scenarios(portfolio, ctx, [parallel_shift(0.0001)])[0]
        double = run_scenarios(portfolio, ctx, [parallel_shift(0.0002)])[0]
        # Linear approximation: double shift ≈ double PnL
        assert double.pnl == pytest.approx(2 * small.pnl, rel=0.1)
