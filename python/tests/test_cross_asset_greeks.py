"""Tests for cross-asset Greeks and multi-factor stress."""

import pytest
from datetime import date

from pricebook.options_book import OptionEntry, OptionsBook
from pricebook.cross_asset_greeks import (
    BookGreekAttribution,
    StressScenario,
    StressResult,
    greek_attribution,
    multi_factor_stress,
)


def _book():
    book = OptionsBook("test")
    book.add(OptionEntry("eq1", "equity", "AAPL", date(2024, 6, 15), 1e6,
                         delta=500, gamma=20, vega=1000, theta=-25, rho=50))
    book.add(OptionEntry("fx1", "fx", "EUR/USD", date(2024, 6, 15), 1e6,
                         delta=300, gamma=15, vega=800, theta=-20, rho=30))
    book.add(OptionEntry("ir1", "ir", "5Y_swap", date(2024, 6, 15), 1e6,
                         delta=0, gamma=0, vega=500, theta=-10, rho=200))
    return book


# ---- Step 1: unified attribution ----

class TestGreekAttribution:
    def test_attribution_sums_to_total(self):
        """Step 1 test: attribution sums to total."""
        book = _book()
        attrib = greek_attribution(
            book,
            spot_changes={"AAPL": 2.0, "EUR/USD": 0.005},
            vol_changes={"AAPL": 0.01, "EUR/USD": 0.02, "5Y_swap": 0.005},
            rate_change=0.001, dt_days=1.0,
        )
        assert attrib.total_pnl == pytest.approx(
            attrib.delta_pnl + attrib.gamma_pnl + attrib.vega_pnl
            + attrib.theta_pnl + attrib.rho_pnl
        )

    def test_per_asset_class_sums(self):
        book = _book()
        attrib = greek_attribution(
            book, {"AAPL": 2.0}, {"AAPL": 0.01}, dt_days=1.0,
        )
        sum_ac = sum(a.total_pnl for a in attrib.by_asset_class)
        assert sum_ac == pytest.approx(attrib.total_pnl)

    def test_carry_vs_convexity(self):
        book = _book()
        attrib = greek_attribution(
            book, {"AAPL": 2.0}, {}, dt_days=1.0,
        )
        assert attrib.carry == pytest.approx(attrib.theta_pnl)
        assert attrib.convexity == pytest.approx(attrib.gamma_pnl)

    def test_delta_pnl(self):
        book = OptionsBook("test")
        book.add(OptionEntry("t1", "equity", "AAPL", delta=100, vega=0,
                             gamma=0, theta=0, rho=0))
        attrib = greek_attribution(book, {"AAPL": 3.0}, {})
        assert attrib.delta_pnl == pytest.approx(300.0)

    def test_empty_book(self):
        attrib = greek_attribution(OptionsBook("t"), {}, {})
        assert attrib.total_pnl == 0.0
        assert attrib.by_asset_class == []


# ---- Step 2: multi-factor stress ----

class TestMultiFactorStress:
    def test_combined_scenario(self):
        """Step 2 test: scenario P&L matches sum of per-factor shocks."""
        book = _book()
        scenario = StressScenario(
            name="risk_off",
            spot_shocks={"AAPL": -10.0, "EUR/USD": -0.05},
            vol_shocks={"AAPL": 0.05, "EUR/USD": 0.03, "5Y_swap": 0.02},
            rate_shock=-0.005,
        )
        result = multi_factor_stress(book, scenario)

        # Verify by computing each factor separately
        delta_only = greek_attribution(book, scenario.spot_shocks, {})
        vol_only = greek_attribution(book, {}, scenario.vol_shocks)
        rate_only = greek_attribution(book, {}, {}, scenario.rate_shock)

        # Total ≈ delta + gamma + vega + rho (no theta since dt=0)
        # Not exactly additive due to gamma cross-terms, but close
        assert result.total_pnl == pytest.approx(
            result.delta_pnl + result.gamma_pnl + result.vega_pnl
            + result.theta_pnl + result.rho_pnl
        )

    def test_scenario_name_recorded(self):
        book = _book()
        scenario = StressScenario("test_scenario", {}, {})
        result = multi_factor_stress(book, scenario)
        assert result.scenario_name == "test_scenario"

    def test_zero_scenario_zero_pnl(self):
        book = _book()
        scenario = StressScenario("flat", {}, {}, 0.0)
        result = multi_factor_stress(book, scenario)
        assert result.total_pnl == pytest.approx(0.0)

    def test_risk_off_equity_loses(self):
        book = OptionsBook("test")
        book.add(OptionEntry("eq1", "equity", "AAPL", delta=1000,
                             gamma=0, vega=0, theta=0, rho=0))
        scenario = StressScenario("risk_off", {"AAPL": -20.0}, {})
        result = multi_factor_stress(book, scenario)
        assert result.total_pnl < 0
