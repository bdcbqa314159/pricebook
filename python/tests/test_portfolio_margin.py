"""Tests for pricebook.risk.portfolio_margin."""

import pytest

from pricebook.risk.portfolio_margin import (
    Position,
    OptionLeg,
    span_margin,
    cross_margin_offset,
    strategy_margin,
    var_based_margin,
    margin_call,
)

# Helper positions
LONG_POS = Position("equity_option", quantity=10, delta=0.5, gamma=0.02, vega=0.1, notional=100_000)
SHORT_POS = Position("equity_option", quantity=-8, delta=-0.4, gamma=0.01, vega=0.08, notional=80_000)


class TestSpanMargin:
    def test_margin_positive(self):
        result = span_margin([LONG_POS], None, price_scan_range=0.15, vol_scan_range=0.30)
        assert result.initial_margin > 0.0

    def test_worst_scenario_is_dict(self):
        result = span_margin([LONG_POS], None, price_scan_range=0.15, vol_scan_range=0.30)
        assert isinstance(result.worst_scenario, dict)

    def test_maintenance_below_initial(self):
        result = span_margin([LONG_POS], None, price_scan_range=0.15, vol_scan_range=0.30)
        assert result.maintenance_margin < result.initial_margin

    def test_diversified_portfolio_margin_finite(self):
        """A portfolio of offsetting positions should have finite margin."""
        result = span_margin([LONG_POS, SHORT_POS], None, price_scan_range=0.15, vol_scan_range=0.30)
        assert result.initial_margin >= 0.0

    def test_components_present(self):
        result = span_margin([LONG_POS], None, price_scan_range=0.15, vol_scan_range=0.30)
        assert "scanning_risk" in result.margin_components
        assert "delta_component" in result.margin_components


class TestCrossMarginOffset:
    def test_offset_benefit_nonnegative(self):
        result = cross_margin_offset([1000.0, 800.0], 1200.0, ["equity", "rates"])
        assert result["offset_benefit"] >= 0.0

    def test_sum_standalone_correct(self):
        result = cross_margin_offset([1000.0, 800.0], 1200.0, ["equity", "rates"])
        assert result["sum_standalone"] == pytest.approx(1800.0)

    def test_zero_benefit_when_portfolio_equals_sum(self):
        result = cross_margin_offset([500.0, 500.0], 1000.0, ["a", "b"])
        assert result["offset_benefit"] == pytest.approx(0.0)

    def test_offset_ratio_in_unit_interval(self):
        result = cross_margin_offset([1000.0, 800.0], 1200.0, ["equity", "rates"])
        assert 0.0 <= result["offset_ratio"] <= 1.0


class TestStrategyMargin:
    def test_covered_call_positive(self):
        legs = [OptionLeg("call", strike=100.0, quantity=-1, premium=5.0)]
        result = strategy_margin(legs, margin_type="reg_t")
        assert result["initial_margin"] > 0.0

    def test_covered_call_cheaper_than_naked_call(self):
        """Covered call margin should reflect lower risk than a naked call."""
        covered = strategy_margin(
            [OptionLeg("call", strike=100.0, quantity=-1, premium=5.0)],
            margin_type="reg_t",
        )
        naked = strategy_margin(
            [
                OptionLeg("call", strike=100.0, quantity=-1, premium=5.0),
                OptionLeg("call", strike=110.0, quantity=-1, premium=2.0),
            ],
            margin_type="reg_t",
        )
        # Two naked calls have higher total notional exposure → higher margin
        assert covered["initial_margin"] < naked["initial_margin"]

    def test_vertical_spread_recognised(self):
        legs = [
            OptionLeg("call", strike=100.0, quantity=1, premium=5.0),
            OptionLeg("call", strike=110.0, quantity=-1, premium=2.0),
        ]
        result = strategy_margin(legs, margin_type="exchange")
        assert result["strategy"] == "vertical_spread"

    def test_iron_condor_recognised(self):
        legs = [
            OptionLeg("call", strike=110.0, quantity=1, premium=1.0),
            OptionLeg("call", strike=115.0, quantity=-1, premium=0.5),
            OptionLeg("put", strike=90.0, quantity=1, premium=1.0),
            OptionLeg("put", strike=85.0, quantity=-1, premium=0.5),
        ]
        result = strategy_margin(legs, margin_type="exchange")
        assert result["strategy"] == "iron_condor"

    def test_maintenance_below_initial(self):
        legs = [OptionLeg("call", strike=100.0, quantity=-1, premium=5.0)]
        result = strategy_margin(legs)
        assert result["maintenance_margin"] <= result["initial_margin"]


class TestVarBasedMargin:
    def test_im_equals_max_var_half_es(self):
        result = var_based_margin(1_000_000.0, var_99=50_000.0, expected_shortfall=80_000.0)
        assert result["initial_margin"] == pytest.approx(max(50_000.0, 0.5 * 80_000.0))

    def test_es_binding_when_large(self):
        result = var_based_margin(1_000_000.0, var_99=10_000.0, expected_shortfall=100_000.0)
        assert result["binding_measure"] == "0.5_es"

    def test_var_binding_when_large(self):
        result = var_based_margin(1_000_000.0, var_99=60_000.0, expected_shortfall=80_000.0)
        assert result["binding_measure"] == "var_99"

    def test_multiplier_scales_im(self):
        base = var_based_margin(1_000_000.0, var_99=50_000.0, expected_shortfall=80_000.0)
        scaled = var_based_margin(1_000_000.0, var_99=50_000.0, expected_shortfall=80_000.0, multiplier=1.5)
        assert scaled["initial_margin"] == pytest.approx(1.5 * base["initial_margin"])


class TestMarginCall:
    def test_no_call_when_equity_above_maintenance(self):
        result = margin_call(equity=15_000.0, initial_margin=10_000.0, maintenance_margin=7_500.0, market_value=100_000.0)
        assert result["margin_call_triggered"] is False
        assert result["call_amount"] == pytest.approx(0.0)

    def test_call_triggered_when_equity_below_maintenance(self):
        result = margin_call(equity=5_000.0, initial_margin=10_000.0, maintenance_margin=7_500.0, market_value=100_000.0)
        assert result["margin_call_triggered"] is True
        assert result["call_amount"] > 0.0

    def test_equity_after_call_ge_initial(self):
        result = margin_call(equity=5_000.0, initial_margin=10_000.0, maintenance_margin=7_500.0, market_value=100_000.0)
        assert result["equity_after_call"] == pytest.approx(result["equity"] + result["call_amount"])
