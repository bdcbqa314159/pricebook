"""Tests for DCF / enterprise valuation."""

import math
import pytest

from pricebook.pe.dcf import (
    WACCInputs, TerminalValue, EVBridge, DCFResult, FootballField,
    DCFModel, compute_wacc, terminal_value_perpetuity,
    terminal_value_exit_multiple, ev_to_equity,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def wacc_inputs():
    return WACCInputs(
        risk_free_rate=0.04,
        equity_risk_premium=0.05,
        beta=1.2,
        cost_of_debt=0.05,
        tax_rate=0.25,
        debt_to_total=0.40,
    )


@pytest.fixture
def base_model(wacc_inputs):
    return DCFModel(
        fcfs=[50, 55, 60, 65, 70],
        wacc_inputs=wacc_inputs,
        terminal_growth=0.02,
        net_debt=200,
    )


# ═══════════════════════════════════════════════════════════════
# WACC
# ═══════════════════════════════════════════════════════════════

class TestWACC:
    def test_cost_of_equity(self, wacc_inputs):
        # Re = 4% + 1.2 × 5% = 10%
        assert abs(wacc_inputs.cost_of_equity - 0.10) < 1e-10

    def test_wacc_calculation(self, wacc_inputs):
        # WACC = 0.60 × 10% + 0.40 × 5% × (1 - 0.25)
        #      = 0.06 + 0.015 = 7.5%
        assert abs(wacc_inputs.wacc - 0.075) < 1e-10

    def test_compute_wacc_function(self, wacc_inputs):
        assert abs(compute_wacc(wacc_inputs) - wacc_inputs.wacc) < 1e-15

    def test_to_dict(self, wacc_inputs):
        d = wacc_inputs.to_dict()
        assert "wacc" in d
        assert "cost_of_equity" in d


# ═══════════════════════════════════════════════════════════════
# Terminal Value
# ═══════════════════════════════════════════════════════════════

class TestTerminalValue:
    def test_perpetuity_gordon(self):
        # TV = 100 × (1 + 0.02) / (0.10 - 0.02) = 102 / 0.08 = 1275
        tv = terminal_value_perpetuity(100, 0.10, 0.02)
        assert abs(tv.value - 1275.0) < 0.01
        assert tv.method == "perpetuity_growth"

    def test_perpetuity_growth_ge_wacc(self):
        with pytest.raises(ValueError):
            terminal_value_perpetuity(100, 0.05, 0.05)
        with pytest.raises(ValueError):
            terminal_value_perpetuity(100, 0.05, 0.06)

    def test_exit_multiple(self):
        tv = terminal_value_exit_multiple(100, 8.0)
        assert tv.value == 800.0
        assert tv.method == "exit_multiple"
        assert tv.exit_multiple == 8.0

    def test_tv_to_dict(self):
        tv = terminal_value_perpetuity(100, 0.10, 0.02)
        d = tv.to_dict()
        assert "method" in d
        assert "growth_rate" in d


# ═══════════════════════════════════════════════════════════════
# EV Bridge
# ═══════════════════════════════════════════════════════════════

class TestEVBridge:
    def test_simple_bridge(self):
        b = ev_to_equity(1000, 400)
        assert b.equity_value == 600

    def test_with_minorities(self):
        b = ev_to_equity(1000, 400, minority=50)
        assert b.equity_value == 550

    def test_with_associates(self):
        b = ev_to_equity(1000, 400, associates=30)
        assert b.equity_value == 630

    def test_per_share(self):
        b = ev_to_equity(1000, 400, shares=100)
        assert b.equity_value_per_share == 6.0

    def test_no_shares(self):
        b = ev_to_equity(1000, 400)
        assert b.equity_value_per_share is None


# ═══════════════════════════════════════════════════════════════
# DCF Model
# ═══════════════════════════════════════════════════════════════

class TestDCFModel:
    def test_value_perpetuity(self, base_model):
        r = base_model.value("perpetuity_growth")
        assert r.enterprise_value > 0
        assert r.pv_fcfs > 0
        assert r.pv_terminal > 0
        assert r.ev_bridge.equity_value == r.enterprise_value - 200

    def test_value_exit_multiple(self, wacc_inputs):
        model = DCFModel(
            fcfs=[50, 55, 60, 65, 70],
            wacc_inputs=wacc_inputs,
            terminal_ebitda=100,
            terminal_multiple=8.0,
            net_debt=200,
        )
        r = model.value("exit_multiple")
        assert r.enterprise_value > 0
        assert r.terminal_value.method == "exit_multiple"

    def test_exit_multiple_missing_params(self, base_model):
        with pytest.raises(ValueError):
            base_model.value("exit_multiple")

    def test_tv_dominates(self, base_model):
        r = base_model.value()
        # Terminal value typically dominates (60-80% of EV)
        tv_pct = r.pv_terminal / r.enterprise_value
        assert tv_pct > 0.4  # at minimum

    def test_higher_growth_higher_value(self, wacc_inputs):
        m_low = DCFModel([50]*5, wacc_inputs, terminal_growth=0.01).value()
        m_high = DCFModel([50]*5, wacc_inputs, terminal_growth=0.03).value()
        assert m_high.enterprise_value > m_low.enterprise_value

    def test_empty_fcfs(self, wacc_inputs):
        with pytest.raises(ValueError):
            DCFModel([], wacc_inputs)

    def test_to_dict(self, base_model):
        d = base_model.value().to_dict()
        assert "enterprise_value" in d
        assert "ev_bridge" in d
        assert "terminal_value" in d


# ═══════════════════════════════════════════════════════════════
# Scenario Analysis
# ═══════════════════════════════════════════════════════════════

class TestScenarioAnalysis:
    def test_scenarios(self, base_model):
        results = base_model.scenario_analysis({
            "bull": {"terminal_growth": 0.03},
            "base": {},
            "bear": {"terminal_growth": 0.01},
        })
        assert len(results) == 3
        evs = {r.scenario: r.enterprise_value for r in results}
        assert evs["bull"] > evs["base"] > evs["bear"]

    def test_scenario_with_fcf_override(self, base_model):
        results = base_model.scenario_analysis({
            "upside": {"fcfs": [60, 70, 80, 90, 100]},
            "downside": {"fcfs": [40, 35, 30, 25, 20]},
        })
        assert results[0].enterprise_value > results[1].enterprise_value


# ═══════════════════════════════════════════════════════════════
# Football Field
# ═══════════════════════════════════════════════════════════════

class TestFootballField:
    def test_football_field_has_methods(self, base_model):
        ff = base_model.football_field()
        assert len(ff.methods) >= 1
        assert len(ff.low) == len(ff.methods)
        assert len(ff.mid) == len(ff.methods)
        assert len(ff.high) == len(ff.methods)

    def test_football_field_ordering(self, base_model):
        ff = base_model.football_field()
        for i in range(len(ff.methods)):
            assert ff.low[i] <= ff.mid[i] <= ff.high[i]

    def test_football_field_with_exit_multiple(self, wacc_inputs):
        model = DCFModel(
            fcfs=[50]*5, wacc_inputs=wacc_inputs,
            terminal_growth=0.02,
            terminal_ebitda=80, terminal_multiple=8.0,
            net_debt=200,
        )
        ff = model.football_field()
        assert len(ff.methods) >= 2  # perpetuity + exit multiple + wacc

    def test_to_dict(self, base_model):
        d = base_model.football_field().to_dict()
        assert "methods" in d
        assert "low" in d


# ═══════════════════════════════════════════════════════════════
# Hand Calculations
# ═══════════════════════════════════════════════════════════════

class TestHandCalc:
    def test_known_dcf(self):
        """Hand-verified DCF: 3 years of FCF=100, WACC=10%, g=2%."""
        wacc = WACCInputs(0.04, 0.05, 1.2, 0.05, 0.25, 0.40)
        # WACC = 7.5%
        assert abs(wacc.wacc - 0.075) < 1e-10

        model = DCFModel(fcfs=[100, 100, 100], wacc_inputs=wacc,
                         terminal_growth=0.02, net_debt=0)
        r = model.value()

        # PV of FCFs: 100/1.075 + 100/1.075^2 + 100/1.075^3
        pv1 = 100 / 1.075
        pv2 = 100 / 1.075 ** 2
        pv3 = 100 / 1.075 ** 3
        expected_pv_fcfs = pv1 + pv2 + pv3
        assert abs(r.pv_fcfs - expected_pv_fcfs) < 0.01

        # TV = 100 × 1.02 / (0.075 - 0.02) = 1854.545
        expected_tv = 100 * 1.02 / 0.055
        # PV of TV = TV / 1.075^3
        expected_pv_tv = expected_tv / 1.075 ** 3
        assert abs(r.pv_terminal - expected_pv_tv) < 0.01
