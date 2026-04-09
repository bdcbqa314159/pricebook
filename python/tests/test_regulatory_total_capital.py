"""Tests for total capital aggregation and unified RegulatoryPortfolio."""

import pytest

from pricebook.regulatory.total_capital import (
    calculate_total_rwa, calculate_capital_ratios,
    RegulatoryPortfolio, RegulatoryPosition,
)


# ---- Total RWA aggregation ----

class TestTotalRWA:
    def test_credit_only_sa(self):
        exposures = [
            {"ead": 10_000_000, "asset_class": "corporate", "rating": "BBB"},
            {"ead": 5_000_000, "asset_class": "corporate", "rating": "A"},
        ]
        r = calculate_total_rwa(credit_exposures_sa=exposures)
        assert r["credit_risk"]["sa"] > 0
        assert r["total_rwa"] > 0

    def test_credit_only_irb(self):
        exposures = [
            {"ead": 10_000_000, "pd": 0.01, "lgd": 0.45, "maturity": 5},
        ]
        r = calculate_total_rwa(credit_exposures_irb=exposures, apply_output_floor=False)
        assert r["credit_risk"]["irb"] > 0
        assert r["credit_risk"]["approach"] == "F-IRB"

    def test_with_market_risk(self):
        delta_pos = {"GIRR": [{"bucket": "USD", "sensitivity": 1_000_000, "risk_weight": 11}]}
        r = calculate_total_rwa(trading_positions=delta_pos)
        assert r["market_risk"]["rwa"] > 0

    def test_with_operational_risk(self):
        r = calculate_total_rwa(business_indicator=2_000_000_000)
        assert r["operational_risk"]["rwa"] > 0

    def test_full_aggregation(self):
        r = calculate_total_rwa(
            credit_exposures_sa=[
                {"ead": 100_000_000, "asset_class": "corporate", "rating": "BBB"},
            ],
            cva_counterparties=[
                {"ead": 10_000_000, "rating": "BBB", "maturity": 5},
            ],
            business_indicator=1_000_000_000,
        )
        assert r["credit_risk"]["rwa"] > 0
        assert r["cva_risk"]["rwa"] > 0
        assert r["operational_risk"]["rwa"] > 0
        assert r["total_rwa"] > 0

    def test_output_floor_binding(self):
        sa = [{"ead": 100_000_000, "asset_class": "corporate", "rating": "BBB"}]  # 75M
        irb = [{"ead": 100_000_000, "pd": 0.001, "lgd": 0.45, "maturity": 1}]  # very small
        r = calculate_total_rwa(
            credit_exposures_sa=sa,
            credit_exposures_irb=irb,
            apply_output_floor=True,
            floor_year=2028,
        )
        assert "output_floor" in r
        assert r["output_floor"]["floor_is_binding"]


# ---- Capital ratios ----

class TestCapitalRatios:
    def test_compliant(self):
        r = calculate_capital_ratios(
            total_rwa=100_000_000_000,
            cet1_capital=12_000_000_000,
            at1_capital=2_000_000_000,
            tier2_capital=2_000_000_000,
        )
        assert r["cet1_ratio_pct"] == pytest.approx(12.0)
        assert r["tier1_ratio_pct"] == pytest.approx(14.0)
        assert r["total_ratio_pct"] == pytest.approx(16.0)
        assert r["compliance"]["cet1"]
        assert r["compliance"]["tier1"]
        assert r["compliance"]["total"]

    def test_breach(self):
        r = calculate_capital_ratios(
            total_rwa=100_000_000_000,
            cet1_capital=4_000_000_000,
        )
        assert not r["compliance"]["cet1"]

    def test_with_buffers(self):
        r = calculate_capital_ratios(
            total_rwa=100_000_000_000,
            cet1_capital=10_000_000_000,
            countercyclical_buffer=0.01,
            gsib_buffer=0.02,
        )
        # Combined CET1 = 4.5% + 2.5% + 1% + 2% = 10%
        assert r["requirements"]["combined_cet1_pct"] == pytest.approx(10.0)
        # CET1 ratio = 10%, compliant exactly
        assert r["compliance"]["cet1"]


# ---- Regulatory Portfolio ----

class TestRegulatoryPortfolio:
    def test_create_empty(self):
        port = RegulatoryPortfolio(name="Test")
        assert port.name == "Test"
        assert port.n_positions == 0
        assert port.total_notional == 0

    def test_add_positions(self):
        port = RegulatoryPortfolio()
        port.add("Apple", notional=10_000_000, rating="AA", tenor_years=5)
        port.add("Microsoft", notional=15_000_000, rating="AAA", tenor_years=7)
        assert port.n_positions == 2
        assert port.n_issuers == 2
        assert port.total_notional == 25_000_000

    def test_chaining(self):
        port = (RegulatoryPortfolio()
                .add("A", notional=1_000_000, rating="BBB")
                .add("B", notional=2_000_000, rating="A"))
        assert port.n_positions == 2

    def test_to_irc_positions(self):
        port = RegulatoryPortfolio()
        port.add("Apple", notional=10_000_000, rating="AA", tenor_years=5)
        irc_pos = port.to_irc_positions()
        assert len(irc_pos) == 1
        assert irc_pos[0].issuer == "Apple"

    def test_irc(self):
        port = RegulatoryPortfolio()
        port.add("Corp_A", notional=10_000_000, rating="BBB", tenor_years=5)
        port.add("Corp_B", notional=5_000_000, rating="BB", tenor_years=3)
        r = port.irc(num_simulations=5_000)
        assert r["irc"] >= 0
        assert r["num_positions"] == 2

    def test_var(self):
        port = RegulatoryPortfolio()
        port.add("Apple", notional=10_000_000, rating="AA")
        r = port.var(confidence=0.99)
        assert r["var_pct"] > 0
        assert r["var_abs"] > 0

    def test_credit_rwa_sa(self):
        port = RegulatoryPortfolio()
        port.add("Apple", notional=10_000_000, rating="A")
        r = port.credit_rwa("sa")
        assert r["credit_risk"]["rwa"] > 0
        assert r["credit_risk"]["approach"] == "SA-CR"

    def test_risk_summary(self):
        port = RegulatoryPortfolio(name="Test Book")
        port.add("Corp_A", notional=10_000_000, rating="BBB", tenor_years=5)
        port.add("Corp_B", notional=5_000_000, rating="A", tenor_years=3)
        r = port.risk_summary()
        assert r["name"] == "Test Book"
        assert r["n_positions"] == 2
        assert r["n_issuers"] == 2
        assert r["total_notional"] == 15_000_000
        assert r["credit_rwa"] > 0
        assert r["total_rwa"] > 0
        assert r["irc"] >= 0
