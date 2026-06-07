"""Tests for pricebook.commodity.carbon_credit and pricebook.commodity.freight."""

import math
import pytest

from pricebook.commodity.carbon_credit import (
    CarbonFuturesResult,
    CarbonOptionResult,
    carbon_futures_price,
    carbon_option_price,
    compliance_value,
)
from pricebook.commodity.freight import (
    FFAResult,
    FreightOptionResult,
    ffa_price,
    freight_option_price,
    time_charter_equivalent,
)


# ---------------------------------------------------------------------------
# carbon_futures_price
# ---------------------------------------------------------------------------

def test_carbon_futures_fair_value_exceeds_spot_when_positive_carry():
    """F > S when rate + storage > convenience yield."""
    result = carbon_futures_price(spot=50.0, rate=0.05, storage_cost=0.005,
                                  convenience_yield=0.01, T=1.0)
    assert result.fair_value > 50.0


def test_carbon_futures_returns_result_type():
    result = carbon_futures_price(spot=50.0, rate=0.05, storage_cost=0.005,
                                  convenience_yield=0.01, T=1.0)
    assert isinstance(result, CarbonFuturesResult)


def test_carbon_futures_roundtrip():
    """Fair value ≈ spot when net carry is zero (r + u = y)."""
    result = carbon_futures_price(spot=50.0, rate=0.05, storage_cost=0.0,
                                  convenience_yield=0.05, T=1.0)
    assert result.fair_value == pytest.approx(50.0, rel=1e-6)


def test_carbon_futures_carry_cost_sign():
    """carry_cost > 0 when positive carry."""
    result = carbon_futures_price(spot=50.0, rate=0.05, storage_cost=0.005,
                                  convenience_yield=0.01, T=1.0)
    assert result.carry_cost > 0.0


# ---------------------------------------------------------------------------
# carbon_option_price
# ---------------------------------------------------------------------------

def test_carbon_option_call_positive():
    result = carbon_option_price(spot=50.0, strike=50.0, rate=0.05,
                                 vol=0.30, T=1.0, option_type="call")
    assert result.price > 0.0


def test_carbon_option_delta_in_unit_interval_for_call():
    result = carbon_option_price(spot=50.0, strike=50.0, rate=0.05,
                                 vol=0.30, T=1.0, option_type="call")
    assert 0.0 < result.delta < 1.0


def test_carbon_option_returns_result_type():
    result = carbon_option_price(spot=50.0, strike=50.0, rate=0.05,
                                 vol=0.30, T=1.0, option_type="call")
    assert isinstance(result, CarbonOptionResult)


# ---------------------------------------------------------------------------
# compliance_value
# ---------------------------------------------------------------------------

def test_compliance_value_surplus_positive():
    result = compliance_value(allowances_held=1000.0, emissions=800.0,
                              spot_price=50.0, penalty_per_ton=100.0)
    assert result["position"] > 0
    assert result["position_value"] > 0.0


def test_compliance_value_deficit_negative_position():
    result = compliance_value(allowances_held=800.0, emissions=1000.0,
                              spot_price=50.0, penalty_per_ton=100.0)
    assert result["position"] < 0
    assert result["position_value"] < 0.0


def test_compliance_value_surplus_zero_compliance_cost():
    result = compliance_value(allowances_held=1000.0, emissions=800.0,
                              spot_price=50.0, penalty_per_ton=100.0)
    assert result["compliance_cost"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ffa_price
# ---------------------------------------------------------------------------

def test_ffa_price_avg_settlement_equals_mean():
    rates = [10000.0, 12000.0, 11000.0, 13000.0]
    result = ffa_price(spot_rate=10000.0, forward_curve_rates=rates, T=0.25)
    assert result.fair_value == pytest.approx(sum(rates) / len(rates), rel=1e-10)


def test_ffa_price_returns_ffa_result():
    result = ffa_price(spot_rate=10000.0, forward_curve_rates=[11000.0], T=0.25)
    assert isinstance(result, FFAResult)


def test_ffa_price_basis_definition():
    rates = [12000.0]
    result = ffa_price(spot_rate=10000.0, forward_curve_rates=rates, T=0.25)
    assert result.basis == pytest.approx(result.fair_value - 10000.0, rel=1e-10)


# ---------------------------------------------------------------------------
# freight_option_price
# ---------------------------------------------------------------------------

def test_freight_option_call_positive():
    result = freight_option_price(ffa_rate=12000.0, strike=12000.0,
                                  vol=0.30, T=1.0, r=0.05, option_type="call")
    assert result.price > 0.0


def test_freight_option_put_positive():
    result = freight_option_price(ffa_rate=12000.0, strike=12000.0,
                                  vol=0.30, T=1.0, r=0.05, option_type="put")
    assert result.price > 0.0


def test_freight_option_returns_result_type():
    result = freight_option_price(ffa_rate=12000.0, strike=12000.0,
                                  vol=0.30, T=1.0, r=0.05)
    assert isinstance(result, FreightOptionResult)


# ---------------------------------------------------------------------------
# time_charter_equivalent
# ---------------------------------------------------------------------------

def test_tce_basic_math():
    """TCE = (revenue - costs) / total_days."""
    revenue_per_day = 15000.0
    days_at_sea = 20.0
    days_in_port = 5.0
    voyage_costs = 50000.0
    port_costs = 20000.0

    result = time_charter_equivalent(
        revenue_per_day=revenue_per_day,
        voyage_costs=voyage_costs,
        port_costs=port_costs,
        days_at_sea=days_at_sea,
        days_in_port=days_in_port,
    )
    expected_gross = revenue_per_day * days_at_sea
    expected_net = expected_gross - voyage_costs - port_costs
    expected_tce = expected_net / (days_at_sea + days_in_port)

    assert result["tce"] == pytest.approx(expected_tce, rel=1e-10)
    assert result["gross_revenue"] == pytest.approx(expected_gross, rel=1e-10)
