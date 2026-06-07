"""Tests for pricebook.structured.cpdo."""

import pytest
from pricebook.structured.cpdo import (
    CPDOResult,
    CPDOMCResult,
    cpdo_simulate,
    cpdo_monte_carlo,
    cpdo_rating,
)

# ---------------------------------------------------------------------------
# Shared baseline parameters
# ---------------------------------------------------------------------------
BASE = dict(
    initial_nav=100.0,
    target_coupon=0.01,
    spread_income=0.04,
    spread_vol=0.05,
    max_leverage=15.0,
    n_periods=252,
    gap_risk=0.02,
    seed=42,
)


# ---------------------------------------------------------------------------
# cpdo_simulate
# ---------------------------------------------------------------------------

def test_simulate_returns_cpdo_result():
    result = cpdo_simulate(**BASE)
    assert isinstance(result, CPDOResult)


def test_simulate_nav_path_not_empty():
    result = cpdo_simulate(**BASE)
    assert len(result.nav_path) > 0


def test_simulate_leverage_path_length_matches_nav_path():
    result = cpdo_simulate(**BASE)
    assert len(result.leverage_path) == len(result.nav_path)


def test_simulate_nav_path_length_equals_n_periods_plus_one():
    result = cpdo_simulate(**BASE)
    assert len(result.nav_path) == BASE["n_periods"] + 1


def test_simulate_final_nav_non_negative():
    result = cpdo_simulate(**BASE)
    assert result.final_nav >= 0.0


def test_simulate_defaulted_is_bool():
    result = cpdo_simulate(**BASE)
    assert isinstance(result.defaulted, bool)


def test_simulate_to_dict_contains_expected_keys():
    result = cpdo_simulate(**BASE)
    d = result.to_dict()
    assert {"final_nav", "coupon_paid", "defaulted", "target_nav", "n_periods"} <= d.keys()


# ---------------------------------------------------------------------------
# cpdo_monte_carlo
# ---------------------------------------------------------------------------

MC_BASE = dict(**BASE, n_paths=500)


def test_mc_returns_cpdo_mc_result():
    result = cpdo_monte_carlo(**MC_BASE)
    assert isinstance(result, CPDOMCResult)


def test_mc_probabilities_sum_leq_one():
    result = cpdo_monte_carlo(**MC_BASE)
    assert result.success_prob + result.default_prob <= 1.0 + 1e-10


def test_mc_rating_is_string():
    result = cpdo_monte_carlo(**MC_BASE)
    assert isinstance(result.rating_implied, str)


def test_mc_higher_spread_vol_raises_default_prob():
    low_vol = cpdo_monte_carlo(**{**MC_BASE, "spread_vol": 0.02})
    high_vol = cpdo_monte_carlo(**{**MC_BASE, "spread_vol": 0.30})
    assert high_vol.default_prob >= low_vol.default_prob


def test_mc_higher_max_leverage_raises_default_prob():
    low_lev = cpdo_monte_carlo(**{**MC_BASE, "max_leverage": 3.0})
    high_lev = cpdo_monte_carlo(**{**MC_BASE, "max_leverage": 30.0})
    assert high_lev.default_prob >= low_lev.default_prob


# ---------------------------------------------------------------------------
# cpdo_rating
# ---------------------------------------------------------------------------

def test_rating_zero_prob_is_aaa():
    assert cpdo_rating(0.0) == "AAA"


def test_rating_one_prob_is_d():
    assert cpdo_rating(1.0) == "D"


def test_rating_moderate_prob_is_investment_grade():
    # 0.05 (5%) → BBB bucket
    rating = cpdo_rating(0.05)
    assert rating.startswith("BBB")


def test_rating_high_prob_is_speculative():
    rating = cpdo_rating(0.30)
    assert rating in {"B+", "B", "B-", "CCC/CC", "D"}
