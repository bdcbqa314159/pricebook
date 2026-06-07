"""Tests for pricebook.options.bermudan_swaption_g2pp."""

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.vasicek import G2PlusPlus
from pricebook.options.bermudan_swaption_g2pp import (
    bermudan_swaption_g2pp_tree,
    bermudan_swaption_g2pp_lsm,
    g2pp_vs_hw1f_bermudan,
)

REF = date(2024, 1, 15)
curve = DiscountCurve.flat(REF, 0.04)

g2pp = G2PlusPlus(a=0.05, b=0.08, sigma1=0.01, sigma2=0.008, rho=-0.7, curve=curve)

ATM_STRIKE = 0.04
EXERCISE_YEARS = [1.0, 2.0, 3.0, 4.0]
SWAP_END = 7.0
N_STEPS = 30  # small for speed


# ---------------------------------------------------------------------------
# bermudan_swaption_g2pp_tree
# ---------------------------------------------------------------------------


def test_bermudan_tree_price_positive():
    """Bermudan swaption tree price must be strictly positive."""
    result = bermudan_swaption_g2pp_tree(
        g2pp,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        n_steps=N_STEPS,
    )
    assert result.price > 0.0


def test_bermudan_tree_price_ge_european():
    """Bermudan price must be >= its European (last-exercise-only) benchmark."""
    result = bermudan_swaption_g2pp_tree(
        g2pp,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        n_steps=N_STEPS,
    )
    assert result.price >= result.european_price - 1e-8


def test_bermudan_tree_early_exercise_premium_nonneg():
    """Early exercise premium must be non-negative."""
    result = bermudan_swaption_g2pp_tree(
        g2pp,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        n_steps=N_STEPS,
    )
    assert result.early_exercise_premium >= -1e-8


# ---------------------------------------------------------------------------
# bermudan_swaption_g2pp_lsm
# ---------------------------------------------------------------------------


def test_bermudan_lsm_price_positive():
    """Bermudan swaption LSM price must be strictly positive."""
    result = bermudan_swaption_g2pp_lsm(
        g2pp,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        n_paths=5000,
        seed=42,
    )
    assert result.price > 0.0


# ---------------------------------------------------------------------------
# g2pp_vs_hw1f_bermudan
# ---------------------------------------------------------------------------


def test_g2pp_vs_hw1f_bermudan_returns_two_factor_premium_key():
    """g2pp_vs_hw1f_bermudan result dict must contain the two_factor_premium key."""
    result = g2pp_vs_hw1f_bermudan(
        curve=curve,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        hw_a=0.05,
        hw_sigma=0.01,
        g2pp_params={
            "a": 0.05, "b": 0.08,
            "sigma1": 0.01, "sigma2": 0.008,
            "rho": -0.7,
        },
        n_steps=N_STEPS,
    )
    assert "two_factor_premium" in result


def test_g2pp_vs_hw1f_bermudan_prices_positive():
    """Both HW1F and G2++ prices returned by comparison function must be positive."""
    result = g2pp_vs_hw1f_bermudan(
        curve=curve,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        hw_a=0.05,
        hw_sigma=0.01,
        g2pp_params={
            "a": 0.05, "b": 0.08,
            "sigma1": 0.01, "sigma2": 0.008,
            "rho": -0.7,
        },
        n_steps=N_STEPS,
    )
    assert result["hw1f_price"] > 0.0
    assert result["g2pp_price"] > 0.0


def test_bermudan_tree_method_label():
    """Result method label should be 'tree'."""
    result = bermudan_swaption_g2pp_tree(
        g2pp,
        exercise_years=EXERCISE_YEARS,
        swap_end_years=SWAP_END,
        strike=ATM_STRIKE,
        n_steps=N_STEPS,
    )
    assert result.method == "tree"
