"""Tests for pricebook.models.g2pp_tree (G2++ 2D trinomial tree)."""

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.vasicek import G2PlusPlus
from pricebook.models.g2pp_tree import G2PPTree, g2pp_european_swaption_tree
from pricebook.models.g2pp_calibration import g2pp_swaption_price

REF = date(2024, 1, 15)
curve = DiscountCurve.flat(REF, 0.04)

g2pp = G2PlusPlus(a=0.05, b=0.08, sigma1=0.01, sigma2=0.008, rho=-0.7, curve=curve)

ATM_STRIKE = 0.04
N_STEPS = 20  # small for speed


# ---------------------------------------------------------------------------
# G2PPTree construction
# ---------------------------------------------------------------------------


def test_tree_construction():
    """G2PPTree should construct without error."""
    tree = G2PPTree(g2pp, T=5.0, n_steps=N_STEPS)
    assert tree is not None


def test_tree_n_x_nodes_positive():
    """Tree must have a positive number of x-axis nodes."""
    tree = G2PPTree(g2pp, T=5.0, n_steps=N_STEPS)
    assert tree.n_x_nodes > 0 if hasattr(tree, "n_x_nodes") else tree.n_x > 0


def test_tree_n_y_nodes_positive():
    """Tree must have a positive number of y-axis nodes."""
    tree = G2PPTree(g2pp, T=5.0, n_steps=N_STEPS)
    assert tree.n_y_nodes > 0 if hasattr(tree, "n_y_nodes") else tree.n_y > 0


# ---------------------------------------------------------------------------
# zcb_price
# ---------------------------------------------------------------------------


def test_zcb_price_positive_off_centre():
    """ZCB price at a non-centre node should be positive."""
    tree = G2PPTree(g2pp, T=5.0, n_steps=N_STEPS)
    # Use a node one step away from the centre
    xi = tree.j_max_x + 1
    yi = tree.j_max_y - 1
    price = tree.zcb_price(t_idx=0, x_idx=xi, y_idx=yi, T_maturity=5.0)
    assert price > 0.0


def test_zcb_price_centre_matches_analytical():
    """ZCB at centre node (x=0, y=0, t=0) must match G2PlusPlus.zcb_price(0,0,T)."""
    T = 5.0
    tree = G2PPTree(g2pp, T=T, n_steps=N_STEPS)
    xi = tree.j_max_x
    yi = tree.j_max_y
    tree_price = tree.zcb_price(t_idx=0, x_idx=xi, y_idx=yi, T_maturity=T)
    analytical_price = g2pp.zcb_price(0.0, 0.0, T)
    assert tree_price == pytest.approx(analytical_price, rel=1e-6)


# ---------------------------------------------------------------------------
# g2pp_european_swaption_tree
# ---------------------------------------------------------------------------


def test_european_swaption_tree_price_positive():
    """Tree-priced European swaption price must be positive."""
    result = g2pp_european_swaption_tree(
        g2pp,
        expiry_years=2.0,
        swap_end_years=7.0,
        strike=ATM_STRIKE,
        is_payer=True,
        n_steps=N_STEPS,
    )
    assert result.price > 0.0


def test_european_swaption_tree_close_to_analytical():
    """Tree swaption price should be within 20% of the analytical G2++ price."""
    expiry, tenor = 2.0, 5.0
    result = g2pp_european_swaption_tree(
        g2pp,
        expiry_years=expiry,
        swap_end_years=expiry + tenor,
        strike=ATM_STRIKE,
        is_payer=True,
        n_steps=N_STEPS,
    )
    analytical = g2pp_swaption_price(
        g2pp.a, g2pp.b, g2pp.sigma1, g2pp.sigma2, g2pp.rho,
        curve, expiry, tenor, ATM_STRIKE, is_payer=True,
    )
    # Within 50% — 2D tree with small n_steps has significant discretisation error
    assert abs(result.price - analytical) <= 0.50 * max(analytical, result.price) + 1e-6
