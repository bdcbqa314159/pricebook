"""Tests for pricebook.fixed_income.xccy_swaption."""

import math
import pytest
from datetime import date

from tests.conftest import make_flat_curve
from pricebook.fixed_income.xccy_swaption import (
    xccy_forward_spread,
    xccy_swaption_bachelier,
    xccy_swaption_black,
)

REF = date(2025, 1, 2)

# Domestic (higher rate) and foreign (lower rate) curves for a positive spread
DOM_RATE = 0.05
FOR_RATE = 0.02
FX_SPOT = 1.10   # domestic per foreign (e.g. USD/EUR ≈ 1.10)


def _curves():
    dom = make_flat_curve(REF, DOM_RATE)
    frn = make_flat_curve(REF, FOR_RATE)
    return dom, frn


# ---------------------------------------------------------------------------
# xccy_forward_spread
# ---------------------------------------------------------------------------

def test_forward_spread_positive_when_dom_gt_for():
    """Forward xccy spread is positive when domestic rate exceeds foreign rate."""
    dom, frn = _curves()
    spread = xccy_forward_spread(REF, dom, frn, FX_SPOT,
                                  start_years=1.0, tenor_years=5.0)
    assert spread > 0.0


def test_forward_spread_zero_when_rates_equal():
    """Forward spread is zero when domestic and foreign curves are identical."""
    curve = make_flat_curve(REF, 0.04)
    spread = xccy_forward_spread(REF, curve, curve, FX_SPOT,
                                  start_years=0.5, tenor_years=3.0)
    assert spread == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# xccy_swaption_black
# ---------------------------------------------------------------------------

def test_black_payer_price_positive():
    """Black payer xccy swaption price is strictly positive."""
    dom, frn = _curves()
    fwd = xccy_forward_spread(REF, dom, frn, FX_SPOT, start_years=1.0, tenor_years=5.0)
    res = xccy_swaption_black(
        REF, dom, frn, FX_SPOT,
        strike_spread=fwd,  # ATM
        vol=0.30,
        expiry_years=1.0, swap_tenor_years=5.0,
        notional=1_000_000.0, is_payer=True,
    )
    assert res.price > 0.0


def test_black_receiver_price_positive():
    """Black receiver xccy swaption price is strictly positive."""
    dom, frn = _curves()
    fwd = xccy_forward_spread(REF, dom, frn, FX_SPOT, start_years=1.0, tenor_years=5.0)
    res = xccy_swaption_black(
        REF, dom, frn, FX_SPOT,
        strike_spread=fwd,
        vol=0.30,
        expiry_years=1.0, swap_tenor_years=5.0,
        notional=1_000_000.0, is_payer=False,
    )
    assert res.price > 0.0


def test_black_forward_spread_stored_in_result():
    """forward_spread in the result matches xccy_forward_spread helper."""
    dom, frn = _curves()
    expected_fwd = xccy_forward_spread(REF, dom, frn, FX_SPOT, start_years=1.0, tenor_years=5.0)
    res = xccy_swaption_black(
        REF, dom, frn, FX_SPOT,
        strike_spread=0.001,
        vol=0.30, expiry_years=1.0, swap_tenor_years=5.0,
    )
    assert res.forward_spread == pytest.approx(expected_fwd, rel=1e-6)


def test_black_payer_receiver_parity_approximate():
    """For an ATM option: payer price ≈ receiver price (symmetric around forward)."""
    dom, frn = _curves()
    fwd = xccy_forward_spread(REF, dom, frn, FX_SPOT, start_years=1.0, tenor_years=5.0)
    payer = xccy_swaption_black(
        REF, dom, frn, FX_SPOT, strike_spread=fwd, vol=0.30,
        expiry_years=1.0, swap_tenor_years=5.0, is_payer=True,
    )
    receiver = xccy_swaption_black(
        REF, dom, frn, FX_SPOT, strike_spread=fwd, vol=0.30,
        expiry_years=1.0, swap_tenor_years=5.0, is_payer=False,
    )
    # ATM payer ≈ ATM receiver for Black-76 on lognormal spread
    assert payer.price == pytest.approx(receiver.price, rel=0.02)


# ---------------------------------------------------------------------------
# xccy_swaption_bachelier
# ---------------------------------------------------------------------------

def test_bachelier_payer_price_positive():
    """Bachelier payer xccy swaption price is positive."""
    dom, frn = _curves()
    fwd = xccy_forward_spread(REF, dom, frn, FX_SPOT, start_years=1.0, tenor_years=5.0)
    # Normal vol in spread units: ~5 bp/yr
    res = xccy_swaption_bachelier(
        REF, dom, frn, FX_SPOT,
        strike_spread=fwd,
        vol_normal=0.0005,
        expiry_years=1.0, swap_tenor_years=5.0,
        notional=1_000_000.0, is_payer=True,
    )
    assert res.price > 0.0


def test_bachelier_close_to_black_atm():
    """At-the-money, Bachelier and Black should agree to within ~5%."""
    dom, frn = _curves()
    fwd = xccy_forward_spread(REF, dom, frn, FX_SPOT, start_years=1.0, tenor_years=5.0)
    # Convert a 30% lognormal vol to approximate normal vol: vol_n ≈ F * vol_ln
    vol_ln = 0.30
    vol_n = fwd * vol_ln  # first-order approximation

    black_res = xccy_swaption_black(
        REF, dom, frn, FX_SPOT, strike_spread=fwd, vol=vol_ln,
        expiry_years=1.0, swap_tenor_years=5.0, notional=1_000_000.0,
    )
    bach_res = xccy_swaption_bachelier(
        REF, dom, frn, FX_SPOT, strike_spread=fwd, vol_normal=vol_n,
        expiry_years=1.0, swap_tenor_years=5.0, notional=1_000_000.0,
    )
    assert bach_res.price == pytest.approx(black_res.price, rel=0.05)


def test_bachelier_annuity_positive():
    """Annuity (sum of discount factors * accrual fractions) is positive."""
    dom, frn = _curves()
    res = xccy_swaption_bachelier(
        REF, dom, frn, FX_SPOT,
        strike_spread=0.001, vol_normal=0.0005,
        expiry_years=1.0, swap_tenor_years=5.0,
        notional=1_000_000.0,
    )
    assert res.annuity > 0.0
