"""Tests for pricebook.structured.equity_linked_note."""

import math
import pytest

from pricebook.structured.equity_linked_note import (
    bear_eln,
    buffered_eln,
    capped_eln,
    digital_eln,
    worst_of_eln,
)

# Common parameters
SPOT = 1000.0
RATE = 0.05
DIV = 0.02
VOL = 0.20
T = 1.0
NOTIONAL = 1000.0


def _bond_floor(rate: float, T: float, notional: float) -> float:
    return notional * math.exp(-rate * T)


# ---------------------------------------------------------------------------
# buffered_eln
# ---------------------------------------------------------------------------

def test_buffered_eln_price_below_notional():
    """Buffered ELN investor pays less than notional (gives up some upside for buffer)."""
    res = buffered_eln(spot=SPOT, rate=RATE, vol=VOL, T=T, buffer=0.80, coupon=0.03,
                        notional=NOTIONAL)
    assert res.price < NOTIONAL


def test_buffered_eln_price_positive():
    """Price is strictly positive."""
    res = buffered_eln(spot=SPOT, rate=RATE, vol=VOL, T=T, buffer=0.85, coupon=0.02,
                        notional=NOTIONAL)
    assert res.price > 0.0


def test_buffered_eln_max_loss_matches_buffer():
    """Maximum loss equals (1 - buffer) * notional."""
    buffer = 0.80
    res = buffered_eln(spot=SPOT, rate=RATE, vol=VOL, T=T, buffer=buffer, coupon=0.02,
                        notional=NOTIONAL)
    assert res.max_loss == pytest.approx((1.0 - buffer) * NOTIONAL, rel=1e-9)


def test_buffered_eln_bond_floor_correct():
    """Bond floor equals the ZCB value of the notional."""
    res = buffered_eln(spot=SPOT, rate=RATE, vol=VOL, T=T, buffer=0.80, coupon=0.0,
                        notional=NOTIONAL)
    expected_floor = _bond_floor(RATE, T, NOTIONAL)
    assert res.bond_floor == pytest.approx(expected_floor, rel=1e-9)


# ---------------------------------------------------------------------------
# capped_eln
# ---------------------------------------------------------------------------

def test_capped_eln_price_above_bond_floor():
    """Capped ELN price exceeds the zero-coupon bond floor."""
    res = capped_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                      cap=1.20, notional=NOTIONAL)
    assert res.price > res.bond_floor


def test_capped_eln_higher_cap_higher_price():
    """A higher cap level increases the ELN price (wider call spread)."""
    low = capped_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                      cap=1.10, notional=NOTIONAL)
    high = capped_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                       cap=1.30, notional=NOTIONAL)
    assert high.price > low.price


def test_capped_eln_price_positive():
    """Capped ELN price is always positive."""
    res = capped_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                      cap=1.20, notional=NOTIONAL)
    assert res.price > 0.0


# ---------------------------------------------------------------------------
# bear_eln
# ---------------------------------------------------------------------------

def test_bear_eln_price_increases_with_vol():
    """Bear ELN price rises with implied volatility (ATM put increases with vol)."""
    low_vol = bear_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=0.10, T=T, notional=NOTIONAL)
    high_vol = bear_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=0.35, T=T, notional=NOTIONAL)
    assert high_vol.price > low_vol.price


def test_bear_eln_option_component_positive():
    """Option component of bear ELN is positive (long put)."""
    res = bear_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T, notional=NOTIONAL)
    assert res.option_component > 0.0


def test_bear_eln_price_above_bond_floor():
    """Bear ELN price is above the bond floor because of the long put."""
    res = bear_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T, notional=NOTIONAL)
    assert res.price > res.bond_floor


# ---------------------------------------------------------------------------
# digital_eln
# ---------------------------------------------------------------------------

def test_digital_eln_price_increases_with_coupon():
    """Higher digital coupon → higher ELN price."""
    low = digital_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                       barrier=SPOT * 1.05, coupon_if_above=0.03, notional=NOTIONAL)
    high = digital_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                        barrier=SPOT * 1.05, coupon_if_above=0.08, notional=NOTIONAL)
    assert high.price > low.price


def test_digital_eln_price_above_bond_floor():
    """Digital ELN price is above the bond floor (positive coupon component)."""
    res = digital_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                       barrier=SPOT * 1.05, coupon_if_above=0.05, notional=NOTIONAL)
    assert res.price > res.bond_floor


def test_digital_eln_max_loss_zero():
    """Capital-protected structure: max_loss = 0."""
    res = digital_eln(spot=SPOT, rate=RATE, div_yield=DIV, vol=VOL, T=T,
                       barrier=SPOT * 1.05, coupon_if_above=0.05, notional=NOTIONAL)
    assert res.max_loss == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# worst_of_eln vs single-underlying
# ---------------------------------------------------------------------------

def test_worst_of_eln_cheaper_than_single_underlying():
    """Worst-of ELN (basket, corr < 1) is cheaper than a single-asset ELN."""
    spots = [SPOT, SPOT]
    vols = [VOL, VOL]
    corr = [[1.0, 0.5], [0.5, 1.0]]
    divs = [DIV, DIV]
    barrier = 0.85
    coupon = 0.07

    # Single underlying: all-correlated basket collapses to single asset
    single_res = worst_of_eln(
        spots=[SPOT], vols=[VOL], correlations=[[1.0]],
        rate=RATE, div_yields=[DIV], T=T,
        barrier=barrier, coupon=coupon, notional=NOTIONAL,
        n_paths=20_000, seed=42,
    )
    # Worst-of basket (two imperfectly correlated assets)
    basket_res = worst_of_eln(
        spots=spots, vols=vols, correlations=corr,
        rate=RATE, div_yields=divs, T=T,
        barrier=barrier, coupon=coupon, notional=NOTIONAL,
        n_paths=20_000, seed=42,
    )
    assert basket_res.price < single_res.price
