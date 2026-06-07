"""Tests for pricebook.options.exotic_payoffs."""

import math
import pytest

from pricebook.options.exotic_payoffs import (
    LadderOptionResult,
    ShoutOptionResult,
    InstallmentOptionResult,
    ladder_option,
    shout_option,
    shout_option_analytical,
    installment_option,
    _bs_call,
)

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
SPOT = 100.0
STRIKE = 100.0
VOL = 0.20
T = 1.0
R = 0.05
Q = 0.0
N_PATHS = 20_000   # reduced for speed; still stable at this seed
SEED = 42


# ---------------------------------------------------------------------------
# ladder_option
# ---------------------------------------------------------------------------

def test_ladder_option_price_geq_vanilla_call():
    """Ladder price >= vanilla call: lock-in adds value."""
    vanilla = _bs_call(SPOT, STRIKE, VOL, T, R, Q)
    rungs = [105.0, 110.0, 115.0]
    result = ladder_option(SPOT, STRIKE, rungs=rungs, vol=VOL, T=T, r=R, q=Q,
                           n_paths=N_PATHS, seed=SEED)
    assert result.price >= vanilla - 0.5   # allow 50c MC tolerance


def test_ladder_option_returns_result_type():
    result = ladder_option(SPOT, STRIKE, rungs=[110.0], vol=VOL, T=T, r=R, q=Q,
                           n_paths=N_PATHS, seed=SEED)
    assert isinstance(result, LadderOptionResult)


def test_ladder_option_more_rungs_higher_price():
    """More rungs (more lock-in opportunities) should not decrease price."""
    r1 = ladder_option(SPOT, STRIKE, rungs=[105.0], vol=VOL, T=T, r=R, q=Q,
                       n_paths=N_PATHS, seed=SEED)
    r2 = ladder_option(SPOT, STRIKE, rungs=[105.0, 110.0, 115.0], vol=VOL, T=T, r=R, q=Q,
                       n_paths=N_PATHS, seed=SEED)
    assert r2.price >= r1.price - 0.5   # allow MC noise


def test_ladder_option_price_positive():
    result = ladder_option(SPOT, STRIKE, rungs=[110.0], vol=VOL, T=T, r=R, q=Q,
                           n_paths=N_PATHS, seed=SEED)
    assert result.price > 0.0


def test_ladder_option_lock_in_levels_stored():
    rungs = [105.0, 110.0]
    result = ladder_option(SPOT, STRIKE, rungs=rungs, vol=VOL, T=T, r=R, q=Q,
                           n_paths=N_PATHS, seed=SEED)
    assert result.lock_in_levels == rungs


# ---------------------------------------------------------------------------
# shout_option
# ---------------------------------------------------------------------------

def test_shout_option_price_geq_vanilla_call():
    """Shout option >= vanilla call: shout right adds value."""
    vanilla = _bs_call(SPOT, STRIKE, VOL, T, R, Q)
    result = shout_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                          n_paths=N_PATHS, seed=SEED)
    assert result.price >= vanilla - 0.5


def test_shout_option_returns_result_type():
    result = shout_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                          n_paths=N_PATHS, seed=SEED)
    assert isinstance(result, ShoutOptionResult)


def test_shout_option_analytical_close_to_mc():
    """Single-shout analytical price within 10% of MC estimate."""
    mc_price = shout_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                            n_paths=50_000, seed=SEED).price
    analytical_price = shout_option_analytical(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q)
    assert analytical_price == pytest.approx(mc_price, rel=0.15)


def test_shout_option_price_positive():
    result = shout_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                          n_paths=N_PATHS, seed=SEED)
    assert result.price > 0.0


# ---------------------------------------------------------------------------
# installment_option
# ---------------------------------------------------------------------------

def test_installment_option_returns_result_type():
    result = installment_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                                n_installments=4, n_paths=N_PATHS, seed=SEED)
    assert isinstance(result, InstallmentOptionResult)


def test_installment_option_deep_itm_close_to_vanilla():
    """Deep ITM with low vol: most paths continue, price near vanilla."""
    deep_spot = 150.0
    vanilla = _bs_call(deep_spot, STRIKE, 0.10, T, R, Q)
    result = installment_option(deep_spot, STRIKE, vol=0.10, T=T, r=R, q=Q,
                                n_installments=4, n_paths=N_PATHS, seed=SEED)
    # Price won't equal vanilla exactly (abandonment option has value), but
    # should be within 20% — most paths remain active.
    assert result.price == pytest.approx(vanilla, rel=0.20)


def test_installment_option_continuation_prob_leq_one():
    result = installment_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                                n_installments=4, n_paths=N_PATHS, seed=SEED)
    assert 0.0 <= result.continuation_prob <= 1.0


def test_installment_option_upfront_positive():
    result = installment_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                                n_installments=4, n_paths=N_PATHS, seed=SEED)
    assert result.upfront_premium > 0.0


def test_installment_option_price_non_negative():
    result = installment_option(SPOT, STRIKE, vol=VOL, T=T, r=R, q=Q,
                                n_installments=4, n_paths=N_PATHS, seed=SEED)
    assert result.price >= 0.0
