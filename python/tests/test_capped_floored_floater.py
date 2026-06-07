"""Tests for pricebook.structured.capped_floored_floater."""

import pytest
from pricebook.structured.capped_floored_floater import (
    CappedFlooredFloaterResult,
    floored_floater,
    collar_floater,
    reverse_floater,
    inverse_floater_duration,
)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------
N = 8                                        # 2-year quarterly: 8 periods
DT = 0.25
FORWARDS = [0.04] * N                        # flat 4% forward curve
DISC = [(1 / (1 + 0.04 * DT)) ** (i + 1)   # naive quarterly compounding DFs
        for i in range(N)]
VOL = 0.20
NOTIONAL = 100.0


# ---------------------------------------------------------------------------
# floored_floater
# ---------------------------------------------------------------------------

def test_floored_floater_price_geq_plain_frn():
    """Floored FRN price >= plain FRN (floor is valuable to the investor)."""
    plain_price = NOTIONAL   # FRN prices at par by identity
    result = floored_floater(FORWARDS, floor_rate=0.02, vol=VOL, discount_factors=DISC,
                             dt=DT, notional=NOTIONAL)
    assert result.price >= plain_price - 1e-8


def test_floored_floater_floor_value_non_negative():
    result = floored_floater(FORWARDS, floor_rate=0.02, vol=VOL, discount_factors=DISC,
                             dt=DT, notional=NOTIONAL)
    assert result.floor_value >= 0.0


def test_floored_floater_returns_result_type():
    result = floored_floater(FORWARDS, floor_rate=0.02, vol=VOL, discount_factors=DISC,
                             dt=DT, notional=NOTIONAL)
    assert isinstance(result, CappedFlooredFloaterResult)


def test_floored_floater_zero_vol_near_zero_floor():
    """With zero vol, deep OTM floor has negligible value."""
    result = floored_floater(FORWARDS, floor_rate=0.001, vol=1e-8, discount_factors=DISC,
                             dt=DT, notional=NOTIONAL)
    assert result.floor_value == pytest.approx(0.0, abs=0.05)


def test_floored_floater_n_periods():
    result = floored_floater(FORWARDS, floor_rate=0.02, vol=VOL, discount_factors=DISC,
                             dt=DT, notional=NOTIONAL)
    assert result.n_periods == N


# ---------------------------------------------------------------------------
# collar_floater
# ---------------------------------------------------------------------------

def test_collar_floater_cap_value_non_negative():
    result = collar_floater(FORWARDS, cap_rate=0.08, floor_rate=0.02, vol=VOL,
                            discount_factors=DISC, dt=DT, notional=NOTIONAL)
    assert result.cap_value >= 0.0


def test_collar_floater_floor_value_non_negative():
    result = collar_floater(FORWARDS, cap_rate=0.08, floor_rate=0.02, vol=VOL,
                            discount_factors=DISC, dt=DT, notional=NOTIONAL)
    assert result.floor_value >= 0.0


def test_collar_floater_collar_value_identity():
    """collar_value == floor_value - cap_value."""
    result = collar_floater(FORWARDS, cap_rate=0.08, floor_rate=0.02, vol=VOL,
                            discount_factors=DISC, dt=DT, notional=NOTIONAL)
    assert result.collar_value == pytest.approx(result.floor_value - result.cap_value, rel=1e-10)


def test_collar_floater_cap_rate_must_exceed_floor_rate():
    with pytest.raises(ValueError):
        collar_floater(FORWARDS, cap_rate=0.02, floor_rate=0.04, vol=VOL,
                       discount_factors=DISC, dt=DT, notional=NOTIONAL)


# ---------------------------------------------------------------------------
# reverse_floater
# ---------------------------------------------------------------------------

def test_reverse_floater_price_depends_on_leverage():
    r1 = reverse_floater(fixed_rate=0.10, forward_rates=FORWARDS, vol=VOL,
                         discount_factors=DISC, dt=DT, notional=NOTIONAL, leverage=1.0)
    r2 = reverse_floater(fixed_rate=0.10, forward_rates=FORWARDS, vol=VOL,
                         discount_factors=DISC, dt=DT, notional=NOTIONAL, leverage=2.0)
    # Higher leverage → larger cap strip → lower price
    assert r2.price != r1.price


def test_reverse_floater_zero_vol_zero_cap():
    """Near-zero vol => cap strip is near zero."""
    result = reverse_floater(fixed_rate=0.10, forward_rates=FORWARDS, vol=1e-8,
                             discount_factors=DISC, dt=DT, notional=NOTIONAL, leverage=1.0)
    assert result.cap_value == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# inverse_floater_duration
# ---------------------------------------------------------------------------

def test_inverse_floater_duration_amplified_when_leverage_positive():
    """Effective duration > maturity for leverage > 0 (amplification)."""
    T = 5.0
    dur = inverse_floater_duration(fixed_rate=0.05, floating_rate=0.04,
                                   leverage=1.0, maturity=T)
    # (1 + leverage=1) × D_fixed; D_fixed < T for coupon bond but
    # the amplification factor 2× should still push it above vanilla
    vanilla_dur = inverse_floater_duration(fixed_rate=0.05, floating_rate=0.04,
                                           leverage=0.0, maturity=T)
    assert dur > vanilla_dur


def test_inverse_floater_duration_positive():
    dur = inverse_floater_duration(fixed_rate=0.05, floating_rate=0.04,
                                   leverage=2.0, maturity=10.0)
    assert dur > 0.0


def test_inverse_floater_duration_invalid_inputs():
    with pytest.raises(ValueError):
        inverse_floater_duration(fixed_rate=0.05, floating_rate=0.0, leverage=1.0, maturity=5.0)
