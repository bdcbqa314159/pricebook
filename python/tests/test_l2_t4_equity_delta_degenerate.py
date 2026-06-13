"""Regression for L2 Wave-2 audit — `equity_delta` degenerate-input branch.

Pre-fix the ``T <= 0 or vol <= 0`` branch had three coupled bugs:

1. Compared ``spot`` to ``strike`` instead of ``forward`` to ``strike``.
   For non-zero dividend yield, ``forward = spot · exp((r - q)·T)`` can
   differ from spot; under zero vol the terminal spot equals forward,
   so payoff intrinsic depends on FORWARD, not spot.

2. The ITM magnitude was a literal ``1.0`` instead of ``exp(-q·T)``.
   Spot delta of a deep ITM call is ``exp(-q·T)`` (the dividend discount),
   not unity.  Inconsistent with `black76_delta` which returns ``±df``.

3. ATM (forward == strike) returned 0 for both call and put instead of
   the standard ``±0.5·exp(-q·T)`` limit.

Post-fix all three cases match `black76_delta`'s convention scaled by
``exp(-q·T)`` (the spot-to-forward chain rule).
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.options.equity_option import equity_delta


class TestEquityDeltaDegenerate_HighDividendCase:
    """High dividend yield: forward < spot.  An option that is ITM on
    spot can still be OTM on forward."""

    def test_call_itm_on_spot_but_otm_on_forward_returns_zero(self):
        """Spot=100, strike=99, r=0, q=0.10, T=1 → forward = 100·exp(-0.10) ≈ 90.48 < 99.
        The call is OTM on forward, so under zero-vol the payoff is 0 and
        delta is 0.  Pre-fix this returned 1.0 (used spot > strike)."""
        d = equity_delta(spot=100.0, strike=99.0, rate=0.0, vol=0.0, T=1.0,
                         option_type=OptionType.CALL, div_yield=0.10)
        assert d == pytest.approx(0.0, abs=1e-12), \
            f"deep-dividend ITM-on-spot/OTM-on-forward should give delta 0, got {d}"

    def test_put_otm_on_spot_but_itm_on_forward_returns_negative_exp_minus_qT(self):
        """Spot=100, strike=99, q=0.10 → forward ≈ 90.48 < 99.
        Put is ITM on forward, so delta should be -exp(-qT) = -exp(-0.10) ≈ -0.9048.
        Pre-fix this returned 0.0 (used spot < strike → false)."""
        d = equity_delta(spot=100.0, strike=99.0, rate=0.0, vol=0.0, T=1.0,
                         option_type=OptionType.PUT, div_yield=0.10)
        assert d == pytest.approx(-math.exp(-0.10), abs=1e-12), \
            f"expected -exp(-0.10) ≈ -0.9048, got {d}"


class TestEquityDeltaDegenerate_Magnitude:
    """Magnitude must equal ``exp(-q·T)`` for clear ITM cases, not literal 1.0."""

    def test_itm_call_magnitude_is_exp_minus_qT(self):
        """Pre-fix returned 1.0 (no dividend scaling).  Post-fix returns exp(-q·T)."""
        d = equity_delta(spot=150.0, strike=100.0, rate=0.05, vol=0.0, T=2.0,
                         option_type=OptionType.CALL, div_yield=0.03)
        # forward = 150·exp((0.05-0.03)·2) = 150·exp(0.04) ≈ 156.12 > 100 → ITM
        assert d == pytest.approx(math.exp(-0.03 * 2.0), abs=1e-12)
        assert d != pytest.approx(1.0)

    def test_itm_put_magnitude_is_minus_exp_minus_qT(self):
        d = equity_delta(spot=50.0, strike=100.0, rate=0.05, vol=0.0, T=2.0,
                         option_type=OptionType.PUT, div_yield=0.03)
        assert d == pytest.approx(-math.exp(-0.03 * 2.0), abs=1e-12)


class TestEquityDeltaDegenerate_ATM:
    """At-the-money (forward == strike) should give ±0.5·exp(-qT), matching
    the standard ATM-at-expiry limit and consistent with `black76_delta`."""

    def test_atm_call_returns_half_exp_minus_qT(self):
        # Pick spot, r, q, T such that forward = strike.  forward = S·exp((r-q)T).
        # With r = q, forward = S regardless of T.
        d = equity_delta(spot=100.0, strike=100.0, rate=0.05, vol=0.0, T=2.0,
                         option_type=OptionType.CALL, div_yield=0.05)
        assert d == pytest.approx(0.5 * math.exp(-0.05 * 2.0), abs=1e-12)

    def test_atm_put_returns_minus_half_exp_minus_qT(self):
        d = equity_delta(spot=100.0, strike=100.0, rate=0.05, vol=0.0, T=2.0,
                         option_type=OptionType.PUT, div_yield=0.05)
        assert d == pytest.approx(-0.5 * math.exp(-0.05 * 2.0), abs=1e-12)


class TestEquityDeltaInteriorUnchanged:
    """Spot-delta in the interior (T > 0, vol > 0) must be unaffected by the fix."""

    def test_interior_delta_unchanged(self):
        d = equity_delta(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
                         option_type=OptionType.CALL, div_yield=0.02)
        # ATM call delta with these params is approximately exp(-q)·N(d1) ≈ 0.59
        assert 0.55 < d < 0.65
