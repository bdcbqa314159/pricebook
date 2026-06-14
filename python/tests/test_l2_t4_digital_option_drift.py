"""Regression for L2 T4 audit of `desks.api.digital_option`:

Pre-fix the d2 formula used ``ln(spot/strike)`` directly, omitting the
risk-neutral drift term ``r·T = -ln(df)``.  For df=1 (no rates) the
formula was correct.  For df<1 (positive rates), the ATM-forward
digital should price at ~0.5 × df × payout, but the spot-anchored
formula gave a different (skewed) value.

Fix: build the forward ``F = spot / df`` and use ``ln(F/strike)`` so
the d2 formula matches Black-76 risk-neutral pricing.
"""

from __future__ import annotations

import math

import pytest

import pricebook.desks.api as pb


class TestDigitalATMForward:
    """At forward = strike (ATM-F), the digital call should price at
    exactly 0.5 × df × payout (probability that lognormal forward
    finishes above strike is 0.5 with no drift)."""

    def test_atm_forward_call_is_half_df(self):
        """spot=95, df=exp(-0.05), so F=95×exp(0.05)≈99.87.
        Use strike=99.87 (≈ATMF) → digital call ≈ 0.5 × df."""
        r = 0.05
        T = 1.0
        df = math.exp(-r * T)
        spot = 95.0
        forward = spot / df
        strike = forward  # ATMF
        price = pb.digital_option(
            spot, strike, vol=0.20, T=T, payout=1.0,
            option_type="call", df=df,
        )
        # ATM-F → N(d2) = N(-σ√T/2) — not exactly 0.5 because of
        # the variance drift in d2.  But for vol=0.20, T=1 the
        # correction is small enough that we can pin within tolerance.
        # The price should be df · N(d2) where d2 = -σ√T/2 = -0.10.
        # N(-0.10) ≈ 0.4602.
        expected = df * 0.4602
        assert price == pytest.approx(expected, rel=1e-3)

    def test_drift_consistency_call_put(self):
        """Call + put = df × payout (binary parity)."""
        spot, strike, vol, T = 100.0, 100.0, 0.20, 0.5
        df = math.exp(-0.04 * T)
        c = pb.digital_option(spot, strike, vol, T, payout=1.0,
                              option_type="call", df=df)
        p = pb.digital_option(spot, strike, vol, T, payout=1.0,
                              option_type="put", df=df)
        assert c + p == pytest.approx(df, rel=1e-9)

    def test_no_rates_unchanged(self):
        """With df=1.0 (no rates), the fix is a no-op."""
        spot, strike, vol, T = 100.0, 100.0, 0.20, 1.0
        price = pb.digital_option(spot, strike, vol, T, payout=1.0,
                                   option_type="call", df=1.0)
        # ATM with no rates: N(-σ√T/2) = N(-0.10) ≈ 0.4602.
        assert price == pytest.approx(0.4602, rel=1e-3)

    def test_positive_rates_increase_call(self):
        """For ITM-spot, positive rates push forward higher → call ↑."""
        spot, strike, vol, T = 100.0, 100.0, 0.20, 1.0
        no_rate = pb.digital_option(spot, strike, vol, T, payout=1.0,
                                     option_type="call", df=1.0)
        with_rate = pb.digital_option(spot, strike, vol, T, payout=1.0,
                                       option_type="call",
                                       df=math.exp(-0.05 * T))
        # Higher rates → forward up → ITM call probability up.
        # But discount factor df<1 reduces the present value.
        # Net effect: pre-fix would IGNORE the forward shift and just
        # apply df; post-fix correctly reflects both.
        # The ratio (with_rate / df) is the risk-neutral probability,
        # which must be > no_rate (ITM more likely under forward-shift).
        prob_with_rate = with_rate / math.exp(-0.05)
        assert prob_with_rate > no_rate
