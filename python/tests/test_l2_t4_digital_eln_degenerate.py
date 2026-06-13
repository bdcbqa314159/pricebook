"""Regression for L2 Wave-2 audit — `_digital_call_bs` in
`structured.equity_linked_note` had two coupled bugs in the degenerate
``T <= 0 or vol <= 0`` branch:

(a) Indicator compared ``spot`` to ``strike``, but at vol=0/T>0 the
    terminal value equals ``forward = spot·exp((r-q)T)``, not spot.
    An OTM-spot but ITM-forward digital silently returned 0.
(b) Missing discount factor — even when ITM, the digital pays $1 at
    maturity and must be present-valued.

Post-fix:
- T = 0: returns ``1.0 if spot > strike else 0.0`` (immediate, no discount).
- T > 0, vol = 0: returns ``df · I(forward > strike)`` with ATM half-step.
- T > 0, vol > 0: unchanged (Black-Scholes ``df · N(d2)``).
"""

from __future__ import annotations

import math

import pytest

from pricebook.structured.equity_linked_note import _digital_call_bs


class TestDigitalCallVolZeroTPositive:
    def test_itm_forward_otm_spot_returns_df(self):
        # spot=95, rate=0.10, T=1 → forward = 95·exp(0.10) ≈ 105.0 > 100 (ITM fwd)
        # But spot (95) < strike (100): pre-fix returned 0 (OTM spot indicator).
        price = _digital_call_bs(spot=95.0, rate=0.10, div_yield=0.0, vol=0.0,
                                 T=1.0, strike=100.0)
        assert price == pytest.approx(math.exp(-0.10), abs=1e-12)
        # Pre-fix would have returned 0.0 here.
        assert price > 0.0

    def test_itm_call_discounts_by_df(self):
        # spot=120, strike=100, rate=0.05, T=1 → forward ≈ 126.2 > 100 → ITM fwd.
        # Pre-fix returned 1.0 (no discount); correct value is exp(-0.05) ≈ 0.9512.
        price = _digital_call_bs(spot=120.0, rate=0.05, div_yield=0.0, vol=0.0,
                                 T=1.0, strike=100.0)
        assert price == pytest.approx(math.exp(-0.05), abs=1e-12)
        # Pre-fix would have returned 1.0 here (no discount).
        assert price < 1.0

    def test_otm_returns_zero(self):
        # spot=80, rate=0.02, T=1 → forward ≈ 81.6 < 100 → OTM.
        price = _digital_call_bs(spot=80.0, rate=0.02, div_yield=0.0, vol=0.0,
                                 T=1.0, strike=100.0)
        assert price == 0.0

    def test_atm_forward_returns_half_df(self):
        # Engineer forward exactly = strike: spot=100, rate=q=0 → forward=100=K.
        price = _digital_call_bs(spot=100.0, rate=0.0, div_yield=0.0, vol=0.0,
                                 T=1.0, strike=100.0)
        # ATM forward at vol=0: one-sided limit = 0.5 · df = 0.5 · 1.0 = 0.5.
        assert price == pytest.approx(0.5, abs=1e-12)


class TestDigitalCallTZero:
    def test_T_zero_itm_pays_one(self):
        # At T=0 there's no discount and no forward-drift.  Indicator on spot.
        price = _digital_call_bs(spot=110.0, rate=0.05, div_yield=0.0, vol=0.0,
                                 T=0.0, strike=100.0)
        assert price == 1.0

    def test_T_zero_otm_pays_zero(self):
        price = _digital_call_bs(spot=90.0, rate=0.05, div_yield=0.0, vol=0.0,
                                 T=0.0, strike=100.0)
        assert price == 0.0


class TestDigitalCallInteriorUnchanged:
    """Healthy interior path (T>0, vol>0) must match pre-fix."""

    def test_atm_finite(self):
        price = _digital_call_bs(spot=100.0, rate=0.05, div_yield=0.0, vol=0.20,
                                 T=1.0, strike=100.0)
        # Approx exp(-0.05) · N(d2 ≈ +0.15) ≈ 0.95 · 0.56 ≈ 0.53.
        assert 0.45 < price < 0.65

    def test_dividend_yield_affects_forward(self):
        # With div_yield=0.05, forward = spot·exp((r-q)·T) = spot·exp(0) = spot.
        # So ATM forward ≈ ATM spot.
        price = _digital_call_bs(spot=100.0, rate=0.05, div_yield=0.05, vol=0.20,
                                 T=1.0, strike=100.0)
        # d2 = (log(1) - 0.5·σ²·T)/(σ√T) = -σ√T/2 = -0.1 → N(-0.1) ≈ 0.46
        # price ≈ exp(-0.05) · 0.46 ≈ 0.437.
        assert 0.40 < price < 0.48
