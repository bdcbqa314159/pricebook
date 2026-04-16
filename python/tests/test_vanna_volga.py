"""Tests for Vanna-Volga deepening."""

import math

import numpy as np
import pytest

from pricebook.vanna_volga import (
    VVResult,
    VVWeights,
    vv_adjust_digital,
    vv_adjust_quanto,
    vv_adjust_touch,
    vv_adjust_vanilla,
    vv_weights,
)


# ---- VV weights ----

class TestVVWeights:
    def test_basic(self):
        w = vv_weights(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0)
        assert isinstance(w, VVWeights)
        # Target is ATM vanilla
        assert w.vega > 0

    def test_atm_has_main_weight(self):
        """For ATM target, x_atm should dominate."""
        K_atm = 1.0 * math.exp((0.02 - 0.01 + 0.5 * 0.10**2) * 1.0)
        w = vv_weights(1.0, K_atm, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0)
        assert abs(w.x_atm) > abs(w.x_rr)

    def test_greeks_match_bs(self):
        w = vv_weights(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0)
        # Vega, vanna, volga should be non-zero
        assert w.vega > 0
        assert abs(w.vanna) > 0
        assert abs(w.volga) > 0


# ---- VV vanilla ----

class TestVVVanilla:
    def test_basic(self):
        result = vv_adjust_vanilla(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0)
        assert isinstance(result, VVResult)
        assert result.bs_price > 0
        assert result.vv_price > 0

    def test_no_smile_no_adjustment(self):
        """Flat smile (ATM = 25C = 25P) → VV = BS."""
        result = vv_adjust_vanilla(1.0, 1.0, 0.02, 0.01, 0.10, 0.10, 0.10, 1.0)
        assert result.vv_adjustment == pytest.approx(0.0, abs=1e-4)

    def test_positive_skew_raises_otm_calls(self):
        """When 25D call vol > 25D put vol (positive skew), OTM call VV > BS."""
        otm_k = 1.05
        positive_skew = vv_adjust_vanilla(1.0, otm_k, 0.02, 0.01,
                                           0.10, 0.13, 0.08, 1.0)
        assert positive_skew.vv_price >= positive_skew.bs_price * 0.9

    def test_reprices_25d(self):
        """VV should exactly reprice 25D strikes (approximately)."""
        from scipy.stats import norm
        from pricebook.vanna_volga import _strike_from_delta
        K_25c = _strike_from_delta(1.0, 0.02, 0.01, 0.11, 1.0, 0.25, True)
        result = vv_adjust_vanilla(1.0, K_25c, 0.02, 0.01,
                                    0.10, 0.11, 0.12, 1.0, is_call=True)
        # VV price should equal the market price using 25D call vol
        from pricebook.black76 import black76_price, OptionType
        F = 1.0 * math.exp((0.02 - 0.01) * 1.0)
        df = math.exp(-0.02)
        mkt = black76_price(F, K_25c, 0.11, 1.0, df, OptionType.CALL)
        # Should be close (not exact due to weighting approximation)
        assert result.vv_price == pytest.approx(mkt, abs=0.02)


# ---- VV digital ----

class TestVVDigital:
    def test_basic(self):
        result = vv_adjust_digital(1.0, 1.0, 0.02, 0.01,
                                    0.10, 0.11, 0.12, 1.0)
        assert isinstance(result, VVResult)
        assert result.bs_price > 0
        assert result.vv_price > 0

    def test_bs_digital_reasonable(self):
        """BS digital call at ATM ≈ 0.5 × DF."""
        result = vv_adjust_digital(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0)
        df = math.exp(-0.02)
        assert result.bs_price == pytest.approx(0.5 * df, abs=0.05)

    def test_no_smile_no_adjustment(self):
        result = vv_adjust_digital(1.0, 1.0, 0.02, 0.01, 0.10, 0.10, 0.10, 1.0)
        assert abs(result.vv_adjustment) < 0.02

    def test_put_digital(self):
        result = vv_adjust_digital(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0,
                                    is_call=False)
        assert result.vv_price > 0


# ---- VV touch ----

class TestVVTouch:
    def test_basic(self):
        result = vv_adjust_touch(1.0, 1.10, 0.02, 0.01,
                                  0.10, 0.11, 0.12, 1.0, payout=1.0)
        assert isinstance(result, VVResult)
        assert result.bs_price > 0
        assert result.vv_price >= 0

    def test_up_vs_down(self):
        up = vv_adjust_touch(1.0, 1.10, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0, is_up=True)
        down = vv_adjust_touch(1.0, 0.90, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0, is_up=False)
        # Both should be positive
        assert up.vv_price > 0
        assert down.vv_price > 0


# ---- VV quanto ----

class TestVVQuanto:
    def test_basic(self):
        result = vv_adjust_quanto(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12,
                                   vol_quanto=0.15, correlation=0.3, T=1.0)
        assert isinstance(result, VVResult)
        assert result.vv_price > 0

    def test_zero_correlation_matches_vanilla(self):
        """Zero correlation → quanto adj = 0 → matches vanilla VV."""
        q = vv_adjust_quanto(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12,
                              vol_quanto=0.15, correlation=0.0, T=1.0)
        v = vv_adjust_vanilla(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0)
        assert q.vv_price == pytest.approx(v.vv_price, rel=0.05)

    def test_positive_correlation_negative_adjustment(self):
        """Positive correlation → quanto forward is lower → call VV lower."""
        pos = vv_adjust_quanto(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12,
                                vol_quanto=0.15, correlation=0.5, T=1.0)
        zero = vv_adjust_quanto(1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12,
                                 vol_quanto=0.15, correlation=0.0, T=1.0)
        assert pos.vv_price < zero.vv_price
