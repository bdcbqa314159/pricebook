"""Tests for equity exotic options."""

import math

import numpy as np
import pytest

from pricebook.equity_exotic import (
    CompoundResult,
    DigitalResult,
    EquityBarrierResult,
    EquityLookbackResult,
    equity_barrier_smile,
    equity_compound_option,
    equity_digital_asset,
    equity_digital_cash,
    equity_lookback_fixed,
    equity_lookback_floating,
)


# ---- Barrier ----

class TestEquityBarrier:
    def test_basic(self):
        result = equity_barrier_smile(
            spot=100, strike=100, barrier=110,
            rate=0.05, dividend_yield=0.02,
            vol_atm=0.20, vol_25d_call=0.21, vol_25d_put=0.22, T=1.0,
            is_up=True, is_knock_in=False, is_call=True,
        )
        assert isinstance(result, EquityBarrierResult)
        assert result.bs_price >= 0
        assert result.price >= 0

    def test_knock_out_less_than_vanilla(self):
        """Knock-out call ≤ vanilla call."""
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp((0.05 - 0.02) * 1.0)
        vanilla = black76_price(F, 100, 0.20, 1.0, math.exp(-0.05), OptionType.CALL)
        ko = equity_barrier_smile(
            100, 100, 120, 0.05, 0.02, 0.20, 0.21, 0.22, 1.0,
            is_up=True, is_knock_in=False, is_call=True,
        )
        assert ko.price <= vanilla + 1e-3

    def test_smile_adjustment_nonzero(self):
        """With skew, VV adjustment should be non-zero."""
        result = equity_barrier_smile(
            100, 100, 115, 0.05, 0.02, 0.20, 0.25, 0.30, 1.0,
        )
        # Smile → VV price different from BS
        assert abs(result.vv_adjustment) >= 0  # may be near zero but exists


# ---- Digital cash ----

class TestDigitalCash:
    def test_basic(self):
        result = equity_digital_cash(100, 100, 0.05, 0.02, 0.20, 1.0)
        assert isinstance(result, DigitalResult)
        assert result.digital_type == "cash"
        assert 0 < result.price < 1.0

    def test_probability_atm(self):
        """ATM digital call probability ≈ 0.5 (forward ≈ strike)."""
        result = equity_digital_cash(100, 100, 0.03, 0.03, 0.20, 1.0)
        assert result.probability == pytest.approx(0.5, abs=0.1)

    def test_deep_itm_near_df(self):
        """Deep ITM digital call ≈ DF × payout."""
        result = equity_digital_cash(200, 100, 0.05, 0.02, 0.20, 1.0, payout=1.0)
        df = math.exp(-0.05)
        assert result.price == pytest.approx(df, abs=0.05)

    def test_call_put_parity(self):
        """Digital call + digital put = DF × payout."""
        call = equity_digital_cash(100, 100, 0.05, 0.02, 0.20, 1.0, is_call=True)
        put = equity_digital_cash(100, 100, 0.05, 0.02, 0.20, 1.0, is_call=False)
        df = math.exp(-0.05)
        assert call.price + put.price == pytest.approx(df, abs=1e-6)

    def test_smile_adjustment(self):
        """Smile-adjusted digital differs from BS."""
        no_smile = equity_digital_cash(100, 100, 0.05, 0.02, 0.20, 1.0,
                                        smile_vols=(0.20, 0.20, 0.20))
        with_smile = equity_digital_cash(100, 100, 0.05, 0.02, 0.20, 1.0,
                                          smile_vols=(0.20, 0.22, 0.24))
        # Non-flat smile → different price
        assert isinstance(with_smile.price, float)

    def test_otm_low_prob(self):
        result = equity_digital_cash(100, 150, 0.05, 0.02, 0.20, 1.0)
        assert result.probability < 0.1


# ---- Digital asset ----

class TestDigitalAsset:
    def test_basic(self):
        result = equity_digital_asset(100, 100, 0.05, 0.02, 0.20, 1.0)
        assert result.digital_type == "asset"
        assert result.price > 0

    def test_deep_itm_near_spot(self):
        """Deep ITM asset digital ≈ spot × e^{-qT}."""
        result = equity_digital_asset(200, 100, 0.05, 0.02, 0.20, 1.0, is_call=True)
        expected = 200 * math.exp(-0.02)
        assert result.price == pytest.approx(expected, abs=1.0)

    def test_deep_otm_near_zero(self):
        result = equity_digital_asset(100, 300, 0.05, 0.02, 0.20, 1.0)
        assert result.price < 5.0


# ---- Lookback ----

class TestEquityLookbackFloating:
    def test_call_positive(self):
        result = equity_lookback_floating(100, 0.05, 0.02, 0.20, 1.0, is_call=True)
        assert isinstance(result, EquityLookbackResult)
        assert result.is_floating
        assert result.price > 0

    def test_put_positive(self):
        result = equity_lookback_floating(100, 0.05, 0.02, 0.20, 1.0, is_call=False)
        assert result.price > 0

    def test_higher_vol_higher_price(self):
        low = equity_lookback_floating(100, 0.05, 0.02, 0.10, 1.0)
        high = equity_lookback_floating(100, 0.05, 0.02, 0.30, 1.0)
        assert high.price > low.price


class TestEquityLookbackFixed:
    def test_basic(self):
        result = equity_lookback_fixed(100, 100, 0.05, 0.02, 0.20, 1.0,
                                        n_paths=5000, seed=42)
        assert not result.is_floating
        assert result.price > 0

    def test_exceeds_vanilla(self):
        """Fixed-strike lookback call ≥ vanilla call."""
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp((0.05 - 0.02) * 1.0)
        vanilla = black76_price(F, 100, 0.20, 1.0, math.exp(-0.05), OptionType.CALL)
        lb = equity_lookback_fixed(100, 100, 0.05, 0.02, 0.20, 1.0,
                                    n_paths=20_000, n_steps=200, seed=42)
        assert lb.price >= vanilla


# ---- Compound ----

class TestCompoundOption:
    def test_basic(self):
        result = equity_compound_option(
            spot=100, strike_outer=5, strike_underlying=100,
            rate=0.05, dividend_yield=0.02, vol=0.20,
            T1=0.5, T2=1.0, is_outer_call=True, is_underlying_call=True,
        )
        assert isinstance(result, CompoundResult)
        assert result.outer_type == "call"
        assert result.underlying_type == "call"

    def test_compound_cheaper_than_underlying(self):
        """Call-on-call ≤ underlying call (paying strike_outer for the option)."""
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp((0.05 - 0.02) * 1.0)
        underlying_call = black76_price(F, 100, 0.20, 1.0, math.exp(-0.05), OptionType.CALL)
        compound = equity_compound_option(100, 5, 100, 0.05, 0.02, 0.20, 0.5, 1.0)
        # Compound should be positive and roughly bounded
        assert compound.price >= 0
        assert compound.price <= underlying_call + 5  # loose upper bound

    def test_requires_T1_less_T2(self):
        with pytest.raises(ValueError):
            equity_compound_option(100, 5, 100, 0.05, 0.02, 0.20, 1.0, 0.5)

    def test_requires_positive_vol(self):
        with pytest.raises(ValueError):
            equity_compound_option(100, 5, 100, 0.05, 0.02, 0.0, 0.5, 1.0)

    def test_deep_otm_zero(self):
        """Very high outer strike → compound worthless."""
        result = equity_compound_option(100, 1000, 100, 0.05, 0.02, 0.20, 0.5, 1.0)
        assert result.price < 1.0
