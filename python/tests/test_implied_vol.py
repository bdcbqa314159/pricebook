"""Tests for implied volatility solvers."""

import pytest
import math

from pricebook.implied_vol import implied_vol_black76, implied_vol_bachelier
from pricebook.black76 import OptionType, black76_price, bachelier_price


F, K, T, DF = 100.0, 100.0, 1.0, math.exp(-0.05)


class TestBlack76ImpliedVol:
    @pytest.mark.parametrize("vol", [0.05, 0.10, 0.20, 0.40, 0.80])
    def test_round_trip_call(self, vol):
        price = black76_price(F, K, vol, T, DF, OptionType.CALL)
        recovered = implied_vol_black76(price, F, K, T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol, abs=1e-8)

    @pytest.mark.parametrize("vol", [0.05, 0.10, 0.20, 0.40, 0.80])
    def test_round_trip_put(self, vol):
        price = black76_price(F, K, vol, T, DF, OptionType.PUT)
        recovered = implied_vol_black76(price, F, K, T, DF, OptionType.PUT)
        assert recovered == pytest.approx(vol, abs=1e-8)

    def test_otm_call(self):
        vol = 0.25
        price = black76_price(F, 120.0, vol, T, DF, OptionType.CALL)
        recovered = implied_vol_black76(price, F, 120.0, T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol, abs=1e-8)

    def test_itm_put(self):
        vol = 0.25
        price = black76_price(F, 80.0, vol, T, DF, OptionType.PUT)
        recovered = implied_vol_black76(price, F, 80.0, T, DF, OptionType.PUT)
        assert recovered == pytest.approx(vol, abs=1e-8)

    def test_deep_otm(self):
        vol = 0.20
        price = black76_price(F, 150.0, vol, T, DF, OptionType.CALL)
        recovered = implied_vol_black76(price, F, 150.0, T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol, abs=1e-6)

    def test_short_expiry(self):
        vol = 0.30
        short_T = 0.01
        price = black76_price(F, K, vol, short_T, DF, OptionType.CALL)
        recovered = implied_vol_black76(price, F, K, short_T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol, abs=1e-6)

    def test_price_below_intrinsic_raises(self):
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_vol_black76(0.001, F, 80.0, T, DF, OptionType.CALL)

    def test_price_above_upper_bound_raises(self):
        with pytest.raises(ValueError, match="exceeds upper bound"):
            implied_vol_black76(DF * F + 1.0, F, K, T, DF, OptionType.CALL)

    def test_zero_T_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            implied_vol_black76(5.0, F, K, 0.0, DF, OptionType.CALL)


class TestBachelierImpliedVol:
    @pytest.mark.parametrize("vol_n", [1.0, 5.0, 10.0, 20.0])
    def test_round_trip_call(self, vol_n):
        price = bachelier_price(F, K, vol_n, T, DF, OptionType.CALL)
        recovered = implied_vol_bachelier(price, F, K, T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol_n, abs=1e-6)

    @pytest.mark.parametrize("vol_n", [1.0, 5.0, 10.0, 20.0])
    def test_round_trip_put(self, vol_n):
        price = bachelier_price(F, K, vol_n, T, DF, OptionType.PUT)
        recovered = implied_vol_bachelier(price, F, K, T, DF, OptionType.PUT)
        assert recovered == pytest.approx(vol_n, abs=1e-6)

    def test_otm(self):
        vol_n = 8.0
        price = bachelier_price(F, 120.0, vol_n, T, DF, OptionType.CALL)
        recovered = implied_vol_bachelier(price, F, 120.0, T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol_n, abs=1e-5)

    def test_negative_forward(self):
        """Bachelier handles negative rates/forwards."""
        vol_n = 5.0
        price = bachelier_price(-0.5, 0.0, vol_n, T, DF, OptionType.PUT)
        recovered = implied_vol_bachelier(price, -0.5, 0.0, T, DF, OptionType.PUT)
        assert recovered == pytest.approx(vol_n, abs=1e-5)

    def test_price_below_intrinsic_raises(self):
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_vol_bachelier(0.001, F, 80.0, T, DF, OptionType.CALL)

    def test_zero_T_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            implied_vol_bachelier(5.0, F, K, 0.0, DF, OptionType.CALL)
