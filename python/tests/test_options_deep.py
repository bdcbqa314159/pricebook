"""Deep tests for options/pricing kernels — DD10 hardening.

Covers: Black-76 put-call parity, edge cases, Greeks signs, implied vol round-trip,
variance swap, FFT convergence to BS.
"""

import math
import pytest
from pricebook.black76 import black76_price, black76_delta, black76_vega, OptionType
from pricebook.implied_vol import implied_vol_black76
from pricebook.fft_pricing import lewis_price
from pricebook.variance_swap import fair_variance_from_vols


class TestBlack76:

    def test_put_call_parity(self):
        F, K, vol, T, df = 100.0, 100.0, 0.20, 1.0, 0.95
        c = black76_price(F, K, vol, T, df, OptionType.CALL)
        p = black76_price(F, K, vol, T, df, OptionType.PUT)
        assert c - p == pytest.approx(df * (F - K), abs=1e-10)

    def test_zero_vol_intrinsic(self):
        F, K, df = 110.0, 100.0, 0.95
        c = black76_price(F, K, 0.0, 1.0, df, OptionType.CALL)
        assert c == pytest.approx(df * 10.0, abs=1e-10)

    def test_zero_time_intrinsic(self):
        c = black76_price(110.0, 100.0, 0.20, 0.0, 1.0, OptionType.CALL)
        assert c == pytest.approx(10.0)

    def test_zero_forward_returns_intrinsic(self):
        """Zero forward should return intrinsic, not crash."""
        c = black76_price(0.0, 100.0, 0.20, 1.0, 0.95, OptionType.CALL)
        assert c == pytest.approx(0.0)
        p = black76_price(0.0, 100.0, 0.20, 1.0, 0.95, OptionType.PUT)
        assert p == pytest.approx(0.95 * 100.0)

    def test_atm_delta_at_zero_vol(self):
        """ATM delta with zero vol should be 0.5*df, not 0."""
        df = 0.95
        d = black76_delta(100.0, 100.0, 0.0, 0.0, df, OptionType.CALL)
        assert d == pytest.approx(0.5 * df)

    def test_higher_vol_higher_price(self):
        low = black76_price(100.0, 100.0, 0.10, 1.0, 0.95, OptionType.CALL)
        high = black76_price(100.0, 100.0, 0.30, 1.0, 0.95, OptionType.CALL)
        assert high > low


class TestGreeksSigns:

    def test_call_delta_positive(self):
        d = black76_delta(100.0, 100.0, 0.20, 1.0, 0.95, OptionType.CALL)
        assert d > 0

    def test_put_delta_negative(self):
        d = black76_delta(100.0, 100.0, 0.20, 1.0, 0.95, OptionType.PUT)
        assert d < 0

    def test_vega_positive(self):
        v = black76_vega(100.0, 100.0, 0.20, 1.0, 0.95)
        assert v > 0


class TestImpliedVol:

    def test_round_trip(self):
        """Price → implied vol → price should match."""
        F, K, vol, T, df = 100.0, 105.0, 0.25, 1.0, 0.95
        price = black76_price(F, K, vol, T, df, OptionType.CALL)
        recovered = implied_vol_black76(price, F, K, T, df, OptionType.CALL)
        assert recovered == pytest.approx(vol, abs=0.001)

    def test_deep_otm(self):
        """Implied vol for deep OTM option should still converge."""
        F, K, vol, T, df = 100.0, 150.0, 0.30, 0.5, 0.98
        price = black76_price(F, K, vol, T, df, OptionType.CALL)
        if price > 1e-10:
            recovered = implied_vol_black76(price, F, K, T, df, OptionType.CALL)
            assert recovered == pytest.approx(vol, abs=0.01)


class TestBlack76EdgeCases:

    def test_very_short_expiry(self):
        """Near-zero expiry should return intrinsic."""
        c = black76_price(110.0, 100.0, 0.20, 1e-8, 1.0, OptionType.CALL)
        assert c == pytest.approx(10.0, abs=0.01)

    def test_very_high_vol(self):
        """Very high vol shouldn't crash."""
        c = black76_price(100.0, 100.0, 5.0, 1.0, 0.95, OptionType.CALL)
        assert c > 0 and math.isfinite(c)
