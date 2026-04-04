"""Tests for shifted SABR and normal vol swaptions."""

import math
import pytest

from pricebook.sabr import (
    sabr_implied_vol,
    sabr_price,
    shifted_sabr_implied_vol,
    shifted_sabr_price,
    sabr_normal_vol,
)
from pricebook.black76 import OptionType, bachelier_price


F, K, T = 0.02, 0.02, 5.0  # low rate environment
ALPHA, BETA, RHO, NU = 0.015, 0.5, -0.3, 0.4


class TestShiftedSABR:
    def test_zero_shift_matches_standard(self):
        vol_std = sabr_implied_vol(F, K, T, ALPHA, BETA, RHO, NU)
        vol_shifted = shifted_sabr_implied_vol(F, K, T, ALPHA, BETA, RHO, NU, shift=0.0)
        assert vol_shifted == pytest.approx(vol_std)

    def test_shift_handles_negative_rates(self):
        """With shift, can price options on negative forwards."""
        vol = shifted_sabr_implied_vol(-0.01, -0.005, T, ALPHA, BETA, RHO, NU, shift=0.03)
        assert vol > 0

    def test_price_positive(self):
        price = shifted_sabr_price(F, K, T, 0.95, ALPHA, BETA, RHO, NU, shift=0.02)
        assert price > 0

    def test_price_matches_standard_at_zero_shift(self):
        p_std = sabr_price(F, K, T, 0.95, ALPHA, BETA, RHO, NU)
        p_shifted = shifted_sabr_price(F, K, T, 0.95, ALPHA, BETA, RHO, NU, shift=0.0)
        assert p_shifted == pytest.approx(p_std)

    def test_otm_call(self):
        price = shifted_sabr_price(F, 0.03, T, 0.95, ALPHA, BETA, RHO, NU, shift=0.02)
        assert price > 0
        atm = shifted_sabr_price(F, F, T, 0.95, ALPHA, BETA, RHO, NU, shift=0.02)
        assert price < atm

    def test_put(self):
        price = shifted_sabr_price(F, K, T, 0.95, ALPHA, BETA, RHO, NU,
                                   shift=0.02, option_type=OptionType.PUT)
        assert price > 0


class TestNormalVol:
    def test_positive(self):
        nvol = sabr_normal_vol(F, K, T, ALPHA, BETA, RHO, NU)
        assert nvol > 0

    def test_scales_with_forward(self):
        """Normal vol ≈ lognormal vol * F for ATM."""
        lognormal = sabr_implied_vol(F, K, T, ALPHA, BETA, RHO, NU)
        normal = sabr_normal_vol(F, K, T, ALPHA, BETA, RHO, NU)
        assert normal == pytest.approx(lognormal * F, rel=0.20)

    def test_with_shift(self):
        nvol = sabr_normal_vol(F, K, T, ALPHA, BETA, RHO, NU, shift=0.02)
        assert nvol > 0

    def test_bachelier_pricing(self):
        """Normal vol can be used with Bachelier pricer."""
        nvol = sabr_normal_vol(F, K, T, ALPHA, BETA, RHO, NU)
        price = bachelier_price(F, K, nvol, T, df=0.95)
        assert price > 0

    def test_sabr_black_vs_bachelier_consistency(self):
        """SABR Black price ≈ Bachelier price with converted normal vol."""
        df = 0.95
        black_price = sabr_price(F, K, T, df, ALPHA, BETA, RHO, NU)
        nvol = sabr_normal_vol(F, K, T, ALPHA, BETA, RHO, NU)
        bach_price = bachelier_price(F, K, nvol, T, df)
        # Approximate equivalence for ATM
        assert bach_price == pytest.approx(black_price, rel=0.15)
