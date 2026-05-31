"""Tests for NIG and CGMY Lévy processes."""

import pytest
import math
import cmath
import numpy as np

from pricebook.models.levy_processes import (
    NIGProcess, CGMYProcess,
    nig_char_func, cgmy_char_func,
)
from pricebook.models.char_func_protocol import validate_char_func, extract_cumulants


# ═══════════════════════════════════════════════════════════════
# NIG Process
# ═══════════════════════════════════════════════════════════════

class TestNIGProcess:
    def test_construction(self):
        nig = NIGProcess(alpha=15, beta=-5, delta=0.5)
        assert nig.alpha == 15
        assert nig.beta == -5

    def test_alpha_gt_beta(self):
        with pytest.raises(ValueError, match="alpha > |beta|"):
            NIGProcess(alpha=5, beta=6, delta=0.5)

    def test_delta_positive(self):
        with pytest.raises(ValueError, match="delta > 0"):
            NIGProcess(alpha=15, beta=-5, delta=-1)

    def test_char_func_phi_zero(self):
        nig = NIGProcess(alpha=15, beta=-5, delta=0.5)
        phi = nig.char_func(T=1.0)
        assert abs(phi(0.0) - 1.0) < 1e-10

    def test_char_func_validates(self):
        phi = nig_char_func(0.05, 15, -5, 0.5, 1.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_negative_beta_negative_skew(self):
        """Negative β → negative skewness."""
        phi = nig_char_func(0.05, 15, -5, 0.5, 1.0)
        info = extract_cumulants(phi)
        assert info.skewness < 0

    def test_positive_excess_kurtosis(self):
        """NIG always has positive excess kurtosis."""
        phi = nig_char_func(0.05, 15, -5, 0.5, 1.0)
        info = extract_cumulants(phi)
        assert info.excess_kurtosis > 0

    def test_terminal_mean_riskneutral(self):
        """Under RN measure, E[S_T] ≈ S_0·exp(rT)."""
        nig = NIGProcess(alpha=15, beta=-5, delta=0.5)
        rate = 0.05
        st = nig.terminal(100, rate, 1.0, 100_000, seed=42)
        expected = 100 * math.exp(rate)
        assert np.mean(st) == pytest.approx(expected, rel=0.03)

    def test_cos_vs_mc(self):
        """COS pricing via NIG char func vs MC."""
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, strike, rate, T = 100, 100, 0.05, 1.0
        nig = NIGProcess(alpha=15, beta=-5, delta=0.5)

        st = nig.terminal(spot, rate, T, 200_000, seed=42)
        mc_call = math.exp(-rate * T) * np.maximum(st - strike, 0).mean()

        phi = nig_char_func(rate, 15, -5, 0.5, T)
        cos_call = cos_price(phi, spot, strike, rate, T, OptionType.CALL, N=256)

        assert cos_call == pytest.approx(mc_call, rel=0.05)

    def test_complex_u(self):
        phi = nig_char_func(0.05, 15, -5, 0.5, 1.0)
        val = phi(complex(5.0, -1.5))
        assert math.isfinite(val.real)
        assert math.isfinite(val.imag)

    def test_to_dict(self):
        nig = NIGProcess(alpha=15, beta=-5, delta=0.5)
        d = nig.to_dict()
        assert d["type"] == "nig"
        assert d["alpha"] == 15


# ═══════════════════════════════════════════════════════════════
# CGMY Process
# ═══════════════════════════════════════════════════════════════

class TestCGMYProcess:
    def test_construction(self):
        cgmy = CGMYProcess(C=1.0, G=5.0, M=10.0, Y=0.5)
        assert cgmy.C == 1.0

    def test_c_positive(self):
        with pytest.raises(ValueError, match="C > 0"):
            CGMYProcess(C=-1, G=5, M=10, Y=0.5)

    def test_y_lt_2(self):
        with pytest.raises(ValueError, match="Y < 2"):
            CGMYProcess(C=1, G=5, M=10, Y=2.5)

    def test_char_func_phi_zero(self):
        cgmy = CGMYProcess(C=1.0, G=5.0, M=10.0, Y=0.5)
        phi = cgmy.char_func(T=1.0)
        assert abs(phi(0.0) - 1.0) < 1e-10

    def test_char_func_validates(self):
        phi = cgmy_char_func(0.05, 1.0, 5.0, 10.0, 0.5, 1.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_y_near_zero_is_vg(self):
        """CGMY with Y→0 should approximate VG."""
        from pricebook.models.char_func_protocol import vg_char_func

        # VG params: σ=0.2, ν=0.25, θ=-0.14
        # Map to CGMY: C=1/ν=4, G=..., M=...
        # Exact mapping: C = 1/ν, G = (√(θ²ν²/4 + σ²ν/2) - θν/2)⁻¹ etc.
        # Just verify the Y=0 char func doesn't explode
        phi = cgmy_char_func(0.05, 4.0, 5.0, 10.0, 0.0, 1.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_asymmetric_gm(self):
        """G < M → more negative jumps → negative skew."""
        phi = cgmy_char_func(0.05, 1.0, 5.0, 10.0, 0.5, 1.0)
        info = extract_cumulants(phi)
        assert info.skewness < 0  # G < M → left-skewed

    def test_positive_excess_kurtosis(self):
        phi = cgmy_char_func(0.05, 1.0, 5.0, 10.0, 0.5, 1.0)
        info = extract_cumulants(phi)
        assert info.excess_kurtosis > 0

    def test_terminal_positive(self):
        """CGMY MC terminal values should be positive."""
        cgmy = CGMYProcess(C=1.0, G=5.0, M=10.0, Y=0.5)
        st = cgmy.terminal(100, 0.05, 1.0, 10_000, seed=42)
        assert np.all(st > 0)
        assert np.mean(st) > 50  # reasonable range

    def test_cos_pricing(self):
        """COS pricing via CGMY char func produces reasonable price."""
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, strike, rate, T = 100, 100, 0.05, 1.0
        phi = cgmy_char_func(rate, 1.0, 5.0, 10.0, 0.5, T)
        cos_call = cos_price(phi, spot, strike, rate, T, OptionType.CALL, N=256)

        assert 5 < cos_call < 25  # reasonable ATM call

    def test_complex_u(self):
        phi = cgmy_char_func(0.05, 1.0, 5.0, 10.0, 0.5, 1.0)
        val = phi(complex(5.0, -1.5))
        assert math.isfinite(val.real)

    def test_to_dict(self):
        cgmy = CGMYProcess(C=1.0, G=5.0, M=10.0, Y=0.5)
        d = cgmy.to_dict()
        assert d["type"] == "cgmy"
        assert d["Y"] == 0.5


# ═══════════════════════════════════════════════════════════════
# Cross-model consistency
# ═══════════════════════════════════════════════════════════════

class TestCrossModel:
    def test_nig_heavier_tails_than_bs(self):
        """NIG should produce higher OTM put prices than Black-Scholes."""
        from pricebook.models.cos_method import cos_price, bs_char_func
        from pricebook.models.black76 import OptionType

        spot, rate, T = 100, 0.05, 1.0
        strike = 80  # deep OTM put

        phi_bs = bs_char_func(rate, 0.0, 0.20, T)
        put_bs = cos_price(phi_bs, spot, strike, rate, T, OptionType.PUT, N=256)

        phi_nig = nig_char_func(rate, 15, -5, 0.5, T)
        put_nig = cos_price(phi_nig, spot, strike, rate, T, OptionType.PUT, N=256)

        assert put_nig > put_bs  # fat tails → higher OTM puts

    def test_cgmy_heavier_tails_than_bs(self):
        """CGMY should produce higher OTM put prices than Black-Scholes."""
        from pricebook.models.cos_method import cos_price, bs_char_func
        from pricebook.models.black76 import OptionType

        spot, rate, T = 100, 0.05, 1.0
        strike = 80

        phi_bs = bs_char_func(rate, 0.0, 0.20, T)
        put_bs = cos_price(phi_bs, spot, strike, rate, T, OptionType.PUT, N=256)

        phi_cgmy = cgmy_char_func(rate, 1.0, 5.0, 10.0, 0.5, T)
        put_cgmy = cos_price(phi_cgmy, spot, strike, rate, T, OptionType.PUT, N=256)

        assert put_cgmy > put_bs
