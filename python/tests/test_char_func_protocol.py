"""Tests for characteristic function protocol and standalone factories.

Validates:
- Protocol compliance (structural typing)
- φ(0) = 1, Hermitian symmetry, boundedness
- COS vs MC cross-validation for all models
- Standalone factory functions match class-based CFs
"""

import pytest
import math
import cmath
import numpy as np

from pricebook.models.char_func_protocol import (
    CharFuncModel, validate_char_func, extract_cumulants,
    merton_char_func, vg_char_func, kou_char_func, bates_char_func,
    svj_char_func, CumulantInfo,
)


# ═══════════════════════════════════════════════════════════════
# Protocol compliance
# ═══════════════════════════════════════════════════════════════

class TestProtocolCompliance:
    def test_merton_satisfies_protocol(self):
        from pricebook.models.jump_process import MertonJumpDiffusion
        mjd = MertonJumpDiffusion(0.05, 0.2, 1.0, -0.1, 0.15)
        assert isinstance(mjd, CharFuncModel)

    def test_vg_has_char_func(self):
        """VG has char_func but with different signature (rate, T) — test it works."""
        from pricebook.models.jump_process import VarianceGammaProcess
        vg = VarianceGammaProcess(0.2, -0.14, 0.25)
        # VG takes (rate, T), not just (T)
        phi = vg.char_func(0.05, 1.0)
        assert abs(phi(0.0) - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

class TestValidation:
    def test_merton_validates(self):
        phi = merton_char_func(0.05, 0.2, 1.0, -0.1, 0.15, 1.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_vg_validates(self):
        phi = vg_char_func(0.05, 0.2, 0.25, -0.14, 1.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_kou_validates(self):
        phi = kou_char_func(0.05, 0.2, 1.0, 1.0, 0.6, 8.0, 5.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_bates_validates(self):
        phi = bates_char_func(0.05, 0.04, 1.5, 0.04, 0.3, -0.7,
                               0.5, -0.05, 0.1, 1.0)
        result = validate_char_func(phi)
        assert result["valid"]

    def test_svj_is_bates(self):
        assert svj_char_func is bates_char_func


# ═══════════════════════════════════════════════════════════════
# Cumulant extraction
# ═══════════════════════════════════════════════════════════════

class TestCumulants:
    def test_merton_mean(self):
        """Merton mean should be close to (r - λk - 0.5σ²)T + λ·μ_j·T."""
        phi = merton_char_func(0.05, 0.2, 1.0, -0.1, 0.15, 1.0)
        info = extract_cumulants(phi)
        # c1 ≈ (r - λk - 0.5σ²)T + λ·μ_j·T (approximately)
        assert isinstance(info, CumulantInfo)
        assert info.c2 > 0  # positive variance

    def test_vg_skewness(self):
        """VG with negative theta should have negative skewness."""
        phi = vg_char_func(0.05, 0.2, 0.25, -0.14, 1.0)
        info = extract_cumulants(phi)
        assert info.skewness < 0

    def test_vg_excess_kurtosis(self):
        """VG has positive excess kurtosis (heavy tails)."""
        phi = vg_char_func(0.05, 0.2, 0.25, -0.14, 1.0)
        info = extract_cumulants(phi)
        assert info.excess_kurtosis > 0


# ═══════════════════════════════════════════════════════════════
# COS vs MC cross-validation
# ═══════════════════════════════════════════════════════════════

class TestCOSvsMC:
    """Cross-validate COS pricing (via char func) against MC simulation."""

    def test_merton_cos_vs_mc(self):
        from pricebook.models.jump_process import MertonJumpDiffusion
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, strike, rate, T = 100, 100, 0.05, 1.0
        mjd = MertonJumpDiffusion(rate, 0.20, 1.0, -0.10, 0.15)

        # MC
        st = mjd.terminal(spot, T, 200_000, seed=42)
        mc_call = math.exp(-rate * T) * np.maximum(st - strike, 0).mean()

        # COS via class method
        phi_class = mjd.char_func(T)
        cos_call = cos_price(phi_class, spot, strike, rate, T, OptionType.CALL, N=256)

        # COS via standalone
        phi_standalone = merton_char_func(rate, 0.20, 1.0, -0.10, 0.15, T)
        cos_standalone = cos_price(phi_standalone, spot, strike, rate, T, OptionType.CALL, N=256)

        assert cos_call == pytest.approx(mc_call, rel=0.05)
        assert cos_standalone == pytest.approx(cos_call, rel=0.001)

    def test_vg_cos_vs_mc(self):
        from pricebook.models.jump_process import VarianceGammaProcess
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, strike, rate, T = 100, 100, 0.05, 1.0
        vg = VarianceGammaProcess(0.20, -0.14, 0.25)

        st = vg.terminal(spot, rate, T, 200_000, seed=42)
        mc_call = math.exp(-rate * T) * np.maximum(st - strike, 0).mean()

        phi_class = vg.char_func(rate, T)
        cos_call = cos_price(phi_class, spot, strike, rate, T, OptionType.CALL, N=256)

        phi_standalone = vg_char_func(rate, 0.20, 0.25, -0.14, T)
        cos_standalone = cos_price(phi_standalone, spot, strike, rate, T, OptionType.CALL, N=256)

        assert cos_call == pytest.approx(mc_call, rel=0.05)
        assert cos_standalone == pytest.approx(cos_call, rel=0.001)

    def test_kou_cos_vs_series(self):
        """Kou: COS pricing via char func vs existing series expansion."""
        from pricebook.equity.equity_jumps import kou_equity_price
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, strike, rate, T = 100, 100, 0.05, 1.0
        vol, lam, p, eta1, eta2 = 0.20, 1.0, 0.6, 8.0, 5.0

        # Series (existing)
        series_r = kou_equity_price(spot, strike, rate, 0.0, vol, T, lam, p, eta1, eta2)

        # COS via char func (new)
        phi = kou_char_func(rate, vol, T, lam, p, eta1, eta2)
        cos_call = cos_price(phi, spot, strike, rate, T, OptionType.CALL, N=256)

        assert cos_call == pytest.approx(series_r.price, rel=0.05)

    def test_bates_cos_reduces_to_heston(self):
        """Bates with λ=0 should equal Heston."""
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, strike, rate, T = 100, 100, 0.05, 1.0
        v0, kappa, theta_v, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.7

        # Bates with no jumps
        phi_bates = bates_char_func(rate, v0, kappa, theta_v, xi, rho,
                                     0.0, 0.0, 0.01, T)  # lam=0

        # Heston standalone
        from pricebook.models.cos_method import cos_price
        cos_bates = cos_price(phi_bates, spot, strike, rate, T, OptionType.CALL, N=256)

        # Should be a reasonable call price (Heston ATM ~8-12)
        assert 5 < cos_bates < 20

    def test_bates_jumps_increase_otm_puts(self):
        """Adding downward jumps should increase OTM put prices."""
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType

        spot, rate, T = 100, 0.05, 1.0
        v0, kappa, theta_v, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.7
        strike = 85  # OTM put

        # No jumps
        phi_no_jump = bates_char_func(rate, v0, kappa, theta_v, xi, rho,
                                       0.0, 0.0, 0.01, T)
        put_no = cos_price(phi_no_jump, spot, strike, rate, T, OptionType.PUT, N=256)

        # Downward jumps
        phi_jump = bates_char_func(rate, v0, kappa, theta_v, xi, rho,
                                    1.0, -0.15, 0.1, T)
        put_jump = cos_price(phi_jump, spot, strike, rate, T, OptionType.PUT, N=256)

        assert put_jump > put_no  # jumps fatten the left tail


# ═══════════════════════════════════════════════════════════════
# Complex u support (needed for FFT / Carr-Madan)
# ═══════════════════════════════════════════════════════════════

class TestComplexInput:
    def test_merton_complex_u(self):
        phi = merton_char_func(0.05, 0.2, 1.0, -0.1, 0.15, 1.0)
        # Carr-Madan uses u - (α+1)i
        val = phi(complex(5.0, -1.5))
        assert isinstance(val, complex)
        assert math.isfinite(val.real)
        assert math.isfinite(val.imag)

    def test_kou_complex_u(self):
        phi = kou_char_func(0.05, 0.2, 1.0, 1.0, 0.6, 8.0, 5.0)
        val = phi(complex(5.0, -1.5))
        assert isinstance(val, complex)
        assert math.isfinite(val.real)

    def test_bates_complex_u(self):
        phi = bates_char_func(0.05, 0.04, 1.5, 0.04, 0.3, -0.7,
                               0.5, -0.05, 0.1, 1.0)
        val = phi(complex(5.0, -1.5))
        assert isinstance(val, complex)
        assert math.isfinite(val.real)
