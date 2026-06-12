"""Regression for L2 Tier-2 T2.5 — `wavelet_transform` handles non-power-of-2 input.

Pre-fix, `wavelet_transform` crashed on non-power-of-2 input lengths:

    >>> from pricebook.numerical._fourier import wavelet_transform
    >>> wavelet_transform(np.arange(7))
    ValueError: operands could not be broadcast together with shapes (4,) (3,)

The DWT halves the signal at each level.  For an odd length, `x[0::2]` and
`x[1::2]` have different sizes, breaking the Haar lifting step.

Post-fix pads to the next power of 2 (also accounting for the requested
`levels`) so the halving stays clean.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._fourier import WaveletType, wavelet_transform


class TestNonPowerOfTwo:
    @pytest.mark.parametrize("n", [3, 5, 7, 9, 11, 17, 100, 1000])
    def test_haar_runs_on_arbitrary_length(self, n):
        x = np.arange(n, dtype=float)
        result = wavelet_transform(x, levels=2, wavelet=WaveletType.HAAR)
        # Coefficients vector should be at least as long as the padded input.
        assert len(result.coefficients) >= n

    @pytest.mark.parametrize("n", [3, 5, 11, 17, 31])
    def test_db2_runs_on_arbitrary_length(self, n):
        x = np.arange(n, dtype=float)
        result = wavelet_transform(x, levels=2, wavelet=WaveletType.DB2)
        assert len(result.coefficients) >= n

    def test_power_of_two_unchanged(self):
        """Power-of-2 input: padding is a no-op, behaviour identical to pre-fix."""
        x = np.arange(16, dtype=float)
        result = wavelet_transform(x, levels=2)
        # 16 = 2^4 → 4 levels available; we ask for 2.
        # Coefficient vector should have length 16 (no padding).
        assert len(result.coefficients) == 16

    def test_minimum_length_2(self):
        """Length-2 input is exactly one level of decomposition."""
        x = np.array([1.0, 3.0])
        result = wavelet_transform(x, levels=1, wavelet=WaveletType.HAAR)
        # Haar: approx = (1+3)/sqrt(2) ≈ 2.828, detail = (1-3)/sqrt(2) ≈ -1.414.
        import math
        # coefficients ordered: [approx, detail] (final approximation comes first
        # after the reverse-concat in the implementation).
        assert math.isclose(result.coefficients[0], 4 / math.sqrt(2), abs_tol=1e-12)
        assert math.isclose(result.coefficients[1], -2 / math.sqrt(2), abs_tol=1e-12)

    def test_levels_capped_by_padded_size(self):
        """Asking for more levels than log2(padded_n) just halts at half<1."""
        x = np.arange(7, dtype=float)
        # Padded to 8, so log2=3 levels max.  Asking for 5 should just run 3.
        result = wavelet_transform(x, levels=5, wavelet=WaveletType.HAAR)
        assert len(result.coefficients) >= 7

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="≥ 2"):
            wavelet_transform(np.array([1.0]), levels=1)
