"""Extended Fourier methods: fractional FFT, Hilbert, wavelet, CF class.

    from pricebook.numerical import fractional_fft, hilbert_transform, wavelet_transform
    from pricebook.numerical import CharacteristicFunction
    from pricebook.numerical import FourierMethod, WaveletType
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class FourierMethod(Enum):
    """Fourier pricing / transform methods."""
    FFT = "fft"
    FRACTIONAL_FFT = "fractional_fft"
    COS = "cos"
    LEWIS = "lewis"


class WaveletType(Enum):
    """Available wavelet basis functions."""
    HAAR = "haar"
    DB2 = "db2"


def fractional_fft(
    x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fractional FFT (chirp-z transform).

    Computes DFT at non-uniform frequency spacing:
    X_k = sum_n x_n exp(-2 pi i alpha k n / N)

    Standard FFT has alpha = 1. Fractional allows arbitrary frequency grids.

    Uses Bluestein's algorithm: O(N log N) via convolution.
    """
    N = len(x)
    M = 1
    while M < 2 * N:
        M *= 2  # next power of 2

    # Chirp: w_n = exp(-pi i alpha n^2 / N)
    n = np.arange(N)
    chirp = np.exp(-1j * math.pi * alpha * n ** 2 / N)

    # Convolution via FFT
    a = np.zeros(M, dtype=complex)
    a[:N] = x * chirp

    b = np.zeros(M, dtype=complex)
    b[:N] = np.conj(chirp)
    b[M - N + 1:] = np.conj(chirp[1:])[::-1]

    A = np.fft.fft(a)
    B = np.fft.fft(b)
    C = np.fft.ifft(A * B)

    return C[:N] * chirp  # complex -- caller takes .real if needed


def hilbert_transform(x: np.ndarray) -> np.ndarray:
    """Hilbert transform via FFT.

    Returns the analytic signal: x_a = x + i H[x].
    The imaginary part is the Hilbert transform.

    Useful for envelope detection, instantaneous frequency.
    """
    N = len(x)
    X = np.fft.fft(x)

    # Multiply by -i sign(freq)
    h = np.zeros(N)
    if N > 0:
        h[0] = 1
        if N % 2 == 0:
            h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[1:(N + 1) // 2] = 2

    return np.fft.ifft(X * h)


@dataclass
class WaveletResult:
    """Wavelet transform result."""
    coefficients: np.ndarray
    levels: int
    wavelet: str

    def to_dict(self) -> dict:
        return {"levels": self.levels, "wavelet": self.wavelet,
                "n_coefficients": len(self.coefficients)}


def wavelet_transform(
    x: np.ndarray,
    levels: int = 3,
    wavelet: WaveletType | str = WaveletType.HAAR,
) -> WaveletResult:
    """Discrete wavelet transform (DWT) via lifting scheme.

    Decomposes signal into approximation + detail coefficients
    at multiple resolution levels.

    Args:
        x: input signal (length should be power of 2 for best results).
        levels: number of decomposition levels.
        wavelet: WaveletType enum or string name.
    """
    if isinstance(wavelet, str):
        wavelet = WaveletType(wavelet.lower())

    wavelet_name = wavelet.value
    x = np.asarray(x, dtype=float).copy()

    # Fix T2.5: pad to the next power of 2.  Pre-fix, non-power-of-2 inputs
    # crashed at `x[0::2] + x[1::2]` because the slices had different lengths
    # for odd `len(x)`.  After `levels` halvings the size must also stay even,
    # so we pad up to `max(2**ceil(log2(n)), 2**levels)`.
    n = len(x)
    if n < 2:
        raise ValueError(f"wavelet_transform: input length must be ≥ 2, got {n}")
    target = max(1 << (n - 1).bit_length(), 1 << levels)
    if target != n:
        x = np.concatenate([x, np.zeros(target - n)])
    n = target
    all_coeffs = []

    for level in range(levels):
        half = len(x) // 2
        if half < 1:
            break

        if wavelet == WaveletType.HAAR:
            # Haar: a = (x[even] + x[odd]) / sqrt(2), d = (x[even] - x[odd]) / sqrt(2)
            approx = (x[0::2] + x[1::2]) / math.sqrt(2)
            detail = (x[0::2] - x[1::2]) / math.sqrt(2)
        elif wavelet == WaveletType.DB2:
            # Daubechies-2 (4 coefficients)
            h = np.array([
                (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
                (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
                (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
                (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
            ])
            g = np.array([h[3], -h[2], h[1], -h[0]])
            approx = np.zeros(half)
            detail = np.zeros(half)
            for k in range(half):
                for j in range(4):
                    idx = (2 * k + j) % len(x)
                    approx[k] += h[j] * x[idx]
                    detail[k] += g[j] * x[idx]
        else:
            raise ValueError(f"unknown wavelet: {wavelet!r}")

        all_coeffs.append(detail)
        x = approx

    all_coeffs.append(x)  # final approximation
    coefficients = np.concatenate(all_coeffs[::-1])

    return WaveletResult(coefficients, levels, wavelet_name)


@dataclass
class CFResult:
    """Characteristic function evaluation result."""
    cumulants: dict[str, float]
    density: np.ndarray | None
    density_grid: np.ndarray | None

    def to_dict(self) -> dict:
        return {"cumulants": self.cumulants}


class CharacteristicFunction:
    """Unified characteristic function interface.

    Wraps a CF phi(u) and provides cumulant extraction, density recovery,
    and pricing via COS/FFT methods.

    Usage:
        cf = CharacteristicFunction(bs_cf, T=1.0)
        cf.cumulants()        # mean, variance, skew, kurtosis
        cf.density(x_grid)    # risk-neutral density
        cf.price_european(spot, strike, rate)  # via COS method
    """

    def __init__(self, cf_func, T: float = 1.0):
        """
        Args:
            cf_func: callable(u) -> complex, the characteristic function phi(u).
            T: time parameter.
        """
        self.cf = cf_func
        self.T = T

    def cumulants(self, max_order: int = 4) -> dict[str, float]:
        """Extract cumulants from CF via finite differences on log(phi).

        kappa_n = (-i)^n (d^n/du^n log phi)(0).  Since log phi(u) for real u
        of a real-valued density is `iuμ + (iu)² σ²/2 + ...` with imaginary
        coefficient at odd orders and real at even orders, the cumulant
        contractions become:

            kappa_1 = Im(log_phi'(0))           (odd)
            kappa_2 = −Re(log_phi''(0))          (even, sign from i² = −1)
            kappa_3 = −Im(log_phi'''(0))         (odd, sign from (-i)³ = i then Im)
            kappa_4 = Re(log_phi''''(0))         (even)

        Fix T3.5: pre-fix `c3 = +Im(stencil) / 2h³`.  The correct sign is
        NEGATIVE (since (-i)³ · d³/du³ giving Im → kappa_3 = −Im(...)).
        Pre-fix skewness had the wrong sign for non-symmetric distributions.

        Fix T3.6: pre-fix used `h = 1e-4` for the 4th-derivative stencil.
        h⁴ = 1e-16 is at machine epsilon, so the numerator (5-point stencil
        subject to cancellation) was dominated by round-off noise.  Use a
        larger h for high-order derivatives: rule of thumb h ~ ε^(1/(n+1))
        for n-th derivative.  Here ε ≈ 1e-16, so:
          n=2: h ~ 3e-6 → use 1e-4 (the existing 2nd-deriv choice)
          n=3: h ~ 1e-4
          n=4: h ~ 6e-4 → use 1e-3
        """
        h2 = 1e-4   # 2nd-derivative stencil
        h3 = 1e-3   # 3rd-derivative stencil (T3.6 wider h)
        h4 = 1e-2   # 4th-derivative stencil (T3.6 wider h)
        log_cf = lambda u: np.log(self.cf(u + 0j))

        c1 = float(np.imag(log_cf(h2) - log_cf(-h2)) / (2 * h2))
        c2 = float(np.real(-log_cf(h2) - log_cf(-h2) + 2 * log_cf(0)) / (h2 ** 2))

        result = {"mean": c1, "variance": max(c2, 0)}

        if max_order >= 3:
            # 5-point central stencil for f'''(0).
            d3 = np.imag(log_cf(2*h3) - 2*log_cf(h3)
                          + 2*log_cf(-h3) - log_cf(-2*h3)) / (2 * h3**3)
            # Fix T3.5: sign correction — kappa_3 = −Im(log_phi'''(0)).
            c3 = float(-d3)
            result["skewness"] = c3 / max(c2, 1e-20) ** 1.5 if c2 > 0 else 0.0

        if max_order >= 4:
            d4 = np.real(log_cf(2*h4) - 4*log_cf(h4) + 6*log_cf(0)
                          - 4*log_cf(-h4) + log_cf(-2*h4)) / h4**4
            c4 = float(d4)
            result["excess_kurtosis"] = c4 / max(c2, 1e-20) ** 2 if c2 > 0 else 0.0

        return result

    def density(self, x_grid: np.ndarray, n_quad: int = 200) -> np.ndarray:
        """Recover density via Fourier inversion.

        f(x) = (1/2pi) integral exp(-iux) phi(u) du

        Fix T4-CF1: pre-fix this method had two robustness gaps.
        (a) ``n_quad=1`` raised ``IndexError`` at ``du = u[1] - u[0]``
            because the linspace had only one point.
        (b) ``density(scalar_x)`` raised ``TypeError`` at ``len(x)`` because
            ``np.asarray(scalar)`` produces a 0-d array (which has no len()).
        Both now raise clearer errors / handle the scalar case explicitly.
        """
        if n_quad < 2:
            raise ValueError(
                f"density: n_quad must be >= 2 (got {n_quad}); need at "
                "least two quadrature points for the trapezoidal rule."
            )

        x = np.atleast_1d(np.asarray(x_grid))   # accept scalars
        u_max = 50.0
        u = np.linspace(0, u_max, n_quad)
        du = u[1] - u[0]

        density = np.zeros(len(x))
        for k, xk in enumerate(x):
            integrand = np.real(self.cf(u) * np.exp(-1j * u * xk))
            # Fix T1.2: np.trapz was removed in NumPy 2.x; use np.trapezoid.
            density[k] = np.trapezoid(integrand, dx=du) / math.pi

        return np.maximum(density, 0)

    def price_european(
        self,
        spot: float,
        strike: float,
        rate: float,
        is_call: bool = True,
        n_terms: int = 128,
    ) -> float:
        """Price European option via COS method."""
        from pricebook.models.cos_method import cos_price
        from pricebook.models.black76 import OptionType
        otype = OptionType.CALL if is_call else OptionType.PUT
        return cos_price(self.cf, spot, strike, rate, self.T, otype, N=n_terms)

    def to_dict(self) -> dict:
        c = self.cumulants(2)
        return {"T": self.T, "mean": c["mean"], "variance": c["variance"]}
