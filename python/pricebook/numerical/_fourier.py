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
    n = len(x)
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

        kappa_n = (-i)^n (d^n/du^n log phi)(0)
        """
        h = 1e-4
        log_cf = lambda u: np.log(self.cf(u + 0j))

        c1 = float(np.imag(log_cf(h) - log_cf(-h)) / (2 * h))
        c2 = float(np.real(-log_cf(h) - log_cf(-h) + 2 * log_cf(0)) / (h ** 2))

        result = {"mean": c1, "variance": max(c2, 0)}

        if max_order >= 3:
            c3 = float(np.imag(log_cf(2*h) - 2*log_cf(h) + 2*log_cf(-h) - log_cf(-2*h)) / (2 * h**3))
            result["skewness"] = c3 / max(c2, 1e-20) ** 1.5 if c2 > 0 else 0.0

        if max_order >= 4:
            c4 = float(np.real(log_cf(2*h) - 4*log_cf(h) + 6*log_cf(0) - 4*log_cf(-h) + log_cf(-2*h)) / h**4)
            result["excess_kurtosis"] = c4 / max(c2, 1e-20) ** 2 if c2 > 0 else 0.0

        return result

    def density(self, x_grid: np.ndarray, n_quad: int = 200) -> np.ndarray:
        """Recover density via Fourier inversion.

        f(x) = (1/2pi) integral exp(-iux) phi(u) du
        """
        x = np.asarray(x_grid)
        u_max = 50.0
        u = np.linspace(0, u_max, n_quad)
        du = u[1] - u[0]

        density = np.zeros(len(x))
        for k, xk in enumerate(x):
            integrand = np.real(self.cf(u) * np.exp(-1j * u * xk))
            density[k] = np.trapz(integrand, dx=du) / math.pi

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
