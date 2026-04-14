"""Rough paths: efficient fBM, path signatures, rough Heston.

Phase M12 slices 208-210 consolidated.

* :func:`fbm_circulant` — fBM via circulant embedding (O(N log N)).
* :func:`path_signature` — truncated iterated integrals up to level N.
* :func:`log_signature` — compact representation via logarithm.
* :func:`rough_heston_cf` — characteristic function via fractional Riccati.

References:
    Lyons, *Differential Equations Driven by Rough Signals*, 1998.
    Chevyrev & Kormilitzin, *A Primer on the Signature Method*, 2016.
    El Euch & Rosenbaum, *The Characteristic Function of Rough Heston*, 2019.
    Dietrich & Newsam, *Fast and Exact Simulation of Stationary Gaussian
    Processes through Circulant Embedding*, 1997.
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass

import numpy as np


# ---- Efficient fBM via circulant embedding ----

@dataclass
class FBMResult:
    """Fractional Brownian motion paths."""
    paths: np.ndarray   # (n_paths, n_steps+1)
    times: np.ndarray   # (n_steps+1,)
    hurst: float


def fbm_circulant(
    hurst: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
) -> FBMResult:
    """Fractional Brownian motion via circulant embedding.

    O(N log N) complexity vs O(N³) for Cholesky. Embeds the
    Toeplitz autocovariance matrix into a circulant matrix, then
    uses FFT to sample.

    The autocovariance of fBM increments is:
        γ(k) = 0.5(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})

    Args:
        hurst: Hurst parameter H ∈ (0, 1). H=0.5 → standard BM.
        T: time horizon.
        n_steps: number of time steps.
        n_paths: number of independent paths.
        seed: random seed.

    Reference:
        Dietrich & Newsam, SIAM J. Sci. Comput. 18(4), 1997.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    H = hurst

    # Autocovariance of fBM increments
    n = n_steps
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
                    + np.abs(k - 1) ** (2 * H))

    # Embed in circulant: first row = [γ(0), γ(1), ..., γ(n-1), 0, γ(n-1), ..., γ(1)]
    m = 2 * n
    row = np.zeros(m)
    row[:n] = gamma
    row[n] = 0.0
    row[n + 1:] = gamma[1:][::-1]

    # Eigenvalues of circulant via FFT
    eigenvalues = np.fft.fft(row).real
    # Ensure non-negative (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_eig = np.sqrt(eigenvalues)

    # Scale by dt^H
    scale = dt ** H

    paths = np.zeros((n_paths, n_steps + 1))
    for p in range(n_paths):
        # Generate complex Gaussian in frequency domain
        z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        z[0] = z[0].real * math.sqrt(2)
        if m % 2 == 0:
            z[m // 2] = z[m // 2].real * math.sqrt(2)

        # Multiply by sqrt of eigenvalues and invert
        # IFFT includes 1/m factor; compensate with sqrt(m)
        w = (np.fft.ifft(sqrt_eig * z) * math.sqrt(m / 2)).real[:n]

        # Cumulative sum → fBM path
        paths[p, 0] = 0.0
        paths[p, 1:] = np.cumsum(w * scale)

    times = np.linspace(0, T, n_steps + 1)
    return FBMResult(paths, times, hurst)


# ---- Path signatures ----

@dataclass
class SignatureResult:
    """Truncated path signature."""
    signature: list[np.ndarray]  # level k → array of iterated integrals
    depth: int
    path_dim: int


def path_signature(
    path: np.ndarray,
    depth: int = 3,
) -> SignatureResult:
    """Compute the truncated signature of a path up to level N.

    The signature of a d-dimensional path X: [0,T] → R^d is:
        S(X)^k_{i₁...i_k} = ∫...∫ dX^{i₁}...dX^{i_k}

    Level 0: scalar 1.
    Level 1: ∫ dX^i = X(T) − X(0) (increments).
    Level 2: ∫∫ dX^i dX^j (iterated integrals).

    Uses the Chen identity for efficient computation via
    piecewise-linear approximation.

    Args:
        path: (n_steps+1, d) path values. For 1D, shape (n_steps+1,).
        depth: maximum signature level.

    Reference:
        Chevyrev & Kormilitzin, 2016, Section 2.
    """
    path = np.atleast_2d(np.asarray(path, dtype=float))
    if path.ndim == 1:
        path = path.reshape(-1, 1)
    if path.shape[0] < path.shape[1]:
        path = path.T  # ensure (time, dim)

    n_steps, d = path.shape
    increments = np.diff(path, axis=0)  # (n_steps-1, d)

    # Level 0: scalar 1
    sig = [np.array([1.0])]

    # Level 1: path increment
    sig.append(path[-1] - path[0])

    if depth > 3:
        raise ValueError(
            f"depth={depth} requested but only levels 0-3 are implemented. "
            "Use depth <= 3."
        )

    # Compute level 2 explicitly
    if depth >= 2:
        level2 = np.zeros((d, d))
        running = np.zeros(d)  # running level-1 signature
        for j in range(len(increments)):
            dx = increments[j]
            level2 += np.outer(running, dx)
            running += dx
        sig.append(level2.ravel())

    # Level 3
    if depth >= 3:
        level3 = np.zeros((d, d, d))
        running1 = np.zeros(d)
        running2 = np.zeros((d, d))
        for j in range(len(increments)):
            dx = increments[j]
            for a in range(d):
                for b in range(d):
                    level3[a, b, :] += running2[a, b] * dx
            running2 += np.outer(running1, dx)
            running1 += dx
        sig.append(level3.ravel())

    return SignatureResult(sig, min(depth, 3), d)


def log_signature(sig: SignatureResult) -> list[np.ndarray]:
    """Log-signature: logarithm of the signature in the tensor algebra.

    For depth ≤ 2, log(S) = S₁ + S₂ − 0.5 S₁ ⊗ S₁.

    The log-signature is a more compact representation that captures
    the same information with fewer coefficients.
    """
    result = [sig.signature[0]]  # level 0

    if sig.depth >= 1:
        result.append(sig.signature[1])  # level 1 = same

    if sig.depth >= 2:
        s1 = sig.signature[1]
        s2 = sig.signature[2]
        d = len(s1)
        # log(S)₂ = S₂ − 0.5 S₁ ⊗ S₁
        s1_tensor = np.outer(s1, s1).ravel()
        result.append(s2 - 0.5 * s1_tensor)

    return result


# ---- Rough Heston characteristic function ----

def rough_heston_cf(
    u: complex,
    T: float,
    hurst: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rate: float = 0.0,
    n_steps: int = 100,
) -> complex:
    """Characteristic function of log(S_T/S_0) under the rough Heston model.

    The rough Heston replaces the standard Heston variance SDE with a
    Volterra process driven by a fractional kernel K(t) = t^{H-1/2}.

    The CF satisfies a fractional Riccati equation solved via Adams scheme:
        ψ(u, T) = exp(iu(r−q)T + θ ∫₀ᵀ h(s) ds + v₀ I^α h(T))

    where h solves the fractional Riccati and α = H + 0.5.

    Simplified implementation using forward Euler on the Riccati.

    Args:
        u: Fourier variable.
        hurst: Hurst parameter H ∈ (0, 0.5) for rough vol.
        v0: initial variance.
        kappa: mean-reversion speed.
        theta: long-run variance.
        xi: vol-of-vol.
        rho: correlation.
        n_steps: discretisation steps for Riccati.

    Reference:
        El Euch & Rosenbaum, Mathematical Finance 29(1), 2019.
    """
    alpha = hurst + 0.5
    dt = T / n_steps

    # Riccati nonlinearity: F(h) = 0.5(iu − u²) + (iuρξ − κ)h + 0.5ξ²h²
    a_coeff = 0.5 * (1j * u - u * u)
    b_coeff = 1j * u * rho * xi - kappa
    c_coeff = 0.5 * xi * xi

    def F(h):
        return a_coeff + b_coeff * h + c_coeff * h * h

    # Forward Euler on the Volterra integral equation:
    # h(t) = (1/Γ(α)) ∫₀ᵗ (t−s)^{α−1} F(h(s)) ds
    h = np.zeros(n_steps + 1, dtype=complex)
    integral_h = 0.0 + 0j  # ∫₀ᵀ h(s) ds (for θ term)
    volterra_h = 0.0 + 0j  # I^α h(T) (for v₀ term)

    gamma_alpha = math.gamma(alpha)

    for i in range(n_steps):
        t_i = (i + 1) * dt
        # Volterra convolution: h(t) = (1/Γ(α)) Σ (t−t_j)^{α−1} F(h_j) dt
        conv = 0.0 + 0j
        for j in range(i + 1):
            kernel = (t_i - j * dt) ** (alpha - 1) if t_i > j * dt else 0.0
            conv += kernel * F(h[j]) * dt
        h[i + 1] = conv / gamma_alpha

        integral_h += h[i + 1] * dt

    # Fractional integral of h at T for the v₀ term
    for j in range(n_steps + 1):
        kernel = (T - j * dt) ** (alpha - 1) if T > j * dt else 0.0
        volterra_h += kernel * h[j] * dt / gamma_alpha

    log_cf = 1j * u * rate * T + theta * integral_h + v0 * volterra_h
    return cmath.exp(log_cf)
