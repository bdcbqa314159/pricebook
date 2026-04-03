"""
Jump processes: Poisson, compound Poisson, Merton jump-diffusion, Variance Gamma.

    from pricebook.jump_process import PoissonProcess, MertonJumpDiffusion

    pp = PoissonProcess(intensity=5.0, seed=42)
    counts = pp.sample(T=1.0, n_paths=10000)

    mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=1.0,
                               jump_mean=-0.1, jump_std=0.15)
    st = mjd.terminal(S0=100, T=1.0, n_paths=50000, seed=42)
"""

from __future__ import annotations

import math
import cmath

import numpy as np


class PoissonProcess:
    """Homogeneous Poisson process N(t) with constant intensity λ.

    E[N(t)] = λt, Var[N(t)] = λt.
    """

    def __init__(self, intensity: float, seed: int | None = 42):
        if intensity < 0:
            raise ValueError(f"intensity must be non-negative, got {intensity}")
        self.intensity = intensity
        self._rng = np.random.default_rng(seed)

    def sample(self, T: float, n_paths: int) -> np.ndarray:
        """Number of events in [0, T] per path. Shape: (n_paths,)."""
        return self._rng.poisson(self.intensity * T, size=n_paths)

    def inter_arrivals(self, n_events: int, n_paths: int) -> np.ndarray:
        """Exponential inter-arrival times. Shape: (n_paths, n_events)."""
        if self.intensity <= 0:
            return np.full((n_paths, n_events), np.inf)
        return self._rng.exponential(1.0 / self.intensity, size=(n_paths, n_events))


class CompoundPoissonProcess:
    """Compound Poisson: X(t) = sum of J_i for i=1..N(t).

    N(t) is Poisson(λt), J_i are iid jump sizes.

    Args:
        intensity: Poisson rate λ.
        jump_mean: mean of jump size distribution.
        jump_std: std of jump size distribution (normal jumps).
    """

    def __init__(
        self,
        intensity: float,
        jump_mean: float = 0.0,
        jump_std: float = 0.1,
        seed: int | None = 42,
    ):
        self.intensity = intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self._rng = np.random.default_rng(seed)

    def sample(self, T: float, n_paths: int) -> np.ndarray:
        """Total jump value X(T) per path. Shape: (n_paths,)."""
        N = self._rng.poisson(self.intensity * T, size=n_paths)
        N_max = max(int(N.max()), 1) if n_paths > 0 else 1
        jumps = self._rng.normal(self.jump_mean, self.jump_std, (n_paths, N_max))
        mask = np.arange(N_max) < N[:, None]
        return (jumps * mask).sum(axis=1)


class MertonJumpDiffusion:
    """Merton jump-diffusion model.

    dS/S = (mu - λk)*dt + sigma*dW + J*dN

    where J ~ N(jump_mean, jump_std^2), k = E[e^J - 1].
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lam: float,
        jump_mean: float,
        jump_std: float,
    ):
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        # k = E[e^J - 1] for lognormal jumps
        self.k = math.exp(jump_mean + 0.5 * jump_std**2) - 1

    def terminal(
        self, S0: float, T: float, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """Simulate terminal values S(T). Shape: (n_paths,)."""
        rng = np.random.default_rng(seed)

        # Diffusion part
        drift = (self.mu - self.lam * self.k - 0.5 * self.sigma**2) * T
        diffusion = self.sigma * math.sqrt(T) * rng.standard_normal(n_paths)

        # Jump part
        N = rng.poisson(self.lam * T, size=n_paths)
        N_max = max(int(N.max()), 1)
        jumps = rng.normal(self.jump_mean, self.jump_std, (n_paths, N_max))
        mask = np.arange(N_max) < N[:, None]
        jump_sum = (jumps * mask).sum(axis=1)

        return S0 * np.exp(drift + diffusion + jump_sum)

    def char_func(self, T: float):
        """Characteristic function of log(S_T/S_0) for COS pricing."""
        mu, sigma, lam = self.mu, self.sigma, self.lam
        jm, js = self.jump_mean, self.jump_std
        k = self.k

        def phi(u: float) -> complex:
            # Diffusion part
            diff = 1j * u * (mu - lam * k - 0.5 * sigma**2) * T \
                   - 0.5 * u**2 * sigma**2 * T
            # Jump part: λT * (E[e^{iuJ}] - 1) = λT * (exp(iu*jm - 0.5*u^2*js^2) - 1)
            jump_cf = cmath.exp(1j * u * jm - 0.5 * u**2 * js**2) - 1
            return cmath.exp(diff + lam * T * jump_cf)

        return phi


class VarianceGammaProcess:
    """Variance Gamma: BM evaluated at a Gamma subordinator.

    Parameters: sigma (vol), theta (drift), nu (variance of time change).
    VG with nu→0 → Black-Scholes.
    """

    def __init__(self, sigma: float, theta: float, nu: float):
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.sigma = sigma
        self.theta = theta
        self.nu = nu

    def terminal(
        self, S0: float, rate: float, T: float, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """Simulate terminal S(T) under risk-neutral measure."""
        rng = np.random.default_rng(seed)

        # Gamma time change: G ~ Gamma(T/nu, nu)
        shape = T / self.nu
        scale = self.nu
        G = rng.gamma(shape, scale, size=n_paths)

        # BM evaluated at Gamma time
        Z = rng.standard_normal(n_paths)
        X = self.theta * G + self.sigma * np.sqrt(G) * Z

        # Risk-neutral drift adjustment
        omega = (1.0 / self.nu) * math.log(1 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu)
        drift = (rate + omega) * T

        return S0 * np.exp(drift + X)

    def char_func(self, rate: float, T: float):
        """Characteristic function of log(S_T/S_0)."""
        sigma, theta, nu = self.sigma, self.theta, self.nu
        omega = (1.0 / nu) * math.log(1 - theta * nu - 0.5 * sigma**2 * nu)

        def phi(u: float) -> complex:
            # Risk-neutral drift
            drift = 1j * u * (rate + omega) * T
            # VG char func: (1 - iu*theta*nu + 0.5*u^2*sigma^2*nu)^{-T/nu}
            inner = 1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu
            return cmath.exp(drift) * inner ** (-T / nu)

        return phi
