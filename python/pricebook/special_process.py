"""
Special stochastic processes for rate and credit models.

CIR (square-root): dX = kappa*(theta-X)*dt + xi*sqrt(X)*dW
Ornstein-Uhlenbeck: dX = -a*X*dt + sigma*dW
Bessel: d-dimensional radial process
Gamma/IG subordinators: non-decreasing Lévy processes for time changes.

    from pricebook.special_process import CIRProcess, OUProcess, GammaProcess

    cir = CIRProcess(kappa=2, theta=0.04, xi=0.3, seed=42)
    paths = cir.sample(x0=0.04, T=5.0, n_steps=500, n_paths=10000)
"""

from __future__ import annotations

import math

import numpy as np


class CIRProcess:
    """Cox-Ingersoll-Ross (square-root) process.

    dX = kappa*(theta - X)*dt + xi*sqrt(X)*dW

    Feller condition: 2*kappa*theta > xi^2 ensures X stays positive.
    """

    def __init__(self, kappa: float, theta: float, xi: float, seed: int | None = 42):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.feller = 2 * kappa * theta > xi**2
        self._rng = np.random.default_rng(seed)

    def sample(
        self, x0: float, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Simulate CIR paths via Euler (with absorption at 0).

        Returns: shape (n_paths, n_steps + 1).
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        sqrt_dt = math.sqrt(dt)

        for i in range(n_steps):
            x = paths[:, i]
            x_pos = np.maximum(x, 0.0)
            dW = sqrt_dt * self._rng.standard_normal(n_paths)
            dx = self.kappa * (self.theta - x_pos) * dt + self.xi * np.sqrt(x_pos) * dW
            paths[:, i + 1] = np.maximum(x + dx, 0.0)

        return paths

    def mean(self, x0: float, t: float) -> float:
        """E[X(t)] = theta + (x0 - theta) * exp(-kappa*t)."""
        return self.theta + (x0 - self.theta) * math.exp(-self.kappa * t)

    def variance(self, x0: float, t: float) -> float:
        """Var[X(t)] analytical formula."""
        k, th, xi = self.kappa, self.theta, self.xi
        ekt = math.exp(-k * t)
        return (x0 * xi**2 * ekt / k * (1 - ekt)
                + th * xi**2 / (2 * k) * (1 - ekt)**2)


class OUProcess:
    """Ornstein-Uhlenbeck: mean-reverting Gaussian process.

    dX = -a*(X - mu)*dt + sigma*dW

    Exact simulation: X(t+dt) = mu + (X(t)-mu)*exp(-a*dt)
                       + sigma*sqrt((1-exp(-2*a*dt))/(2*a)) * Z
    """

    def __init__(self, a: float, mu: float = 0.0, sigma: float = 1.0, seed: int | None = 42):
        self.a = a
        self.mu = mu
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def sample(
        self, x0: float, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Exact simulation of OU paths. Shape: (n_paths, n_steps + 1)."""
        dt = T / n_steps
        e_adt = math.exp(-self.a * dt)
        if self.a > 0:
            std = self.sigma * math.sqrt((1 - math.exp(-2 * self.a * dt)) / (2 * self.a))
        else:
            std = self.sigma * math.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0

        for i in range(n_steps):
            Z = self._rng.standard_normal(n_paths)
            paths[:, i + 1] = self.mu + (paths[:, i] - self.mu) * e_adt + std * Z

        return paths

    def stationary_mean(self) -> float:
        return self.mu

    def stationary_variance(self) -> float:
        """Var[X_∞] = sigma^2 / (2*a)."""
        if self.a <= 0:
            return float("inf")
        return self.sigma**2 / (2 * self.a)


class BesselProcess:
    """d-dimensional Bessel process: R(t) = ||W(t)|| where W is d-dim BM.

    For d ≥ 2, R(t) > 0 a.s. Connection: R^2 is a squared Bessel (CIR-like).
    """

    def __init__(self, dimension: int, seed: int | None = 42):
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self.d = dimension
        self._rng = np.random.default_rng(seed)

    def sample(
        self, r0: float, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Simulate Bessel paths via squared Bessel (more stable).

        Squared Bessel: dY = d*dt + 2*sqrt(Y)*dW, then R = sqrt(Y).
        Shape: (n_paths, n_steps + 1).
        """
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        Y = np.zeros((n_paths, n_steps + 1))
        Y[:, 0] = r0**2

        for i in range(n_steps):
            y = np.maximum(Y[:, i], 0.0)
            dW = sqrt_dt * self._rng.standard_normal(n_paths)
            Y[:, i + 1] = np.maximum(y + self.d * dt + 2.0 * np.sqrt(y) * dW, 0.0)

        return np.sqrt(Y)

    def mean_squared(self, r0: float, t: float) -> float:
        """E[R(t)^2] = r0^2 + d*t."""
        return r0**2 + self.d * t


class GammaProcess:
    """Gamma subordinator: non-decreasing Lévy process.

    G(t) ~ Gamma(shape=t/scale, scale=scale).
    E[G(t)] = t, Var[G(t)] = scale*t (when parameterised as rate=1/scale).

    Used as time change for Variance Gamma.
    """

    def __init__(self, variance_rate: float = 1.0, seed: int | None = 42):
        """Args:
            variance_rate: nu parameter. E[G(1)]=1, Var[G(1)]=nu.
        """
        if variance_rate <= 0:
            raise ValueError(f"variance_rate must be positive, got {variance_rate}")
        self.nu = variance_rate
        self._rng = np.random.default_rng(seed)

    def sample(self, T: float, n_paths: int) -> np.ndarray:
        """G(T) per path. Shape: (n_paths,). E[G(T)]=T, Var[G(T)]=nu*T."""
        shape = T / self.nu
        scale = self.nu
        return self._rng.gamma(shape, scale, size=n_paths)


class InverseGaussianProcess:
    """Inverse Gaussian subordinator.

    IG(t) with mean t and variance delta*t.
    Non-decreasing, used as time change for NIG process.
    """

    def __init__(self, delta: float = 1.0, seed: int | None = 42):
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = delta
        self._rng = np.random.default_rng(seed)

    def sample(self, T: float, n_paths: int) -> np.ndarray:
        """IG(T) per path. Shape: (n_paths,)."""
        # Parameterisation: mean=T, shape=T^2/delta
        from scipy.stats import invgauss
        mu_param = T
        shape = T**2 / (self.delta * T) if T > 0 else 1.0
        # scipy invgauss: mu = mean/scale, scale
        return invgauss.rvs(mu=mu_param / shape, scale=shape, size=n_paths,
                            random_state=self._rng)
