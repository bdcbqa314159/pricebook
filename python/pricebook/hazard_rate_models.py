"""Stochastic hazard rate models: Hull-White, Black-Karasinski, CIR++.

Extends :mod:`pricebook.stochastic_credit` with short-rate-style models
applied to the default intensity λ(t).

* :class:`HWHazardRate` — Gaussian mean-reverting intensity, exact calibration.
* :class:`BKHazardRate` — log-normal intensity via trinomial tree.
* :class:`CIRPlusPlus` — shifted CIR for exact calibration to survival curve.
* :class:`TwoFactorIntensity` — level + slope decomposition.

References:
    Hull & White, *Pricing Interest-Rate-Derivative Securities*, RFS, 1990
    (applied to hazard rates).
    Black & Karasinski, *Bond and Option Pricing when Short Rates are
    Lognormal*, Financial Analysts J., 1991.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 3-4 (CIR++).
    Schönbucher, *Credit Derivatives Pricing Models*, Ch. 7-8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


# ---- Hull-White hazard rate ----

@dataclass
class HWHazardResult:
    """Hull-White hazard rate simulation result."""
    lambda_paths: np.ndarray  # (n_paths, n_steps+1)
    survival_mc: float
    times: np.ndarray


class HWHazardRate:
    """Hull-White model for default intensity.

    dλ = (θ(t) − aλ) dt + σ dW

    Gaussian: can go negative (use for short horizons or with floor).
    θ(t) is calibrated to match the market survival curve exactly:
        θ(t) = ∂f/∂t + a f(t) + σ²/(2a)(1 − e^{−2at})
    where f(t) = −∂ln Q(t)/∂t is the instantaneous forward hazard rate.

    Args:
        a: mean-reversion speed.
        sigma: volatility of the intensity.
        market_hazards: list of (time, hazard_rate) pillars for calibration.
    """

    def __init__(
        self,
        a: float,
        sigma: float,
        market_hazards: list[tuple[float, float]] | None = None,
    ):
        self.a = a
        self.sigma = sigma
        self._market_hazards = market_hazards or []

    def theta(self, t: float) -> float:
        """Time-dependent drift θ(t) for exact calibration.

        If no market hazards provided, uses a flat forward hazard.
        """
        f = self._forward_hazard(t)
        df_dt = self._forward_hazard_deriv(t)
        return df_dt + self.a * f + self.sigma**2 / (2 * self.a) * (
            1 - math.exp(-2 * self.a * t)
        ) if self.a > 0 else f

    def _forward_hazard(self, t: float) -> float:
        """Instantaneous forward hazard rate from market pillars."""
        if not self._market_hazards:
            return 0.02  # flat default
        # Piecewise constant interpolation
        for i, (ti, hi) in enumerate(self._market_hazards):
            if t <= ti:
                return hi
        return self._market_hazards[-1][1]

    def _forward_hazard_deriv(self, t: float) -> float:
        """Numerical derivative of forward hazard."""
        eps = 1e-4
        return (self._forward_hazard(t + eps) - self._forward_hazard(t - eps)) / (2 * eps)

    def survival_analytical(self, lam0: float, T: float) -> float:
        """Analytical survival probability (affine model).

        Q(0,T) = exp(−A(T) − B(T)λ₀)
        where B(T) = (1 − e^{−aT})/a and A(T) involves θ(t) integral.

        Simplified: uses numerical integration of θ.
        """
        n = max(int(T * 100), 10)
        dt = T / n
        B = (1 - math.exp(-self.a * T)) / self.a if self.a > 0 else T

        # A(T) = ∫₀ᵀ [θ(s)B(T−s) − 0.5σ²B(T−s)²] ds
        A = 0.0
        for i in range(n):
            s = (i + 0.5) * dt
            B_ts = (1 - math.exp(-self.a * (T - s))) / self.a if self.a > 0 else (T - s)
            A += (self.theta(s) * B_ts - 0.5 * self.sigma**2 * B_ts**2) * dt

        return math.exp(-A - B * lam0)

    def simulate(
        self,
        lam0: float,
        T: float,
        n_steps: int = 100,
        n_paths: int = 10_000,
        seed: int | None = None,
    ) -> HWHazardResult:
        """Simulate HW intensity paths via Euler."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = lam0

        for i in range(n_steps):
            t = i * dt
            lam = paths[:, i]
            dW = rng.standard_normal(n_paths) * sqrt_dt
            paths[:, i + 1] = lam + (self.theta(t) - self.a * lam) * dt + self.sigma * dW

        # MC survival: E[exp(−∫λ ds)]
        integral = np.sum(paths[:, :-1], axis=1) * dt
        surv = float(np.mean(np.exp(-integral)))

        return HWHazardResult(paths, surv, times)


# ---- Black-Karasinski hazard rate ----

@dataclass
class BKHazardResult:
    """Black-Karasinski hazard rate result."""
    lambda_paths: np.ndarray
    survival_mc: float
    times: np.ndarray


class BKHazardRate:
    """Black-Karasinski model for default intensity.

    d(ln λ) = (θ(t) − a ln λ) dt + σ dW

    Log-normal: λ > 0 always. No analytical survival formula →
    use trinomial tree or MC simulation.

    Args:
        a: mean-reversion speed (on log-intensity).
        sigma: volatility of log-intensity.
        market_hazards: for calibration of θ(t).
    """

    def __init__(
        self,
        a: float,
        sigma: float,
        market_hazards: list[tuple[float, float]] | None = None,
    ):
        self.a = a
        self.sigma = sigma
        self._market_hazards = market_hazards or []

    def _target_log_hazard(self, t: float) -> float:
        """Target log(λ) from market pillars."""
        if not self._market_hazards:
            return math.log(0.02)
        for ti, hi in self._market_hazards:
            if t <= ti:
                return math.log(max(hi, 1e-10))
        return math.log(max(self._market_hazards[-1][1], 1e-10))

    def simulate(
        self,
        lam0: float,
        T: float,
        n_steps: int = 100,
        n_paths: int = 10_000,
        seed: int | None = None,
    ) -> BKHazardResult:
        """Simulate BK intensity paths via Euler on log-intensity."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        log_lam = np.full((n_paths, n_steps + 1), math.log(max(lam0, 1e-10)))

        for i in range(n_steps):
            t = i * dt
            x = log_lam[:, i]
            theta_t = self._target_log_hazard(t) * self.a  # simplified calibration
            dW = rng.standard_normal(n_paths) * sqrt_dt
            log_lam[:, i + 1] = x + (theta_t - self.a * x) * dt + self.sigma * dW

        paths = np.exp(log_lam)

        # MC survival
        integral = np.sum(paths[:, :-1], axis=1) * dt
        surv = float(np.mean(np.exp(-integral)))

        return BKHazardResult(paths, surv, times)

    def trinomial_tree_survival(
        self,
        lam0: float,
        T: float,
        n_steps: int = 50,
    ) -> float:
        """Survival probability via trinomial tree on log(λ).

        Mirrors the Hull-White trinomial tree but in log-space.
        Probabilities calibrated to match drift and variance.
        """
        dt = T / n_steps
        dx = self.sigma * math.sqrt(3 * dt)
        if dx == 0:
            return math.exp(-lam0 * T)

        x0 = math.log(max(lam0, 1e-10))
        j_max = int(math.ceil(0.184 / (self.a * dt))) if self.a * dt > 0 else n_steps

        # Build tree nodes
        n_nodes = 2 * j_max + 1
        x_vals = np.array([x0 + j * dx for j in range(-j_max, j_max + 1)])

        # Arrow-Debreu prices (probability × discount)
        Q = np.zeros(n_nodes)
        mid = j_max
        Q[mid] = 1.0

        for step in range(n_steps):
            t = step * dt
            Q_new = np.zeros(n_nodes)

            for j in range(n_nodes):
                if Q[j] < 1e-20:
                    continue

                x_j = x_vals[j]
                lam_j = math.exp(x_j)

                # Discount by survival over this step
                disc = math.exp(-lam_j * dt)

                # Drift
                mu = self._target_log_hazard(t) - self.a * x_j

                # Probabilities (Kamrad-Ritchken style)
                eta = mu * dt / dx
                p_up = (1.0 / 6.0 + (eta**2 + eta) / 2.0)
                p_mid = (2.0 / 3.0 - eta**2)
                p_dn = (1.0 / 6.0 + (eta**2 - eta) / 2.0)

                # Clamp probabilities
                p_up = max(min(p_up, 1.0), 0.0)
                p_dn = max(min(p_dn, 1.0), 0.0)
                p_mid = 1.0 - p_up - p_dn
                p_mid = max(p_mid, 0.0)

                # Distribute
                j_up = min(j + 1, n_nodes - 1)
                j_dn = max(j - 1, 0)
                Q_new[j_up] += Q[j] * disc * p_up
                Q_new[j] += Q[j] * disc * p_mid
                Q_new[j_dn] += Q[j] * disc * p_dn

            Q = Q_new

        return float(Q.sum())


# ---- CIR++ (shifted CIR) ----

@dataclass
class CIRPPResult:
    """CIR++ hazard rate result."""
    lambda_paths: np.ndarray
    survival_mc: float
    times: np.ndarray
    phi: np.ndarray  # deterministic shift function


class CIRPlusPlus:
    """CIR++ model: λ(t) = x(t) + φ(t).

    x follows CIR: dx = κ(θ − x)dt + ξ√x dW
    φ(t) is a deterministic shift calibrated so that the model
    matches the market survival curve exactly.

    φ(t) = λ_market(t) − x_bar(t)

    where x_bar(t) = E[x(t)] = θ + (x₀ − θ)e^{−κt}.

    Args:
        kappa: CIR mean-reversion.
        theta: CIR long-run level.
        xi: CIR vol-of-vol.
        market_hazards: for calibration of φ(t).
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        xi: float,
        market_hazards: list[tuple[float, float]] | None = None,
    ):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self._market_hazards = market_hazards or []

    def phi(self, t: float, x0: float) -> float:
        """Deterministic shift φ(t) = λ_market(t) − E[x(t)]."""
        x_bar = self.theta + (x0 - self.theta) * math.exp(-self.kappa * t)
        lam_market = self._market_hazard(t)
        return lam_market - x_bar

    def _market_hazard(self, t: float) -> float:
        if not self._market_hazards:
            return 0.02
        for ti, hi in self._market_hazards:
            if t <= ti:
                return hi
        return self._market_hazards[-1][1]

    def simulate(
        self,
        x0: float,
        T: float,
        n_steps: int = 100,
        n_paths: int = 10_000,
        seed: int | None = None,
    ) -> CIRPPResult:
        """Simulate CIR++ paths: λ(t) = x(t) + φ(t)."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        x_paths = np.zeros((n_paths, n_steps + 1))
        x_paths[:, 0] = x0
        phi_vals = np.zeros(n_steps + 1)
        phi_vals[0] = self.phi(0, x0)

        for i in range(n_steps):
            t = (i + 1) * dt
            x = np.maximum(x_paths[:, i], 0.0)
            dW = rng.standard_normal(n_paths) * sqrt_dt
            x_paths[:, i + 1] = x + self.kappa * (self.theta - x) * dt + self.xi * np.sqrt(x) * dW
            x_paths[:, i + 1] = np.maximum(x_paths[:, i + 1], 0.0)
            phi_vals[i + 1] = self.phi(t, x0)

        # λ = x + φ
        lambda_paths = x_paths + phi_vals[np.newaxis, :]

        # MC survival
        integral = np.sum(lambda_paths[:, :-1], axis=1) * dt
        surv = float(np.mean(np.exp(-integral)))

        return CIRPPResult(lambda_paths, surv, times, phi_vals)


# ---- Two-factor intensity ----

@dataclass
class TwoFactorResult:
    """Two-factor intensity result."""
    lambda_paths: np.ndarray
    x1_paths: np.ndarray
    x2_paths: np.ndarray
    survival_mc: float
    times: np.ndarray


class TwoFactorIntensity:
    """Two-factor intensity: λ(t) = x₁(t) + x₂(t) + φ(t).

    dx₁ = −a₁ x₁ dt + σ₁ dW₁  (level factor, slow mean-reversion)
    dx₂ = −a₂ x₂ dt + σ₂ dW₂  (slope factor, fast mean-reversion)
    corr(dW₁, dW₂) = ρ

    This gives a richer term structure of hazard rate volatility
    than single-factor models.
    """

    def __init__(
        self,
        a1: float,
        sigma1: float,
        a2: float,
        sigma2: float,
        rho: float = 0.0,
        base_hazard: float = 0.02,
    ):
        self.a1 = a1
        self.sigma1 = sigma1
        self.a2 = a2
        self.sigma2 = sigma2
        self.rho = rho
        self.base_hazard = base_hazard

    def simulate(
        self,
        T: float,
        n_steps: int = 100,
        n_paths: int = 10_000,
        seed: int | None = None,
    ) -> TwoFactorResult:
        """Simulate two-factor intensity paths."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        x1 = np.zeros((n_paths, n_steps + 1))
        x2 = np.zeros((n_paths, n_steps + 1))

        for i in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            dW1 = z1 * sqrt_dt
            dW2 = (self.rho * z1 + math.sqrt(1 - self.rho**2) * z2) * sqrt_dt

            x1[:, i + 1] = x1[:, i] - self.a1 * x1[:, i] * dt + self.sigma1 * dW1
            x2[:, i + 1] = x2[:, i] - self.a2 * x2[:, i] * dt + self.sigma2 * dW2

        lambda_paths = x1 + x2 + self.base_hazard

        integral = np.sum(np.maximum(lambda_paths[:, :-1], 0.0), axis=1) * dt
        surv = float(np.mean(np.exp(-integral)))

        return TwoFactorResult(lambda_paths, x1, x2, surv, times)
