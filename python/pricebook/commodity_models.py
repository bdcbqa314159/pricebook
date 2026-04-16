"""Stochastic models for commodity prices.

* :class:`SchwartzOneFactor` — mean-reverting log-spot (Schwartz 1997 Model 1).
* :class:`GibsonSchwartz` — two-factor spot + stochastic convenience yield.
* :class:`SchwartzSmith` — long-short decomposition (short mean-reverts, long is BM).
* :class:`CommodityJumpDiffusion` — Merton-style jumps for commodity spikes.

References:
    Schwartz, *The Stochastic Behavior of Commodity Prices: Implications for
    Valuation and Hedging*, J. Finance, 1997 (the three models).
    Gibson & Schwartz, *Stochastic Convenience Yield and the Pricing of Oil
    Contingent Claims*, J. Finance, 1990.
    Schwartz & Smith, *Short-Term Variations and Long-Term Dynamics in
    Commodity Prices*, Mgmt. Sci., 2000.
    Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 2-3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Schwartz one-factor ----

@dataclass
class SchwartzOneFactorResult:
    """Schwartz one-factor simulation result."""
    spot_paths: np.ndarray
    times: np.ndarray
    mean_terminal: float
    long_run_level: float


class SchwartzOneFactor:
    """Schwartz (1997) Model 1: mean-reverting log-spot.

    Dynamics:
        d(log S) = κ(μ − log S) dt + σ dW

    Analytical forward price:
        F(S, T) = exp(e^{-κT} × log S + (1 − e^{-κT}) × μ + σ²(1 − e^{-2κT})/(4κ))

    In the risk-neutral measure, μ absorbs the market price of risk.

    Args:
        kappa: mean-reversion speed.
        mu: long-run log-price level.
        sigma: volatility of log-spot.
    """

    def __init__(self, kappa: float, mu: float, sigma: float):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma

    def forward_price(self, spot: float, T: float) -> float:
        """Analytical forward (futures) price F(S, T)."""
        if T <= 0:
            return spot
        e = math.exp(-self.kappa * T)
        log_F = e * math.log(spot) + (1 - e) * self.mu + \
                self.sigma**2 * (1 - math.exp(-2 * self.kappa * T)) / (4 * self.kappa)
        return math.exp(log_F)

    def simulate(
        self,
        spot: float,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> SchwartzOneFactorResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        log_S = np.full((n_paths, n_steps + 1), math.log(spot))

        for step in range(n_steps):
            dW = rng.standard_normal(n_paths) * sqrt_dt
            log_S[:, step + 1] = log_S[:, step] \
                + self.kappa * (self.mu - log_S[:, step]) * dt \
                + self.sigma * dW

        spot_paths = np.exp(log_S)
        mean_term = float(spot_paths[:, -1].mean())
        long_run = math.exp(self.mu)

        return SchwartzOneFactorResult(spot_paths, times, mean_term, long_run)


# ---- Gibson-Schwartz two-factor ----

@dataclass
class GibsonSchwartzResult:
    """Gibson-Schwartz two-factor simulation result."""
    spot_paths: np.ndarray
    convenience_yield_paths: np.ndarray
    times: np.ndarray
    mean_terminal_spot: float
    mean_terminal_delta: float


class GibsonSchwartz:
    """Gibson-Schwartz (1990) two-factor commodity model.

    Dynamics:
        dS/S = (r − δ) dt + σ_S dW_S
        dδ = κ(α − δ) dt + σ_δ dW_δ
        corr(dW_S, dW_δ) = ρ dt

    Convenience yield δ is stochastic and mean-reverting. In the
    risk-neutral measure α absorbs market price of convenience yield risk.

    Forward price (closed form):
        F(S, δ, T) = S × exp(A(T) − B(T) × δ)
    where
        B(T) = (1 − e^{-κT}) / κ
        A(T) = (r − α + σ_δ²/(2κ²) − ρ σ_S σ_δ / κ) × T
               + B(T) × (α − σ_δ²/κ² + ρ σ_S σ_δ / κ)
               − σ_δ² B(T)² / (4κ)

    Args:
        r: risk-free rate.
        kappa: mean reversion of convenience yield.
        alpha: long-run convenience yield.
        sigma_s: spot vol.
        sigma_delta: convenience yield vol.
        rho: correlation between spot and convenience yield.
    """

    def __init__(
        self,
        r: float,
        kappa: float,
        alpha: float,
        sigma_s: float,
        sigma_delta: float,
        rho: float = 0.0,
    ):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if not -1 < rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        self.r = r
        self.kappa = kappa
        self.alpha = alpha
        self.sigma_s = sigma_s
        self.sigma_delta = sigma_delta
        self.rho = rho

    def forward_price(self, spot: float, convenience_yield: float, T: float) -> float:
        """Analytical forward price."""
        if T <= 0:
            return spot
        k = self.kappa
        B = (1 - math.exp(-k * T)) / k

        # A(T) from Gibson-Schwartz (with risk-neutral drift adjustments)
        A = ((self.r - self.alpha + self.sigma_delta**2 / (2 * k**2)
              - self.rho * self.sigma_s * self.sigma_delta / k) * T
             + B * (self.alpha - self.sigma_delta**2 / k**2
                    + self.rho * self.sigma_s * self.sigma_delta / k)
             - self.sigma_delta**2 * B**2 / (4 * k))

        return spot * math.exp(A - B * convenience_yield)

    def simulate(
        self,
        spot: float,
        convenience_yield: float,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> GibsonSchwartzResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        S = np.zeros((n_paths, n_steps + 1))
        delta = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = spot
        delta[:, 0] = convenience_yield

        for step in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + math.sqrt(1 - self.rho**2) * rng.standard_normal(n_paths)

            # Spot dynamics
            drift_s = (self.r - delta[:, step] - 0.5 * self.sigma_s**2) * dt
            S[:, step + 1] = S[:, step] * np.exp(drift_s + self.sigma_s * z1 * sqrt_dt)

            # Convenience yield dynamics (OU)
            delta[:, step + 1] = (delta[:, step]
                                   + self.kappa * (self.alpha - delta[:, step]) * dt
                                   + self.sigma_delta * z2 * sqrt_dt)

        return GibsonSchwartzResult(
            spot_paths=S,
            convenience_yield_paths=delta,
            times=times,
            mean_terminal_spot=float(S[:, -1].mean()),
            mean_terminal_delta=float(delta[:, -1].mean()),
        )


# ---- Schwartz-Smith two-factor ----

@dataclass
class SchwartzSmithResult:
    """Schwartz-Smith long-short decomposition result."""
    spot_paths: np.ndarray          # exp(chi + xi)
    short_term_paths: np.ndarray    # chi (mean-reverting)
    long_term_paths: np.ndarray     # xi (Brownian)
    times: np.ndarray


class SchwartzSmith:
    """Schwartz-Smith (2000) long-short two-factor decomposition.

    log S_t = χ_t + ξ_t  where
        dχ = −κ χ dt + σ_χ dW_χ (short-term mean-reverting)
        dξ = μ_ξ dt + σ_ξ dW_ξ  (long-term Brownian with drift)
        corr(dW_χ, dW_ξ) = ρ dt

    χ captures transient deviations (spikes, news); ξ captures the
    permanent component. Forward price = E^Q[S_T] accessible in closed form.

    Args:
        kappa: mean reversion of short-term factor.
        sigma_chi: short-term vol.
        mu_xi: long-term drift (risk-neutral).
        sigma_xi: long-term vol.
        rho: correlation.
    """

    def __init__(
        self,
        kappa: float,
        sigma_chi: float,
        mu_xi: float,
        sigma_xi: float,
        rho: float = 0.0,
    ):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        self.kappa = kappa
        self.sigma_chi = sigma_chi
        self.mu_xi = mu_xi
        self.sigma_xi = sigma_xi
        self.rho = rho

    def forward_price(
        self,
        chi0: float,
        xi0: float,
        T: float,
    ) -> float:
        """Analytical forward price.

        E[log S_T] = e^{-κT} × χ₀ + ξ₀ + μ_ξ × T
        Var[log S_T] = σ_χ²(1−e^{-2κT})/(2κ) + σ_ξ² T
                       + 2 × ρ σ_χ σ_ξ (1−e^{-κT})/κ

        F(T) = exp(E[log S_T] + 0.5 × Var[log S_T])
        """
        if T <= 0:
            return math.exp(chi0 + xi0)

        e = math.exp(-self.kappa * T)
        mean = e * chi0 + xi0 + self.mu_xi * T

        var = (self.sigma_chi**2 * (1 - math.exp(-2 * self.kappa * T))
               / (2 * self.kappa)
               + self.sigma_xi**2 * T
               + 2 * self.rho * self.sigma_chi * self.sigma_xi
                 * (1 - e) / self.kappa)

        return math.exp(mean + 0.5 * var)

    def simulate(
        self,
        chi0: float,
        xi0: float,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> SchwartzSmithResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        chi = np.full((n_paths, n_steps + 1), chi0)
        xi = np.full((n_paths, n_steps + 1), xi0)

        for step in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + math.sqrt(1 - self.rho**2) * rng.standard_normal(n_paths)

            chi[:, step + 1] = chi[:, step] - self.kappa * chi[:, step] * dt \
                + self.sigma_chi * z1 * sqrt_dt
            xi[:, step + 1] = xi[:, step] + self.mu_xi * dt \
                + self.sigma_xi * z2 * sqrt_dt

        log_S = chi + xi
        spot = np.exp(log_S)

        return SchwartzSmithResult(
            spot_paths=spot,
            short_term_paths=chi,
            long_term_paths=xi,
            times=times,
        )


# ---- Jump-diffusion for commodities ----

@dataclass
class CommodityJumpResult:
    """Commodity jump-diffusion simulation result."""
    spot_paths: np.ndarray
    n_jumps: np.ndarray             # number of jumps per path
    times: np.ndarray
    mean_jumps_per_path: float


class CommodityJumpDiffusion:
    """Merton-style jump-diffusion for commodities.

    Natural for power prices (spikes) and oil crashes.

    Dynamics:
        dS/S = (μ − λκ) dt + σ dW + (Y − 1) dN

    where N ~ Poisson(λ), log Y ~ N(m, δ²).
    For mean-reverting commodities (power), add OU drift.

    Args:
        mu: drift (risk-neutral, typically r).
        sigma: diffusive vol.
        lambda_jump: jump intensity (per year).
        mu_jump: mean log-jump size.
        sigma_jump: vol of log-jump.
        kappa_mr: optional mean-reversion strength (0 for GBM+jumps).
        mu_lr: long-run level if mean-reverting.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lambda_jump: float,
        mu_jump: float,
        sigma_jump: float,
        kappa_mr: float = 0.0,
        mu_lr: float = 0.0,
    ):
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.kappa_mr = kappa_mr
        self.mu_lr = mu_lr

    def simulate(
        self,
        spot: float,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> CommodityJumpResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        kappa_j = math.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0

        S = np.full((n_paths, n_steps + 1), spot)
        jump_counts = np.zeros(n_paths, dtype=int)

        for step in range(n_steps):
            z = rng.standard_normal(n_paths)
            n_j = rng.poisson(self.lambda_jump * dt, n_paths)
            jump_counts += n_j

            # Aggregate log-jump per path
            log_jump = np.zeros(n_paths)
            for p_idx in np.where(n_j > 0)[0]:
                jumps = rng.normal(self.mu_jump, self.sigma_jump, n_j[p_idx])
                log_jump[p_idx] = jumps.sum()

            # Drift: GBM + optional mean reversion
            if self.kappa_mr > 0:
                drift = self.kappa_mr * (self.mu_lr - np.log(np.maximum(S[:, step], 1e-10))) * dt \
                    - self.lambda_jump * kappa_j * dt \
                    - 0.5 * self.sigma**2 * dt
            else:
                drift = (self.mu - self.lambda_jump * kappa_j - 0.5 * self.sigma**2) * dt

            S[:, step + 1] = S[:, step] * np.exp(drift + self.sigma * z * sqrt_dt + log_jump)

        return CommodityJumpResult(
            spot_paths=S,
            n_jumps=jump_counts,
            times=times,
            mean_jumps_per_path=float(jump_counts.mean()),
        )
