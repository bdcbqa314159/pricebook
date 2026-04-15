"""Short rate model deepening: BK rates, CIR++ rates, Cheyette, affine.

Extends the existing Hull-White and Vasicek models with:

* :class:`BKRateModel` — Black-Karasinski (log-normal, always positive).
* :class:`CIRPPRateModel` — CIR++ with exact calibration to yield curve.
* :class:`CheyetteModel` — Markovian HJM with state variables (x, y).
* :class:`AffineModel` — unified Dai-Singleton A_m(n) framework.

References:
    Black & Karasinski, *Bond and Option Pricing when Short Rates are
    Lognormal*, Financial Analysts J., 1991.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 3-4.
    Cheyette, *Term Structure Dynamics and Mortgage Valuation*,
    J. Fixed Income, 1992.
    Dai & Singleton, *Specification Analysis of Affine Term Structure
    Models*, J. Finance, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


# ---- Black-Karasinski for rates ----

@dataclass
class BKRateResult:
    """BK rate model result."""
    rate_paths: np.ndarray
    zcb_price: float
    times: np.ndarray


class BKRateModel:
    """Black-Karasinski short rate model.

    d(ln r) = (θ(t) − a ln r) dt + σ dW

    Log-normal: r > 0 always. No analytical ZCB formula → tree or MC.
    θ(t) calibrated to fit the initial yield curve.

    Args:
        a: mean-reversion speed on log-rate.
        sigma: volatility of log-rate.
        market_rates: [(time, zero_rate)] for calibration of θ(t).
    """

    def __init__(self, a: float, sigma: float,
                 market_rates: list[tuple[float, float]] | None = None):
        self.a = a
        self.sigma = sigma
        self._market_rates = market_rates or [(1, 0.05)]

    def _target_log_rate(self, t: float) -> float:
        for ti, ri in self._market_rates:
            if t <= ti:
                return math.log(max(ri, 1e-10))
        return math.log(max(self._market_rates[-1][1], 1e-10))

    def simulate(self, r0: float, T: float, n_steps: int = 100,
                 n_paths: int = 10_000, seed: int | None = None) -> BKRateResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        log_r = np.full((n_paths, n_steps + 1), math.log(max(r0, 1e-10)))

        for i in range(n_steps):
            t = i * dt
            x = log_r[:, i]
            theta = self._target_log_rate(t) * self.a
            dW = rng.standard_normal(n_paths) * sqrt_dt
            log_r[:, i + 1] = x + (theta - self.a * x) * dt + self.sigma * dW

        paths = np.exp(log_r)
        integral = np.sum(paths[:, :-1], axis=1) * dt
        zcb = float(np.mean(np.exp(-integral)))

        return BKRateResult(paths, zcb, times)


# ---- CIR++ for rates ----

@dataclass
class CIRPPRateResult:
    """CIR++ rate model result."""
    rate_paths: np.ndarray
    zcb_price: float
    zcb_analytical: float
    times: np.ndarray


class CIRPPRateModel:
    """CIR++ short rate: r(t) = x(t) + φ(t).

    dx = κ(θ − x)dt + ξ√x dW
    φ(t) calibrated so that the model matches the market yield curve.

    Analytical ZCB for the CIR component; φ shifts to match market.

    Args:
        kappa: CIR mean-reversion.
        theta: CIR long-run level.
        xi: CIR vol.
        market_rates: [(time, zero_rate)] for calibration.
    """

    def __init__(self, kappa: float, theta: float, xi: float,
                 market_rates: list[tuple[float, float]] | None = None):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self._market_rates = market_rates or [(1, 0.05)]

    def _market_rate(self, t: float) -> float:
        for ti, ri in self._market_rates:
            if t <= ti:
                return ri
        return self._market_rates[-1][1]

    def phi(self, t: float, x0: float) -> float:
        x_bar = self.theta + (x0 - self.theta) * math.exp(-self.kappa * t)
        return self._market_rate(t) - x_bar

    def zcb_analytical(self, r0: float, T: float) -> float:
        """CIR analytical ZCB + shift adjustment."""
        gamma = math.sqrt(self.kappa**2 + 2 * self.xi**2)
        exp_gt = math.exp(gamma * T)
        denom = (gamma + self.kappa) * (exp_gt - 1) + 2 * gamma
        B = 2.0 * (exp_gt - 1.0) / denom
        exponent = 2.0 * self.kappa * self.theta / (self.xi**2)
        A_num = 2.0 * gamma * math.exp((self.kappa + gamma) * T / 2.0)
        A = (A_num / denom) ** exponent
        # CIR ZCB on the x component
        x0 = max(r0 - self.phi(0, r0), 0.01)
        cir_zcb = A * math.exp(-B * x0)
        # Shift adjustment: multiply by exp(-∫φ ds) ≈ exp(-φ(T/2) × T)
        phi_avg = self.phi(T / 2, x0)
        return cir_zcb * math.exp(-phi_avg * T)

    def simulate(self, r0: float, T: float, n_steps: int = 100,
                 n_paths: int = 10_000, seed: int | None = None) -> CIRPPRateResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        x0 = max(r0 - self.phi(0, r0), 0.01)
        x = np.full((n_paths, n_steps + 1), x0)

        for i in range(n_steps):
            xc = np.maximum(x[:, i], 0.0)
            dW = rng.standard_normal(n_paths) * sqrt_dt
            x[:, i + 1] = xc + self.kappa * (self.theta - xc) * dt + self.xi * np.sqrt(xc) * dW
            x[:, i + 1] = np.maximum(x[:, i + 1], 0.0)

        # r = x + φ
        phi_vals = np.array([self.phi(t, x0) for t in times])
        rate_paths = x + phi_vals[np.newaxis, :]

        integral = np.sum(rate_paths[:, :-1], axis=1) * dt
        zcb_mc = float(np.mean(np.exp(-integral)))
        zcb_a = self.zcb_analytical(r0, T)

        return CIRPPRateResult(rate_paths, zcb_mc, zcb_a, times)


# ---- Cheyette model ----

@dataclass
class CheyetteResult:
    """Cheyette model result."""
    rate_paths: np.ndarray
    x_paths: np.ndarray
    y_paths: np.ndarray
    zcb_price: float
    times: np.ndarray


class CheyetteModel:
    """Cheyette (Markovian HJM) with state variables (x, y).

    dx = (y − κx) dt + σ(t) dW
    dy = (σ(t)² − 2κy) dt
    r(t) = f(0, t) + x(t)

    where f(0, t) is the initial forward rate curve. The model is
    Markovian in (x, y), unlike general HJM which is infinite-dimensional.

    Args:
        kappa: mean-reversion.
        sigma: volatility (constant for simplicity; can be time-dependent).
        forward_rates: [(time, forward_rate)] initial curve.
    """

    def __init__(self, kappa: float, sigma: float,
                 forward_rates: list[tuple[float, float]] | None = None):
        self.kappa = kappa
        self.sigma = sigma
        self._forward_rates = forward_rates or [(1, 0.05), (5, 0.05), (10, 0.05)]

    def _f0(self, t: float) -> float:
        for ti, fi in self._forward_rates:
            if t <= ti:
                return fi
        return self._forward_rates[-1][1]

    def zcb_analytical(self, T: float) -> float:
        """Analytical ZCB: P(0,T) = exp(−∫₀ᵀ f(0,s) ds)."""
        n = max(int(T * 20), 1)
        dt = T / n
        return math.exp(-sum(self._f0(i * dt) * dt for i in range(n)))

    def simulate(self, T: float, n_steps: int = 100,
                 n_paths: int = 10_000, seed: int | None = None) -> CheyetteResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        x = np.zeros((n_paths, n_steps + 1))
        y = np.zeros((n_paths, n_steps + 1))

        for i in range(n_steps):
            t = i * dt
            sig = self.sigma
            dW = rng.standard_normal(n_paths) * sqrt_dt
            x[:, i + 1] = x[:, i] + (y[:, i] - self.kappa * x[:, i]) * dt + sig * dW
            y[:, i + 1] = y[:, i] + (sig**2 - 2 * self.kappa * y[:, i]) * dt

        # r(t) = f(0,t) + x(t)
        f0_vals = np.array([self._f0(t) for t in times])
        rate_paths = f0_vals[np.newaxis, :] + x

        integral = np.sum(rate_paths[:, :-1], axis=1) * dt
        zcb = float(np.mean(np.exp(-integral)))

        return CheyetteResult(rate_paths, x, y, zcb, times)


# ---- Affine term structure (Dai-Singleton) ----

@dataclass
class AffineResult:
    """Affine model result."""
    zcb_price: float
    A_coeff: float
    B_coeffs: np.ndarray
    model_class: str


class AffineModel:
    """Unified affine term structure model A_m(n).

    n factors, m with square-root (CIR-type), n−m Gaussian (Vasicek-type).

    dr_i = κ_i(θ_i − r_i)dt + σ_i √(α_i + β_i r_i) dW_i

    For CIR-type (i < m): α=0, β=1 (square-root diffusion).
    For Vasicek-type (i ≥ m): α=1, β=0 (constant diffusion).

    ZCB: P(t,T) = exp(A(τ) − Σ B_i(τ) r_i(t)) where τ = T−t.

    Args:
        kappas: mean-reversion speeds.
        thetas: long-run levels.
        sigmas: volatilities.
        m: number of CIR-type factors (remaining are Vasicek).
    """

    def __init__(self, kappas: list[float], thetas: list[float],
                 sigmas: list[float], m: int = 0):
        self.n = len(kappas)
        self.m = m
        self.kappas = np.array(kappas)
        self.thetas = np.array(thetas)
        self.sigmas = np.array(sigmas)

    @property
    def model_class(self) -> str:
        return f"A_{self.m}({self.n})"

    def zcb_price(self, r0: list[float] | np.ndarray, T: float) -> AffineResult:
        """Analytical ZCB via Riccati ODEs.

        For each factor, solve the Riccati ODE for B_i(τ):
        - CIR: dB/dτ = 1 − κB − 0.5σ²B²
        - Vasicek: dB/dτ = 1 − κB (linear → exact)

        A(τ) = Σ κ_i θ_i ∫₀^τ B_i(s) ds − 0.5 Σ_{Vasicek} σ_i² ∫₀^τ B_i²(s) ds
        """
        r = np.asarray(r0, dtype=float)
        B = np.zeros(self.n)
        A = 0.0

        # Solve via Euler on the Riccati
        n_ode = max(int(T * 100), 10)
        dtau = T / n_ode

        B_vals = np.zeros(self.n)
        A_val = 0.0

        for step in range(n_ode):
            for i in range(self.n):
                if i < self.m:
                    # CIR: dB/dτ = 1 − κB − 0.5σ²B²
                    dB = 1.0 - self.kappas[i] * B_vals[i] - 0.5 * self.sigmas[i]**2 * B_vals[i]**2
                else:
                    # Vasicek: dB/dτ = 1 − κB
                    dB = 1.0 - self.kappas[i] * B_vals[i]
                B_vals[i] += dB * dtau

                # A contribution: A = −Σ κθ ∫B ds + 0.5 Σ_{Vasicek} σ² ∫B² ds
                A_val -= self.kappas[i] * self.thetas[i] * B_vals[i] * dtau
                if i >= self.m:
                    A_val += 0.5 * self.sigmas[i]**2 * B_vals[i]**2 * dtau

        zcb = math.exp(A_val - float(B_vals @ r))

        return AffineResult(zcb, A_val, B_vals, self.model_class)
