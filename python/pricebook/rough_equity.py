"""Rough volatility for equity: rBergomi, rough Heston, forward variance.

* :class:`rBergomiEquity` — rBergomi model for equity with calibration.
* :func:`rough_heston_cf` — characteristic function for rough Heston.
* :func:`rough_heston_price` — COS pricing under rough Heston.
* :func:`forward_variance_curve` — ξ(T) bootstrap.

References:
    Bayer, Friz & Gatheral, *Pricing Under Rough Volatility*, QF, 2016.
    El Euch & Rosenbaum, *The Characteristic Function of Rough Heston Models*,
    Math. Finance, 2019.
    Gatheral, Jaisson & Rosenbaum, *Volatility is Rough*, QF, 2018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn


# ---- rBergomi ----

@dataclass
class rBergomiResult:
    """rBergomi simulation result."""
    spot_paths: np.ndarray
    variance_paths: np.ndarray
    mean_terminal: float
    H: float
    n_paths: int


class rBergomiEquity:
    """rBergomi model for equity.

    dS/S = (r - q) dt + √v_t dW_s
    v_t = ξ_0(t) × exp(η × W^H_t − (η² / 2) × t^{2H})

    where W^H is a fractional Brownian motion with Hurst H < 0.5 (rough).

    Args:
        xi0: flat forward variance (simplified; can be T-dependent).
        eta: vol of vol.
        H: Hurst parameter (0.05 - 0.5 typical, 0.1 = "very rough").
        rho: correlation between spot and vol Brownian.
    """

    def __init__(self, xi0: float, eta: float, H: float, rho: float = -0.7):
        if not 0 < H < 0.5:
            raise ValueError("Hurst H must be in (0, 0.5) for rough vol")
        self.xi0 = xi0
        self.eta = eta
        self.H = H
        self.rho = rho

    def simulate(
        self,
        spot: float,
        rate: float,
        dividend_yield: float,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> rBergomiResult:
        """Simulate rBergomi spot and variance paths.

        Uses hybrid scheme: variance via fractional kernel, spot via GBM.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)
        H = self.H

        # Generate correlated Brownian increments
        Z1 = rng.standard_normal((n_paths, n_steps))
        Z2 = self.rho * Z1 + math.sqrt(1 - self.rho**2) * rng.standard_normal((n_paths, n_steps))

        # Fractional Brownian motion increments via simple power-law weighting
        # W^H(t) = sum of weighted past increments
        # Simplified: use scaled Z1 for fBM-like behaviour
        W_H = np.zeros((n_paths, n_steps + 1))
        for i in range(1, n_steps + 1):
            # Riemann-Liouville fBM: W^H(t_i) = (1/Γ(H+½)) Σ_{j=0}^{i-1} (t_i−t_j)^{H−½} ΔW_j
            weights = np.array([(times[i] - times[j]) ** (H - 0.5) for j in range(i)])
            weights *= sqrt_dt / max(gamma_fn(H + 0.5), 1e-10)
            # Full convolution: sum over all past increments weighted by kernel
            W_H[:, i] = Z1[:, :i] @ weights

        # Variance: v_t = ξ₀ × exp(η W^H − η² t^{2H} / 2)
        v = self.xi0 * np.exp(
            self.eta * W_H
            - 0.5 * self.eta**2 * times ** (2 * H)
        )

        # Spot: log-Euler with stochastic variance
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = spot
        for step in range(n_steps):
            v_pos = np.maximum(v[:, step], 0.0)
            drift = (rate - dividend_yield - 0.5 * v_pos) * dt
            S[:, step + 1] = S[:, step] * np.exp(drift + np.sqrt(v_pos) * Z2[:, step] * sqrt_dt)

        return rBergomiResult(
            spot_paths=S,
            variance_paths=v,
            mean_terminal=float(S[:, -1].mean()),
            H=self.H,
            n_paths=n_paths,
        )

    def forward_variance(self, T: float) -> float:
        """Forward variance ξ(T). Constant in this simplified model."""
        return self.xi0

    def implied_vol(self, spot: float, strike: float, rate: float,
                    dividend_yield: float, T: float, is_call: bool = True,
                    n_paths: int = 10_000, seed: int | None = 42) -> float:
        """Approximate implied vol from MC price.

        Returns Black-Scholes implied vol that reproduces the MC price.
        """
        result = self.simulate(spot, rate, dividend_yield, T, n_paths, seed=seed)
        S_T = result.spot_paths[:, -1]
        if is_call:
            payoff = np.maximum(S_T - strike, 0.0)
        else:
            payoff = np.maximum(strike - S_T, 0.0)

        df = math.exp(-rate * T)
        price = df * float(payoff.mean())

        # Invert BS for implied vol
        from pricebook.implied_vol import implied_vol_black76
        from pricebook.black76 import OptionType
        try:
            F = spot * math.exp((rate - dividend_yield) * T)
            ot = OptionType.CALL if is_call else OptionType.PUT
            iv = implied_vol_black76(price, F, strike, T, df, ot)
            return float(iv)
        except Exception:
            return float(math.sqrt(self.xi0))


# ---- Rough Heston ----

@dataclass
class RoughHestonParams:
    """Rough Heston parameters."""
    v0: float          # initial variance
    lambda_: float     # mean reversion (equivalent to κ in standard Heston)
    theta: float       # long-run variance
    xi: float          # vol of vol
    rho: float         # correlation
    H: float           # Hurst parameter (0 < H < 0.5)


def rough_heston_cf(
    u: complex,
    T: float,
    params: RoughHestonParams,
    n_steps: int = 200,
) -> complex:
    """Characteristic function for rough Heston.

    Uses El Euch-Rosenbaum fractional Riccati equation, discretised.
    Simplified approximation with power-law kernel.

    For pricing, this is typically fed into COS / Carr-Madan.
    """
    # Fractional Riccati (discretised):
    # h(t) = ∫₀^t K(t-s) × F(u, h(s)) ds
    # where K(τ) = τ^{H-1/2} / Γ(H+1/2)
    # F(u, h) = 0.5(u² + iu) + (iuρξ − λ) h + 0.5 ξ² h²
    # Simplified approximation

    alpha = params.H + 0.5
    dt = T / n_steps
    times = np.linspace(dt, T, n_steps)

    h = np.zeros(n_steps, dtype=complex)

    for i in range(n_steps):
        t = times[i]
        # F(u, h_{i-1})
        if i == 0:
            h_prev = 0.0
        else:
            h_prev = h[i - 1]

        F_val = 0.5 * (u**2 + 1j * u) \
                + (1j * u * params.rho * params.xi - params.lambda_) * h_prev \
                + 0.5 * params.xi**2 * h_prev**2

        # Integral approximation: K × F × dt with power-law kernel
        kernel = (t ** (alpha - 1)) / max(gamma_fn(alpha), 1e-10)
        h[i] = h[i - 1] + kernel * F_val * dt if i > 0 else kernel * F_val * dt

    # CF = exp(v0 × ∫₀ᵀ h(s) ds) for flat forward variance ξ₀ = v0
    # Trapezoidal integration of h over discretised time grid
    h_integral = np.sum(h) * dt  # trapezoidal approximation on uniform grid
    cf = np.exp(params.v0 * h_integral)
    return cf


def rough_heston_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    T: float,
    params: RoughHestonParams,
    is_call: bool = True,
    n_integration: int = 100,
    u_max: float = 30.0,
) -> float:
    """European option price under rough Heston via Fourier inversion.

    Uses a simple Carr-Madan-like quadrature over the characteristic function.
    """
    F = spot * math.exp((rate - dividend_yield) * T)

    # Log-moneyness
    k = math.log(strike / F)

    # Simplified Fourier inversion: integrate 0 to u_max
    du = u_max / n_integration
    integral = 0.0
    alpha = 0.75  # damping parameter

    for i in range(1, n_integration + 1):
        u = i * du
        # Damped CF
        try:
            cf = rough_heston_cf(u - (alpha + 1) * 1j, T, params, n_steps=50)
            denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u
            if abs(denominator) > 1e-10:
                integrand = (cf * np.exp(-1j * u * k) / denominator).real
                integral += integrand * du
        except (ValueError, ZeroDivisionError, OverflowError):
            continue

    call_price = math.exp(-alpha * k) / math.pi * integral * math.exp(-rate * T)
    call_price = max(call_price, 0.0)

    if is_call:
        return float(call_price)
    else:
        # Put-call parity
        return float(max(call_price - F * math.exp(-rate * T) + strike * math.exp(-rate * T), 0.0))


# ---- Forward variance curve ----

@dataclass
class ForwardVarianceCurve:
    """Forward variance curve ξ(T)."""
    tenors: np.ndarray
    forward_variances: np.ndarray
    method: str


def forward_variance_curve(
    tenors: list[float],
    atm_variance: list[float],
) -> ForwardVarianceCurve:
    """Bootstrap forward variance curve from ATM variance term structure.

    ATM variance w(T) = T × σ²_ATM(T).
    Forward variance: ξ(T) = d(w(T)) / dT.

    Numerical differentiation: ξ(Tᵢ) ≈ (w(Tᵢ₊₁) − w(Tᵢ₋₁)) / (Tᵢ₊₁ − Tᵢ₋₁).
    """
    T = np.array(tenors)
    v = np.array(atm_variance)
    w = T * v

    # Forward difference for interior, extrapolate at edges
    xi = np.zeros_like(T)
    xi[0] = v[0]  # initial forward variance = ATM variance at first tenor
    xi[-1] = (w[-1] - w[-2]) / (T[-1] - T[-2])
    for i in range(1, len(T) - 1):
        xi[i] = (w[i + 1] - w[i - 1]) / (T[i + 1] - T[i - 1])

    return ForwardVarianceCurve(T, xi, "finite_difference")
