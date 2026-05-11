"""Advanced dividend modelling for equity.

* :class:`BuhlerStochasticDividend` — stochastic dividend yield (Bühler).
* :func:`dividend_curve_bootstrap` — build dividend curve from futures.
* :func:`implied_dividend_yield` — from put-call parity.
* :func:`dividend_basis_trade` — cash-vs-futures basis analytics.
* :func:`dividend_hedge_ratio` — hedge spot via dividend futures.

References:
    Bühler, *Stochastic Proportional Dividends*, WP, 2010.
    Allen, Granzia & Otsuki, *Equity Structured Products*, RBC, 2011.
    Kragt, *Managing Dividend Risk*, Risk, 2015.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Stochastic dividend ----

@dataclass
class BuhlerResult:
    """Bühler stochastic dividend simulation result."""
    spot_paths: np.ndarray
    dividend_yield_paths: np.ndarray
    mean_terminal_spot: float
    mean_terminal_yield: float


class BuhlerStochasticDividend:
    """Bühler stochastic dividend yield model.

    dS/S = (r − q(t)) dt + σ_S dW_S + (jumps at ex-dates)
    dq   = κ_q (θ_q − q) dt + ξ_q dW_q
    dW_S dW_q = ρ dt

    Args:
        r: risk-free rate.
        q0: initial dividend yield.
        kappa_q, theta_q, xi_q: dividend yield OU parameters.
        sigma_s: spot vol.
        rho: correlation between spot and dividend yield.
    """

    def __init__(
        self,
        r: float,
        q0: float,
        kappa_q: float,
        theta_q: float,
        xi_q: float,
        sigma_s: float,
        rho: float = 0.0,
    ):
        self.r = r
        self.q0 = q0
        self.kappa_q = kappa_q
        self.theta_q = theta_q
        self.xi_q = xi_q
        self.sigma_s = sigma_s
        self.rho = rho

    def simulate(
        self,
        spot: float,
        T: float,
        n_paths: int = 5000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> BuhlerResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        S = np.zeros((n_paths, n_steps + 1))
        q = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = spot
        q[:, 0] = self.q0

        for step in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + math.sqrt(1 - self.rho**2) * rng.standard_normal(n_paths)

            # Dividend yield OU (can go negative; clamp)
            q_new = (q[:, step]
                     + self.kappa_q * (self.theta_q - q[:, step]) * dt
                     + self.xi_q * z2 * sqrt_dt)
            q[:, step + 1] = np.maximum(q_new, 0.0)

            # Spot
            drift = (self.r - q[:, step] - 0.5 * self.sigma_s**2) * dt
            S[:, step + 1] = S[:, step] * np.exp(drift + self.sigma_s * z1 * sqrt_dt)

        return BuhlerResult(
            spot_paths=S,
            dividend_yield_paths=q,
            mean_terminal_spot=float(S[:, -1].mean()),
            mean_terminal_yield=float(q[:, -1].mean()),
        )

    def implied_forward(self, spot: float, T: float,
                         n_paths: int = 10_000, seed: int | None = 42) -> float:
        """Model-implied forward from MC."""
        result = self.simulate(spot, T, n_paths, seed=seed)
        return float(result.spot_paths[:, -1].mean())

    def simulate_via_engine(
        self,
        spot: float,
        T: float,
        n_paths: int = 5000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> BuhlerResult:
        """Simulate via unified MC engine (GBM + OU)."""
        from pricebook.mc_migrate import gbm_paths, ou_paths

        q_paths = ou_paths(self.q0, self.kappa_q, self.theta_q, self.xi_q,
                           T, n_steps, n_paths, seed or 42)
        s_paths = gbm_paths(spot, self.r, self.sigma_s, T, n_steps, n_paths,
                            (seed or 42) + 1)

        return BuhlerResult(
            spot_paths=s_paths,
            dividend_yield_paths=np.maximum(q_paths, 0.0),
            mean_terminal_spot=float(s_paths[:, -1].mean()),
            mean_terminal_yield=float(np.maximum(q_paths[:, -1], 0.0).mean()),
        )


# ---- Dividend curve ----

@dataclass
class DividendCurve:
    """Bootstrapped dividend curve from futures."""
    tenors: np.ndarray
    cumulative_dividends: np.ndarray    # ∫₀^T D(s) ds
    implied_yields: np.ndarray           # average yield to T
    method: str


def dividend_curve_bootstrap(
    spot: float,
    rate: float,
    tenors: list[float],
    div_futures_prices: list[float],
) -> DividendCurve:
    """Bootstrap dividend curve from index / single-stock dividend futures.

    Dividend future F^D(T) pays the cumulative dividends over [0, T]:
        F^D(T) ≈ S₀ × (1 − e^{-q̄T}) × e^{(r-q̄)T}

    For simplicity, we bootstrap average yield per period from futures.

    Args:
        spot: current spot.
        rate: risk-free rate.
        tenors: futures maturities.
        div_futures_prices: prices of dividend futures (cumulative div).
    """
    T = np.array(tenors)
    D = np.array(div_futures_prices)

    # Average yield: q̄ such that S×q̄×T ≈ D(T) (simplification)
    yields = np.where(T > 0, D / (spot * T), 0.0)

    return DividendCurve(
        tenors=T,
        cumulative_dividends=D,
        implied_yields=yields,
        method="linear_bootstrap",
    )


# ---- Implied dividend yield ----

@dataclass
class ImpliedDividendResult:
    """Implied dividend yield from put-call parity."""
    implied_yield: float
    forward: float
    call_price: float
    put_price: float
    residual: float


def implied_dividend_yield(
    spot: float,
    strike: float,
    call_price: float,
    put_price: float,
    rate: float,
    T: float,
) -> ImpliedDividendResult:
    """Extract implied dividend yield from put-call parity.

    Parity: C − P = S e^{-qT} − K e^{-rT}
    Solve for q:
        e^{-qT} = (C − P + K e^{-rT}) / S
        q = -log((C − P + K e^{-rT}) / S) / T

    Args:
        spot: current spot.
        strike: common strike of call and put.
        call_price, put_price: market option prices.
        rate: risk-free rate.
        T: time to expiry.
    """
    if T <= 0 or spot <= 0:
        return ImpliedDividendResult(0.0, spot, call_price, put_price, 0.0)

    implied_forward = (call_price - put_price + strike * math.exp(-rate * T)) / math.exp(-rate * T)

    arg = (call_price - put_price + strike * math.exp(-rate * T)) / spot
    if arg <= 0:
        return ImpliedDividendResult(0.0, spot, call_price, put_price, 1.0)

    q = -math.log(arg) / T

    return ImpliedDividendResult(
        implied_yield=float(q),
        forward=float(implied_forward),
        call_price=call_price,
        put_price=put_price,
        residual=0.0,
    )


# ---- Dividend basis ----

@dataclass
class DividendBasisResult:
    """Dividend basis (cash vs futures) analytics."""
    cash_dividend_pv: float
    future_price: float
    basis: float                    # future - PV(cash)
    basis_per_year: float
    method: str


def dividend_basis_trade(
    cash_dividends: list[tuple[float, float]],  # (time, amount)
    rate: float,
    T: float,
    future_price: float,
) -> DividendBasisResult:
    """Dividend basis: cash dividend PV vs dividend future.

    Cash leg: PV of individual discrete dividends.
    Future leg: price of dividend future (cumulative to T).
    Basis: future − PV(cash) (should be small if consistent).

    Args:
        cash_dividends: list of (pay_time, amount) pairs.
        rate: discount rate.
        T: futures maturity.
        future_price: market futures price.
    """
    cash_pv = sum(amt * math.exp(-rate * t) for t, amt in cash_dividends if t <= T)
    basis = future_price - cash_pv
    basis_per_year = basis / T if T > 0 else 0.0

    return DividendBasisResult(
        cash_dividend_pv=float(cash_pv),
        future_price=float(future_price),
        basis=float(basis),
        basis_per_year=float(basis_per_year),
        method="pv_basis",
    )


# ---- Hedge ratio ----

@dataclass
class DividendHedgeResult:
    """Dividend hedge ratio result."""
    spot_sensitivity: float
    future_hedge_ratio: float   # # of dividend futures per unit of spot risk
    residual_risk: float


def dividend_hedge_ratio(
    spot: float,
    dividend_yield: float,
    T: float,
    beta_div: float = 1.0,
) -> DividendHedgeResult:
    """Hedge ratio for dividend risk using dividend futures.

    Spot sensitivity to dividends: ∂S/∂q ≈ −S × T (for small shifts).
    Dividend futures sensitivity: ∂F^D/∂q ≈ S × T.

    Hedge ratio: to hedge spot's dividend exposure, short β × ∂S/∂q of futures.

    Args:
        spot: current spot.
        dividend_yield: current yield.
        T: horizon (futures maturity).
        beta_div: dividend beta (stock-specific vs index).
    """
    spot_sens = -spot * T
    future_sens = spot * T  # approximate

    if abs(future_sens) > 1e-10:
        hedge_ratio = -spot_sens * beta_div / future_sens
    else:
        hedge_ratio = 0.0

    residual = spot_sens + hedge_ratio * future_sens

    return DividendHedgeResult(
        spot_sensitivity=float(spot_sens),
        future_hedge_ratio=float(hedge_ratio),
        residual_risk=float(residual),
    )
