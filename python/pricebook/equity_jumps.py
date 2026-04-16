"""Equity jump models: Kou, SVJ, regime switching, Merton hybrid.

* :func:`kou_equity_price` — Kou double-exponential jumps for equity.
* :class:`SVJEquityModel` — Stochastic vol + jumps (Bates for equity).
* :class:`RegimeSwitchingEquity` — bull/bear regime model.
* :func:`merton_equity_hybrid` — Merton jump-diffusion for equity.

References:
    Kou, *A Jump-Diffusion Model for Option Pricing*, Mgmt Sci, 2002.
    Bates, *Post-'87 Crash Fears in the S&P 500 Futures Option Market*, JE, 2000.
    Duffie-Pan-Singleton, *Transform Analysis and Option Pricing*, Econometrica, 2000.
    Hamilton, *A New Approach to the Economic Analysis of Nonstationary Time
    Series and the Business Cycle*, Econometrica, 1989.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


# ---- Kou equity ----

@dataclass
class KouResult:
    """Kou jump-diffusion pricing result."""
    price: float
    lambda_jump: float
    p: float                 # probability of up-jump
    eta1: float              # up-jump rate (1/mean jump size)
    eta2: float              # down-jump rate
    n_terms: int


def kou_equity_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    lambda_jump: float,
    p: float,
    eta1: float,
    eta2: float,
    is_call: bool = True,
    n_terms: int = 40,
) -> KouResult:
    """Kou double-exponential jump-diffusion for equity.

    Dynamics:
        dS/S = (r − q − λζ) dt + σ dW + (Y − 1) dN
    where N is Poisson(λ), Y = exp(ξ), ξ ~ double exp:
        p × Exp(η₁) (up-jump) + (1−p) × (−Exp(η₂)) (down-jump)

    ζ = E[Y − 1] = p × η₁/(η₁−1) + (1−p) × η₂/(η₂+1) − 1.

    Pricing via series expansion (Kou 2002):
        C = Σ e^{-λ'T} (λ'T)ⁿ / n! × E[C_n]

    For equity the typical calibration has η₁ > η₂ — asymmetric crashes.

    Args:
        lambda_jump: jump intensity per year.
        p: probability jump is up (typically 0.3-0.5 for equity).
        eta1: up-jump rate (must be > 1).
        eta2: down-jump rate.
        n_terms: series truncation.
    """
    if eta1 <= 1:
        raise ValueError("eta1 must be > 1 for finite expected up-jump")

    # Mean jump size E[Y - 1]
    zeta = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

    # Adjusted intensity (martingale condition)
    lambda_prime = lambda_jump * (1 + zeta)

    # Series: sum of Poisson-weighted BS-like prices
    # For each n jumps, approximate jump contribution to drift+vol
    total = 0.0
    log_lambda_T = math.log(max(lambda_prime * T, 1e-300))
    log_factorial = 0.0

    for n in range(n_terms):
        if n > 0:
            log_factorial += math.log(n)

        log_weight = -lambda_prime * T + n * log_lambda_T - log_factorial
        if log_weight < -50:
            continue
        weight = math.exp(log_weight)

        # Conditional on n jumps:
        # Log-return ~ N((r-q-λζ)T + n × E[ξ], σ²T + n × Var[ξ])
        # where E[ξ] = p/η₁ - (1-p)/η₂, Var[ξ] = p/η₁² + (1-p)/η₂²
        mean_xi = p / eta1 - (1 - p) / eta2
        var_xi = p * 2 / eta1**2 + (1 - p) * 2 / eta2**2  # (times 2 for exponential var)

        mean_n = (rate - dividend_yield - lambda_jump * zeta) * T + n * mean_xi
        var_n = vol**2 * T + n * var_xi
        sigma_n = math.sqrt(var_n)

        # Adjusted rate for BS
        r_n = mean_n / T + 0.5 * sigma_n**2 / T
        sigma_bs = math.sqrt(var_n / T)

        F = spot * math.exp(r_n * T - dividend_yield * T)
        df = math.exp(-rate * T)
        opt = OptionType.CALL if is_call else OptionType.PUT
        try:
            bs_price = black76_price(F, strike, sigma_bs, T, df, opt)
            total += weight * bs_price
        except Exception:
            continue

    return KouResult(
        price=float(max(total, 0.0)),
        lambda_jump=lambda_jump,
        p=p, eta1=eta1, eta2=eta2,
        n_terms=n_terms,
    )


# ---- SVJ (Stochastic Vol + Jumps) ----

@dataclass
class SVJResult:
    """SVJ pricing result."""
    price: float
    spot_paths: np.ndarray
    variance_paths: np.ndarray
    n_jumps_total: int


class SVJEquityModel:
    """SVJ model: Heston + Merton jumps for equity.

    dS/S = (r − q − λ κ) dt + √v dW_s + (Y − 1) dN
    dv   = κ_v(θ_v − v) dt + ξ √v dW_v
    dW_s dW_v = ρ dt
    ln Y ~ N(μ_j, σ_j²)

    Args:
        v0, kappa_v, theta_v, xi_v, rho: Heston parameters.
        lambda_jump: jump intensity.
        mu_j: mean log-jump size.
        sigma_j: vol of log-jump.
    """

    def __init__(
        self,
        v0: float,
        kappa_v: float,
        theta_v: float,
        xi_v: float,
        rho: float,
        lambda_jump: float,
        mu_j: float,
        sigma_j: float,
    ):
        self.v0 = v0
        self.kappa_v = kappa_v
        self.theta_v = theta_v
        self.xi_v = xi_v
        self.rho = rho
        self.lambda_jump = lambda_jump
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def simulate_option(
        self,
        spot: float,
        strike: float,
        rate: float,
        dividend_yield: float,
        T: float,
        is_call: bool = True,
        n_paths: int = 10_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> SVJResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        kappa_j = math.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0

        S = np.full(n_paths, spot)
        v = np.full(n_paths, self.v0)

        paths = np.zeros((n_paths, n_steps + 1))
        var_paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        var_paths[:, 0] = self.v0

        n_jumps_total = 0

        for step in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + math.sqrt(1 - self.rho**2) * rng.standard_normal(n_paths)

            n_jumps = rng.poisson(self.lambda_jump * dt, n_paths)
            n_jumps_total += int(n_jumps.sum())

            log_jump = np.zeros(n_paths)
            for p_idx in np.where(n_jumps > 0)[0]:
                jumps = rng.normal(self.mu_j, self.sigma_j, n_jumps[p_idx])
                log_jump[p_idx] = jumps.sum()

            v_pos = np.maximum(v, 0.0)
            drift = (rate - dividend_yield - self.lambda_jump * kappa_j - 0.5 * v_pos) * dt
            S = S * np.exp(drift + np.sqrt(v_pos) * z1 * sqrt_dt + log_jump)
            v = np.maximum(v + self.kappa_v * (self.theta_v - v) * dt
                           + self.xi_v * np.sqrt(v_pos) * z2 * sqrt_dt, 0.0)

            paths[:, step + 1] = S
            var_paths[:, step + 1] = v

        if is_call:
            payoff = np.maximum(paths[:, -1] - strike, 0.0)
        else:
            payoff = np.maximum(strike - paths[:, -1], 0.0)

        df = math.exp(-rate * T)
        price = df * float(payoff.mean())

        return SVJResult(price, paths, var_paths, n_jumps_total)


# ---- Regime switching ----

@dataclass
class RegimeResult:
    """Regime-switching equity result."""
    spot_paths: np.ndarray
    regime_paths: np.ndarray
    mean_regime_duration: dict[int, float]


class RegimeSwitchingEquity:
    """Bull/bear/crisis regime-switching equity model.

    Each regime has its own drift and vol. Markov transitions
    governed by generator Q.

    Args:
        regime_drifts: drift per regime (e.g., bull: 0.10, bear: -0.05, crisis: -0.20).
        regime_vols: vol per regime (e.g., bull: 0.15, bear: 0.25, crisis: 0.50).
        transition_rates: K×K generator matrix.
        initial_regime: starting regime.
    """

    def __init__(
        self,
        regime_drifts: list[float],
        regime_vols: list[float],
        transition_rates: np.ndarray,
        initial_regime: int = 0,
    ):
        self.drifts = np.array(regime_drifts)
        self.vols = np.array(regime_vols)
        self.Q = np.array(transition_rates)
        self.n_regimes = len(regime_drifts)
        self.initial_regime = initial_regime

    def simulate(
        self,
        spot: float,
        T: float,
        n_paths: int = 5000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> RegimeResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        S = np.full(n_paths, spot)
        regime = np.full(n_paths, self.initial_regime, dtype=int)

        paths = np.zeros((n_paths, n_steps + 1))
        reg_paths = np.zeros((n_paths, n_steps + 1), dtype=int)
        paths[:, 0] = spot
        reg_paths[:, 0] = self.initial_regime

        for step in range(n_steps):
            u = rng.random(n_paths)
            for p in range(n_paths):
                r = regime[p]
                leave_rate = -self.Q[r, r]
                if u[p] < leave_rate * dt:
                    other = np.array([self.Q[r, k] for k in range(self.n_regimes) if k != r])
                    if other.sum() > 0:
                        probs = other / other.sum()
                        u2 = rng.random()
                        cum = 0.0
                        other_regs = [kk for kk in range(self.n_regimes) if kk != r]
                        for k_idx, k in enumerate(other_regs):
                            cum += probs[k_idx]
                            if u2 < cum:
                                regime[p] = k
                                break

            drifts_p = self.drifts[regime]
            vols_p = self.vols[regime]
            z = rng.standard_normal(n_paths)
            S = S * np.exp((drifts_p - 0.5 * vols_p**2) * dt + vols_p * z * sqrt_dt)

            paths[:, step + 1] = S
            reg_paths[:, step + 1] = regime

        mean_dur = {}
        for r in range(self.n_regimes):
            counts = (reg_paths == r).sum(axis=1)
            mean_dur[r] = float(counts.mean()) * dt

        return RegimeResult(paths, reg_paths, mean_dur)


# ---- Merton hybrid ----

@dataclass
class MertonHybridResult:
    """Merton jump-diffusion equity result."""
    price: float
    jump_contribution: float
    diffusion_contribution: float


def merton_equity_hybrid(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    lambda_jump: float,
    jump_mean: float,
    jump_vol: float,
    is_call: bool = True,
    n_terms: int = 30,
) -> MertonHybridResult:
    """Merton jump-diffusion for equity (crash risk premium).

    Price via infinite Poisson-weighted sum of BS prices with modified
    parameters per jump count.
    """
    kappa = math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    lambda_prime = lambda_jump * (1 + kappa)
    lambda_T = lambda_prime * T

    total = 0.0
    diffusion_only = 0.0
    log_lambda_T = math.log(max(lambda_T, 1e-300))
    log_fact = 0.0

    for n in range(n_terms):
        if n > 0:
            log_fact += math.log(n)

        log_w = -lambda_T + n * log_lambda_T - log_fact
        if log_w < -50:
            continue
        w = math.exp(log_w)

        sigma_n = math.sqrt(vol**2 + n * jump_vol**2 / T)
        r_n = rate - lambda_jump * kappa + n * (jump_mean + 0.5 * jump_vol**2) / T

        F = spot * math.exp((r_n - dividend_yield) * T)
        df = math.exp(-r_n * T)
        opt = OptionType.CALL if is_call else OptionType.PUT
        try:
            bs = black76_price(F, strike, sigma_n, T, df, opt)
            total += w * bs
            if n == 0:
                diffusion_only = w * bs
        except Exception:
            continue

    total = max(total, 0.0)
    jump_part = total - diffusion_only

    return MertonHybridResult(
        price=float(total),
        jump_contribution=float(jump_part),
        diffusion_contribution=float(diffusion_only),
    )
