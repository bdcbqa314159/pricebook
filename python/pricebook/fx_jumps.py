"""FX jumps and regime switching: Merton, Bates, regime, intervention.

Extends :mod:`pricebook.jump_process` with FX-specific applications:

* :func:`merton_fx_price` — Merton jump-diffusion for FX options.
* :class:`BatesFXModel` — Heston + jumps FX model.
* :class:`RegimeSwitchingVol` — Markov vol regime (low/normal/crisis).
* :func:`fx_intervention_adjustment` — peg break / intervention risk premium.

References:
    Merton, *Option Pricing when Underlying Stock Returns are Discontinuous*,
    JFE, 1976.
    Bates, *Jumps and Stochastic Volatility: Exchange Rate Processes*, RFS, 1996.
    Hamilton, *A New Approach to the Economic Analysis of Nonstationary Time Series*,
    Econometrica, 1989.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.black76 import black76_price, OptionType


# ---- Merton FX ----

@dataclass
class MertonFXResult:
    """Merton jump-diffusion FX pricing result."""
    price: float
    n_terms: int
    lambda_jump: float
    jump_mean: float
    jump_vol: float


def merton_fx_price(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    lambda_jump: float,
    jump_mean: float,
    jump_vol: float,
    is_call: bool = True,
    n_terms: int = 30,
) -> MertonFXResult:
    """Merton jump-diffusion for FX options.

    Dynamics:
        dS/S = (r_d − r_f − λκ) dt + σ dW + (Y − 1) dN

    where N is Poisson(λ), ln Y ~ N(m, δ²), κ = E[Y − 1] = exp(m + δ²/2) − 1.

    Pricing via infinite sum of Black-Scholes prices:
        V = Σ_{n=0}^∞ e^{−λ'T} (λ'T)ⁿ / n! × BS(S, K, T, σ_n, r_n)
    where
        λ' = λ(1 + κ)
        σ_n² = σ² + n δ² / T
        r_n = r_d − λκ + n × (m + δ²/2) / T

    Args:
        lambda_jump: jump intensity.
        jump_mean: mean of log jump size m.
        jump_vol: vol of log jump size δ.
    """
    kappa = math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    lambda_prime = lambda_jump * (1 + kappa)
    lambda_T = lambda_prime * T

    price = 0.0
    # Pre-compute Poisson probabilities via log for stability
    log_lambda_T = math.log(max(lambda_T, 1e-300))
    log_factorial = 0.0

    for n in range(n_terms):
        if n > 0:
            log_factorial += math.log(n)

        # Poisson weight: e^{-λ'T} (λ'T)^n / n!
        log_weight = -lambda_T + n * log_lambda_T - log_factorial
        if log_weight < -50:  # underflow
            continue
        weight = math.exp(log_weight)

        # Modified parameters
        sigma_n_sq = vol**2 + n * jump_vol**2 / T
        sigma_n = math.sqrt(sigma_n_sq)

        r_n = rate_dom - lambda_jump * kappa + n * (jump_mean + 0.5 * jump_vol**2) / T

        # BS price with adjusted vol and rate
        F = spot * math.exp((r_n - rate_for) * T)
        df = math.exp(-r_n * T)
        opt = OptionType.CALL if is_call else OptionType.PUT
        bs_price = black76_price(F, strike, sigma_n, T, df, opt)

        price += weight * bs_price

    return MertonFXResult(float(max(price, 0.0)), n_terms,
                          lambda_jump, jump_mean, jump_vol)


# ---- Bates FX ----

@dataclass
class BatesFXResult:
    """Bates (Heston + jumps) FX simulation result."""
    price: float
    paths: np.ndarray
    variance_paths: np.ndarray
    n_jumps_total: int


class BatesFXModel:
    """Bates model for FX: Heston + Merton jumps.

    dS/S = (r_d − r_f − λκ) dt + √v dW_s + (Y − 1) dN
    dv   = κ_v(θ_v − v) dt + ξ √v dW_v
    dW_s dW_v = ρ dt

    Args:
        v0, kappa_v, theta_v, xi, rho: Heston variance parameters.
        lambda_jump, jump_mean, jump_vol: jump parameters.
    """

    def __init__(
        self,
        v0: float,
        kappa_v: float,
        theta_v: float,
        xi: float,
        rho: float,
        lambda_jump: float,
        jump_mean: float,
        jump_vol: float,
    ):
        self.v0 = v0
        self.kappa_v = kappa_v
        self.theta_v = theta_v
        self.xi = xi
        self.rho = rho
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_vol = jump_vol

    def simulate_option(
        self,
        spot: float,
        strike: float,
        rate_dom: float,
        rate_for: float,
        T: float,
        is_call: bool = True,
        n_paths: int = 10_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> BatesFXResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        kappa_jump = math.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0

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

            # Poisson jump count this step
            n_jumps = rng.poisson(self.lambda_jump * dt, n_paths)
            n_jumps_total += int(n_jumps.sum())

            # Jump multiplier: Y - 1 where ln Y ~ N(m, δ²) per jump
            log_jump = np.zeros(n_paths)
            for p in np.where(n_jumps > 0)[0]:
                jumps = rng.normal(self.jump_mean, self.jump_vol, n_jumps[p])
                log_jump[p] = jumps.sum()

            v_pos = np.maximum(v, 0.0)
            drift = (rate_dom - rate_for - self.lambda_jump * kappa_jump - 0.5 * v_pos) * dt
            S = S * np.exp(drift + np.sqrt(v_pos) * z1 * sqrt_dt + log_jump)
            v = np.maximum(v + self.kappa_v * (self.theta_v - v) * dt
                           + self.xi * np.sqrt(v_pos) * z2 * sqrt_dt, 0.0)

            paths[:, step + 1] = S
            var_paths[:, step + 1] = v

        if is_call:
            payoff = np.maximum(paths[:, -1] - strike, 0.0)
        else:
            payoff = np.maximum(strike - paths[:, -1], 0.0)

        df = math.exp(-rate_dom * T)
        price = df * float(payoff.mean())

        return BatesFXResult(price, paths, var_paths, n_jumps_total)


# ---- Regime switching ----

@dataclass
class RegimeSwitchingResult:
    """Regime-switching simulation result."""
    price: float
    regime_paths: np.ndarray    # (n_paths, n_steps+1) regime indicators
    spot_paths: np.ndarray
    mean_regime_duration: dict[int, float]


class RegimeSwitchingVol:
    """Markov regime-switching FX vol model.

    K regimes (e.g. K=3: low, normal, crisis). In each regime, spot follows GBM
    with regime-specific vol. Regime transitions governed by K×K generator Q.

    Args:
        regime_vols: vol in each regime.
        transition_rates: K×K generator matrix (row sums to 0).
        initial_regime: starting regime.
    """

    def __init__(
        self,
        regime_vols: list[float],
        transition_rates: np.ndarray,
        initial_regime: int = 0,
    ):
        self.regime_vols = np.array(regime_vols)
        self.Q = np.array(transition_rates)
        self.n_regimes = len(regime_vols)
        self.initial_regime = initial_regime

    def simulate(
        self,
        spot: float,
        rate_dom: float,
        rate_for: float,
        T: float,
        n_paths: int = 5000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> RegimeSwitchingResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        S = np.full(n_paths, spot)
        regime = np.full(n_paths, self.initial_regime, dtype=int)

        paths = np.zeros((n_paths, n_steps + 1))
        reg_paths = np.zeros((n_paths, n_steps + 1), dtype=int)
        paths[:, 0] = spot
        reg_paths[:, 0] = self.initial_regime

        # Transition probabilities per dt (approximate): I + Q*dt
        for step in range(n_steps):
            # Sample regime transitions
            u = rng.random(n_paths)
            for p in range(n_paths):
                r = regime[p]
                # Leaving rate = -Q[r,r]
                leave_rate = -self.Q[r, r]
                if u[p] < leave_rate * dt:
                    # Transition: distribute among other regimes
                    other_rates = np.array([self.Q[r, k] for k in range(self.n_regimes) if k != r])
                    if other_rates.sum() > 0:
                        probs = other_rates / other_rates.sum()
                        u2 = rng.random()
                        cum = 0.0
                        for k_idx, k in enumerate([kk for kk in range(self.n_regimes) if kk != r]):
                            cum += probs[k_idx]
                            if u2 < cum:
                                regime[p] = k
                                break

            # Spot evolution under current regime
            vols_p = self.regime_vols[regime]
            z = rng.standard_normal(n_paths)
            drift = (rate_dom - rate_for - 0.5 * vols_p**2) * dt
            S = S * np.exp(drift + vols_p * z * sqrt_dt)

            paths[:, step + 1] = S
            reg_paths[:, step + 1] = regime

        # Mean duration per regime
        mean_dur = {}
        for r in range(self.n_regimes):
            counts = (reg_paths == r).sum(axis=1)
            mean_dur[r] = float(counts.mean()) * dt

        return RegimeSwitchingResult(
            price=0.0,  # caller can compute
            regime_paths=reg_paths,
            spot_paths=paths,
            mean_regime_duration=mean_dur,
        )


# ---- Intervention / peg break ----

@dataclass
class InterventionResult:
    """FX peg break / intervention adjustment result."""
    base_price: float
    intervention_adjusted_price: float
    expected_loss_from_break: float
    break_probability: float


def fx_intervention_adjustment(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    break_intensity: float,          # Poisson rate per year
    break_jump_size: float,          # typical devaluation on break (e.g. -0.20)
    is_call: bool = True,
) -> InterventionResult:
    """Intervention/peg-break risk adjustment for FX option.

    Model: pegged FX with Poisson break intensity λ_b; on break, spot jumps
    by `break_jump_size` (log return). This is essentially a Merton model
    with specific parameters.

    Useful for EMFX and pegged currencies (HKD, SAR, historically CHF/EUR).

    Args:
        break_intensity: λ_b (probability of break per year).
        break_jump_size: log return on break (negative for devaluation).
    """
    # Base price under diffusion only
    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)
    opt = OptionType.CALL if is_call else OptionType.PUT
    base_price = black76_price(F, strike, vol, T, df, opt)

    # Intervention-adjusted via Merton-like model
    adjusted = merton_fx_price(
        spot, strike, rate_dom, rate_for, vol, T,
        lambda_jump=break_intensity,
        jump_mean=break_jump_size,
        jump_vol=abs(break_jump_size) * 0.3,  # uncertain break size
        is_call=is_call, n_terms=20,
    )

    # P(at least one break)
    break_prob = 1.0 - math.exp(-break_intensity * T)

    # Expected loss from break (approximately)
    expected_loss = spot * (1 - math.exp(break_jump_size)) * break_prob

    return InterventionResult(
        base_price=float(base_price),
        intervention_adjusted_price=adjusted.price,
        expected_loss_from_break=float(expected_loss),
        break_probability=float(break_prob),
    )
