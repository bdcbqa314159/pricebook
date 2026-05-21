"""Convertible bond with joint equity-credit dynamics (C8).

Models the feedback loop: stock drops → credit spread widens → convertible
value drops further. Uses correlated (equity, hazard) Monte Carlo.

    from pricebook.credit.convertible_equity_credit import (
        convertible_equity_credit_price, EquityCreditConvertibleResult,
        equity_credit_greeks,
    )

The joint process:
    dS/S = (r - q) dt + σ_S dW_S                    [equity GBM]
    dλ   = κ(θ - λ) dt + ξ√λ dW_λ                   [hazard CIR]
    corr(dW_S, dW_λ) = ρ                             [negative: stock down → hazard up]

Default occurs when cumulative hazard ∫λ ds exceeds an exponential threshold.
On default, holder receives recovery × face.

References:
    Ayache, Forsyth & Vetzal (2003). Valuation of Convertible Bonds with Credit Risk.
    Davis & Lischka (2002). Convertible Bonds with Market Risk and Credit Risk.
    Tsiveriotis & Fernandes (1998). Valuing Convertible Bonds with Credit Risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class EquityCreditConvertibleResult:
    """Result of joint equity-credit convertible pricing."""
    price: float
    bond_floor: float            # straight risky bond value
    conversion_value: float      # conversion_ratio × spot
    conversion_premium: float
    # Greeks
    delta: float                 # ∂V/∂S
    gamma: float                 # ∂²V/∂S²
    vega: float                  # ∂V/∂σ (per 1% vol)
    cs01: float                  # ∂V/∂λ₀ (per 1bp hazard)
    rho_sensitivity: float       # ∂V/∂ρ (per 0.1 correlation)
    # Diagnostics
    default_prob: float          # simulated default probability
    avg_hazard: float            # average hazard across paths
    n_paths: int

    def to_dict(self) -> dict:
        return vars(self)


def convertible_equity_credit_price(
    spot: float,
    notional: float,
    coupon_rate: float,
    conversion_ratio: float,
    maturity_years: float,
    risk_free_rate: float,
    equity_vol: float,
    hazard_rate_0: float,
    hazard_mean_reversion: float = 1.0,
    hazard_long_run: float | None = None,
    hazard_vol: float = 0.10,
    equity_credit_corr: float = -0.30,
    recovery: float = 0.40,
    dividend_yield: float = 0.0,
    n_paths: int = 20_000,
    n_steps: int | None = None,
    seed: int = 42,
    _compute_greeks: bool = True,
) -> EquityCreditConvertibleResult:
    """Price a convertible bond with joint equity-credit MC.

    Args:
        spot: current stock price.
        notional: face value.
        coupon_rate: annual coupon.
        conversion_ratio: shares received on conversion per unit notional.
        maturity_years: maturity in years.
        risk_free_rate: risk-free rate.
        equity_vol: equity volatility.
        hazard_rate_0: initial hazard rate (e.g. 0.02 = 200bp CDS).
        hazard_mean_reversion: κ — speed of mean-reversion for hazard.
        hazard_long_run: θ — long-run hazard level (default = hazard_rate_0).
        hazard_vol: ξ — vol of hazard (CIR diffusion).
        equity_credit_corr: ρ — correlation between equity and hazard shocks.
            Negative = stock down → hazard up (wrong-way risk, realistic).
        recovery: recovery rate on default.
        dividend_yield: continuous dividend yield.
        n_paths: MC paths.
        n_steps: time steps (default = monthly).
        seed: random seed.
    """
    if n_steps is None:
        n_steps = max(int(maturity_years * 12), 12)

    if hazard_long_run is None:
        hazard_long_run = hazard_rate_0

    rng = np.random.default_rng(seed)
    dt = maturity_years / n_steps
    sqrt_dt = math.sqrt(dt)

    rho = equity_credit_corr
    kappa = hazard_mean_reversion
    theta = hazard_long_run
    xi = hazard_vol

    # Coupon schedule
    coupons_per_year = 2
    coupon_amount = notional * coupon_rate / coupons_per_year
    coupon_step_interval = max(n_steps // max(int(maturity_years * coupons_per_year), 1), 1)

    # Simulate joint (S, λ) paths
    S = np.full((n_paths, n_steps + 1), float(spot))
    lam = np.full((n_paths, n_steps + 1), float(hazard_rate_0))
    cum_hazard = np.zeros(n_paths)
    default_time_step = np.full(n_paths, n_steps + 1, dtype=int)  # step of default

    # Exponential threshold for default timing
    exp_threshold = rng.exponential(1.0, n_paths)

    drift_s = (risk_free_rate - dividend_yield - 0.5 * equity_vol ** 2) * dt

    for step in range(n_steps):
        # Correlated normals
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_s = z1
        z_lam = rho * z1 + math.sqrt(1 - rho ** 2) * z2

        # Equity: GBM
        S[:, step + 1] = S[:, step] * np.exp(drift_s + equity_vol * sqrt_dt * z_s)

        # Hazard: CIR (Euler, floored at 0)
        lam_curr = np.maximum(lam[:, step], 0.0)
        dlam = kappa * (theta - lam_curr) * dt + xi * np.sqrt(lam_curr) * sqrt_dt * z_lam
        lam[:, step + 1] = np.maximum(lam_curr + dlam, 0.0)

        # Cumulative hazard for default check
        cum_hazard += lam[:, step + 1] * dt

        # Check for default: ∫λ ds > Exp(1) threshold
        newly_defaulted = (cum_hazard >= exp_threshold) & (default_time_step == n_steps + 1)
        default_time_step[newly_defaulted] = step + 1

    # Backward induction with LSM
    V = np.maximum(float(notional), conversion_ratio * S[:, -1]).astype(float)

    # Mark defaulted paths at terminal
    defaulted = default_time_step <= n_steps
    V[defaulted & (default_time_step == n_steps)] = recovery * notional

    for step in range(n_steps - 1, -1, -1):
        # Discount at risk-free rate (credit risk handled by default simulation)
        V *= math.exp(-risk_free_rate * dt)

        # Coupons
        if step > 0 and step % coupon_step_interval == 0:
            alive = default_time_step > step
            V[alive] += coupon_amount

        # Default at this step → recovery
        just_defaulted = default_time_step == step + 1
        V[just_defaulted] = recovery * notional * math.exp(-risk_free_rate * (n_steps - step) * dt)

        # Conversion decision (only for alive paths)
        alive = default_time_step > step + 1
        conv_val = conversion_ratio * S[:, step]

        if step > 0:
            s_alive = S[alive, step]
            if len(s_alive) > 10 and s_alive.std() > 1e-10:
                X = np.column_stack([np.ones(alive.sum()), s_alive, s_alive ** 2])
                try:
                    coeffs = np.linalg.lstsq(X, V[alive], rcond=None)[0]
                    continuation = X @ coeffs
                except np.linalg.LinAlgError:
                    continuation = V[alive]
                convert_mask = conv_val[alive] > continuation
                V_alive = V[alive].copy()
                V_alive[convert_mask] = conv_val[alive][convert_mask]
                V[alive] = V_alive
            else:
                V[alive] = np.maximum(V[alive], conv_val[alive])

    price = float(V.mean())

    # Bond floor: risky straight bond
    bond_floor = _risky_bond_floor(
        notional, coupon_rate, maturity_years, risk_free_rate,
        hazard_rate_0, recovery, coupons_per_year,
    )

    conversion_val = conversion_ratio * spot
    premium = (price - conversion_val) / max(conversion_val, 1e-6)
    default_prob = float(defaulted.sum()) / n_paths
    avg_hazard = float(lam[:, -1].mean())

    # Greeks via bump-and-reprice (common-random-numbers)
    if _compute_greeks:
        delta, gamma = _compute_delta_gamma(
            spot, notional, coupon_rate, conversion_ratio, maturity_years,
            risk_free_rate, equity_vol, hazard_rate_0, kappa, theta, xi, rho,
            recovery, dividend_yield, n_paths, n_steps, seed, price,
        )
        vega = _compute_vega(
            spot, notional, coupon_rate, conversion_ratio, maturity_years,
            risk_free_rate, equity_vol, hazard_rate_0, kappa, theta, xi, rho,
            recovery, dividend_yield, n_paths, n_steps, seed, price,
        )
        cs01 = _compute_cs01(
            spot, notional, coupon_rate, conversion_ratio, maturity_years,
            risk_free_rate, equity_vol, hazard_rate_0, kappa, theta, xi, rho,
            recovery, dividend_yield, n_paths, n_steps, seed, price,
        )
        rho_sens = _compute_rho_sens(
            spot, notional, coupon_rate, conversion_ratio, maturity_years,
            risk_free_rate, equity_vol, hazard_rate_0, kappa, theta, xi, rho,
            recovery, dividend_yield, n_paths, n_steps, seed, price,
        )
    else:
        delta = gamma = vega = cs01 = rho_sens = 0.0

    return EquityCreditConvertibleResult(
        price=price,
        bond_floor=bond_floor,
        conversion_value=conversion_val,
        conversion_premium=premium,
        delta=delta,
        gamma=gamma,
        vega=vega,
        cs01=cs01,
        rho_sensitivity=rho_sens,
        default_prob=default_prob,
        avg_hazard=avg_hazard,
        n_paths=n_paths,
    )


def _risky_bond_floor(notional, coupon_rate, T, r, h, recovery, freq):
    """Straight risky bond value (no conversion)."""
    pv = 0.0
    cpn = notional * coupon_rate / freq
    n = int(T * freq)
    prev_surv = 1.0
    for i in range(1, n + 1):
        t = i / freq
        surv = math.exp(-h * t)
        df = math.exp(-r * t)
        pv += cpn * df * surv
        pv += recovery * notional * df * (prev_surv - surv)
        prev_surv = surv
    pv += notional * math.exp(-r * T) * math.exp(-h * T)
    return pv


def _reprice(spot, notional, coupon_rate, cr, T, r, vol, h0, k, th, xi, rho,
             rec, q, np_, ns, seed):
    """Quick reprice for Greeks (skip Greek computation to avoid recursion)."""
    result = convertible_equity_credit_price(
        spot, notional, coupon_rate, cr, T, r, vol, h0, k, th, xi, rho,
        rec, q, np_, ns, seed, _compute_greeks=False,
    )
    return result.price


def _compute_delta_gamma(spot, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho,
                          rec, q, np_, ns, seed, base_price):
    bump = spot * 0.01
    p_up = _reprice(spot + bump, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho, rec, q, np_, ns, seed)
    p_dn = _reprice(spot - bump, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho, rec, q, np_, ns, seed)
    delta = (p_up - p_dn) / (2 * bump)
    gamma = (p_up - 2 * base_price + p_dn) / (bump ** 2)
    return delta, gamma


def _compute_vega(spot, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho,
                   rec, q, np_, ns, seed, base_price):
    bump = 0.01
    p_up = _reprice(spot, notional, cpn, cr, T, r, vol + bump, h0, k, th, xi, rho, rec, q, np_, ns, seed)
    return p_up - base_price  # per 1% vol


def _compute_cs01(spot, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho,
                   rec, q, np_, ns, seed, base_price):
    bump = 0.0001  # 1bp
    p_up = _reprice(spot, notional, cpn, cr, T, r, vol, h0 + bump, k, th + bump, xi, rho, rec, q, np_, ns, seed)
    return p_up - base_price  # per 1bp hazard


def _compute_rho_sens(spot, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho,
                       rec, q, np_, ns, seed, base_price):
    bump = 0.10
    rho_bumped = max(-0.99, min(0.99, rho + bump))
    p_up = _reprice(spot, notional, cpn, cr, T, r, vol, h0, k, th, xi, rho_bumped, rec, q, np_, ns, seed)
    return p_up - base_price  # per 0.1 correlation
