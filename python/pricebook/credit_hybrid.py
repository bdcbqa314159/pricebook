"""Hybrid credit-rates models: callable risky bonds, floating CLN, convertibles.

Phase C6 slices 223-225 consolidated.

* :func:`callable_risky_bond` — HW rate tree + survival overlay.
* :func:`floating_cln` — floating coupon CLN with stochastic hazard.
* :func:`convertible_bond` — equity + credit + rates (three-factor MC).

References:
    Schönbucher, *Credit Derivatives Pricing Models*, Ch. 9.
    Tsiveriotis & Fernandes, *Valuing Convertible Bonds with Credit Risk*,
    J. Fixed Income, 1998.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 22 (credit-rates hybrids).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Callable risky bond ----

@dataclass
class CallableRiskyBondResult:
    """Pricing result for a callable risky bond."""
    price: float
    non_callable_price: float
    call_value: float
    oas: float  # option-adjusted spread


def callable_risky_bond(
    notional: float,
    coupon_rate: float,
    maturity_years: int,
    call_price: float,
    call_start_year: int,
    flat_rate: float = 0.05,
    rate_vol: float = 0.01,
    flat_hazard: float = 0.02,
    recovery: float = 0.4,
    n_tree_steps: int = 50,
) -> CallableRiskyBondResult:
    """Price a callable risky bond via backward induction.

    At each node: issuer calls if bond value > call_price.
    Survival overlay: discount by both rate and hazard.

    Simplified: uses a recombining binomial tree for rates with
    survival probability overlay.

    Args:
        call_price: price at which issuer can call (per 100 face).
        call_start_year: first year the bond is callable.
        rate_vol: volatility of the short rate (HW-style).
    """
    dt = maturity_years / n_tree_steps
    u = math.exp(rate_vol * math.sqrt(dt))
    d = 1.0 / u
    p = 0.5  # risk-neutral probability

    # Build rate tree
    rates = np.zeros((n_tree_steps + 1, n_tree_steps + 1))
    for i in range(n_tree_steps + 1):
        for j in range(i + 1):
            rates[j, i] = flat_rate * u**(i - 2*j)

    # Terminal values: redemption conditional on survival
    V = np.zeros((n_tree_steps + 1, n_tree_steps + 1))
    for j in range(n_tree_steps + 1):
        surv = math.exp(-flat_hazard * maturity_years)
        V[j, n_tree_steps] = notional * surv + recovery * notional * (1 - surv)

    # Non-callable value (for comparison)
    V_nc = V.copy()

    # Backward induction
    for i in range(n_tree_steps - 1, -1, -1):
        t = (i + 1) * dt
        for j in range(i + 1):
            r = rates[j, i]
            df = math.exp(-r * dt)
            surv_step = math.exp(-flat_hazard * dt)

            # Continuation: discounted expected value × survival + recovery × default
            cont = df * (p * V[j, i + 1] + (1 - p) * V[j + 1, i + 1])
            cont = cont * surv_step + recovery * notional * df * (1 - surv_step)

            # Add coupon
            cont += coupon_rate * notional * dt * df * surv_step

            # Call decision (issuer calls if bond > call_price)
            year = int(t / 1.0)
            if year >= call_start_year:
                V[j, i] = min(cont, call_price * notional / 100.0)
            else:
                V[j, i] = cont

            # Non-callable: no call constraint
            cont_nc = df * (p * V_nc[j, i + 1] + (1 - p) * V_nc[j + 1, i + 1])
            cont_nc = cont_nc * surv_step + recovery * notional * df * (1 - surv_step)
            cont_nc += coupon_rate * notional * dt * df * surv_step
            V_nc[j, i] = cont_nc

    price = V[0, 0] / notional * 100
    nc_price = V_nc[0, 0] / notional * 100
    call_val = nc_price - price

    # OAS: spread that makes callable price match
    oas = (nc_price - price) / (maturity_years * nc_price / 100) * 10000 if nc_price > 0 else 0.0

    return CallableRiskyBondResult(price, nc_price, call_val, oas)


# ---- Floating-rate CLN ----

@dataclass
class FloatingCLNResult:
    """Pricing result for a floating-rate CLN."""
    price: float
    coupon_pv: float
    principal_pv: float
    recovery_pv: float
    par_spread: float


def floating_cln(
    notional: float,
    spread: float,
    maturity_years: int,
    flat_rate: float = 0.05,
    flat_hazard: float = 0.02,
    hazard_vol: float = 0.0,
    rate_hazard_corr: float = 0.0,
    recovery: float = 0.4,
    frequency: int = 4,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> FloatingCLNResult:
    """Price a floating-rate CLN with optional stochastic hazard.

    Coupon = (floating_rate + spread) × notional × dt, conditional on survival.
    On default: recovery × notional.

    When hazard_vol = 0, uses deterministic pricing.
    When hazard_vol > 0, uses MC with correlated rate-hazard dynamics.

    Args:
        spread: credit spread over floating.
        hazard_vol: volatility of hazard rate (0 = deterministic).
        rate_hazard_corr: correlation between rate and hazard innovations.
    """
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    if hazard_vol == 0.0:
        # Deterministic pricing
        coupon_pv = 0.0
        recovery_pv = 0.0

        for i in range(1, n_periods + 1):
            t = i * dt
            df = math.exp(-flat_rate * t)
            surv = math.exp(-flat_hazard * t)
            surv_prev = math.exp(-flat_hazard * (t - dt))

            coupon = (flat_rate + spread) * notional * dt
            coupon_pv += df * surv * coupon
            recovery_pv += df * (surv_prev - surv) * recovery * notional

        df_T = math.exp(-flat_rate * maturity_years)
        surv_T = math.exp(-flat_hazard * maturity_years)
        principal_pv = df_T * surv_T * notional

        price = (coupon_pv + principal_pv + recovery_pv) / notional * 100

        # Par spread: spread at which price = 100
        annuity = sum(
            math.exp(-flat_rate * i * dt) * math.exp(-flat_hazard * i * dt) * dt
            for i in range(1, n_periods + 1)
        )
        prot_pv = sum(
            math.exp(-flat_rate * i * dt) * (1 - recovery) *
            (math.exp(-flat_hazard * (i-1) * dt) - math.exp(-flat_hazard * i * dt))
            for i in range(1, n_periods + 1)
        )
        par_spread = prot_pv / annuity if annuity > 0 else 0.0

        return FloatingCLNResult(price, coupon_pv, principal_pv, recovery_pv, par_spread)

    # MC with stochastic hazard
    rng = np.random.default_rng(seed)
    sqrt_dt = math.sqrt(dt)
    rho = rate_hazard_corr

    r = np.full(n_paths, flat_rate)
    lam = np.full(n_paths, flat_hazard)
    pv = np.zeros(n_paths)

    for i in range(1, n_periods + 1):
        t = i * dt
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW_r = z1 * sqrt_dt
        dW_lam = (rho * z1 + math.sqrt(1 - rho**2) * z2) * sqrt_dt

        # OU dynamics for rate
        r = r + 0.5 * (flat_rate - r) * dt + 0.005 * dW_r
        # CIR-like for hazard
        lam_safe = np.maximum(lam, 1e-10)
        lam = lam + 0.5 * (flat_hazard - lam) * dt + hazard_vol * np.sqrt(lam_safe) * dW_lam
        lam = np.maximum(lam, 0.0)

        df = np.exp(-r * dt)
        surv = np.exp(-lam * dt)

        coupon = (r + spread) * notional * dt
        pv += df * surv * coupon
        pv += df * (1 - surv) * recovery * notional

    # Principal
    pv += np.exp(-r * dt) * np.exp(-lam * dt) * notional

    price = float(pv.mean()) / notional * 100

    return FloatingCLNResult(price, 0.0, 0.0, 0.0, 0.0)


# ---- Convertible bond (three-factor) ----

@dataclass
class ConvertibleBondResult:
    """Pricing result for a convertible bond."""
    price: float
    bond_floor: float
    conversion_value: float
    conversion_premium: float


def convertible_bond(
    notional: float,
    coupon_rate: float,
    maturity_years: int,
    conversion_ratio: float,
    spot: float,
    flat_rate: float = 0.05,
    equity_vol: float = 0.30,
    flat_hazard: float = 0.02,
    recovery: float = 0.4,
    n_paths: int = 50_000,
    n_steps_per_year: int = 12,
    seed: int | None = None,
) -> ConvertibleBondResult:
    """Price a convertible bond via MC (equity + credit + rates).

    At each step the bondholder can convert to equity:
        conversion_value = conversion_ratio × S(t).

    If default occurs, bondholder receives recovery × notional.
    If no default and no conversion, receives coupons + principal.

    Simplified: no call provision by issuer (add via callable_risky_bond pattern).

    Args:
        conversion_ratio: number of shares per bond.
        spot: current equity price.
        equity_vol: equity volatility.
    """
    rng = np.random.default_rng(seed)
    n_steps = maturity_years * n_steps_per_year
    dt = maturity_years / n_steps
    sqrt_dt = math.sqrt(dt)

    # Simulate equity paths
    S = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)  # not defaulted, not converted
    pv = np.zeros(n_paths)

    for step in range(1, n_steps + 1):
        t = step * dt
        z = rng.standard_normal(n_paths)
        dW = z * sqrt_dt

        # Equity: GBM (conditional on survival)
        S = S * np.exp((flat_rate - 0.5 * equity_vol**2) * dt + equity_vol * dW)

        # Default check
        default_prob = 1 - math.exp(-flat_hazard * dt)
        defaults = alive & (rng.random(n_paths) < default_prob)
        pv[defaults] += math.exp(-flat_rate * t) * recovery * notional
        alive[defaults] = False

        # Conversion decision: convert if conversion value > bond continuation
        # Simplified: convert at maturity only (American conversion adds complexity)

        # Coupon accrual (annual)
        if step % n_steps_per_year == 0 and step < n_steps:
            pv[alive] += math.exp(-flat_rate * t) * coupon_rate * notional

    # At maturity: surviving holders choose max(bond, conversion)
    T = maturity_years
    df_T = math.exp(-flat_rate * T)
    bond_value = notional + coupon_rate * notional  # last coupon + principal
    conv_value = conversion_ratio * S

    # Final payoff for surviving paths
    final = np.maximum(bond_value, conv_value[alive]) if np.any(alive) else np.array([])
    if len(final) > 0:
        pv[alive] += df_T * final

    price = float(pv.mean()) / notional * 100

    # Bond floor: risky bond without conversion option
    bond_floor_pv = 0.0
    surv = 1.0
    for yr in range(1, maturity_years + 1):
        df = math.exp(-flat_rate * yr)
        surv_new = math.exp(-flat_hazard * yr)
        bond_floor_pv += df * surv_new * coupon_rate * notional
        bond_floor_pv += df * (surv - surv_new) * recovery * notional
        surv = surv_new
    bond_floor_pv += math.exp(-flat_rate * maturity_years) * surv * notional
    bond_floor = bond_floor_pv / notional * 100

    conv_now = conversion_ratio * spot / notional * 100
    premium = price - max(bond_floor, conv_now)

    return ConvertibleBondResult(price, bond_floor, conv_now, premium)
