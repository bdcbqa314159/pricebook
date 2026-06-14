"""Bermudan barrier options — early exercise combined with barrier knock-out/knock-in.

Combines the Longstaff-Schwartz (LSM) algorithm for Bermudan/American early exercise
with continuous barrier monitoring on every simulation step. The library has European
barriers and American options separately; this module provides the combined product.

Key design decisions:
- Barrier is checked at *every* simulation step, not only at exercise dates.
- LSM regression uses normalised polynomial basis [1, S, S^2] restricted to ITM paths.
- American limit: pass exercise_dates=None to make every step an exercise date.
- Knock-in variants are priced as: knock_in = vanilla_Bermudan − knock_out.

References:
    Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas*, 2nd ed.
        McGraw-Hill. Ch. 4 (barrier options), Ch. 5 (American options).
    Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
        Springer. Ch. 8 (American and Bermudan by simulation, LSM).
    Longstaff, F. A. & Schwartz, E. S. (2001). Valuing American options by
        simulation: a simple least-squares approach. *Review of Financial
        Studies*, 14(1), 113–147.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BermudanBarrierResult:
    """Pricing result for a Bermudan barrier option."""

    price: float
    """LSM price of the Bermudan barrier option."""

    delta: float
    """Finite-difference delta (bump = 1 % of spot)."""

    gamma: float
    """Finite-difference gamma."""

    vega: float
    """Finite-difference vega (bump = 1 vol point)."""

    barrier_hit_prob: float
    """Fraction of paths that hit the barrier before expiry."""

    early_exercise_prob: float
    """Fraction of paths where early exercise was optimal (on surviving paths)."""

    european_barrier_price: float
    """European barrier price for comparison (same inputs, no early exercise)."""


@dataclass
class BermudanDoubleBarrierResult:
    """Pricing result for a Bermudan double-barrier option."""

    price: float
    """LSM price of the Bermudan double-barrier option."""

    delta: float
    """Finite-difference delta."""

    gamma: float
    """Finite-difference gamma."""

    vega: float
    """Finite-difference vega."""

    upper_barrier_hit_prob: float
    """Fraction of paths that hit the upper barrier."""

    lower_barrier_hit_prob: float
    """Fraction of paths that hit the lower barrier."""

    early_exercise_prob: float
    """Fraction of surviving paths where early exercise was optimal."""

    european_double_barrier_price: float
    """European double-barrier price without early exercise."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _intrinsic(S: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S - strike, 0.0)
    return np.maximum(strike - S, 0.0)


def _european_barrier_mc(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    barrier_type: str,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> float:
    """European barrier via plain MC (no early exercise)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, spot, dtype=float)
    survived = np.ones(n_paths, dtype=bool)   # knock-out tracking
    activated = np.zeros(n_paths, dtype=bool)  # knock-in tracking

    mu_dt = (r - q - 0.5 * vol * vol) * dt
    sig_sdt = vol * math.sqrt(dt)
    is_up = "up" in barrier_type
    is_out = "out" in barrier_type

    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S *= np.exp(mu_dt + sig_sdt * Z)
        hit = (S >= barrier) if is_up else (S <= barrier)
        survived &= ~hit
        activated |= hit

    payoff = _intrinsic(S, strike, option_type)
    mask = survived if is_out else activated
    return float(math.exp(-r * T) * (payoff * mask).mean())


def _lsm_bermudan_barrier_core(
    spot: float,
    strike: float,
    barrier: float | None,
    upper_barrier: float | None,
    lower_barrier: float | None,
    vol: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    barrier_type: str,   # used only when barrier is not None
    exercise_dates: np.ndarray | None,
    n_paths: int,
    n_steps: int,
    seed: int,
    n_basis: int = 3,
) -> tuple[float, float, float, float]:
    """Core LSM with barrier monitoring.

    Returns:
        (price, barrier_hit_prob, early_exercise_prob, barrier_hit_upper_prob)
        barrier_hit_upper_prob is meaningful only for double-barrier case.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu_dt = (r - q - 0.5 * vol * vol) * dt
    sig_sdt = vol * math.sqrt(dt)

    # Determine exercise date set (1-indexed steps, step n_steps = expiry excluded)
    if exercise_dates is None:
        ex_set = set(range(1, n_steps + 1))
    else:
        ex_set = set(int(round(t * n_steps / T)) for t in exercise_dates
                     if 0 < t < T)

    # Build full path matrix  shape: (n_paths, n_steps+1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    Z_all = rng.standard_normal((n_paths, n_steps))
    for step in range(n_steps):
        paths[:, step + 1] = paths[:, step] * np.exp(mu_dt + sig_sdt * Z_all[:, step])

    # Barrier monitoring: build survived mask step-by-step
    survived = np.ones(n_paths, dtype=bool)
    upper_hit = np.zeros(n_paths, dtype=bool)
    lower_hit = np.zeros(n_paths, dtype=bool)

    is_up = barrier is not None and "up" in barrier_type
    is_out = barrier is not None and "out" in barrier_type

    for step in range(1, n_steps + 1):
        S_step = paths[:, step]
        if barrier is not None:
            hit = (S_step >= barrier) if is_up else (S_step <= barrier)
            survived &= ~hit
        if upper_barrier is not None:
            hit_up = S_step >= upper_barrier
            upper_hit |= hit_up
            survived &= ~hit_up
        if lower_barrier is not None:
            hit_lo = S_step <= lower_barrier
            lower_hit |= hit_lo
            survived &= ~hit_lo

    # Terminal payoff (only surviving paths)
    terminal_payoff = _intrinsic(paths[:, -1], strike, option_type) * survived

    # LSM backward pass
    # cashflow[i] = present value (at time 0) of the cash received on path i
    cashflow = terminal_payoff * math.exp(-r * T)
    # timing[i] = step at which path i receives its cash (n_steps = at expiry)
    cash_step = np.full(n_paths, n_steps, dtype=float)

    # Rebuild step-by-step survival for backward pass
    # (need the survival status at each interior step to decide exercise)
    step_survived_at = np.empty((n_paths, n_steps + 1), dtype=bool)
    step_survived_at[:, 0] = True
    sv = np.ones(n_paths, dtype=bool)
    for step in range(1, n_steps + 1):
        S_step = paths[:, step]
        if barrier is not None:
            hit = (S_step >= barrier) if is_up else (S_step <= barrier)
            sv &= ~hit
        if upper_barrier is not None:
            sv &= S_step < upper_barrier
        if lower_barrier is not None:
            sv &= S_step > lower_barrier
        step_survived_at[:, step] = sv.copy()

    early_exercised = np.zeros(n_paths, dtype=bool)

    for step in range(n_steps - 1, 0, -1):
        if step not in ex_set:
            continue

        s = paths[:, step]
        alive = step_survived_at[:, step]
        intrinsic = _intrinsic(s, strike, option_type)
        itm = (intrinsic > 0) & alive

        if itm.sum() < n_basis + 1:
            continue

        s_itm = s[itm]
        s_norm = s_itm / spot
        basis = np.column_stack([s_norm ** k for k in range(n_basis)])

        # Convert ``cashflow[i]`` (stored as PV-at-t=0) to value at the
        # CURRENT step ``step``, so the regression target is in the same
        # units as ``intrinsic[itm]`` for the exercise comparison below.
        #
        # PV_at_0 = future_cf · exp(-r · cash_step · dt)
        # value_at_step = future_cf · exp(-r · (cash_step - step) · dt)
        #               = PV_at_0 · exp(+r · step · dt)
        #
        # Fix T4-BBARR1: pre-fix the conversion was written as
        # ``cashflow[itm] * exp(-r * (cash_step - step) * dt)``, which
        # algebraically equals ``correct × exp(-r · cash_step · dt)`` —
        # over-discounted by an extra factor.  The regression target was
        # systematically biased downward (heavily for paths whose
        # cashflow sits at the terminal step), so the LSM
        # over-exercised early.  Correct expression: just un-discount
        # PV-at-0 back to the current step.
        cont_at_step = cashflow[itm] * np.exp(r * step * dt)

        coeff = np.linalg.lstsq(basis, cont_at_step, rcond=None)[0]
        cont_hat = basis @ coeff

        exercise_now = intrinsic[itm] > cont_hat
        paths_itm = np.where(itm)[0]
        ex_idx = paths_itm[exercise_now]

        if len(ex_idx) > 0:
            cashflow[ex_idx] = intrinsic[itm][exercise_now] * math.exp(-r * step * dt)
            cash_step[ex_idx] = step
            early_exercised[ex_idx] = True

    # cashflow is already in PV(t=0) terms for each path; non-surviving paths get 0
    all_pv = cashflow.copy()
    all_pv[~survived & ~early_exercised] = 0.0
    price = float(all_pv.mean())

    barrier_hit_prob = float(1.0 - survived.mean()) if barrier is not None or upper_barrier is not None or lower_barrier is not None else 0.0
    early_ex_prob = float(early_exercised[survived].mean()) if survived.sum() > 0 else 0.0
    upper_hit_prob = float(upper_hit.mean())

    return price, barrier_hit_prob, early_ex_prob, upper_hit_prob


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bermudan_barrier_option(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    exercise_dates: list[float] | None = None,
    n_paths: int = 100_000,
    n_steps: int = 500,
    seed: int = 42,
) -> BermudanBarrierResult:
    """Bermudan barrier option priced via LSM with continuous barrier monitoring.

    Early exercise is permitted only at the specified exercise dates. The barrier
    is monitored at every simulation step (continuous monitoring approximation).
    For American behaviour (exercise at any time), pass exercise_dates=None.

    Knock-in variants are priced via parity: knock_in = Bermudan_vanilla − knock_out.

    Args:
        spot: current underlying price.
        strike: option strike.
        barrier: barrier level.
        vol: flat implied volatility (annualised).
        T: time to expiry in years.
        r: continuously-compounded risk-free rate.
        q: continuous dividend yield.
        option_type: "call" or "put".
        barrier_type: "down-and-out", "up-and-out", "down-and-in", "up-and-in".
        exercise_dates: list of exercise times in years; None = American.
        n_paths: number of Monte Carlo paths.
        n_steps: number of time steps for simulation and barrier monitoring.
        seed: random seed for reproducibility.

    Returns:
        BermudanBarrierResult with price, Greeks, and diagnostic statistics.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if vol <= 0:
        raise ValueError(f"vol must be positive, got {vol}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    option_type = option_type.lower()
    barrier_type = barrier_type.lower().replace(" ", "-")

    is_knockout = "out" in barrier_type
    ex_arr = np.array(exercise_dates) if exercise_dates is not None else None

    # Knock-out pricing
    if is_knockout:
        price, hit_prob, ex_prob, _ = _lsm_bermudan_barrier_core(
            spot, strike, barrier, None, None,
            vol, T, r, q, option_type, barrier_type,
            ex_arr, n_paths, n_steps, seed,
        )
    else:
        # Knock-in = vanilla Bermudan - knock-out
        ko_type = barrier_type.replace("-in", "-out")
        ko_price, hit_prob, _, _ = _lsm_bermudan_barrier_core(
            spot, strike, barrier, None, None,
            vol, T, r, q, option_type, ko_type,
            ex_arr, n_paths, n_steps, seed,
        )
        vanilla_price, _, ex_prob, _ = _lsm_bermudan_barrier_core(
            spot, strike, None, None, None,
            vol, T, r, q, option_type, "",
            ex_arr, n_paths, n_steps, seed,
        )
        price = vanilla_price - ko_price

    # European barrier for comparison
    eur_barrier = _european_barrier_mc(
        spot, strike, barrier, vol, T, r, q,
        option_type, barrier_type, n_paths, n_steps, seed,
    )

    # Bump-and-reprice Greeks (cheaper: smaller n_paths)
    _g_paths = max(n_paths // 5, 10_000)
    bump_s = spot * 0.01
    bump_v = 0.01

    def _price(s: float, v: float) -> float:
        if is_knockout:
            p, _, _, _ = _lsm_bermudan_barrier_core(
                s, strike, barrier, None, None,
                v, T, r, q, option_type, barrier_type,
                ex_arr, _g_paths, n_steps, seed,
            )
        else:
            ko_type = barrier_type.replace("-in", "-out")
            ko_p, _, _, _ = _lsm_bermudan_barrier_core(
                s, strike, barrier, None, None,
                v, T, r, q, option_type, ko_type,
                ex_arr, _g_paths, n_steps, seed,
            )
            van_p, _, _, _ = _lsm_bermudan_barrier_core(
                s, strike, None, None, None,
                v, T, r, q, option_type, "",
                ex_arr, _g_paths, n_steps, seed,
            )
            p = van_p - ko_p
        return p

    p_up = _price(spot + bump_s, vol)
    p_dn = _price(spot - bump_s, vol)
    p_vu = _price(spot, vol + bump_v)

    delta = (p_up - p_dn) / (2 * bump_s)
    gamma = (p_up - 2 * price + p_dn) / (bump_s ** 2)
    vega = p_vu - price

    return BermudanBarrierResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        barrier_hit_prob=hit_prob,
        early_exercise_prob=ex_prob,
        european_barrier_price=eur_barrier,
    )


def american_barrier_option(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "put",
    barrier_type: str = "up-and-out",
    n_paths: int = 100_000,
    n_steps: int = 500,
    seed: int = 42,
) -> BermudanBarrierResult:
    """American barrier option — Bermudan barrier with all steps as exercise dates.

    Equivalent to calling bermudan_barrier_option with exercise_dates=None.
    This is a convenience wrapper for the most common use case.

    Args:
        spot: current underlying price.
        strike: option strike.
        barrier: barrier level.
        vol: flat implied volatility.
        T: time to expiry in years.
        r: risk-free rate.
        q: dividend yield.
        option_type: "call" or "put".
        barrier_type: "down-and-out", "up-and-out", "down-and-in", "up-and-in".
        n_paths: number of MC paths.
        n_steps: number of time steps.
        seed: random seed.

    Returns:
        BermudanBarrierResult (exercise_dates=None gives American behaviour).
    """
    return bermudan_barrier_option(
        spot=spot, strike=strike, barrier=barrier, vol=vol, T=T,
        r=r, q=q, option_type=option_type, barrier_type=barrier_type,
        exercise_dates=None,
        n_paths=n_paths, n_steps=n_steps, seed=seed,
    )


def bermudan_double_barrier(
    spot: float,
    strike: float,
    upper_barrier: float,
    lower_barrier: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    exercise_dates: list[float] | None = None,
    n_paths: int = 100_000,
    n_steps: int = 500,
    seed: int = 42,
) -> BermudanDoubleBarrierResult:
    """Bermudan double-barrier option (knock-out at either barrier).

    The option is extinguished if the spot touches either the upper or the lower
    barrier at any simulation step. Early exercise is permitted at exercise_dates.

    Args:
        spot: current underlying price.
        strike: option strike.
        upper_barrier: upper knock-out level (must be > spot).
        lower_barrier: lower knock-out level (must be < spot).
        vol: flat implied volatility.
        T: time to expiry in years.
        r: risk-free rate.
        q: dividend yield.
        option_type: "call" or "put".
        exercise_dates: list of exercise times; None = American.
        n_paths: number of MC paths.
        n_steps: number of time steps.
        seed: random seed.

    Returns:
        BermudanDoubleBarrierResult with price, Greeks, and diagnostics.
    """
    if upper_barrier <= spot:
        raise ValueError(f"upper_barrier={upper_barrier} must be > spot={spot}")
    if lower_barrier >= spot:
        raise ValueError(f"lower_barrier={lower_barrier} must be < spot={spot}")

    ex_arr = np.array(exercise_dates) if exercise_dates is not None else None

    price, hit_prob, ex_prob, upper_hit_prob = _lsm_bermudan_barrier_core(
        spot, strike, None, upper_barrier, lower_barrier,
        vol, T, r, q, option_type, "",
        ex_arr, n_paths, n_steps, seed,
    )
    lower_hit_prob = hit_prob - upper_hit_prob

    # European double-barrier for comparison
    rng_e = np.random.default_rng(seed)
    dt = T / n_steps
    S_e = np.full(n_paths, spot, dtype=float)
    sv_e = np.ones(n_paths, dtype=bool)
    mu_dt = (r - q - 0.5 * vol * vol) * dt
    sig_sdt = vol * math.sqrt(dt)
    for _ in range(n_steps):
        Z = rng_e.standard_normal(n_paths)
        S_e *= np.exp(mu_dt + sig_sdt * Z)
        sv_e &= (S_e < upper_barrier) & (S_e > lower_barrier)
    eur_db = float(math.exp(-r * T) * (_intrinsic(S_e, strike, option_type) * sv_e).mean())

    # Greeks via bump
    _g_paths = max(n_paths // 5, 10_000)
    bump_s = spot * 0.01
    bump_v = 0.01

    def _price(s: float, v: float) -> float:
        p, _, _, _ = _lsm_bermudan_barrier_core(
            s, strike, None, upper_barrier, lower_barrier,
            v, T, r, q, option_type, "", ex_arr, _g_paths, n_steps, seed,
        )
        return p

    p_up = _price(spot + bump_s, vol)
    p_dn = _price(spot - bump_s, vol)
    p_vu = _price(spot, vol + bump_v)
    delta = (p_up - p_dn) / (2 * bump_s)
    gamma = (p_up - 2 * price + p_dn) / (bump_s ** 2)
    vega = p_vu - price

    return BermudanDoubleBarrierResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        upper_barrier_hit_prob=float(upper_hit_prob),
        lower_barrier_hit_prob=float(max(lower_hit_prob, 0.0)),
        early_exercise_prob=ex_prob,
        european_double_barrier_price=eur_db,
    )


def barrier_exercise_interaction(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "put",
    barrier_type: str = "down-and-out",
    n_paths: int = 80_000,
    n_steps: int = 200,
    seed: int = 42,
) -> dict:
    """Analyse the interaction between early exercise and barrier knock-out.

    Computes four prices and decomposes how the barrier modifies the early
    exercise premium.  Useful for understanding which effect dominates for a
    given moneyness / barrier distance combination.

    Args:
        spot: current underlying price.
        strike: option strike.
        barrier: barrier level.
        vol: flat implied volatility.
        T: time to expiry.
        r: risk-free rate.
        q: dividend yield.
        option_type: "call" or "put".
        barrier_type: "down-and-out" or "up-and-out".
        n_paths: MC paths (shared across all four runs).
        n_steps: simulation time steps.
        seed: random seed.

    Returns:
        dict with keys:
            european_price           — plain European Black-Scholes MC price.
            barrier_price            — European barrier (no early exercise).
            american_price           — American option (no barrier).
            american_barrier_price   — American + barrier combined.
            early_exercise_premium   — american_price − european_price.
            barrier_discount_vanilla — european_price − barrier_price.
            barrier_discount_american— american_price − american_barrier_price.
            exercise_premium_reduction — how much the barrier reduces the EEP
                                         (positive = barrier shrinks the premium).
    """
    # 1. European plain MC
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu_dt = (r - q - 0.5 * vol * vol) * dt
    sig_sdt = vol * math.sqrt(dt)
    S_e = np.full(n_paths, spot, dtype=float)
    for _ in range(n_steps):
        S_e *= np.exp(mu_dt + sig_sdt * rng.standard_normal(n_paths))
    eur_price = float(math.exp(-r * T) * _intrinsic(S_e, strike, option_type).mean())

    # 2. European barrier
    bar_price = _european_barrier_mc(
        spot, strike, barrier, vol, T, r, q, option_type, barrier_type,
        n_paths, n_steps, seed,
    )

    # 3. American (no barrier)
    am_price_raw, _, _, _ = _lsm_bermudan_barrier_core(
        spot, strike, None, None, None,
        vol, T, r, q, option_type, "",
        None, n_paths, n_steps, seed,
    )

    # 4. American + barrier
    is_ko = "out" in barrier_type
    if is_ko:
        am_bar_price, _, _, _ = _lsm_bermudan_barrier_core(
            spot, strike, barrier, None, None,
            vol, T, r, q, option_type, barrier_type,
            None, n_paths, n_steps, seed,
        )
    else:
        ko_type = barrier_type.replace("-in", "-out")
        ko_p, _, _, _ = _lsm_bermudan_barrier_core(
            spot, strike, barrier, None, None,
            vol, T, r, q, option_type, ko_type,
            None, n_paths, n_steps, seed,
        )
        am_bar_price = am_price_raw - ko_p

    eep_vanilla = am_price_raw - eur_price
    eep_barrier = am_bar_price - bar_price
    exercise_premium_reduction = eep_vanilla - eep_barrier

    return {
        "european_price": eur_price,
        "barrier_price": bar_price,
        "american_price": am_price_raw,
        "american_barrier_price": am_bar_price,
        "early_exercise_premium": eep_vanilla,
        "barrier_discount_vanilla": eur_price - bar_price,
        "barrier_discount_american": am_price_raw - am_bar_price,
        "exercise_premium_reduction": exercise_premium_reduction,
    }
