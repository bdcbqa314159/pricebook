"""Bermudan swaption pricing under LMM.

Extends :mod:`pricebook.bermudan_swaption` with forward-rate-based methods:

* :func:`bermudan_swaption_lmm` — LSM pricing under LMM dynamics.
* :func:`bermudan_exercise_boundary` — extract exercise frontier.
* :func:`bermudan_upper_bound` — Andersen-Broadie dual upper bound.

References:
    Andersen & Piterbarg, *Interest Rate Modeling*, Vol. 3, Ch. 18.
    Longstaff & Schwartz, *Valuing American Options by Simulation*, RFS, 2001.
    Andersen & Broadie, *Primal-Dual Simulation for Upper Bounds*, Ops. Res., 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Bermudan swaption under LMM ----

@dataclass
class BermudanLMMResult:
    """Bermudan swaption pricing result under LMM."""
    price: float
    exercise_rate: float        # fraction of paths that exercise early
    mean_exercise_time: float   # average exercise time (conditional on exercise)
    n_exercise_dates: int


@dataclass
class ExerciseBoundary:
    """Exercise boundary: swap rate threshold at each exercise date."""
    exercise_times: np.ndarray
    boundary_rates: np.ndarray    # critical swap rate at each date
    exercise_counts: np.ndarray   # paths exercising at each date


@dataclass
class BermudanBoundsResult:
    """Lower and upper bounds for Bermudan swaption."""
    lower_bound: float   # LSM price
    upper_bound: float   # Andersen-Broadie dual
    gap: float           # upper - lower


def _simulate_lmm_paths(
    forward_rates: np.ndarray,
    inst_vols: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    dt_tenor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate LMM forward rate paths.

    Returns (n_paths, n_steps+1, n_fwd) array.
    """
    n_fwd = len(forward_rates)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    F = np.zeros((n_paths, n_steps + 1, n_fwd))
    F[:, 0, :] = forward_rates

    for step in range(n_steps):
        dW = rng.standard_normal((n_paths, n_fwd)) * sqrt_dt
        for j in range(n_fwd):
            L = np.maximum(F[:, step, j], 1e-10)
            # Lognormal LMM dynamics (simplified drift)
            F[:, step + 1, j] = L * np.exp(
                -0.5 * inst_vols[j]**2 * dt + inst_vols[j] * dW[:, j]
            )

    return F


def _swap_rate(forward_rates: np.ndarray, start_idx: int, end_idx: int,
               dt_tenor: float) -> np.ndarray:
    """Compute par swap rate from forward rates.

    swap_rate = (1 - P(T_end)) / annuity
    where P(T_k) = Π_{i=start}^{k-1} 1/(1 + τ F_i)
    """
    n_fwd = forward_rates.shape[-1]
    end_idx = min(end_idx, n_fwd)
    if end_idx <= start_idx:
        return np.zeros(forward_rates.shape[0]) if forward_rates.ndim > 1 else 0.0

    if forward_rates.ndim == 1:
        # Single set of forwards
        disc = 1.0
        annuity = 0.0
        for i in range(start_idx, end_idx):
            disc /= (1 + dt_tenor * forward_rates[i])
            annuity += disc * dt_tenor
        return (1.0 - disc) / annuity if annuity > 0 else 0.0

    # Vectorised over paths
    n_paths = forward_rates.shape[0]
    disc = np.ones(n_paths)
    annuity = np.zeros(n_paths)
    for i in range(start_idx, end_idx):
        disc /= (1 + dt_tenor * np.maximum(forward_rates[:, i], 0.0))
        annuity += disc * dt_tenor

    rate = np.where(annuity > 0, (1.0 - disc) / annuity, 0.0)
    return rate


def _annuity(forward_rates: np.ndarray, start_idx: int, end_idx: int,
             dt_tenor: float) -> np.ndarray:
    """Compute swap annuity from forward rates."""
    n_fwd = forward_rates.shape[-1]
    end_idx = min(end_idx, n_fwd)

    if forward_rates.ndim == 1:
        disc = 1.0
        ann = 0.0
        for i in range(start_idx, end_idx):
            disc /= (1 + dt_tenor * forward_rates[i])
            ann += disc * dt_tenor
        return ann

    n_paths = forward_rates.shape[0]
    disc = np.ones(n_paths)
    ann = np.zeros(n_paths)
    for i in range(start_idx, end_idx):
        disc /= (1 + dt_tenor * np.maximum(forward_rates[:, i], 0.0))
        ann += disc * dt_tenor
    return ann


def bermudan_swaption_lmm(
    forward_rates: list[float],
    inst_vols: list[float],
    strike: float,
    exercise_indices: list[int],
    swap_end_idx: int,
    dt_tenor: float = 0.5,
    is_payer: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 100,
    n_basis: int = 3,
    seed: int | None = None,
) -> BermudanLMMResult:
    """Bermudan swaption priced via LSM under LMM dynamics.

    At each exercise date, the holder can enter a swap paying fixed rate K
    and receiving floating. Exercise value = annuity × (S(t) − K)+ for payer.

    Args:
        forward_rates: initial forward rate curve.
        inst_vols: instantaneous vol per forward rate.
        strike: fixed swap rate.
        exercise_indices: forward indices at which exercise is allowed.
        swap_end_idx: index of the last forward in the swap.
        dt_tenor: tenor spacing.
        is_payer: True = payer swaption (exercise when rates rise).
        n_basis: number of polynomial basis functions for LSM regression.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    fwd = np.array(forward_rates, dtype=float)
    vols = np.array(inst_vols, dtype=float)
    n_fwd = len(fwd)

    T = max(exercise_indices) * dt_tenor + dt_tenor
    F = _simulate_lmm_paths(fwd, vols, T, n_steps, n_paths, dt_tenor, rng)

    # Map exercise indices to simulation steps
    dt_sim = T / n_steps
    exercise_steps = []
    for idx in sorted(exercise_indices):
        t_ex = idx * dt_tenor
        step = min(int(round(t_ex / dt_sim)), n_steps)
        exercise_steps.append((step, idx))

    # Compute exercise values at each exercise date
    n_ex = len(exercise_steps)
    ex_values = np.zeros((n_paths, n_ex))
    ex_swap_rates = np.zeros((n_paths, n_ex))

    for k, (step, start_idx) in enumerate(exercise_steps):
        fwd_at_ex = F[:, step, :]
        sr = _swap_rate(fwd_at_ex, start_idx, swap_end_idx, dt_tenor)
        ann = _annuity(fwd_at_ex, start_idx, swap_end_idx, dt_tenor)
        ex_swap_rates[:, k] = sr

        if is_payer:
            ex_values[:, k] = np.maximum(sr - strike, 0.0) * ann
        else:
            ex_values[:, k] = np.maximum(strike - sr, 0.0) * ann

    # LSM backward induction
    cashflow = ex_values[:, -1].copy()
    exercise_time_idx = np.full(n_paths, n_ex - 1, dtype=int)

    for k in range(n_ex - 2, -1, -1):
        ev = ex_values[:, k]
        itm = ev > 0

        if itm.sum() < n_basis + 1:
            continue

        # Discount cashflow from exercise_time_idx to step k
        step_k = exercise_steps[k][0]
        disc_cf = np.zeros(itm.sum())
        itm_idx = np.where(itm)[0]

        for p_idx, p in enumerate(itm_idx):
            future_k = exercise_time_idx[p]
            future_step = exercise_steps[future_k][0]
            dt_between = (future_step - step_k) * dt_sim
            # Simple discounting using average forward rate
            avg_rate = np.mean(F[p, step_k, :min(n_fwd, 3)])
            disc_cf[p_idx] = cashflow[p] * math.exp(-avg_rate * dt_between)

        # Regression on swap rate
        sr_itm = ex_swap_rates[itm, k]
        sr_mean = sr_itm.mean()
        sr_std = max(sr_itm.std(), 1e-10)
        sr_norm = (sr_itm - sr_mean) / sr_std

        basis = np.column_stack([sr_norm**j for j in range(n_basis)])

        try:
            coeffs = np.linalg.lstsq(basis, disc_cf, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continue

        exercise_mask = ev[itm] > continuation
        exercise_idx = itm_idx[exercise_mask]

        cashflow[exercise_idx] = ev[exercise_idx]
        exercise_time_idx[exercise_idx] = k

    # Discount to time 0
    pv = np.zeros(n_paths)
    for p in range(n_paths):
        k = exercise_time_idx[p]
        step = exercise_steps[k][0]
        t_ex = step * dt_sim
        avg_rate = np.mean(F[p, 0, :min(n_fwd, 3)])
        pv[p] = cashflow[p] * math.exp(-avg_rate * t_ex)

    price = float(pv.mean())

    # Exercise statistics
    exercised = cashflow > 0
    exercise_rate = float(exercised.mean())
    if exercised.sum() > 0:
        ex_times = np.array([exercise_steps[exercise_time_idx[p]][0] * dt_sim
                             for p in range(n_paths) if exercised[p]])
        mean_ex_time = float(ex_times.mean())
    else:
        mean_ex_time = 0.0

    return BermudanLMMResult(price, exercise_rate, mean_ex_time, n_ex)


def bermudan_exercise_boundary(
    forward_rates: list[float],
    inst_vols: list[float],
    strike: float,
    exercise_indices: list[int],
    swap_end_idx: int,
    dt_tenor: float = 0.5,
    is_payer: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 100,
    n_basis: int = 3,
    seed: int | None = None,
) -> ExerciseBoundary:
    """Extract exercise boundary from LSM regression.

    Returns the critical swap rate at each exercise date: exercise when
    swap rate exceeds this boundary (payer) or drops below (receiver).
    """
    rng = np.random.default_rng(seed)
    fwd = np.array(forward_rates, dtype=float)
    vols = np.array(inst_vols, dtype=float)
    n_fwd = len(fwd)

    T = max(exercise_indices) * dt_tenor + dt_tenor
    F = _simulate_lmm_paths(fwd, vols, T, n_steps, n_paths, dt_tenor, rng)

    dt_sim = T / n_steps
    exercise_steps = []
    for idx in sorted(exercise_indices):
        t_ex = idx * dt_tenor
        step = min(int(round(t_ex / dt_sim)), n_steps)
        exercise_steps.append((step, idx))

    n_ex = len(exercise_steps)
    ex_values = np.zeros((n_paths, n_ex))
    ex_swap_rates = np.zeros((n_paths, n_ex))

    for k, (step, start_idx) in enumerate(exercise_steps):
        fwd_at_ex = F[:, step, :]
        sr = _swap_rate(fwd_at_ex, start_idx, swap_end_idx, dt_tenor)
        ann = _annuity(fwd_at_ex, start_idx, swap_end_idx, dt_tenor)
        ex_swap_rates[:, k] = sr
        if is_payer:
            ex_values[:, k] = np.maximum(sr - strike, 0.0) * ann
        else:
            ex_values[:, k] = np.maximum(strike - sr, 0.0) * ann

    # LSM backward to find boundary
    cashflow = ex_values[:, -1].copy()
    exercise_time_idx = np.full(n_paths, n_ex - 1, dtype=int)
    boundary_rates = np.full(n_ex, np.nan)
    exercise_counts = np.zeros(n_ex, dtype=int)

    # Last date: always exercise if ITM
    itm_last = ex_values[:, -1] > 0
    exercise_counts[-1] = int(itm_last.sum())
    if itm_last.sum() > 0:
        boundary_rates[-1] = strike  # at-the-money boundary

    for k in range(n_ex - 2, -1, -1):
        ev = ex_values[:, k]
        itm = ev > 0

        if itm.sum() < n_basis + 1:
            continue

        step_k = exercise_steps[k][0]
        disc_cf = np.zeros(itm.sum())
        itm_idx = np.where(itm)[0]

        for p_idx, p in enumerate(itm_idx):
            future_k = exercise_time_idx[p]
            future_step = exercise_steps[future_k][0]
            dt_between = (future_step - step_k) * dt_sim
            avg_rate = np.mean(F[p, step_k, :min(n_fwd, 3)])
            disc_cf[p_idx] = cashflow[p] * math.exp(-avg_rate * dt_between)

        sr_itm = ex_swap_rates[itm, k]
        sr_mean = sr_itm.mean()
        sr_std = max(sr_itm.std(), 1e-10)
        sr_norm = (sr_itm - sr_mean) / sr_std

        basis = np.column_stack([sr_norm**j for j in range(n_basis)])

        try:
            coeffs = np.linalg.lstsq(basis, disc_cf, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continue

        exercise_mask = ev[itm] > continuation
        exercise_idx = itm_idx[exercise_mask]

        # Find boundary: the swap rate where exercise ≈ continuation
        if exercise_mask.sum() > 0 and (~exercise_mask).sum() > 0:
            if is_payer:
                boundary_rates[k] = float(sr_itm[exercise_mask].min())
            else:
                boundary_rates[k] = float(sr_itm[exercise_mask].max())

        exercise_counts[k] = int(exercise_mask.sum())
        cashflow[exercise_idx] = ev[exercise_idx]
        exercise_time_idx[exercise_idx] = k

    exercise_times = np.array([step * dt_sim for step, _ in exercise_steps])

    return ExerciseBoundary(exercise_times, boundary_rates, exercise_counts)


def bermudan_upper_bound(
    forward_rates: list[float],
    inst_vols: list[float],
    strike: float,
    exercise_indices: list[int],
    swap_end_idx: int,
    dt_tenor: float = 0.5,
    is_payer: bool = True,
    n_paths: int = 10_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> BermudanBoundsResult:
    """Andersen-Broadie dual upper bound for Bermudan swaption.

    Lower bound: LSM price.
    Upper bound: max over exercise dates of discounted payoff.
    (Simplified: uses discounted payoff envelope instead of full
    sub-simulation martingale correction.)

    True price satisfies: lower ≤ V ≤ upper.
    """
    rng = np.random.default_rng(seed)
    fwd = np.array(forward_rates, dtype=float)
    vols = np.array(inst_vols, dtype=float)
    n_fwd = len(fwd)

    T = max(exercise_indices) * dt_tenor + dt_tenor
    F = _simulate_lmm_paths(fwd, vols, T, n_steps, n_paths, dt_tenor, rng)

    dt_sim = T / n_steps
    exercise_steps = []
    for idx in sorted(exercise_indices):
        t_ex = idx * dt_tenor
        step = min(int(round(t_ex / dt_sim)), n_steps)
        exercise_steps.append((step, idx))

    # Upper bound: max discounted payoff over exercise dates
    max_disc_payoff = np.zeros(n_paths)

    for step, start_idx in exercise_steps:
        fwd_at_ex = F[:, step, :]
        sr = _swap_rate(fwd_at_ex, start_idx, swap_end_idx, dt_tenor)
        ann = _annuity(fwd_at_ex, start_idx, swap_end_idx, dt_tenor)

        if is_payer:
            payoff = np.maximum(sr - strike, 0.0) * ann
        else:
            payoff = np.maximum(strike - sr, 0.0) * ann

        t_ex = step * dt_sim
        avg_rate = np.mean(F[:, 0, :min(n_fwd, 3)], axis=1)
        disc_payoff = payoff * np.exp(-avg_rate * t_ex)

        max_disc_payoff = np.maximum(max_disc_payoff, disc_payoff)

    upper = float(max_disc_payoff.mean())

    # Lower bound: LSM price (use same paths via deterministic seed)
    lower_result = bermudan_swaption_lmm(
        forward_rates, inst_vols, strike, exercise_indices, swap_end_idx,
        dt_tenor, is_payer, n_paths, n_steps, seed=seed,
    )
    lower = lower_result.price

    return BermudanBoundsResult(lower, upper, upper - lower)
