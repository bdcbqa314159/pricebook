"""
Bermudan swaption pricing via Hull-White tree and LSM.

A bermudan swaption can be exercised at any coupon date of the
underlying swap. At each exercise date, the holder decides whether
to exercise (enter the swap) or continue holding the option.

    from pricebook.bermudan_swaption import bermudan_swaption_tree, bermudan_swaption_lsm

    price = bermudan_swaption_tree(hw, expiry_years=[1,2,3,4,5],
                                    swap_end=10, strike=0.05)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.hull_white import HullWhite
from pricebook.discount_curve import DiscountCurve
from pricebook.brownian import WienerProcess


def bermudan_swaption_tree(
    hw: HullWhite,
    exercise_years: list[float],
    swap_end_years: float,
    strike: float,
    is_payer: bool = True,
    n_steps: int = 100,
) -> float:
    """Bermudan swaption via Hull-White tree.

    At each exercise date: max(continuation, exercise_value).
    Exercise value = swap PV using analytical HW bond prices.

    Args:
        exercise_years: list of times when exercise is allowed.
        swap_end_years: maturity of the underlying swap.
        strike: fixed rate of the underlying swap.
        is_payer: True = right to pay fixed (call on rate).
    """
    T = max(exercise_years)
    dt = T / n_steps
    a, sigma = hw.a, hw.sigma
    dr = sigma * math.sqrt(3.0 * dt)
    j_max = int(math.ceil(0.1835 / (a * dt)))
    n_nodes = 2 * j_max + 1
    mid = j_max

    exercise_steps = set(int(round(t / dt)) for t in exercise_years)

    Q, _, _, r0 = hw._evolve_state_prices(T, n_steps)

    # Terminal value: 0 (option expires worthless if not exercised)
    values = np.zeros(n_nodes)

    for step in range(n_steps - 1, -1, -1):
        new_values = np.zeros(n_nodes)

        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)
            new_values[idx] = values[idx] * one_step_df

        # Check for exercise at this step
        if (step + 1) in exercise_steps:
            t_exercise = (step + 1) * dt
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                r_j = r0 + j * dr

                # Swap value at exercise: PV of fixed - PV of floating
                # Using analytical HW bond prices
                p_end = hw.zcb_price(t_exercise, swap_end_years, r_j)
                n_payments = max(1, int(swap_end_years - t_exercise))
                annuity = 0.0
                for k in range(1, n_payments + 1):
                    t_pay = t_exercise + k
                    if t_pay <= swap_end_years:
                        annuity += hw.zcb_price(t_exercise, t_pay, r_j)

                swap_pv = (1.0 - p_end) - strike * annuity
                if not is_payer:
                    swap_pv = -swap_pv

                exercise_value = max(swap_pv, 0.0)
                new_values[idx] = max(new_values[idx], exercise_value)

        values = new_values

    return float(values[mid])


def bermudan_swaption_lsm(
    hw: HullWhite,
    exercise_years: list[float],
    swap_end_years: float,
    strike: float,
    is_payer: bool = True,
    n_paths: int = 50_000,
    n_basis: int = 3,
    seed: int = 42,
) -> float:
    """Bermudan swaption via Longstaff-Schwartz MC.

    Simulate short rate paths, compute swap value at each exercise date,
    regress continuation vs exercise.
    """
    exercise_years_sorted = sorted(exercise_years)
    T_max = max(exercise_years)
    n_steps = len(exercise_years_sorted)

    # Simulate rate paths at exercise dates
    rng = np.random.default_rng(seed)
    r0 = hw._forward_rate(0.0)

    # Generate rate at each exercise date via exact OU
    rate_paths = np.zeros((n_paths, n_steps))
    r_prev = np.full(n_paths, r0)

    for i, t in enumerate(exercise_years_sorted):
        t_prev = exercise_years_sorted[i - 1] if i > 0 else 0.0
        dt = t - t_prev
        e_adt = math.exp(-hw.a * dt)
        std = hw.sigma * math.sqrt((1 - e_adt**2) / (2 * hw.a)) if hw.a > 0 else hw.sigma * math.sqrt(dt)
        Z = rng.standard_normal(n_paths)
        rate_paths[:, i] = r0 + (r_prev - r0) * e_adt + std * Z
        r_prev = rate_paths[:, i]

    # Compute exercise value at each exercise date
    exercise_values = np.zeros((n_paths, n_steps))
    for i, t_ex in enumerate(exercise_years_sorted):
        for p in range(n_paths):
            r = rate_paths[p, i]
            p_end = hw.zcb_price(t_ex, swap_end_years, r)
            n_pay = max(1, int(swap_end_years - t_ex))
            annuity = sum(
                hw.zcb_price(t_ex, t_ex + k, r)
                for k in range(1, n_pay + 1) if t_ex + k <= swap_end_years
            )
            swap_pv = (1.0 - p_end) - strike * annuity
            if not is_payer:
                swap_pv = -swap_pv
            exercise_values[p, i] = max(swap_pv, 0.0)

    # LSM backward induction
    df_steps = [math.exp(-r0 * (exercise_years_sorted[i] - (exercise_years_sorted[i-1] if i > 0 else 0)))
                for i in range(n_steps)]

    cashflow = exercise_values[:, -1].copy()
    cashflow_step = np.full(n_paths, n_steps - 1)

    for i in range(n_steps - 2, -1, -1):
        ev = exercise_values[:, i]
        itm = ev > 0

        if itm.sum() < n_basis + 1:
            continue

        # Discount cashflow to step i
        steps_ahead = cashflow_step[itm] - i
        disc_cf = cashflow[itm].copy()
        for s in range(int(steps_ahead.max())):
            mask = steps_ahead > s
            disc_cf[mask] *= df_steps[min(i + s + 1, n_steps - 1)]

        # Regression
        r_itm = rate_paths[itm, i]
        r_norm = r_itm / r0
        basis = np.column_stack([r_norm**k for k in range(n_basis)])

        try:
            coeffs = np.linalg.lstsq(basis, disc_cf, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continue

        exercise = ev[itm] > continuation
        exercise_idx = np.where(itm)[0][exercise]

        cashflow[exercise_idx] = ev[exercise_idx]
        cashflow_step[exercise_idx] = i

    # Discount to time 0
    discount = np.array([math.exp(-r0 * exercise_years_sorted[int(s)]) for s in cashflow_step])
    pv = (cashflow * discount).mean()

    return float(pv)
