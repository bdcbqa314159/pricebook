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
    swap_freq: float = 1.0,
) -> float:
    """Bermudan swaption via Hull-White trinomial tree.

    Uses standard trinomial branching with transition probabilities
    from the Hull-White model. At each exercise date, applies
    max(continuation, exercise_value).

    Args:
        exercise_years: list of times when exercise is allowed.
        swap_end_years: maturity of the underlying swap.
        strike: fixed rate of the underlying swap.
        is_payer: True = right to pay fixed (call on rate).
        swap_freq: payment frequency of the underlying swap (e.g. 0.5 for semi-annual).
    """
    T = max(exercise_years)
    dt = T / n_steps
    a, sigma = hw.a, hw.sigma
    dr = sigma * math.sqrt(3.0 * dt)
    j_max = max(1, int(math.ceil(0.1835 / (a * dt))))
    n_nodes = 2 * j_max + 1
    mid = j_max

    exercise_steps = set(int(round(t / dt)) for t in exercise_years)

    # Get initial short rate
    r0 = hw._forward_rate(0.0)

    # Terminal value: 0 (option expires worthless if not exercised)
    values = np.zeros(n_nodes)

    for step in range(n_steps - 1, -1, -1):
        new_values = np.zeros(n_nodes)

        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)

            # Trinomial transition probabilities
            eta = a * j * dr * dt
            p_up = (1.0 / 6.0) + (j * j * a * a * dt * dt - j * a * dt) / 6.0
            p_mid = 2.0 / 3.0 - j * j * a * a * dt * dt / 3.0
            p_down = (1.0 / 6.0) + (j * j * a * a * dt * dt + j * a * dt) / 6.0

            # Clamp probabilities
            p_up = max(0.0, min(1.0, p_up))
            p_mid = max(0.0, min(1.0, p_mid))
            p_down = max(0.0, min(1.0, p_down))
            p_total = p_up + p_mid + p_down
            if p_total > 0:
                p_up /= p_total
                p_mid /= p_total
                p_down /= p_total

            # Successor node indices (clamped)
            j_up = min(j + 1, j_max)
            j_mid = j
            j_down = max(j - 1, -j_max)

            cont = (p_up * values[j_up + mid]
                     + p_mid * values[j_mid + mid]
                     + p_down * values[j_down + mid])
            new_values[idx] = cont * one_step_df

        # Check for exercise at this step
        if (step + 1) in exercise_steps:
            t_exercise = (step + 1) * dt
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                r_j = r0 + j * dr

                # Swap value at exercise using analytical HW bond prices
                p_end = hw.zcb_price(t_exercise, swap_end_years, r_j)
                annuity = 0.0
                t_pay = t_exercise + swap_freq
                while t_pay <= swap_end_years + 1e-10:
                    annuity += swap_freq * hw.zcb_price(t_exercise, t_pay, r_j)
                    t_pay += swap_freq

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
    swap_freq: float = 1.0,
) -> float:
    """Bermudan swaption via Longstaff-Schwartz MC.

    Simulate short rate paths under HW, compute swap value at each
    exercise date, regress continuation vs exercise. Uses path-dependent
    discounting (integral of short rate along each path).
    """
    exercise_years_sorted = sorted(exercise_years)
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
        # HW mean-reversion target: use theta(t)/a approximation via forward rate
        r_mean = hw._forward_rate(t)
        e_adt = math.exp(-hw.a * dt)
        std = hw.sigma * math.sqrt((1 - e_adt**2) / (2 * hw.a)) if hw.a > 0 else hw.sigma * math.sqrt(dt)
        Z = rng.standard_normal(n_paths)
        rate_paths[:, i] = r_mean + (r_prev - r_mean) * e_adt + std * Z
        r_prev = rate_paths[:, i]

    # Compute exercise value at each exercise date
    exercise_values = np.zeros((n_paths, n_steps))
    for i, t_ex in enumerate(exercise_years_sorted):
        for p in range(n_paths):
            r = rate_paths[p, i]
            p_end = hw.zcb_price(t_ex, swap_end_years, r)
            annuity = 0.0
            t_pay = t_ex + swap_freq
            while t_pay <= swap_end_years + 1e-10:
                annuity += swap_freq * hw.zcb_price(t_ex, t_pay, r)
                t_pay += swap_freq

            swap_pv = (1.0 - p_end) - strike * annuity
            if not is_payer:
                swap_pv = -swap_pv
            exercise_values[p, i] = max(swap_pv, 0.0)

    # Path-dependent discount factors between exercise dates
    # Use average of rate at step i and step i+1 for trapezoidal integration
    disc_factors = np.ones((n_paths, n_steps))
    for i in range(n_steps):
        t_prev = exercise_years_sorted[i - 1] if i > 0 else 0.0
        dt = exercise_years_sorted[i] - t_prev
        if i == 0:
            r_avg = 0.5 * (r0 + rate_paths[:, 0])
        else:
            r_avg = 0.5 * (rate_paths[:, i - 1] + rate_paths[:, i])
        disc_factors[:, i] = np.exp(-r_avg * dt)

    # Cumulative discount to time 0 for each step
    cum_disc = np.ones((n_paths, n_steps))
    cum_disc[:, 0] = disc_factors[:, 0]
    for i in range(1, n_steps):
        cum_disc[:, i] = cum_disc[:, i - 1] * disc_factors[:, i]

    # LSM backward induction
    cashflow = exercise_values[:, -1].copy()
    cashflow_step = np.full(n_paths, n_steps - 1, dtype=int)

    for i in range(n_steps - 2, -1, -1):
        ev = exercise_values[:, i]
        itm = ev > 0

        if itm.sum() < n_basis + 1:
            continue

        # Discount cashflow from cashflow_step to step i (path-dependent)
        disc_cf = np.zeros(itm.sum())
        itm_idx = np.where(itm)[0]
        for k, p in enumerate(itm_idx):
            s = cashflow_step[p]
            # Discount from step s to step i
            df = 1.0
            for j in range(i + 1, s + 1):
                df *= disc_factors[p, j]
            disc_cf[k] = cashflow[p] * df

        # Regression
        r_itm = rate_paths[itm, i]
        r_mean = r_itm.mean()
        r_std = r_itm.std()
        if r_std < 1e-15:
            r_std = 1.0
        r_norm = (r_itm - r_mean) / r_std
        basis = np.column_stack([r_norm**k for k in range(n_basis)])

        try:
            coeffs = np.linalg.lstsq(basis, disc_cf, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continue

        exercise = ev[itm] > continuation
        exercise_idx = itm_idx[exercise]

        cashflow[exercise_idx] = ev[exercise_idx]
        cashflow_step[exercise_idx] = i

    # Discount each cashflow to time 0 using path-dependent discount
    pv_paths = np.zeros(n_paths)
    for p in range(n_paths):
        s = cashflow_step[p]
        pv_paths[p] = cashflow[p] * cum_disc[p, s]

    return float(pv_paths.mean())
