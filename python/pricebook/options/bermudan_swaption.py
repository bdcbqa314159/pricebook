"""
Bermudan swaption pricing via Hull-White tree and LSM.

A bermudan swaption can be exercised at any coupon date of the
underlying swap. At each exercise date, the holder decides whether
to exercise (enter the swap) or continue holding the option.

    from pricebook.options.bermudan_swaption import bermudan_swaption_tree, bermudan_swaption_lsm

    price = bermudan_swaption_tree(hw, expiry_years=[1,2,3,4,5],
                                    swap_end=10, strike=0.05)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.models.hull_white import HullWhite
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.brownian import WienerProcess


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

    Forward-fits per-step α(t) so the tree reprices the initial discount
    curve, then backward-induces with the textbook trinomial branching.
    At each exercise step the holder takes ``max(continuation, exercise)``.

    Args:
        exercise_years: list of times when exercise is allowed.
        swap_end_years: maturity of the underlying swap.
        strike: fixed rate of the underlying swap.
        is_payer: True = right to pay fixed (call on rate).
        swap_freq: payment frequency of the underlying swap (e.g. 0.5 for semi-annual).

    Fix T4-BERM1 (this slice) rolls up three coupled defects:

    1. **Wrong trinomial probabilities**: the drift terms used ``/6``
       (``p_u = 1/6 + (j²a²dt² − j·a·dt)/6``) instead of the textbook
       ``/2`` (Hull §32.4, eq. 32.10).  Net effect: the drift was 3×
       too small — mean-reversion underestimated, vol over-stated.

    2. **Missing time-varying α(t)**: short rate at node (step, j) used
       ``r0 + j·dr`` for every step, with ``r0`` the initial forward
       rate.  The correct HW tree shifts each step's grid by ``α(t)``
       so it reprices the initial ZCB curve (same defect fixed in
       ``tree_european_swaption`` as T1.9).  Pre-fix the tree silently
       mis-matched the input curve for any non-flat term structure.

    3. **Exercise compared to discounted continuation**: the old loop
       computed ``new_values = exp(-r·dt) × Σ P · V_{step+1}`` and then,
       if ``step+1`` was an exercise date, set ``new_values =
       max(new_values, exercise_at_step+1)`` — but the exercise side was
       NOT discounted from step+1 back to step.  Result: exercise
       systematically over-valued by ``exp(+r·dt)``, biasing the option
       UPWARD.  Now exercise is applied AT its own step (modifying
       ``V[step]`` before the next backward discount), eliminating the
       discount mismatch.
    """
    T = max(exercise_years)
    dt = T / n_steps

    # Forward sweep: per-step α(t) and tree geometry from the calibrated
    # HW infrastructure (same code path as ``tree_european_swaption``).
    alphas, dr, j_max = hw.build_tree_alphas(T, n_steps)
    n_nodes = 2 * j_max + 1
    mid = j_max
    a = hw.a

    exercise_steps = {int(round(t / dt)) for t in exercise_years}

    def _exercise_value_at(step: int, j: int) -> float:
        """Swap PV at node (step, j), taking max with 0."""
        t_ex = step * dt
        # Short rate at this node uses α calibrated at the entry side
        # (i.e. the shift applied for the transition INTO this step).
        # For step == n_steps we use alphas[n_steps - 1] as the final
        # shift; for step == 0 we'd use alphas[0] but exercising at
        # t=0 is trivially zero for ATM swaptions.
        alpha_step = alphas[min(step, n_steps - 1)]
        r_j = alpha_step + j * dr
        p_end = hw.zcb_price(t_ex, swap_end_years, r_j)
        annuity = 0.0
        t_pay = t_ex + swap_freq
        while t_pay <= swap_end_years + 1e-10:
            annuity += swap_freq * hw.zcb_price(t_ex, t_pay, r_j)
            t_pay += swap_freq
        swap_pv = (1.0 - p_end) - strike * annuity
        if not is_payer:
            swap_pv = -swap_pv
        return max(swap_pv, 0.0)

    # Backward induction.  V[step, j] = option value at node (step, j).
    # Initialise terminal value to either the exercise payoff (if
    # maturity is an exercise date) or zero.
    values = np.zeros(n_nodes)
    if n_steps in exercise_steps:
        for j in range(-j_max, j_max + 1):
            values[j + mid] = _exercise_value_at(n_steps, j)

    for step in range(n_steps - 1, -1, -1):
        alpha_step = alphas[step]
        new_values = np.zeros(n_nodes)
        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = alpha_step + j * dr
            disc = math.exp(-r_j * dt)

            # Textbook HW trinomial probabilities (Hull eq. 32.10) — the
            # drift term is divided by 2, not 6.
            p_u = 1.0/6 + (j**2 * a**2 * dt**2 - j * a * dt) / 2
            p_d = 1.0/6 + (j**2 * a**2 * dt**2 + j * a * dt) / 2
            p_u = max(0.0, min(1.0, p_u))
            p_d = max(0.0, min(1.0, p_d))
            p_m = max(0.0, 1.0 - p_u - p_d)

            j_up = min(j + 1, j_max)
            j_dn = max(j - 1, -j_max)

            cont = (p_u * values[j_up + mid]
                    + p_m * values[j + mid]
                    + p_d * values[j_dn + mid])
            new_values[idx] = cont * disc

        # Apply exercise opportunity AT this step (modifies new_values
        # before it becomes V[step] in the next iteration).  Skips
        # step 0 because the option price at t=0 is the single root
        # node value, with no holder-vs-continuation choice (you're not
        # exercising "today" against today's continuation — that's the
        # option value).
        if step in exercise_steps and step > 0:
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                ex = _exercise_value_at(step, j)
                if ex > new_values[idx]:
                    new_values[idx] = ex

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

    # Generate rate at each exercise date via exact OU under HW.
    # Fix T4-BERM2: pre-fix the conditional mean used the forward rate
    # ``f(0, t)`` as the OU mean.  But under HW the short rate decomposes
    # as r(t) = α(t) + y(t) where y follows zero-mean OU and α(t) is the
    # Brigo-Mercurio shift (``HullWhite._alpha``).  Conditional mean is
    # ``α(t) + (r_prev − α(t_prev)) · exp(−a·dt)``, not the forward.
    # The difference is the convexity term ``(σ²/2a²)(1 − e^{−at})²`` —
    # small for short tenors / low vol, but biases long-dated paths.
    rate_paths = np.zeros((n_paths, n_steps))
    r_prev = np.full(n_paths, r0)
    alpha_prev = hw._alpha(0.0)

    for i, t in enumerate(exercise_years_sorted):
        t_prev = exercise_years_sorted[i - 1] if i > 0 else 0.0
        dt = t - t_prev
        alpha_t = hw._alpha(t)
        e_adt = math.exp(-hw.a * dt)
        std = hw.sigma * math.sqrt((1 - e_adt**2) / (2 * hw.a)) if hw.a > 0 else hw.sigma * math.sqrt(dt)
        Z = rng.standard_normal(n_paths)
        rate_paths[:, i] = alpha_t + (r_prev - alpha_prev) * e_adt + std * Z
        r_prev = rate_paths[:, i]
        alpha_prev = alpha_t

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


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------


def bermudan_swaption_lsm_via_engine(
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
    """``bermudan_swaption_lsm`` with HW rate paths from unified MC engine."""
    from pricebook.models.mc_migrate import hw_paths as _hw_paths  # noqa: lazy

    exercise_years_sorted = sorted(exercise_years)
    n_steps = len(exercise_years_sorted)
    r0 = hw._forward_rate(0.0)
    T = exercise_years_sorted[-1]

    # Generate HW short-rate paths at fine resolution, then sample at exercise dates
    n_sim_steps = max(n_steps * 10, 100)
    rate_grid = _hw_paths(
        r0=r0, a=hw.a, sigma=hw.sigma, T=T,
        n_steps=n_sim_steps, n_paths=n_paths, seed=seed,
        theta_func=lambda t: hw._forward_rate(t) * hw.a,
    )
    dt_sim = T / n_sim_steps

    # Sample rate at each exercise date
    rate_paths = np.zeros((n_paths, n_steps))
    for i, t in enumerate(exercise_years_sorted):
        idx = min(int(round(t / dt_sim)), n_sim_steps)
        rate_paths[:, i] = rate_grid[:, idx]

    # ---- Rest is identical to the original LSM backward induction ----
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

    disc_factors = np.ones((n_paths, n_steps))
    for i in range(n_steps):
        t_prev = exercise_years_sorted[i - 1] if i > 0 else 0.0
        dt = exercise_years_sorted[i] - t_prev
        if i == 0:
            r_avg = 0.5 * (r0 + rate_paths[:, 0])
        else:
            r_avg = 0.5 * (rate_paths[:, i - 1] + rate_paths[:, i])
        disc_factors[:, i] = np.exp(-r_avg * dt)

    cum_disc = np.ones((n_paths, n_steps))
    cum_disc[:, 0] = disc_factors[:, 0]
    for i in range(1, n_steps):
        cum_disc[:, i] = cum_disc[:, i - 1] * disc_factors[:, i]

    cashflow = exercise_values[:, -1].copy()
    cashflow_step = np.full(n_paths, n_steps - 1, dtype=int)

    for i in range(n_steps - 2, -1, -1):
        ev = exercise_values[:, i]
        itm = ev > 0
        if itm.sum() < n_basis + 1:
            continue
        disc_cf = np.zeros(itm.sum())
        itm_idx = np.where(itm)[0]
        for k, p in enumerate(itm_idx):
            s = cashflow_step[p]
            df = 1.0
            for j in range(i + 1, s + 1):
                df *= disc_factors[p, j]
            disc_cf[k] = cashflow[p] * df
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

    pv_paths = np.zeros(n_paths)
    for p in range(n_paths):
        s = cashflow_step[p]
        pv_paths[p] = cashflow[p] * cum_disc[p, s]

    return float(pv_paths.mean())
