"""Advanced autocall mechanics: discrete observation, memory coupon, step-down.

Extends the base autocall product with retail-standard features:
discrete observation dates, memory coupon accumulation, worst-of
with discrete barriers, and step-down autocall levels.

* :class:`AdvancedAutocallResult` — pricing result.
* :func:`discrete_autocall` — autocall with discrete observation dates.
* :func:`worst_of_discrete_autocall` — worst-of basket, discrete barriers.
* :func:`step_down_autocall` — declining autocall barriers.

References:
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.
    Deng et al., *Pricing Autocallable Products*, SSRN, 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AdvancedAutocallResult:
    """Advanced autocall pricing result."""
    price: float                    # fair value per unit notional
    expected_life: float            # expected life (years)
    autocall_probability: float     # probability of early termination
    coupon_expected: float          # expected total coupon
    memory_coupon_value: float      # value of memory feature
    worst_final_pct: float          # avg worst-of at final date
    n_observations: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def discrete_autocall(
    spot: float,
    autocall_barrier: float,
    coupon_barrier: float,
    put_barrier: float,
    coupon_rate: float,
    observation_dates: list[float],
    vol: float,
    rate: float = 0.04,
    div_yield: float = 0.02,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
    memory_coupon: bool = False,
) -> AdvancedAutocallResult:
    """Autocall with discrete observation dates and optional memory coupon.

    At each observation date:
    - If S ≥ autocall_barrier × S₀: early termination + all coupons.
    - If S ≥ coupon_barrier × S₀: coupon paid (or accumulated if memory).
    - At maturity, if S < put_barrier × S₀: loss = (put_barrier − S/S₀).

    Memory coupon: unpaid coupons accumulate and are paid when
    a subsequent coupon barrier is breached.

    Args:
        spot: initial spot price.
        autocall_barrier: autocall level (fraction of spot, e.g. 1.0).
        coupon_barrier: coupon payment level (fraction of spot, e.g. 0.80).
        put_barrier: capital protection level (fraction of spot, e.g. 0.60).
        coupon_rate: coupon per observation period.
        observation_dates: observation times (years).
        vol: annualised vol.
        rate: risk-free rate.
        div_yield: continuous dividend yield.
        notional: face value.
        n_sims: Monte Carlo paths.
        seed: random seed.
        memory_coupon: if True, unpaid coupons accumulate.
    """
    rng = np.random.default_rng(seed)
    n_obs = len(observation_dates)
    dt_list = [observation_dates[0]] + [
        observation_dates[i] - observation_dates[i - 1]
        for i in range(1, n_obs)
    ]

    total_pv = 0.0
    total_life = 0.0
    n_autocalled = 0
    total_coupon = 0.0
    memory_value_total = 0.0
    worst_final = 0.0

    for _ in range(n_sims):
        S = spot
        unpaid_coupons = 0
        path_pv = 0.0
        called = False
        path_coupon = 0.0

        for i, (t, dt) in enumerate(zip(observation_dates, dt_list)):
            drift = (rate - div_yield - 0.5 * vol ** 2) * dt
            diffusion = vol * math.sqrt(dt) * rng.standard_normal()
            S *= math.exp(drift + diffusion)

            df = math.exp(-rate * t)
            perf = S / spot

            # Autocall check
            if perf >= autocall_barrier:
                # Standard autocall payout:
                # - notional (capital redemption)
                # - current period's coupon (the "autocall coupon")
                # - if memory: PLUS any unpaid coupons accumulated so far.
                # Earlier conditional coupons (paid in past iterations
                # when coupon_barrier was met) are ALREADY in path_pv —
                # we ADD here rather than overwrite.
                #
                # Fix T4-AUTO2: pre-fix this branch OVERWROTE path_pv
                # with ``(notional + (i+1)·coupon) × df`` — wiping out
                # earlier coupon accruals AND paying coupons for every
                # period regardless of whether the coupon barrier was
                # met (for the non-memory case).  The redundant
                # ``if memory_coupon: ... else: ...`` block (both
                # branches set ``total_periods_paid = i + 1``) was a
                # symptom: memory and no-memory got the same payout.
                if memory_coupon:
                    extra_coupons = unpaid_coupons + 1
                else:
                    extra_coupons = 1
                autocall_payment = (notional
                                    + extra_coupons * coupon_rate * notional) * df
                path_pv += autocall_payment
                path_coupon += extra_coupons * coupon_rate
                if memory_coupon and unpaid_coupons > 0:
                    memory_value_total += unpaid_coupons * coupon_rate * notional * df
                called = True
                total_life += t
                n_autocalled += 1
                break

            # Coupon check
            if perf >= coupon_barrier:
                if memory_coupon:
                    path_coupon += (unpaid_coupons + 1) * coupon_rate
                    path_pv += (unpaid_coupons + 1) * coupon_rate * notional * df
                    memory_value_total += unpaid_coupons * coupon_rate * notional * df
                    unpaid_coupons = 0
                else:
                    path_coupon += coupon_rate
                    path_pv += coupon_rate * notional * df
            else:
                if memory_coupon:
                    unpaid_coupons += 1

        if not called:
            # Final observation
            T = observation_dates[-1]
            df = math.exp(-rate * T)
            perf = S / spot
            worst_final += perf

            if perf < put_barrier:
                # Capital loss
                path_pv += notional * perf * df  # recovery
            else:
                path_pv += notional * df  # full notional returned

            total_life += T

        total_pv += path_pv
        total_coupon += path_coupon

    price = total_pv / n_sims
    exp_life = total_life / n_sims
    autocall_prob = n_autocalled / n_sims
    avg_coupon = total_coupon / n_sims
    avg_worst = worst_final / max(n_sims - n_autocalled, 1)
    memory_val = memory_value_total / n_sims

    return AdvancedAutocallResult(
        price=price,
        expected_life=exp_life,
        autocall_probability=autocall_prob,
        coupon_expected=avg_coupon,
        memory_coupon_value=memory_val,
        worst_final_pct=avg_worst * 100,
        n_observations=n_obs,
    )


def worst_of_discrete_autocall(
    spots: list[float],
    autocall_barrier: float,
    coupon_barrier: float,
    put_barrier: float,
    coupon_rate: float,
    observation_dates: list[float],
    vols: list[float],
    correlations: list[list[float]],
    rate: float = 0.04,
    div_yields: list[float] | None = None,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
) -> AdvancedAutocallResult:
    """Worst-of autocall on a basket with discrete barriers.

    Autocall/coupon barriers evaluated on the worst performer.

    Args:
        spots: initial spots per asset.
        vols: annualised vol per asset.
        correlations: correlation matrix.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(spots)
    n_obs = len(observation_dates)
    divs = div_yields or [0.02] * n_assets

    # Cholesky decomposition
    corr = np.array(correlations)
    L = np.linalg.cholesky(corr)

    dt_list = [observation_dates[0]] + [
        observation_dates[i] - observation_dates[i - 1]
        for i in range(1, n_obs)
    ]

    total_pv = 0.0
    total_life = 0.0
    n_autocalled = 0
    total_coupon = 0.0
    worst_sum = 0.0

    for _ in range(n_sims):
        S = np.array(spots, dtype=float)
        path_pv = 0.0
        path_coupon = 0.0
        called = False

        for i, (t, dt) in enumerate(zip(observation_dates, dt_list)):
            Z = rng.standard_normal(n_assets)
            corr_Z = L @ Z
            for j in range(n_assets):
                drift = (rate - divs[j] - 0.5 * vols[j] ** 2) * dt
                diffusion = vols[j] * math.sqrt(dt) * corr_Z[j]
                S[j] *= math.exp(drift + diffusion)

            df = math.exp(-rate * t)
            perfs = S / np.array(spots)
            worst = float(np.min(perfs))

            if worst >= autocall_barrier:
                path_pv = (notional + (i + 1) * coupon_rate * notional) * df
                path_coupon = (i + 1) * coupon_rate
                called = True
                total_life += t
                n_autocalled += 1
                break

            if worst >= coupon_barrier:
                path_coupon += coupon_rate
                path_pv += coupon_rate * notional * df

        if not called:
            T = observation_dates[-1]
            df = math.exp(-rate * T)
            perfs = S / np.array(spots)
            worst = float(np.min(perfs))
            worst_sum += worst

            if worst < put_barrier:
                path_pv += notional * worst * df
            else:
                path_pv += notional * df

            total_life += T

        total_pv += path_pv
        total_coupon += path_coupon

    return AdvancedAutocallResult(
        price=total_pv / n_sims,
        expected_life=total_life / n_sims,
        autocall_probability=n_autocalled / n_sims,
        coupon_expected=total_coupon / n_sims,
        memory_coupon_value=0.0,
        worst_final_pct=worst_sum / max(n_sims - n_autocalled, 1) * 100,
        n_observations=n_obs,
    )


def step_down_autocall(
    spot: float,
    initial_barrier: float,
    step_down_per_period: float,
    coupon_barrier: float,
    put_barrier: float,
    coupon_rate: float,
    observation_dates: list[float],
    vol: float,
    rate: float = 0.04,
    div_yield: float = 0.02,
    notional: float = 100.0,
    n_sims: int = 50_000,
    seed: int = 42,
) -> AdvancedAutocallResult:
    """Step-down autocall: barrier decreases at each observation.

    Barrier at observation i: initial_barrier − i × step_down.
    Makes autocall more likely over time.

    Args:
        initial_barrier: starting autocall level (e.g. 1.0).
        step_down_per_period: barrier decrease per period (e.g. 0.05).
    """
    n_obs = len(observation_dates)
    barriers = [initial_barrier - i * step_down_per_period for i in range(n_obs)]

    rng = np.random.default_rng(seed)
    dt_list = [observation_dates[0]] + [
        observation_dates[i] - observation_dates[i - 1]
        for i in range(1, n_obs)
    ]

    total_pv = 0.0
    total_life = 0.0
    n_autocalled = 0
    total_coupon = 0.0

    for _ in range(n_sims):
        S = spot
        path_pv = 0.0
        path_coupon = 0.0
        called = False

        for i, (t, dt) in enumerate(zip(observation_dates, dt_list)):
            drift = (rate - div_yield - 0.5 * vol ** 2) * dt
            diffusion = vol * math.sqrt(dt) * rng.standard_normal()
            S *= math.exp(drift + diffusion)

            df = math.exp(-rate * t)
            perf = S / spot

            if perf >= barriers[i]:
                path_pv = (notional + (i + 1) * coupon_rate * notional) * df
                path_coupon = (i + 1) * coupon_rate
                called = True
                total_life += t
                n_autocalled += 1
                break

            if perf >= coupon_barrier:
                path_coupon += coupon_rate
                path_pv += coupon_rate * notional * df

        if not called:
            T = observation_dates[-1]
            df = math.exp(-rate * T)
            perf = S / spot
            if perf < put_barrier:
                path_pv += notional * perf * df
            else:
                path_pv += notional * df
            total_life += T

        total_pv += path_pv
        total_coupon += path_coupon

    return AdvancedAutocallResult(
        price=total_pv / n_sims,
        expected_life=total_life / n_sims,
        autocall_probability=n_autocalled / n_sims,
        coupon_expected=total_coupon / n_sims,
        memory_coupon_value=0.0,
        worst_final_pct=0.0,
        n_observations=n_obs,
    )
