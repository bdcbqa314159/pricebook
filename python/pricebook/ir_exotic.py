"""Exotic IR products: TARNs, snowballs, callable range accruals, ratchets, flexi-swaps.

Phase IR1 slices 234-236 consolidated.

* :func:`tarn_price` — Target Redemption Note.
* :func:`snowball_price` — snowball / inverse floater.
* :func:`callable_range_accrual` — range accrual with issuer call.
* :func:`ratchet_cap` — cap with resetting strike.
* :func:`flexi_swap` — holder chooses which periods to exercise.

References:
    Brigo & Mercurio, *Interest Rate Models*, Ch. 15-16.
    Andersen & Piterbarg, *Interest Rate Modeling*, Vol. III, Ch. 19.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- TARN (Target Redemption Note) ----

@dataclass
class TARNResult:
    """TARN pricing result."""
    price: float
    expected_life: float
    target_hit_probability: float
    mean_total_coupon: float


def tarn_price(
    notional: float,
    coupon_rate: float,
    target: float,
    maturity_years: int,
    flat_rate: float = 0.05,
    rate_vol: float = 0.01,
    frequency: int = 4,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> TARNResult:
    """Price a Target Redemption Note via MC.

    Pays coupon each period. When cumulative coupon reaches the target,
    the note redeems at par. If target is never hit, redeems at maturity.

    Path-dependent: early redemption depends on the rate path.

    Args:
        coupon_rate: fixed coupon rate per period.
        target: cumulative coupon target (e.g. 0.20 = 20% of notional).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    pv = np.zeros(n_paths)
    cum_coupon = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    redemption_time = np.full(n_paths, float(maturity_years))

    r = np.full(n_paths, flat_rate)

    for i in range(1, n_periods + 1):
        t = i * dt
        # Rate dynamics (simple OU)
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        r = r + 0.5 * (flat_rate - r) * dt + rate_vol * dW

        # Coupon this period
        coupon = coupon_rate * notional * dt
        df = np.exp(-flat_rate * t)

        # Pay coupon to alive paths
        pv[alive] += df * coupon
        cum_coupon[alive] += coupon_rate * dt

        # Check target hit
        target_hit = alive & (cum_coupon >= target)
        if np.any(target_hit):
            pv[target_hit] += df * notional  # redeem at par
            redemption_time[target_hit] = t
            alive[target_hit] = False

    # Maturity redemption for paths that didn't hit target
    df_T = math.exp(-flat_rate * maturity_years)
    pv[alive] += df_T * notional

    price = float(pv.mean()) / notional * 100
    expected_life = float(redemption_time.mean())
    hit_prob = float((~alive).mean())  # fraction that hit target before maturity

    return TARNResult(price, expected_life, hit_prob, float(cum_coupon.mean()))


# ---- Snowball / inverse floater ----

@dataclass
class SnowballResult:
    """Snowball pricing result."""
    price: float
    mean_final_coupon: float
    mean_total_coupon: float


def snowball_price(
    notional: float,
    initial_coupon: float,
    spread: float,
    maturity_years: int,
    flat_rate: float = 0.05,
    rate_vol: float = 0.01,
    floor: float = 0.0,
    frequency: int = 4,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> SnowballResult:
    """Price a snowball note via MC.

    Coupon_n = max(coupon_{n-1} + spread − r_n, floor).

    When rates fall, the coupon ratchets up (snowball effect).
    When rates rise, the coupon is floored at zero.

    Args:
        initial_coupon: first coupon rate.
        spread: added to previous coupon each period.
        floor: minimum coupon rate (default 0).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    r = np.full(n_paths, flat_rate)
    coupon = np.full(n_paths, initial_coupon)
    pv = np.zeros(n_paths)
    total_coupon = np.zeros(n_paths)

    for i in range(1, n_periods + 1):
        t = i * dt
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        r = r + 0.5 * (flat_rate - r) * dt + rate_vol * dW

        # Snowball coupon
        coupon = np.maximum(coupon + spread * dt - r * dt, floor)
        payment = coupon * notional * dt
        df = math.exp(-flat_rate * t)
        pv += df * payment
        total_coupon += coupon * dt

    # Principal at maturity
    df_T = math.exp(-flat_rate * maturity_years)
    pv += df_T * notional

    price = float(pv.mean()) / notional * 100

    return SnowballResult(price, float(coupon.mean()), float(total_coupon.mean()))


# ---- Callable range accrual ----

@dataclass
class CallableRangeAccrualResult:
    """Callable range accrual result."""
    price: float
    non_callable_price: float
    call_value: float
    accrual_fraction: float


def callable_range_accrual(
    notional: float,
    coupon_rate: float,
    lower: float,
    upper: float,
    maturity_years: int,
    call_start_year: int = 1,
    call_price: float = 100.0,
    flat_rate: float = 0.05,
    rate_vol: float = 0.01,
    frequency: int = 4,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> CallableRangeAccrualResult:
    """Callable range accrual via MC with LSM call decision.

    Coupon accrues when the rate is in [lower, upper].
    Issuer can call at call_price after call_start_year.

    Backward LSM: issuer calls when estimated continuation > par.

    Args:
        lower / upper: range boundaries for rate.
        call_price: per 100 face.
        call_start_year: first callable year.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    # Simulate rate paths
    r_all = np.zeros((n_paths, n_periods + 1))
    r_all[:, 0] = flat_rate
    for i in range(1, n_periods + 1):
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        r_all[:, i] = r_all[:, i - 1] + 0.5 * (flat_rate - r_all[:, i - 1]) * dt + rate_vol * dW

    # Compute non-callable price and per-period coupons
    pv_nc = np.zeros(n_paths)
    coupons = np.zeros((n_paths, n_periods))
    total_in_range = 0

    for i in range(1, n_periods + 1):
        t = i * dt
        in_range = (r_all[:, i] >= lower) & (r_all[:, i] <= upper)
        total_in_range += in_range.sum()
        coupons[:, i - 1] = np.where(in_range, coupon_rate * notional * dt, 0.0)
        df = math.exp(-flat_rate * t)
        pv_nc += df * coupons[:, i - 1]

    df_T = math.exp(-flat_rate * maturity_years)
    pv_nc += df_T * notional

    # LSM backward pass for issuer call decision
    # V[p] = value-to-go for the issuer (what they owe the investor)
    V = np.full(n_paths, df_T * notional)  # terminal principal

    call_decision = np.zeros((n_paths, n_periods), dtype=bool)
    call_start_period = call_start_year * frequency

    for i in range(n_periods, 0, -1):
        t = i * dt
        df = math.exp(-flat_rate * t)
        V += df * coupons[:, i - 1]

        if i >= call_start_period:
            par_val = call_price * notional / 100 * df

            # LSM: regress V on rate
            r_i = r_all[:, i]
            r_norm = (r_i - r_i.mean()) / max(r_i.std(), 1e-10)
            basis = np.column_stack([np.ones(n_paths), r_norm, r_norm**2])
            try:
                coeffs = np.linalg.lstsq(basis, V, rcond=None)[0]
                est_cont = basis @ coeffs
            except np.linalg.LinAlgError:
                est_cont = V

            # Issuer calls when continuation > par (saves money)
            call_decision[:, i - 1] = est_cont > par_val

    # Forward pass: apply call decisions
    alive = np.ones(n_paths, dtype=bool)
    pv_c = np.zeros(n_paths)

    for i in range(1, n_periods + 1):
        t = i * dt
        df = math.exp(-flat_rate * t)
        pv_c += np.where(alive, df * coupons[:, i - 1], 0.0)

        if i >= call_start_period:
            issuer_calls = alive & call_decision[:, i - 1]
            pv_c += np.where(issuer_calls, call_price * notional / 100 * df, 0.0)
            alive &= ~issuer_calls

    pv_c += np.where(alive, df_T * notional, 0.0)

    nc_price = float(pv_nc.mean()) / notional * 100
    c_price = float(pv_c.mean()) / notional * 100
    accrual = total_in_range / (n_paths * n_periods)

    return CallableRangeAccrualResult(c_price, nc_price, nc_price - c_price, accrual)


# ---- Ratchet cap ----

@dataclass
class RatchetCapResult:
    """Ratchet cap result."""
    price: float
    standard_cap_price: float
    ratchet_premium: float


def ratchet_cap(
    notional: float,
    initial_strike: float,
    maturity_years: int,
    flat_rate: float = 0.05,
    rate_vol: float = 0.01,
    frequency: int = 4,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> RatchetCapResult:
    """Price a ratchet cap: strike resets to max(previous fixing, floor).

    Each caplet's strike = max(previous rate fixing, initial_strike).
    In a rising rate environment, the ratchet cap is more valuable
    than a standard cap because the strike follows the rate up.

    Wait — actually ratchet means the strike ratchets DOWN:
    strike_n = min(previous_fixing, strike_{n-1}). This makes it
    more valuable because the strike drops when rates drop.

    Convention here: strike resets to previous fixing if lower.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    r = np.full(n_paths, flat_rate)
    strike = np.full(n_paths, initial_strike)
    pv_ratchet = np.zeros(n_paths)
    pv_standard = np.zeros(n_paths)

    for i in range(1, n_periods + 1):
        t = i * dt
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        r = r + 0.5 * (flat_rate - r) * dt + rate_vol * dW

        df = math.exp(-flat_rate * t)

        # Ratchet caplet payoff
        ratchet_payoff = np.maximum(r - strike, 0.0) * notional * dt
        pv_ratchet += df * ratchet_payoff

        # Standard caplet
        standard_payoff = np.maximum(r - initial_strike, 0.0) * notional * dt
        pv_standard += df * standard_payoff

        # Update ratchet strike: reset to previous fixing
        strike = np.minimum(strike, r)

    price = float(pv_ratchet.mean())
    std_price = float(pv_standard.mean())

    return RatchetCapResult(price, std_price, price - std_price)


# ---- Flexi-swap ----

@dataclass
class FlexiSwapResult:
    """Flexi-swap result."""
    price: float
    vanilla_swap_price: float
    optionality_value: float
    mean_exercises: float


def flexi_swap(
    notional: float,
    fixed_rate: float,
    maturity_years: int,
    max_exercises: int,
    flat_rate: float = 0.05,
    rate_vol: float = 0.01,
    frequency: int = 4,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> FlexiSwapResult:
    """Price a flexi-swap: holder chooses which periods to exercise.

    The holder can exercise (enter swap for that period) up to
    max_exercises times. Rational exercise: exercise when the
    floating rate exceeds the fixed rate.

    Args:
        max_exercises: maximum number of periods the holder can exercise.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    r = np.full(n_paths, flat_rate)
    pv_flexi = np.zeros(n_paths)
    pv_vanilla = np.zeros(n_paths)
    exercises_used = np.zeros(n_paths, dtype=int)

    for i in range(1, n_periods + 1):
        t = i * dt
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        r = r + 0.5 * (flat_rate - r) * dt + rate_vol * dW

        df = math.exp(-flat_rate * t)
        swap_cf = (r - fixed_rate) * notional * dt

        # Vanilla: always exercise
        pv_vanilla += df * swap_cf

        # Flexi: exercise if profitable and exercises remaining
        can_exercise = exercises_used < max_exercises
        should_exercise = can_exercise & (swap_cf > 0)
        pv_flexi[should_exercise] += df * swap_cf[should_exercise]
        exercises_used[should_exercise] += 1

    price = float(pv_flexi.mean())
    vanilla = float(pv_vanilla.mean())

    return FlexiSwapResult(price, vanilla, price - vanilla,
                           float(exercises_used.mean()))
