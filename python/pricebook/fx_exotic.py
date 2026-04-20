"""FX exotic options: touches, lookbacks, Asian FX, range accruals.

Extends :mod:`pricebook.fx_barrier` with path-dependent exotics:

* :func:`fx_one_touch` / :func:`fx_no_touch` — barrier hit / no-hit payoffs.
* :func:`fx_double_touch` / :func:`fx_double_no_touch` — two-barrier versions.
* :func:`fx_lookback_floating` / :func:`fx_lookback_fixed` — Goldman-Sosin-Gatto.
* :func:`fx_asian_geometric` / :func:`fx_asian_arithmetic` — path-averaged FX.
* :func:`fx_range_accrual` — accrues fixed amount per day FX in range.
* :func:`fx_accumulator` — knock-out discount accumulator (KODA).

References:
    Wystup, *FX Options and Structured Products*, Wiley, 2nd ed., 2017.
    Clark, *FX Option Pricing*, Wiley, 2011.
    Goldman, Sosin & Gatto, *Path-Dependent Options*, JF, 1979.
    Kemna & Vorst, *A Pricing Method for Options on Average Asset Values*, JBF, 1990.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.black76 import OptionType, black76_price


# ---- Touch options ----

@dataclass
class TouchResult:
    """One-touch / no-touch option result."""
    price: float
    is_touch: bool      # True for touch, False for no-touch
    is_double: bool     # True for double-barrier
    barrier: float
    barrier_upper: float | None


def fx_one_touch(
    spot: float,
    barrier: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    payout: float = 1.0,
    is_up: bool = True,
    n_paths: int = 10_000,
    n_steps: int = 200,
    seed: int | None = 42,
) -> TouchResult:
    """One-touch: pays `payout` at expiry if spot touches barrier before T.

    Priced via MC with discrete daily-like observation. For a path-dependent
    touch, analytical closed forms (Reiner-Rubinstein) exist but are sensitive
    to drift sign and barrier location; MC is simple and robust.

    Args:
        spot: current FX spot.
        barrier: touch barrier level.
        rate_dom: domestic rate (discounting).
        rate_for: foreign rate (drift adjustment).
        vol: FX volatility.
        T: time to expiry.
        payout: cash payout on touch.
        is_up: True if barrier above spot.
    """
    if T <= 0:
        touched = (is_up and spot >= barrier) or (not is_up and spot <= barrier)
        return TouchResult(payout if touched else 0.0, True, False, barrier, None)

    if (is_up and spot >= barrier) or (not is_up and spot <= barrier):
        # Already touched — paid at expiry (present value of payout)
        return TouchResult(payout * math.exp(-rate_dom * T), True, False, barrier, None)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)

    S = np.full(n_paths, spot)
    touched = np.zeros(n_paths, dtype=bool)

    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        if is_up:
            touched |= S >= barrier
        else:
            touched |= S <= barrier

    df = math.exp(-rate_dom * T)
    price = df * payout * float(touched.mean())
    return TouchResult(price, True, False, barrier, None)


def fx_no_touch(
    spot: float,
    barrier: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    payout: float = 1.0,
    is_up: bool = True,
) -> TouchResult:
    """No-touch: pays `payout` if spot never touches barrier.

    NT = DF × payout − OT  (one-touch parity).
    """
    df = math.exp(-rate_dom * T)
    ot = fx_one_touch(spot, barrier, rate_dom, rate_for, vol, T, payout, is_up)
    price = df * payout - ot.price
    return TouchResult(max(price, 0.0), False, False, barrier, None)


def fx_double_no_touch(
    spot: float,
    barrier_low: float,
    barrier_high: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    payout: float = 1.0,
    n_terms: int = 20,
) -> TouchResult:
    """Double no-touch: pays if spot stays in [barrier_low, barrier_high].

    Hui (1996) series solution:
        DNT = Σ (sin(kπlog(S/L)/log(U/L)) / k) × exp(...) × payout

    For simplicity we use a series truncation.
    """
    if spot <= barrier_low or spot >= barrier_high:
        return TouchResult(0.0, False, True, barrier_low, barrier_high)

    L = math.log(spot / barrier_low)
    U = math.log(barrier_high / barrier_low)
    if U <= 0:
        return TouchResult(0.0, False, True, barrier_low, barrier_high)

    alpha = -0.5 * (2 * (rate_dom - rate_for) / vol**2 - 1)
    beta = -0.25 * (2 * (rate_dom - rate_for) / vol**2 - 1)**2 - 2 * rate_dom / vol**2

    total = 0.0
    for k in range(1, n_terms + 1):
        term = 2 * k * math.pi / (U**2) * (1 - (-1)**k * math.exp(alpha * (math.log(barrier_high/spot) - math.log(spot/barrier_low))))
        term *= math.sin(k * math.pi * L / U)
        term *= math.exp(0.5 * (beta - (k * math.pi / U)**2) * vol**2 * T)
        if k * k > 1e4:
            break
        total += term / (k * math.pi / U)**2 * (k * math.pi / U)

    # Fallback: MC if series unstable
    if total <= 0 or total > 1:
        total = _dnt_mc(spot, barrier_low, barrier_high, rate_dom, rate_for, vol, T)

    price = payout * math.exp(-rate_dom * T) * total
    return TouchResult(float(max(price, 0.0)), False, True, barrier_low, barrier_high)


def _dnt_mc(spot, lo, hi, rd, rf, vol, T, n_paths=20_000, n_steps=100, seed=42):
    """MC fallback for double no-touch."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (rd - rf - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)
    S = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        alive &= (S > lo) & (S < hi)
    return float(alive.mean())


def fx_double_touch(
    spot: float,
    barrier_low: float,
    barrier_high: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    payout: float = 1.0,
) -> TouchResult:
    """Double touch: pays if spot touches either barrier.

    DT = DF × payout − DNT.
    """
    df = math.exp(-rate_dom * T)
    dnt = fx_double_no_touch(spot, barrier_low, barrier_high, rate_dom, rate_for, vol, T, payout)
    price = df * payout - dnt.price
    return TouchResult(max(price, 0.0), True, True, barrier_low, barrier_high)


# ---- Lookback options ----

@dataclass
class LookbackResult:
    """Lookback option result."""
    price: float
    is_floating_strike: bool
    is_call: bool


def fx_lookback_floating(
    spot: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    is_call: bool = True,
    running_extreme: float | None = None,
) -> LookbackResult:
    """Floating-strike lookback: payoff = S_T − m (call) or M − S_T (put).

    Goldman-Sosin-Gatto (1979) closed form for continuously-monitored
    lognormal dynamics.

    Args:
        spot: current spot.
        rate_dom, rate_for: domestic/foreign rates.
        vol: FX vol.
        T: time to expiry.
        is_call: call pays S_T − min; put pays max − S_T.
        running_extreme: observed running min (call) or max (put); defaults to spot.
    """
    if vol <= 0 or T <= 0:
        return LookbackResult(0.0, True, is_call)

    m = running_extreme if running_extreme is not None else spot
    a = rate_dom - rate_for
    b = a - 0.5 * vol**2
    sigma_T = vol * math.sqrt(T)

    if is_call:
        # S × exp(-r_f T) × N(d1) − m × exp(-r_d T) × N(d2) + smile term
        if m >= spot:
            # min ≤ S always
            pass
        d1 = (math.log(spot / m) + (a + 0.5 * vol**2) * T) / sigma_T
        d2 = d1 - sigma_T
        eps = vol**2 / (2 * a) if abs(a) > 1e-10 else sigma_T**2 / 2

        term1 = spot * math.exp(-rate_for * T) * norm.cdf(d1)
        term2 = m * math.exp(-rate_dom * T) * norm.cdf(d2)
        # Extra term from reflection
        if abs(a) > 1e-10:
            term3 = spot * math.exp(-rate_for * T) * eps * (
                norm.cdf(d1) - math.exp(-a * T) * (m/spot)**(2*a/vol**2) * norm.cdf(d1 - 2*a*math.sqrt(T)/vol)
            )
        else:
            term3 = spot * math.exp(-rate_for * T) * sigma_T * norm.pdf(d1)

        price = term1 - term2 + term3
    else:
        # put: max − S_T
        if m <= spot:
            m = spot
        d1 = (math.log(spot / m) + (a + 0.5 * vol**2) * T) / sigma_T
        d2 = d1 - sigma_T
        eps = vol**2 / (2 * a) if abs(a) > 1e-10 else sigma_T**2 / 2

        term1 = m * math.exp(-rate_dom * T) * norm.cdf(-d2)
        term2 = spot * math.exp(-rate_for * T) * norm.cdf(-d1)
        if abs(a) > 1e-10:
            term3 = spot * math.exp(-rate_for * T) * eps * (
                math.exp(-a * T) * (m/spot)**(2*a/vol**2) * norm.cdf(-d1 + 2*a*math.sqrt(T)/vol) - norm.cdf(-d1)
            )
        else:
            term3 = spot * math.exp(-rate_for * T) * sigma_T * norm.pdf(d1)

        price = term1 - term2 + term3

    return LookbackResult(float(max(price, 0.0)), True, is_call)


def fx_lookback_fixed(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    is_call: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> LookbackResult:
    """Fixed-strike lookback: max(max(S) − K, 0) call or max(K − min(S), 0) put.

    Priced via MC (no simple closed form for fixed-strike).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)
    df = math.exp(-rate_dom * T)

    S = np.full(n_paths, spot)
    extremes = np.full(n_paths, spot)

    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        if is_call:
            extremes = np.maximum(extremes, S)
        else:
            extremes = np.minimum(extremes, S)

    if is_call:
        payoff = np.maximum(extremes - strike, 0.0)
    else:
        payoff = np.maximum(strike - extremes, 0.0)

    price = df * float(payoff.mean())
    return LookbackResult(price, False, is_call)


# ---- Asian options ----

@dataclass
class AsianResult:
    """Asian option result."""
    price: float
    is_geometric: bool
    is_call: bool
    n_fixings: int


def fx_asian_geometric(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    n_fixings: int = 12,
    is_call: bool = True,
) -> AsianResult:
    """Geometric Asian FX option — closed form.

    The geometric average of lognormals is lognormal, so Black-76 applies
    with adjusted vol and forward.

    σ_G² = σ² × (2N + 1) / (6(N + 1))  (continuous limit: σ²/3)
    μ_G = (r_d − r_f − σ²/2) × T × (N + 1) / (2N) + adjustment
    """
    if n_fixings < 1:
        n_fixings = 1

    # Adjusted vol for geometric mean
    sigma_g = vol * math.sqrt((2 * n_fixings + 1) / (6 * (n_fixings + 1)))

    # Adjusted drift for geometric average
    # The geometric mean of N fixings at t_i = iT/N has:
    # E[G] = S₀ × exp(avg_drift × T) where
    # avg_drift = (r−q−σ²/2) × (N+1)/(2N) + σ_G²/2
    avg_factor = (n_fixings + 1) / (2 * n_fixings)
    mu_g = (rate_dom - rate_for - 0.5 * vol**2) * avg_factor + 0.5 * sigma_g**2

    # Effective forward for the geometric average
    F_g = spot * math.exp(mu_g * T)

    opt_type = OptionType.CALL if is_call else OptionType.PUT
    df = math.exp(-rate_dom * T)
    price = black76_price(F_g, strike, sigma_g, T, df, opt_type)

    return AsianResult(float(price), True, is_call, n_fixings)


def fx_asian_arithmetic(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    n_fixings: int = 12,
    is_call: bool = True,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> AsianResult:
    """Arithmetic Asian FX — MC with geometric control variate.

    Reduces variance by ~10-100x vs plain MC by using closed-form
    geometric Asian as the control.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_fixings
    drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)
    df = math.exp(-rate_dom * T)

    S = np.full(n_paths, spot)
    sum_S = np.zeros(n_paths)
    sum_log_S = np.zeros(n_paths)

    for _ in range(n_fixings):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        sum_S += S
        sum_log_S += np.log(S)

    arith_avg = sum_S / n_fixings
    geo_avg = np.exp(sum_log_S / n_fixings)

    if is_call:
        arith_payoff = np.maximum(arith_avg - strike, 0.0)
        geo_payoff = np.maximum(geo_avg - strike, 0.0)
    else:
        arith_payoff = np.maximum(strike - arith_avg, 0.0)
        geo_payoff = np.maximum(strike - geo_avg, 0.0)

    # Control variate: exact geometric price
    geo_exact = fx_asian_geometric(spot, strike, rate_dom, rate_for,
                                    vol, T, n_fixings, is_call).price
    geo_mc = df * float(geo_payoff.mean())

    # Adjusted arithmetic price = arith_mc + (geo_exact - geo_mc)
    arith_mc = df * float(arith_payoff.mean())
    price = arith_mc + (geo_exact - geo_mc)

    return AsianResult(float(max(price, 0.0)), False, is_call, n_fixings)


# ---- Range accrual and accumulator ----

@dataclass
class RangeAccrualResult:
    """Range accrual result."""
    price: float
    accrual_rate: float    # fraction of days expected in range
    n_observations: int
    range_low: float
    range_high: float


def fx_range_accrual(
    spot: float,
    range_low: float,
    range_high: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    coupon_per_period: float = 0.01,
    n_observations: int = 252,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> RangeAccrualResult:
    """FX range accrual: pays coupon × fraction_of_days_in_range.

    Path-dependent on daily observations of spot.

    Args:
        range_low/range_high: accrual range.
        coupon_per_period: coupon per observation if in range.
        n_observations: number of daily observations.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_observations
    drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)
    df = math.exp(-rate_dom * T)

    S = np.full(n_paths, spot)
    in_range_count = np.zeros(n_paths)

    for _ in range(n_observations):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        in_range_count += ((S >= range_low) & (S <= range_high)).astype(float)

    accrual_rate = float(in_range_count.mean() / n_observations)
    price = df * coupon_per_period * in_range_count.mean()

    return RangeAccrualResult(float(price), accrual_rate, n_observations,
                              range_low, range_high)


@dataclass
class AccumulatorResult:
    """KODA (knock-out discount accumulator) result."""
    price: float
    knock_out_prob: float
    expected_accumulated: float    # expected notional accumulated
    strike: float
    barrier: float


def fx_accumulator(
    spot: float,
    strike: float,
    barrier_up: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    daily_notional: float = 1.0,
    leverage: float = 2.0,
    n_observations: int = 252,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> AccumulatorResult:
    """FX accumulator (KODA): buy FX daily at discount, knock out on barrier.

    Daily: if spot < strike → accumulate daily_notional × leverage units
           if strike ≤ spot < barrier → accumulate daily_notional units
           if spot ≥ barrier → knock out, stop accumulating

    PV = E[Σ daily_notional × (S_t − strike) × DF_t]

    Args:
        strike: accumulation strike (below spot typically).
        barrier_up: upper knock-out barrier (above spot).
        daily_notional: base units per day.
        leverage: multiplier when spot < strike (downside gearing).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_observations
    drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)

    S = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    accumulated = np.zeros(n_paths)

    for step in range(n_observations):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)

        # Check knock-out
        ko = S >= barrier_up
        alive &= ~ko

        # Daily accrual for alive paths
        t = (step + 1) * dt
        df_t = math.exp(-rate_dom * t)

        # Notional multiplier: leverage if spot < strike
        n_today = np.where(S < strike, daily_notional * leverage, daily_notional)
        # Only active paths accumulate
        n_today = np.where(alive, n_today, 0.0)

        # P&L = (S - K) × notional, discounted
        pv += (S - strike) * n_today * df_t
        accumulated += n_today

    ko_prob = float(1 - alive.mean())
    price = float(pv.mean())
    expected_acc = float(accumulated.mean())

    return AccumulatorResult(price, ko_prob, expected_acc, strike, barrier_up)
