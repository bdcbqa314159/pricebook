"""Rates structured products: CMS spread range accrual, callable step-up bonds.

* :func:`cms_spread_range_accrual` — accrues when CMS10-CMS2 in range.
* :func:`callable_step_up_bond` — issuer-callable bond with increasing coupon.
* :func:`inflation_range_accrual` — accrues when inflation in range.

References:
    Brigo & Mercurio (2006). Interest Rate Models, Ch. 13 (CMS).
    Piterbarg (2004). Computing Deltas of Callable LIBOR Exotics in FDM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. CMS spread range accrual
# ---------------------------------------------------------------------------

@dataclass
class CMSSpreadRangeAccrualResult:
    price: float
    expected_accrual_fraction: float
    coupon_rate: float
    range_low: float
    range_high: float
    n_observations: int

    def to_dict(self) -> dict:
        return vars(self)


def cms_spread_range_accrual(
    notional: float,
    coupon_rate: float,
    range_low: float,
    range_high: float,
    cms_long_rate: float,
    cms_short_rate: float,
    cms_long_vol: float,
    cms_short_vol: float,
    correlation: float,
    rate: float,
    T: float,
    n_observations: int = 252,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> CMSSpreadRangeAccrualResult:
    """CMS spread range accrual: accrues coupon when CMS_long - CMS_short is in range.

    Typically: CMS10 - CMS2. Coupon accrues daily if spread is in [range_low, range_high].
    Payoff = notional x coupon_rate x (days_in_range / total_days).

    A bet on curve steepness staying within bounds.

    Args:
        cms_long_rate: initial long CMS rate (e.g., CMS10 = 4%).
        cms_short_rate: initial short CMS rate (e.g., CMS2 = 3.5%).
        cms_long_vol / cms_short_vol: vols of the two CMS rates.
        correlation: correlation between CMS rates.
        range_low / range_high: accrual range for the spread.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_observations
    sqrt_dt = math.sqrt(dt)

    # Simulate correlated CMS rates as OU processes
    long_rate = np.full(n_paths, cms_long_rate)
    short_rate = np.full(n_paths, cms_short_rate)
    days_in_range = np.zeros(n_paths)

    for _ in range(n_observations):
        z1 = rng.standard_normal(n_paths)
        z2 = correlation * z1 + math.sqrt(max(1 - correlation**2, 0.0)) * rng.standard_normal(n_paths)

        # OU-like mean-reverting dynamics
        long_rate += 0.1 * (cms_long_rate - long_rate) * dt + cms_long_vol * sqrt_dt * z1
        short_rate += 0.1 * (cms_short_rate - short_rate) * dt + cms_short_vol * sqrt_dt * z2

        spread = long_rate - short_rate
        in_range = (spread >= range_low) & (spread <= range_high)
        days_in_range += in_range.astype(float)

    accrual_fraction = days_in_range / n_observations
    df = math.exp(-rate * T)
    payoff = notional * coupon_rate * accrual_fraction
    price = df * float(payoff.mean())
    avg_accrual = float(accrual_fraction.mean())

    return CMSSpreadRangeAccrualResult(
        price, avg_accrual, coupon_rate, range_low, range_high, n_observations)


# ---------------------------------------------------------------------------
# 2. Callable step-up bond
# ---------------------------------------------------------------------------

@dataclass
class CallableStepUpResult:
    price: float
    non_callable_price: float
    call_value: float
    expected_call_time: float
    coupon_schedule: list[float]

    def to_dict(self) -> dict:
        return {"price": self.price, "non_callable": self.non_callable_price,
                "call_value": self.call_value, "expected_call_time": self.expected_call_time,
                "coupons": self.coupon_schedule}


def callable_step_up_bond(
    face: float,
    coupon_schedule: list[float],
    rate: float,
    vol: float,
    T: float,
    call_dates_idx: list[int] | None = None,
    call_price: float = 100.0,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> CallableStepUpResult:
    """Callable bond with step-up coupon schedule.

    Issuer can call (redeem early) at call_price on specified dates.
    Coupons increase over time (step-up) to compensate holder for call risk.

    Priced via MC: simulate rates, at each call date check if issuer
    would call (call if bond value > call_price).

    Args:
        coupon_schedule: list of annual coupon rates per period.
        call_dates_idx: period indices where issuer can call (default: all).
        call_price: redemption price on call (per 100 face).
    """
    n_periods = len(coupon_schedule)
    dt = T / n_periods
    if call_dates_idx is None:
        call_dates_idx = list(range(1, n_periods))

    # Non-callable price: deterministic
    nc_price = 0.0
    for i in range(n_periods):
        t = (i + 1) * dt
        nc_price += coupon_schedule[i] * face / 100 * math.exp(-rate * t)
    nc_price += face * math.exp(-rate * T)

    # MC for callable
    from pricebook.mc_migrate import ou_paths
    r_paths = ou_paths(rate, 0.1, rate, vol, T, n_periods, n_paths, seed or 42)

    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    call_time = np.full(n_paths, T)

    for i in range(n_periods):
        t = (i + 1) * dt
        r_t = r_paths[:, i + 1]
        df_t = np.exp(-r_t * t)

        # Coupon payment for alive paths
        coupon = coupon_schedule[i] * face / 100
        pv += np.where(alive, coupon * df_t, 0.0)

        # Call decision at call dates
        if i in call_dates_idx:
            # Issuer calls if remaining bond value > call_price
            remaining_coupons = sum(
                coupon_schedule[j] * face / 100 * np.exp(-r_t * ((j + 1 - i) * dt))
                for j in range(i + 1, n_periods)
            )
            remaining_principal = face * np.exp(-r_t * ((n_periods - i) * dt))
            bond_value = remaining_coupons + remaining_principal

            should_call = alive & (bond_value > call_price * face / 100)
            pv += np.where(should_call, call_price * face / 100 * df_t, 0.0)
            call_time = np.where(should_call & alive, t, call_time)
            alive &= ~should_call

    # Principal at maturity for surviving paths
    pv += np.where(alive, face * np.exp(-r_paths[:, -1] * T), 0.0)

    price = float(pv.mean())
    call_value = nc_price - price
    avg_call = float(call_time.mean())

    return CallableStepUpResult(price, float(nc_price), float(call_value),
                                  avg_call, coupon_schedule)


# ---------------------------------------------------------------------------
# 3. Inflation-linked range accrual
# ---------------------------------------------------------------------------

@dataclass
class InflationRangeAccrualResult:
    price: float
    expected_accrual_fraction: float
    coupon_rate: float
    inflation_range_low: float
    inflation_range_high: float

    def to_dict(self) -> dict:
        return vars(self)


def inflation_range_accrual(
    notional: float,
    coupon_rate: float,
    inflation_range_low: float,
    inflation_range_high: float,
    initial_inflation: float,
    inflation_vol: float,
    rate: float,
    T: float,
    mean_reversion: float = 0.5,
    n_observations: int = 12,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> InflationRangeAccrualResult:
    """Inflation-linked range accrual: coupon accrues when YoY inflation is in range.

    Payoff = notional x coupon_rate x (months_in_range / total_months).

    Bet on inflation staying within bounds (e.g., 2-3%).

    Args:
        inflation_range_low / high: e.g., 0.02 and 0.03 for 2-3%.
        initial_inflation: current YoY inflation rate.
        inflation_vol: vol of inflation rate.
        mean_reversion: OU mean reversion of inflation.
    """
    from pricebook.mc_migrate import ou_paths

    inf_paths = ou_paths(initial_inflation, mean_reversion, initial_inflation,
                          inflation_vol, T, n_observations, n_paths, seed or 42)

    monitoring = inf_paths[:, 1:]
    in_range = (monitoring >= inflation_range_low) & (monitoring <= inflation_range_high)
    accrual_fraction = in_range.sum(axis=1) / n_observations

    df = math.exp(-rate * T)
    payoff = notional * coupon_rate * accrual_fraction
    price = df * float(payoff.mean())

    return InflationRangeAccrualResult(
        price, float(accrual_fraction.mean()), coupon_rate,
        inflation_range_low, inflation_range_high)
