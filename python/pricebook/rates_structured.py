"""Rates structured products.

* :func:`cms_spread_range_accrual` — accrues when CMS10-CMS2 in range.
* :func:`callable_step_up_bond` — issuer-callable bond with increasing coupon.
* :func:`inflation_range_accrual` — accrues when inflation in range.
* :func:`zc_swaption` — option on a zero-coupon swap rate.
* :func:`inverse_floater` — fixed minus floating (leveraged rate bet).
* :func:`capped_floater` — FRN with coupon cap.

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


# ---------------------------------------------------------------------------
# 4. Zero-coupon swaption
# ---------------------------------------------------------------------------

@dataclass
class ZCSwaptionResult:
    """Zero-coupon swaption pricing result."""
    price: float
    forward_zc_rate: float
    vol: float
    delta: float

    def to_dict(self) -> dict:
        return vars(self)


def zc_swaption(
    forward_zc_rate: float,
    strike: float,
    vol: float,
    T_option: float,
    T_swap: float,
    notional: float = 1_000_000,
    rate: float = 0.04,
    is_payer: bool = True,
) -> ZCSwaptionResult:
    """Option on a zero-coupon swap rate via Black-76.

    A ZC swaption pays (ZC_rate(T_option, T_swap) - K) at T_swap,
    discounted. The zero-coupon rate is the single-period rate
    from T_option to T_swap.

    Args:
        forward_zc_rate: forward zero-coupon rate from T_option to T_swap.
        strike: strike rate.
        vol: Black vol of the ZC rate.
        T_option: option expiry (years).
        T_swap: swap maturity (years).
        is_payer: True for payer (right to pay fixed), False for receiver.
    """
    from pricebook.black76 import black76_price, OptionType

    df = math.exp(-rate * T_swap)
    tau = T_swap - T_option
    opt_type = OptionType.CALL if is_payer else OptionType.PUT
    unit_price = black76_price(forward_zc_rate, strike, vol, T_option, df, opt_type)
    price = float(unit_price * notional * tau)

    # Delta via bump
    bump = forward_zc_rate * 0.01
    up = black76_price(forward_zc_rate + bump, strike, vol, T_option, df, opt_type)
    delta = float((up - unit_price) / bump * notional * tau)

    return ZCSwaptionResult(price, forward_zc_rate, vol, delta)


# ---------------------------------------------------------------------------
# 5. Inverse floater
# ---------------------------------------------------------------------------

@dataclass
class InverseFloaterResult:
    """Inverse floater pricing result."""
    price: float
    fixed_rate: float
    leverage: float
    expected_coupon: float
    floor: float

    def to_dict(self) -> dict:
        return vars(self)


def inverse_floater(
    notional: float,
    fixed_rate: float,
    leverage: float,
    rate: float,
    vol: float,
    T: float,
    n_periods: int = 10,
    floor: float = 0.0,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> InverseFloaterResult:
    """Inverse floater: coupon = max(fixed - leverage × floating, floor).

    Long duration bet: benefits from falling rates.

    Coupon at each period: max(fixed_rate - leverage × r(t), floor).

    Args:
        fixed_rate: fixed component (e.g. 0.08 for 8%).
        leverage: multiplier on floating rate (e.g. 1.0 or 2.0).
        floor: minimum coupon (typically 0).
    """
    from pricebook.mc_migrate import ou_paths

    r_paths = ou_paths(rate, 0.1, rate, vol, T, n_periods, n_paths, seed or 42)
    dt = T / n_periods

    pv = np.zeros(n_paths)
    total_coupon = np.zeros(n_paths)

    for i in range(n_periods):
        t = (i + 1) * dt
        r_t = r_paths[:, i + 1]
        coupon = np.maximum(fixed_rate - leverage * r_t, floor)
        total_coupon += coupon
        df_t = np.exp(-r_t * t)
        pv += notional * coupon * dt * df_t

    # Principal at maturity
    pv += notional * np.exp(-r_paths[:, -1] * T)

    price = float(pv.mean())
    avg_coupon = float(total_coupon.mean() / n_periods)

    return InverseFloaterResult(price, fixed_rate, leverage, avg_coupon, floor)


# ---------------------------------------------------------------------------
# 6. Capped floater
# ---------------------------------------------------------------------------

@dataclass
class CappedFloaterResult:
    """Capped floater pricing result."""
    price: float
    cap_rate: float
    spread: float
    expected_coupon: float
    cap_cost: float         # price difference vs uncapped FRN

    def to_dict(self) -> dict:
        return vars(self)


def capped_floater(
    notional: float,
    spread: float,
    cap_rate: float,
    rate: float,
    vol: float,
    T: float,
    n_periods: int = 10,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> CappedFloaterResult:
    """FRN with coupon cap: coupon = min(floating + spread, cap).

    Issuer is long a cap (benefits when rates rise above cap_rate).
    Holder gives up upside above cap in exchange for higher initial spread.

    Args:
        spread: fixed spread over floating (e.g. 0.005 = 50bp).
        cap_rate: maximum coupon rate.
    """
    from pricebook.mc_migrate import ou_paths

    r_paths = ou_paths(rate, 0.1, rate, vol, T, n_periods, n_paths, seed or 42)
    dt = T / n_periods

    pv_capped = np.zeros(n_paths)
    pv_uncapped = np.zeros(n_paths)
    total_coupon = np.zeros(n_paths)

    for i in range(n_periods):
        t = (i + 1) * dt
        r_t = r_paths[:, i + 1]
        floating_coupon = r_t + spread
        capped_coupon = np.minimum(floating_coupon, cap_rate)
        total_coupon += capped_coupon
        df_t = np.exp(-r_t * t)
        pv_capped += notional * capped_coupon * dt * df_t
        pv_uncapped += notional * floating_coupon * dt * df_t

    # Principal
    pv_capped += notional * np.exp(-r_paths[:, -1] * T)
    pv_uncapped += notional * np.exp(-r_paths[:, -1] * T)

    price = float(pv_capped.mean())
    uncapped_price = float(pv_uncapped.mean())
    avg_coupon = float(total_coupon.mean() / n_periods)
    cap_cost = uncapped_price - price

    return CappedFloaterResult(price, cap_rate, spread, avg_coupon, float(cap_cost))
