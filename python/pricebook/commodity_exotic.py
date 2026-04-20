"""Commodity exotic options: barriers, lookbacks, Asians, quantos.

* :func:`commodity_barrier_smile` — barrier w/ VV smile for commodities.
* :func:`commodity_lookback` — floating/fixed strike lookback.
* :func:`commodity_asian_monthly` — monthly averaging (standard oil/gas settlement).
* :func:`quanto_commodity_option` — FX-hedged commodity payoff.

References:
    Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 5-7.
    Hull, *Options, Futures, and Other Derivatives*, Ch. 26.
    Kemna & Vorst, *A Pricing Method for Options Based on Average Asset Values*,
    JBF, 1990.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


# ---- Commodity barrier with smile ----

@dataclass
class CommodityBarrierResult:
    """Commodity barrier option result."""
    price: float
    bs_price: float
    vv_adjustment: float
    barrier: float
    is_knock_in: bool
    is_up: bool


def commodity_barrier_smile(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    convenience_yield: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    is_up: bool = True,
    is_knock_in: bool = False,
    is_call: bool = True,
) -> CommodityBarrierResult:
    """Commodity barrier option with Vanna-Volga smile adjustment.

    For commodities, the drift is (r − δ) where δ is the convenience yield,
    analogous to dividend yield for equity. Reuses the fx_barrier machinery.

    Args:
        spot: commodity spot.
        strike, barrier: option levels.
        rate: risk-free rate.
        convenience_yield: commodity convenience yield δ.
        vol_atm, vol_25d_call, vol_25d_put: market smile.
        T: time to expiry.
        is_up / is_knock_in / is_call: standard barrier flags.
    """
    from pricebook.fx_barrier import vanna_volga_barrier, fx_barrier_pde

    opt_type = OptionType.CALL if is_call else OptionType.PUT

    bs = fx_barrier_pde(
        spot, strike, barrier, rate, convenience_yield, vol_atm, T,
        is_up=is_up, is_knock_in=is_knock_in, option_type=opt_type,
    )

    vv = vanna_volga_barrier(
        spot, strike, barrier, rate, convenience_yield,
        vol_atm, vol_25d_call, vol_25d_put, T,
        is_up=is_up, is_knock_in=is_knock_in, option_type=opt_type,
    )

    return CommodityBarrierResult(
        price=float(vv),
        bs_price=float(bs),
        vv_adjustment=float(vv - bs),
        barrier=barrier,
        is_knock_in=is_knock_in,
        is_up=is_up,
    )


# ---- Commodity lookback ----

@dataclass
class CommodityLookbackResult:
    """Commodity lookback option result."""
    price: float
    is_floating: bool
    is_call: bool
    n_observations: int


def commodity_lookback(
    spot: float,
    rate: float,
    convenience_yield: float,
    vol: float,
    T: float,
    is_call: bool = True,
    is_floating: bool = True,
    strike: float | None = None,
    n_observations: int = 252,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> CommodityLookbackResult:
    """Commodity lookback option.

    Floating strike call: payoff = S_T − min(S_t) (ATM always).
    Fixed strike call: payoff = max(max(S_t) − K, 0).

    For continuous monitoring, Goldman-Sosin-Gatto closed form applies (floating).
    For discrete monitoring (typical: daily close), MC is used.

    Args:
        is_floating: True for floating strike, False for fixed.
        strike: required if is_floating=False.
        n_observations: number of discrete observations.
    """
    if is_floating:
        # Use GSG closed form via fx_lookback_floating
        from pricebook.fx_exotic import fx_lookback_floating
        r = fx_lookback_floating(spot, rate, convenience_yield, vol, T, is_call)
        return CommodityLookbackResult(r.price, True, is_call, n_observations)

    # Fixed strike: discrete MC
    if strike is None:
        strike = spot

    rng = np.random.default_rng(seed)
    dt = T / n_observations
    drift = (rate - convenience_yield - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)

    S = np.full(n_paths, spot)
    extreme = np.full(n_paths, spot)

    for _ in range(n_observations):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        if is_call:
            extreme = np.maximum(extreme, S)
        else:
            extreme = np.minimum(extreme, S)

    if is_call:
        payoff = np.maximum(extreme - strike, 0.0)
    else:
        payoff = np.maximum(strike - extreme, 0.0)

    df = math.exp(-rate * T)
    price = df * float(payoff.mean())

    return CommodityLookbackResult(price, False, is_call, n_observations)


# ---- Commodity Asian (monthly) ----

@dataclass
class CommodityAsianResult:
    """Commodity Asian option result."""
    price: float
    is_arithmetic: bool
    n_fixings: int
    is_call: bool
    control_variate_adjustment: float


def commodity_asian_monthly(
    spot: float,
    strike: float,
    rate: float,
    convenience_yield: float,
    vol: float,
    T: float,
    n_fixings: int = 12,
    is_call: bool = True,
    is_arithmetic: bool = True,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> CommodityAsianResult:
    """Commodity Asian option — monthly averaging (standard oil/gas settlement).

    Arithmetic average is the market convention but lacks a closed form;
    priced via MC with geometric-average control variate for variance reduction.

    Geometric average is lognormal → closed form via modified Black-76.

    Args:
        n_fixings: number of monthly fixings.
        is_arithmetic: True for arithmetic avg, False for geometric (closed form).
    """
    # Geometric Asian closed form (for control and for geometric pricing)
    sigma_g = vol * math.sqrt((2 * n_fixings + 1) / (6 * (n_fixings + 1)))
    avg_factor = (n_fixings + 1) / (2 * n_fixings)
    mu_g = (rate - convenience_yield - 0.5 * vol**2) * avg_factor + 0.5 * sigma_g**2
    F_g = spot * math.exp(mu_g * T)
    df = math.exp(-rate * T)
    opt_type = OptionType.CALL if is_call else OptionType.PUT
    geo_exact = black76_price(F_g, strike, sigma_g, T, df, opt_type)

    if not is_arithmetic:
        # Return geometric closed form
        return CommodityAsianResult(
            price=float(geo_exact),
            is_arithmetic=False,
            n_fixings=n_fixings,
            is_call=is_call,
            control_variate_adjustment=0.0,
        )

    # Arithmetic: MC with geometric control variate
    rng = np.random.default_rng(seed)
    dt = T / n_fixings
    drift = (rate - convenience_yield - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)

    S = np.full(n_paths, spot)
    sum_S = np.zeros(n_paths)
    sum_log_S = np.zeros(n_paths)

    for _ in range(n_fixings):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        sum_S += S
        sum_log_S += np.log(S)

    arith = sum_S / n_fixings
    geo = np.exp(sum_log_S / n_fixings)

    if is_call:
        arith_payoff = np.maximum(arith - strike, 0.0)
        geo_payoff = np.maximum(geo - strike, 0.0)
    else:
        arith_payoff = np.maximum(strike - arith, 0.0)
        geo_payoff = np.maximum(strike - geo, 0.0)

    arith_mc = df * float(arith_payoff.mean())
    geo_mc = df * float(geo_payoff.mean())

    # Control variate correction
    cv_adjustment = geo_exact - geo_mc
    price = arith_mc + cv_adjustment

    return CommodityAsianResult(
        price=float(max(price, 0.0)),
        is_arithmetic=True,
        n_fixings=n_fixings,
        is_call=is_call,
        control_variate_adjustment=float(cv_adjustment),
    )


# ---- Quanto commodity ----

@dataclass
class QuantoCommodityResult:
    """Quanto commodity option result."""
    price: float
    native_forward: float       # forward in commodity's local currency
    quanto_forward: float       # quanto-adjusted forward
    quanto_adjustment: float
    correlation: float


def quanto_commodity_option(
    spot: float,                # commodity spot in native (USD typically)
    strike: float,              # strike in quanto currency
    fx_spot: float,             # FX spot (quanto currency / native)
    rate_quanto: float,         # rate in quanto currency (for discounting)
    rate_native: float,         # rate in native (for commodity drift)
    convenience_yield: float,
    vol_commodity: float,
    vol_fx: float,
    correlation: float,
    T: float,
    is_call: bool = True,
) -> QuantoCommodityResult:
    """Quanto commodity option: payoff in different currency than commodity.

    E.g. EUR-denominated WTI crude call: strike and payout in EUR while
    commodity is priced in USD.

    Quanto adjustment (Reiner 1992):
        F_quanto = F_native × exp(−ρ × σ_commodity × σ_fx × T)

    Then price via Black-76 on the quanto forward.

    Args:
        spot: commodity spot in native currency.
        fx_spot: FX from native to quanto (e.g. 0.92 EUR/USD).
        rate_quanto: quanto currency rate (for DF).
        rate_native: native rate (impacts forward).
        correlation: correlation between commodity and FX.
    """
    # Native forward
    F_native = spot * math.exp((rate_native - convenience_yield) * T)

    # Quanto adjustment
    quanto_adj = math.exp(-correlation * vol_commodity * vol_fx * T)
    F_quanto = F_native * quanto_adj

    # Option price in quanto currency (no FX conversion for payoff)
    df = math.exp(-rate_quanto * T)
    opt_type = OptionType.CALL if is_call else OptionType.PUT
    price = black76_price(F_quanto, strike, vol_commodity, T, df, opt_type)

    return QuantoCommodityResult(
        price=float(price),
        native_forward=float(F_native),
        quanto_forward=float(F_quanto),
        quanto_adjustment=float(F_quanto - F_native),
        correlation=correlation,
    )
