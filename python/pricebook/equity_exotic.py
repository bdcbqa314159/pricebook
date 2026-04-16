"""Equity exotic options: barriers with smile, digitals, lookbacks, compound.

* :func:`equity_barrier_smile` — barrier with VV smile adjustment.
* :func:`equity_digital_cash` — cash-or-nothing digital with smile.
* :func:`equity_digital_asset` — asset-or-nothing digital.
* :func:`equity_lookback_floating` — Goldman-Sosin-Gatto floating strike.
* :func:`equity_lookback_fixed` — fixed-strike lookback (MC).
* :func:`equity_compound_option` — call-on-call (Geske 1979).

References:
    Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 26.
    Wilmott, *Paul Wilmott on Quantitative Finance*, Vol. 2, Ch. 22.
    Geske, *The Valuation of Compound Options*, JFE, 1979.
    Goldman, Sosin & Gatto, *Path Dependent Options*, JF, 1979.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.black76 import black76_price, OptionType


# ---- Barrier with smile ----

@dataclass
class EquityBarrierResult:
    """Equity barrier option result."""
    price: float
    bs_price: float
    vv_adjustment: float
    barrier: float
    is_knock_in: bool
    is_up: bool


def equity_barrier_smile(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend_yield: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    is_up: bool = True,
    is_knock_in: bool = False,
    is_call: bool = True,
) -> EquityBarrierResult:
    """Equity barrier option with Vanna-Volga smile adjustment.

    For equity, rate_dom = risk-free rate and rate_for = dividend yield.
    Reuses :func:`pricebook.fx_barrier.vanna_volga_barrier` machinery.

    Args:
        spot: equity spot price.
        strike: option strike.
        barrier: barrier level.
        rate: risk-free rate.
        dividend_yield: continuous dividend yield.
        vol_atm, vol_25d_call, vol_25d_put: market smile vols.
        T: time to expiry.
        is_up: True if barrier above spot.
        is_knock_in: True for knock-in, False for knock-out.
        is_call: True for call payoff, False for put.
    """
    from pricebook.fx_barrier import vanna_volga_barrier, fx_barrier_pde

    opt_type = OptionType.CALL if is_call else OptionType.PUT

    bs = fx_barrier_pde(
        spot, strike, barrier, rate, dividend_yield, vol_atm, T,
        is_up=is_up, is_knock_in=is_knock_in, option_type=opt_type,
    )

    vv = vanna_volga_barrier(
        spot, strike, barrier, rate, dividend_yield,
        vol_atm, vol_25d_call, vol_25d_put, T,
        is_up=is_up, is_knock_in=is_knock_in, option_type=opt_type,
    )

    return EquityBarrierResult(
        price=float(vv),
        bs_price=float(bs),
        vv_adjustment=float(vv - bs),
        barrier=barrier,
        is_knock_in=is_knock_in,
        is_up=is_up,
    )


# ---- Digital options ----

@dataclass
class DigitalResult:
    """Digital option result."""
    price: float
    probability: float      # RN probability of finishing ITM
    digital_type: str       # "cash" or "asset"
    is_call: bool


def equity_digital_cash(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    payout: float = 1.0,
    is_call: bool = True,
    smile_vols: tuple[float, float, float] | None = None,
) -> DigitalResult:
    """Cash-or-nothing digital: pays `payout` if S_T > K (call) or S_T < K (put).

    Flat-vol price: payout × DF × N(d2) for call, N(-d2) for put.

    Smile-consistent pricing via call spread approximation:
        D ≈ [C(K-h) − C(K+h)] / (2h) × payout
    (applied automatically when `smile_vols` provided).

    Args:
        smile_vols: optional (atm, 25d_call, 25d_put) for VV smile.
    """
    if vol <= 0 or T <= 0:
        itm = (is_call and spot > strike) or (not is_call and spot < strike)
        return DigitalResult(payout if itm else 0.0, 1.0 if itm else 0.0,
                             "cash", is_call)

    F = spot * math.exp((rate - dividend_yield) * T)
    df = math.exp(-rate * T)
    d2 = (math.log(F / strike) - 0.5 * vol**2 * T) / (vol * math.sqrt(T))

    prob = norm.cdf(d2) if is_call else norm.cdf(-d2)
    price_bs = df * payout * prob

    if smile_vols is None:
        return DigitalResult(float(price_bs), float(prob), "cash", is_call)

    # Smile-adjusted via call spread
    from pricebook.vanna_volga import vv_adjust_vanilla
    atm, c25, p25 = smile_vols
    h = spot * 0.0005
    up = vv_adjust_vanilla(spot, strike + h, rate, dividend_yield,
                            atm, c25, p25, T, True).vv_price
    dn = vv_adjust_vanilla(spot, strike - h, rate, dividend_yield,
                            atm, c25, p25, T, True).vv_price

    digital_call = (dn - up) / (2 * h) * payout
    if is_call:
        price = digital_call
    else:
        price = df * payout - digital_call

    return DigitalResult(float(max(price, 0.0)), float(prob), "cash", is_call)


def equity_digital_asset(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    is_call: bool = True,
) -> DigitalResult:
    """Asset-or-nothing digital: pays S_T if S_T > K (call) or S_T < K (put).

    Flat-vol price: spot × e^{-qT} × N(d1) for call.
    """
    if vol <= 0 or T <= 0:
        itm = (is_call and spot > strike) or (not is_call and spot < strike)
        return DigitalResult(spot if itm else 0.0, 1.0 if itm else 0.0,
                             "asset", is_call)

    F = spot * math.exp((rate - dividend_yield) * T)
    d1 = (math.log(F / strike) + 0.5 * vol**2 * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    if is_call:
        price = spot * math.exp(-dividend_yield * T) * norm.cdf(d1)
        prob = norm.cdf(d2)
    else:
        price = spot * math.exp(-dividend_yield * T) * norm.cdf(-d1)
        prob = norm.cdf(-d2)

    return DigitalResult(float(price), float(prob), "asset", is_call)


# ---- Lookback ----

@dataclass
class EquityLookbackResult:
    """Equity lookback option result."""
    price: float
    is_floating: bool
    is_call: bool


def equity_lookback_floating(
    spot: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    is_call: bool = True,
    running_extreme: float | None = None,
) -> EquityLookbackResult:
    """Floating-strike lookback for equity: payoff S_T − min(S) (call).

    Goldman-Sosin-Gatto (1979) closed form.

    For equity, "rate_for" = dividend_yield (continuous).
    """
    from pricebook.fx_exotic import fx_lookback_floating
    r = fx_lookback_floating(spot, rate, dividend_yield, vol, T, is_call,
                              running_extreme)
    return EquityLookbackResult(r.price, True, is_call)


def equity_lookback_fixed(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    is_call: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 200,
    seed: int | None = 42,
) -> EquityLookbackResult:
    """Fixed-strike lookback: payoff max(max(S) − K, 0) (call)."""
    from pricebook.fx_exotic import fx_lookback_fixed
    r = fx_lookback_fixed(spot, strike, rate, dividend_yield, vol, T,
                           is_call, n_paths, n_steps, seed)
    return EquityLookbackResult(r.price, False, is_call)


# ---- Compound options ----

@dataclass
class CompoundResult:
    """Compound option result (Geske 1979)."""
    price: float
    underlying_type: str        # "call" or "put"
    outer_type: str             # "call" or "put" (e.g. "call on call")


def equity_compound_option(
    spot: float,
    strike_outer: float,
    strike_underlying: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T1: float,                  # time to compound option expiry
    T2: float,                  # time to underlying expiry (T2 > T1)
    is_outer_call: bool = True,
    is_underlying_call: bool = True,
    n_iter: int = 50,
) -> CompoundResult:
    """Compound option: option on an option.

    Geske (1979) closed form for call-on-call (the most common):

        C_compound = S × e^{-qT2} × M(a₁, b₁; √(T1/T2))
                     − K2 × e^{-rT2} × M(a₂, b₂; √(T1/T2))
                     − K1 × e^{-rT1} × N(a₂)

    where:
        S* = spot price at T1 at which C(S*, K2, T2-T1) = K1
        a₁ = [log(S/S*) + (r-q+σ²/2)T1] / (σ√T1),  a₂ = a₁ − σ√T1
        b₁ = [log(S/K2) + (r-q+σ²/2)T2] / (σ√T2),  b₂ = b₁ − σ√T2
        M(·) = bivariate normal CDF

    For simplicity here we use the approach with S* found by Newton iteration.

    Args:
        strike_outer: strike of the outer option (K1).
        strike_underlying: strike of the inner option (K2).
        T1: outer option expiry.
        T2: underlying option expiry.
        is_outer_call: True for "call on X", False for "put on X".
        is_underlying_call: True for X = call, False for X = put.
    """
    if T1 >= T2 or T1 <= 0 or T2 <= 0:
        raise ValueError("Require 0 < T1 < T2")
    if vol <= 0:
        raise ValueError("Require positive vol")

    # Find S* : C(S*, K2, T2-T1) = K1 via bisection
    tau = T2 - T1

    def inner_call(S):
        F = S * math.exp((rate - dividend_yield) * tau)
        df = math.exp(-rate * tau)
        ot = OptionType.CALL if is_underlying_call else OptionType.PUT
        return black76_price(F, strike_underlying, vol, tau, df, ot)

    # Bisection for S*
    lo, hi = 0.01, spot * 10
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        v = inner_call(mid) - strike_outer
        if is_underlying_call:
            if v > 0:
                hi = mid
            else:
                lo = mid
        else:
            if v > 0:
                lo = mid
            else:
                hi = mid
    S_star = 0.5 * (lo + hi)

    # Use MC for compound pricing (bivariate normal closed form is involved;
    # MC is clean and works for all combinations).
    rng = np.random.default_rng(42)
    n_paths = 50_000
    sqrt_T1 = math.sqrt(T1)

    z1 = rng.standard_normal(n_paths)
    S_T1 = spot * np.exp((rate - dividend_yield - 0.5 * vol**2) * T1 + vol * sqrt_T1 * z1)

    # Inner option value at T1 for each path
    inner_vals = np.array([inner_call(s) for s in S_T1])

    # Outer payoff at T1
    if is_outer_call:
        outer_payoff = np.maximum(inner_vals - strike_outer, 0.0)
    else:
        outer_payoff = np.maximum(strike_outer - inner_vals, 0.0)

    df_T1 = math.exp(-rate * T1)
    price = df_T1 * float(outer_payoff.mean())

    outer_str = "call" if is_outer_call else "put"
    under_str = "call" if is_underlying_call else "put"

    return CompoundResult(float(price), under_str, outer_str)
