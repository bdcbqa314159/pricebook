"""FX Greeks deepening: higher-order sensitivities and vega ladders.

Extends :mod:`pricebook.fx_option` with:

* :func:`fx_vanna` — ∂²V/∂S∂σ (delta's sensitivity to vol).
* :func:`fx_volga` — ∂²V/∂σ² (vega convexity).
* :func:`fx_charm` — ∂Δ/∂t (delta decay).
* :func:`fx_dvega_dspot` / :func:`fx_dvega_dvol` — second-order vega sensitivities.
* :func:`fx_vega_ladder` — vega per (expiry, delta) bucket.
* :func:`fx_smile_consistent_greeks` — VV-adjusted Greeks.

References:
    Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017, Ch. 2.
    Clark, *FX Option Pricing*, Wiley, 2011.
    Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 19.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from pricebook.black76 import black76_price, black76_vega, OptionType


# ---- Helper: BS FX pricing ----

def _fx_call_price(S, K, rd, rf, vol, T):
    F = S * math.exp((rd - rf) * T)
    df = math.exp(-rd * T)
    return black76_price(F, K, vol, T, df, OptionType.CALL)


def _fx_put_price(S, K, rd, rf, vol, T):
    F = S * math.exp((rd - rf) * T)
    df = math.exp(-rd * T)
    return black76_price(F, K, vol, T, df, OptionType.PUT)


def _d1(S, K, rd, rf, vol, T):
    return (math.log(S / K) + (rd - rf + 0.5 * vol**2) * T) / (vol * math.sqrt(T))


def _d2(S, K, rd, rf, vol, T):
    return _d1(S, K, rd, rf, vol, T) - vol * math.sqrt(T)


# ---- Higher-order Greeks ----

def fx_vega(spot, strike, rd, rf, vol, T):
    """Vega = ∂V/∂σ. Same for call and put."""
    if vol <= 0 or T <= 0:
        return 0.0
    d1 = _d1(spot, strike, rd, rf, vol, T)
    return spot * math.exp(-rf * T) * norm.pdf(d1) * math.sqrt(T)


def fx_vanna(spot, strike, rd, rf, vol, T):
    """Vanna = ∂²V/∂S∂σ = ∂Δ/∂σ.

    Vanna = -exp(-rf T) × N'(d1) × d2 / σ
    """
    if vol <= 0 or T <= 0:
        return 0.0
    d1 = _d1(spot, strike, rd, rf, vol, T)
    d2 = d1 - vol * math.sqrt(T)
    return -math.exp(-rf * T) * norm.pdf(d1) * d2 / vol


def fx_volga(spot, strike, rd, rf, vol, T):
    """Volga (vomma) = ∂²V/∂σ² = ∂vega/∂σ.

    Volga = vega × d1 × d2 / σ
    """
    if vol <= 0 or T <= 0:
        return 0.0
    d1 = _d1(spot, strike, rd, rf, vol, T)
    d2 = d1 - vol * math.sqrt(T)
    vega = fx_vega(spot, strike, rd, rf, vol, T)
    return vega * d1 * d2 / vol


def fx_charm(spot, strike, rd, rf, vol, T, is_call=True):
    """Charm = ∂Δ/∂t (time derivative of delta).

    For FX call (spot delta):
        charm = -exp(-rf T) × N'(d1) × [2(r_d-r_f)T - d2 σ√T] / (2T σ√T)
              + rf × exp(-rf T) × N(d1)      (for calls)
              - rf × exp(-rf T) × N(-d1)      (for puts)
    """
    if vol <= 0 or T <= 0:
        return 0.0
    d1 = _d1(spot, strike, rd, rf, vol, T)
    d2 = d1 - vol * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    term1 = -math.exp(-rf * T) * norm.pdf(d1) \
            * (2 * (rd - rf) * T - d2 * vol * sqrt_T) / (2 * T * vol * sqrt_T)

    if is_call:
        term2 = rf * math.exp(-rf * T) * norm.cdf(d1)
    else:
        term2 = -rf * math.exp(-rf * T) * norm.cdf(-d1)

    return term1 + term2


def fx_dvega_dspot(spot, strike, rd, rf, vol, T, dS=0.01):
    """DvegaDspot = ∂vega/∂S via finite difference."""
    v_up = fx_vega(spot * (1 + dS), strike, rd, rf, vol, T)
    v_dn = fx_vega(spot * (1 - dS), strike, rd, rf, vol, T)
    return (v_up - v_dn) / (2 * dS * spot)


def fx_dvega_dvol(spot, strike, rd, rf, vol, T):
    """DvegaDvol = volga."""
    return fx_volga(spot, strike, rd, rf, vol, T)


# ---- Greeks ladder ----

@dataclass
class VegaBucket:
    """Vega in one (tenor, delta) bucket."""
    expiry: float
    delta: float            # 0.25, 0.50, -0.25, -0.10, etc.
    strike: float
    vol: float
    vega: float
    notional_weighted: float


@dataclass
class VegaLadder:
    """FX vega ladder per (expiry, delta) bucket."""
    buckets: list[VegaBucket]
    total_vega: float
    tenor_totals: dict[float, float]  # sum by tenor
    delta_totals: dict[float, float]  # sum by delta bucket


def _strike_from_delta_spot(spot, rd, rf, vol, T, delta, is_call):
    sign = 1 if is_call else -1
    a = min(abs(delta) * math.exp(rf * T), 0.9999)
    d1 = sign * norm.ppf(a)
    return spot * math.exp(-d1 * vol * math.sqrt(T) + (rd - rf + 0.5 * vol**2) * T)


def fx_vega_ladder(
    spot: float,
    rate_dom: float,
    rate_for: float,
    tenors: list[float],
    deltas: list[float],
    vol_fn,
    notional: float = 1.0,
) -> VegaLadder:
    """Compute vega ladder across (tenor, delta) buckets.

    Args:
        tenors: list of expiry times (years).
        deltas: list of absolute delta values (e.g. [0.10, 0.25, 0.50]).
            Each delta produces a call strike; for puts include negative deltas.
        vol_fn: callable(T, K) → vol.
        notional: notional for vega scaling.
    """
    buckets = []
    tenor_totals = {T: 0.0 for T in tenors}
    delta_totals = {d: 0.0 for d in deltas}

    for T in tenors:
        for delta in deltas:
            is_call = delta > 0
            abs_delta = abs(delta)
            # Initial strike estimate using ATM vol
            vol_est = vol_fn(T, spot)
            K = _strike_from_delta_spot(spot, rate_dom, rate_for, vol_est, T, abs_delta, is_call)
            # Refine
            vol = vol_fn(T, K)
            K = _strike_from_delta_spot(spot, rate_dom, rate_for, vol, T, abs_delta, is_call)
            vol = vol_fn(T, K)

            v = fx_vega(spot, K, rate_dom, rate_for, vol, T)
            v_weighted = v * notional
            buckets.append(VegaBucket(T, delta, K, vol, v, v_weighted))

            tenor_totals[T] += v_weighted
            delta_totals[delta] += v_weighted

    total = sum(b.notional_weighted for b in buckets)

    return VegaLadder(buckets, total, tenor_totals, delta_totals)


# ---- Smile-consistent Greeks ----

@dataclass
class SmileGreeksResult:
    """Smile-consistent Greeks via VV."""
    delta_bs: float
    delta_smile: float       # VV-adjusted
    vega_bs: float
    vega_smile: float
    vanna_bs: float
    vanna_smile: float
    volga_bs: float
    volga_smile: float


def fx_smile_consistent_greeks(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    is_call: bool = True,
    ds_bump: float = 0.001,
    dvol_bump: float = 0.001,
) -> SmileGreeksResult:
    """Smile-consistent Greeks via Vanna-Volga.

    Standard BS Greeks assume frozen vol. Smile-consistent Greeks
    account for the vol surface moving with spot (the smile's backbone).

    Computed by bumping spot/vol and re-pricing under VV.
    """
    from pricebook.vanna_volga import vv_adjust_vanilla

    def vv_price(S, atm, c, p):
        return vv_adjust_vanilla(S, strike, rate_dom, rate_for, atm, c, p, T, is_call).vv_price

    # BS Greeks (flat vol)
    d1 = _d1(spot, strike, rate_dom, rate_for, vol_atm, T)
    if is_call:
        delta_bs = math.exp(-rate_for * T) * norm.cdf(d1)
    else:
        delta_bs = -math.exp(-rate_for * T) * norm.cdf(-d1)
    vega_bs = fx_vega(spot, strike, rate_dom, rate_for, vol_atm, T)
    vanna_bs = fx_vanna(spot, strike, rate_dom, rate_for, vol_atm, T)
    volga_bs = fx_volga(spot, strike, rate_dom, rate_for, vol_atm, T)

    # VV-adjusted prices at bumped spot
    p_up = vv_price(spot * (1 + ds_bump), vol_atm, vol_25d_call, vol_25d_put)
    p_dn = vv_price(spot * (1 - ds_bump), vol_atm, vol_25d_call, vol_25d_put)
    delta_smile = (p_up - p_dn) / (2 * ds_bump * spot)

    # VV-adjusted vega
    p_vup = vv_price(spot, vol_atm + dvol_bump, vol_25d_call, vol_25d_put)
    p_vdn = vv_price(spot, vol_atm - dvol_bump, vol_25d_call, vol_25d_put)
    vega_smile = (p_vup - p_vdn) / (2 * dvol_bump)

    # Vanna: d(delta)/d(vol)
    p_up_vup = vv_price(spot * (1 + ds_bump), vol_atm + dvol_bump, vol_25d_call, vol_25d_put)
    p_dn_vup = vv_price(spot * (1 - ds_bump), vol_atm + dvol_bump, vol_25d_call, vol_25d_put)
    delta_vup = (p_up_vup - p_dn_vup) / (2 * ds_bump * spot)
    vanna_smile = (delta_vup - delta_smile) / dvol_bump

    # Volga: d(vega)/d(vol)
    p_vup2 = vv_price(spot, vol_atm + 2 * dvol_bump, vol_25d_call, vol_25d_put)
    vega_vup = (p_vup2 - p_vup) / dvol_bump
    volga_smile = (vega_vup - vega_smile) / dvol_bump

    return SmileGreeksResult(
        delta_bs=delta_bs, delta_smile=delta_smile,
        vega_bs=vega_bs, vega_smile=vega_smile,
        vanna_bs=vanna_bs, vanna_smile=vanna_smile,
        volga_bs=volga_bs, volga_smile=volga_smile,
    )
