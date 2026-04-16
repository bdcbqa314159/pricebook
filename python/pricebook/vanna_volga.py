"""Vanna-Volga method: full three-volatility smile adjustment.

Extends :mod:`pricebook.fx_barrier` with:

* :func:`vv_weights` — solve 3×3 system for smile-implied hedge weights.
* :func:`vv_adjust_vanilla` — market-consistent smile for vanilla.
* :func:`vv_adjust_digital` — VV for cash-or-nothing digitals.
* :func:`vv_adjust_touch` — VV for one-touch options.
* :func:`vv_adjust_quanto` — VV for quanto FX options.

References:
    Castagna & Mercurio, *The Vanna-Volga Method for Implied Volatilities*,
    Risk Magazine, 2007.
    Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017.
    Clark, *FX Option Pricing*, Wiley, 2011, Ch. 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.black76 import black76_price, black76_vega, OptionType


# ---- Helper Greeks ----

def _bs_price(S, K, rd, rf, vol, T, is_call=True):
    F = S * math.exp((rd - rf) * T)
    df = math.exp(-rd * T)
    ot = OptionType.CALL if is_call else OptionType.PUT
    return black76_price(F, K, vol, T, df, ot)


def _bs_vega(S, K, rd, rf, vol, T):
    F = S * math.exp((rd - rf) * T)
    df = math.exp(-rd * T)
    return black76_vega(F, K, vol, T, df)


def _bs_vanna(S, K, rd, rf, vol, T, dS=0.005):
    v_up = _bs_vega(S * (1 + dS), K, rd, rf, vol, T)
    v_dn = _bs_vega(S * (1 - dS), K, rd, rf, vol, T)
    return (v_up - v_dn) / (2 * dS * S)


def _bs_volga(S, K, rd, rf, vol, T, dvol=0.001):
    v_up = _bs_vega(S, K, rd, rf, vol + dvol, T)
    v_dn = _bs_vega(S, K, rd, rf, vol - dvol, T)
    return (v_up - v_dn) / (2 * dvol)


def _strike_from_delta(S, rd, rf, vol, T, delta, is_call=True):
    """Invert delta to get strike (spot delta convention)."""
    sign = 1 if is_call else -1
    q = rf
    d1_target = sign * norm.ppf(abs(delta) * math.exp(q * T))
    return S * math.exp(-d1_target * vol * math.sqrt(T) + (rd - rf + 0.5 * vol**2) * T)


# ---- VV weights ----

@dataclass
class VVWeights:
    """Vanna-Volga hedge weights."""
    x_atm: float    # weight on ATM straddle
    x_rr: float     # weight on 25D risk reversal (call vol − put vol)
    x_bf: float     # weight on 25D butterfly
    vega: float
    vanna: float
    volga: float


def vv_weights(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    is_call: bool = True,
) -> VVWeights:
    """Solve 3×3 system for smile-implied VV hedge weights.

    The three hedging instruments: ATM vanilla, 25D call, 25D put.
    For each of vega, vanna, volga, set up the linear system:
        [vega_ATM  vega_25C  vega_25P]   [x_ATM]   [vega_target]
        [vanna_ATM vanna_25C vanna_25P] × [x_RR ]  = [vanna_target]
        [volga_ATM volga_25C volga_25P]   [x_BF ]   [volga_target]

    (Weights on RR and BF are recovered via basis transformation.)

    Args:
        spot: FX spot.
        strike: option strike.
        rate_dom, rate_for: rates.
        vol_atm: ATM vol.
        vol_25d_call / vol_25d_put: market smile vols.
        T: time to expiry.
        is_call: True for call, False for put.
    """
    # ATM strike (delta-neutral)
    K_atm = spot * math.exp((rate_dom - rate_for + 0.5 * vol_atm**2) * T)
    K_25c = _strike_from_delta(spot, rate_dom, rate_for, vol_25d_call, T, 0.25, True)
    K_25p = _strike_from_delta(spot, rate_dom, rate_for, vol_25d_put, T, 0.25, False)

    # Target Greeks at the target strike
    vega_t = _bs_vega(spot, strike, rate_dom, rate_for, vol_atm, T)
    vanna_t = _bs_vanna(spot, strike, rate_dom, rate_for, vol_atm, T)
    volga_t = _bs_volga(spot, strike, rate_dom, rate_for, vol_atm, T)

    # Greeks of hedging instruments (all at ATM vol for consistency)
    def greeks(K):
        v = _bs_vega(spot, K, rate_dom, rate_for, vol_atm, T)
        va = _bs_vanna(spot, K, rate_dom, rate_for, vol_atm, T)
        vo = _bs_volga(spot, K, rate_dom, rate_for, vol_atm, T)
        return v, va, vo

    v_atm, va_atm, vo_atm = greeks(K_atm)
    v_25c, va_25c, vo_25c = greeks(K_25c)
    v_25p, va_25p, vo_25p = greeks(K_25p)

    # Solve 3x3 system
    A = np.array([
        [v_atm, v_25c, v_25p],
        [va_atm, va_25c, va_25p],
        [vo_atm, vo_25c, vo_25p],
    ])
    b = np.array([vega_t, vanna_t, volga_t])

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    return VVWeights(
        x_atm=float(x[0]), x_rr=float(x[1]), x_bf=float(x[2]),
        vega=float(vega_t), vanna=float(vanna_t), volga=float(volga_t),
    )


# ---- VV adjustments ----

@dataclass
class VVResult:
    """Vanna-Volga adjusted price result."""
    bs_price: float             # flat-vol price
    vv_price: float             # smile-adjusted price
    vv_adjustment: float        # vv_price - bs_price
    method: str


def vv_adjust_vanilla(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    is_call: bool = True,
) -> VVResult:
    """Vanna-Volga adjusted vanilla price (market-consistent smile).

    VV_price(K) = BS_price(K, σ_ATM) + smile_cost
    where smile_cost hedges vanna and volga exposures using RR and BF.

    For vanilla K near ATM, VV → BS(σ_ATM).
    For 25D strikes, VV exactly reprices the market smile.
    """
    bs = _bs_price(spot, strike, rate_dom, rate_for, vol_atm, T, is_call)

    K_25c = _strike_from_delta(spot, rate_dom, rate_for, vol_25d_call, T, 0.25, True)
    K_25p = _strike_from_delta(spot, rate_dom, rate_for, vol_25d_put, T, 0.25, False)

    # Market vs BS prices of 25D strikes
    mkt_25c = _bs_price(spot, K_25c, rate_dom, rate_for, vol_25d_call, T, True)
    bs_25c = _bs_price(spot, K_25c, rate_dom, rate_for, vol_atm, T, True)

    mkt_25p = _bs_price(spot, K_25p, rate_dom, rate_for, vol_25d_put, T, False)
    bs_25p = _bs_price(spot, K_25p, rate_dom, rate_for, vol_atm, T, False)

    # VV weights for target strike
    w = vv_weights(spot, strike, rate_dom, rate_for,
                   vol_atm, vol_25d_call, vol_25d_put, T, is_call)

    # Smile cost: weighted sum of market smile costs
    smile_cost = w.x_rr * (mkt_25c - bs_25c) + w.x_bf * (mkt_25p - bs_25p)

    vv_price = bs + smile_cost

    return VVResult(bs, vv_price, smile_cost, "vanna_volga_3x3")


def vv_adjust_digital(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    payout: float = 1.0,
    is_call: bool = True,
) -> VVResult:
    """VV-adjusted cash-or-nothing digital price.

    Digital(K) = -∂Call(K)/∂K ≈ (Call(K-h) - Call(K+h)) / (2h)

    Apply VV to both call prices to get smile-consistent digital.
    """
    h = spot * 0.0005
    call_up = vv_adjust_vanilla(spot, strike + h, rate_dom, rate_for,
                                 vol_atm, vol_25d_call, vol_25d_put, T, True)
    call_dn = vv_adjust_vanilla(spot, strike - h, rate_dom, rate_for,
                                 vol_atm, vol_25d_call, vol_25d_put, T, True)

    # Digital call = -∂C/∂K
    digital_vv = (call_dn.vv_price - call_up.vv_price) / (2 * h) * payout

    # BS digital
    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)
    d2 = (math.log(F / strike) - 0.5 * vol_atm**2 * T) / (vol_atm * math.sqrt(T))
    if is_call:
        digital_bs = df * payout * norm.cdf(d2)
    else:
        digital_bs = df * payout * norm.cdf(-d2)

    if not is_call:
        # Digital put = DF × payout − Digital call
        digital_vv = df * payout - digital_vv

    return VVResult(float(digital_bs), float(max(digital_vv, 0.0)),
                    float(digital_vv - digital_bs), "vv_digital")


def vv_adjust_touch(
    spot: float,
    barrier: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    payout: float = 1.0,
    is_up: bool = True,
) -> VVResult:
    """VV-adjusted one-touch price.

    Uses vanna/volga Greeks of the touch (approximated by digital-like
    behaviour at the barrier) with smile-implied hedging costs.

    For simplicity: treats the touch as a digital with twice the barrier
    probability (reflection principle approximation), then applies VV.

    Args:
        payout: cash payout on touch.
    """
    from pricebook.fx_exotic import fx_one_touch

    # BS touch price (flat vol)
    bs_result = fx_one_touch(spot, barrier, rate_dom, rate_for, vol_atm, T,
                              payout, is_up, seed=42, n_paths=20_000)
    bs_price = bs_result.price

    # Approximate VV adjustment: use digital at barrier as proxy
    # Touch ≈ 2 × digital (reflection principle for ATM)
    digital = vv_adjust_digital(spot, barrier, rate_dom, rate_for,
                                 vol_atm, vol_25d_call, vol_25d_put, T,
                                 payout, is_call=is_up)

    # VV adjustment scaled for touch
    vv_adj = digital.vv_adjustment * 2.0  # rough scaling

    vv_price = max(bs_price + vv_adj, 0.0)

    return VVResult(bs_price, vv_price, vv_price - bs_price, "vv_touch")


def vv_adjust_quanto(
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    vol_quanto: float,
    correlation: float,
    T: float,
    is_call: bool = True,
) -> VVResult:
    """VV-adjusted quanto FX option.

    Quanto adjustment: forward rate is modified by
        F_quanto = F × exp(−ρ × σ_FX × σ_quanto × T)

    Then apply VV with adjusted forward.
    """
    # Quanto-adjusted forward
    F = spot * math.exp((rate_dom - rate_for) * T)
    F_q = F * math.exp(-correlation * vol_atm * vol_quanto * T)

    # Effective spot for VV (inverse of rates)
    spot_q = F_q * math.exp(-(rate_dom - rate_for) * T)

    vv = vv_adjust_vanilla(spot_q, strike, rate_dom, rate_for,
                            vol_atm, vol_25d_call, vol_25d_put, T, is_call)

    bs = _bs_price(spot, strike, rate_dom, rate_for, vol_atm, T, is_call)

    return VVResult(bs, vv.vv_price, vv.vv_price - bs, "vv_quanto")
