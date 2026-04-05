"""FX barrier options and Vanna-Volga smile-consistent pricing.

FX barriers via 1D PDE with FX-specific boundary conditions.
Vanna-Volga adjusts flat-vol barrier prices for smile using
ATM, 25D RR, and 25D BF market quotes.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType, black76_price, black76_vega
from pricebook.finite_difference import fd_barrier_knockout, fd_barrier_knockin


def fx_barrier_pde(
    spot: float,
    strike: float,
    barrier: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    is_up: bool = False,
    is_knock_in: bool = False,
    option_type: OptionType = OptionType.CALL,
    n_spot: int = 200,
    n_time: int = 200,
) -> float:
    """FX barrier option via 1D PDE.

    Uses the existing FD engine with r = rate_dom, q = rate_for.
    """
    barrier_kw = {"barrier_upper": barrier} if is_up else {"barrier_lower": barrier}

    if is_knock_in:
        return fd_barrier_knockin(
            spot=spot, strike=strike,
            rate=rate_dom, vol=vol, T=T,
            option_type=option_type,
            div_yield=rate_for, n_spot=n_spot, n_time=n_time,
            **barrier_kw,
        )
    return fd_barrier_knockout(
        spot=spot, strike=strike,
        rate=rate_dom, vol=vol, T=T,
        option_type=option_type,
        div_yield=rate_for, n_spot=n_spot, n_time=n_time,
        **barrier_kw,
    )


# ---------------------------------------------------------------------------
# Vanna-Volga method
# ---------------------------------------------------------------------------


def _bs_price(S, K, rd, rf, vol, T, is_call=True):
    """Black-Scholes FX option price."""
    F = S * math.exp((rd - rf) * T)
    df = math.exp(-rd * T)
    ot = OptionType.CALL if is_call else OptionType.PUT
    return black76_price(F, K, vol, T, df, ot)


def _bs_vega(S, K, rd, rf, vol, T):
    """Black-Scholes vega."""
    F = S * math.exp((rd - rf) * T)
    df = math.exp(-rd * T)
    return black76_vega(F, K, vol, T, df)


def _bs_vanna(S, K, rd, rf, vol, T, dS=0.01):
    """Vanna = d(vega)/d(spot) via finite difference."""
    v_up = _bs_vega(S * (1 + dS), K, rd, rf, vol, T)
    v_dn = _bs_vega(S * (1 - dS), K, rd, rf, vol, T)
    return (v_up - v_dn) / (2 * dS * S)


def _bs_volga(S, K, rd, rf, vol, T, dvol=0.001):
    """Volga = d(vega)/d(vol) via finite difference."""
    v_up = _bs_vega(S, K, rd, rf, vol + dvol, T)
    v_dn = _bs_vega(S, K, rd, rf, vol - dvol, T)
    return (v_up - v_dn) / (2 * dvol)


def vanna_volga_barrier(
    spot: float,
    strike: float,
    barrier: float,
    rate_dom: float,
    rate_for: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    T: float,
    is_up: bool = False,
    is_knock_in: bool = False,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Vanna-Volga smile-consistent barrier price.

    Adjusts the flat-vol (ATM) barrier price using the cost of hedging
    smile risk with ATM, 25D call, and 25D put vanillas.

    VV_price = BS_barrier(ATM) + vanna_cost + volga_cost
    """
    # Flat-vol barrier price
    bs_barrier = fx_barrier_pde(
        spot, strike, barrier, rate_dom, rate_for, vol_atm, T,
        is_up, is_knock_in, option_type,
    )

    # Smile Greeks of the barrier (using ATM vol)
    vanna_bar = _bs_vanna(spot, strike, rate_dom, rate_for, vol_atm, T)
    volga_bar = _bs_volga(spot, strike, rate_dom, rate_for, vol_atm, T)

    # Cost of vanna: proportional to risk-reversal
    rr = vol_25d_call - vol_25d_put
    vanna_cost = 0.5 * vanna_bar * rr

    # Cost of volga: proportional to butterfly
    bf = 0.5 * (vol_25d_call + vol_25d_put) - vol_atm
    volga_cost = 0.5 * volga_bar * bf

    return max(bs_barrier + vanna_cost + volga_cost, 0.0)
