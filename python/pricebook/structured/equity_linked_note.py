"""Equity-linked notes (ELNs): closed-form and Monte Carlo pricers.

* :func:`buffered_eln`   — downside buffer; loss only below buffer level.
* :func:`capped_eln`     — upside participation capped at a strike level.
* :func:`bear_eln`       — inverse / bear note: profits when index falls.
* :func:`digital_eln`    — enhanced coupon if index above barrier at maturity.
* :func:`twin_win_eln`   — profits from both up and down moves (barrier protected).
* :func:`worst_of_eln`   — MC basket note; coupon unless any underlying breaches barrier.

References:
    Deng (2006). Equity-Linked Notes: Modelling and Pricing.
    Das (2005). Structured Products, Vol. 1, Ch. 4-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import OptionType, _norm_cdf, black76_price


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ELNResult:
    """Pricing output shared by all ELN functions."""
    price: float
    bond_floor: float
    option_component: float
    coupon_value: float
    max_loss: float
    participation_rate: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward(spot: float, rate: float, div_yield: float, T: float) -> float:
    return spot * math.exp((rate - div_yield) * T)


def _df(rate: float, T: float) -> float:
    return math.exp(-rate * T)


def _bs_put(spot: float, rate: float, div_yield: float, vol: float, T: float, strike: float) -> float:
    """Black-Scholes (via Black-76 on forward) put price."""
    F = _forward(spot, rate, div_yield, T)
    df = _df(rate, T)
    return black76_price(F, strike, vol, T, df, OptionType.PUT)


def _bs_call(spot: float, rate: float, div_yield: float, vol: float, T: float, strike: float) -> float:
    """Black-Scholes (via Black-76 on forward) call price."""
    F = _forward(spot, rate, div_yield, T)
    df = _df(rate, T)
    return black76_price(F, strike, vol, T, df, OptionType.CALL)


def _digital_call_bs(spot: float, rate: float, div_yield: float, vol: float, T: float, strike: float) -> float:
    """Cash-or-nothing digital call: pays $1 if S_T > K."""
    if T <= 0 or vol <= 0:
        return 1.0 if spot > strike else 0.0
    F = _forward(spot, rate, div_yield, T)
    df = _df(rate, T)
    sqrt_t = math.sqrt(T)
    d2 = (math.log(F / strike) - 0.5 * vol * vol * T) / (vol * sqrt_t)
    return df * _norm_cdf(d2)


# ---------------------------------------------------------------------------
# 1. Buffered ELN
# ---------------------------------------------------------------------------

def buffered_eln(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    buffer: float,
    coupon: float,
    notional: float = 1000.0,
) -> ELNResult:
    """ELN with downside buffer.

    Investor receives ``coupon`` at maturity unless the index falls below the
    buffer level (expressed as a fraction of spot, e.g. 0.80 for 80%).
    Below the buffer, losses are pass-through.

    Structure::

        Price = ZCB + short put@spot - short put@(buffer * spot)

    where the put spread replicates the loss-below-buffer profile.

    Args:
        spot:     Current index level.
        rate:     Continuously compounded risk-free rate.
        vol:      Index implied volatility.
        T:        Time to maturity in years.
        buffer:   Buffer level as fraction of spot (e.g. 0.80).
        coupon:   Fixed coupon paid at maturity (per notional unit).
        notional: Face value.

    Returns:
        :class:`ELNResult` with decomposed price components.
    """
    df = _df(rate, T)
    bond_floor = notional * df  # ZCB repays principal

    # Short put spread: short put at spot, long put at buffer*spot
    # Investor is short the put spread -> price deducted
    K_low = buffer * spot
    put_atm = _bs_put(spot, rate, 0.0, vol, T, spot)
    put_buf = _bs_put(spot, rate, 0.0, vol, T, K_low)
    put_spread = put_atm - put_buf  # cost of short spread (positive)

    coupon_pv = coupon * notional * df
    option_component = -put_spread * notional / spot  # scaled to notional

    price = bond_floor + coupon_pv + option_component
    max_loss = (1.0 - buffer) * notional

    return ELNResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_component=float(option_component),
        coupon_value=float(coupon_pv),
        max_loss=float(max_loss),
        participation_rate=0.0,
    )


# ---------------------------------------------------------------------------
# 2. Capped ELN
# ---------------------------------------------------------------------------

def capped_eln(
    spot: float,
    rate: float,
    div_yield: float,
    vol: float,
    T: float,
    cap: float,
    participation: float = 1.0,
    notional: float = 1000.0,
) -> ELNResult:
    """Upside participation ELN capped at ``cap`` (fraction of spot).

    Structure::

        Price = ZCB + participation * (call@spot - call@(cap*spot))

    Args:
        spot:          Current index level.
        rate:          Continuously compounded risk-free rate.
        div_yield:     Continuous dividend yield.
        vol:           Index implied volatility.
        T:             Time to maturity in years.
        cap:           Cap level as fraction of spot (e.g. 1.20 for +20%).
        participation: Upside participation factor (default 1.0).
        notional:      Face value.

    Returns:
        :class:`ELNResult` with decomposed price components.
    """
    df = _df(rate, T)
    bond_floor = notional * df

    call_atm = _bs_call(spot, rate, div_yield, vol, T, spot)
    call_cap = _bs_call(spot, rate, div_yield, vol, T, cap * spot)
    call_spread = call_atm - call_cap

    option_component = participation * call_spread * notional / spot
    price = bond_floor + option_component
    max_gain = (cap - 1.0) * participation * notional

    return ELNResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_component=float(option_component),
        coupon_value=0.0,
        max_loss=float(notional),
        participation_rate=float(participation),
    )


# ---------------------------------------------------------------------------
# 3. Bear ELN
# ---------------------------------------------------------------------------

def bear_eln(
    spot: float,
    rate: float,
    div_yield: float,
    vol: float,
    T: float,
    notional: float = 1000.0,
) -> ELNResult:
    """Inverse / bear ELN: pays when the index falls.

    Structure::

        Price = ZCB + put@spot

    The put gives leveraged exposure to index declines.

    Args:
        spot:      Current index level.
        rate:      Continuously compounded risk-free rate.
        div_yield: Continuous dividend yield.
        vol:       Index implied volatility.
        T:         Time to maturity in years.
        notional:  Face value.

    Returns:
        :class:`ELNResult` with decomposed price components.
    """
    df = _df(rate, T)
    bond_floor = notional * df

    put_atm = _bs_put(spot, rate, div_yield, vol, T, spot)
    option_component = put_atm * notional / spot

    price = bond_floor + option_component

    return ELNResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_component=float(option_component),
        coupon_value=0.0,
        max_loss=float(notional),
        participation_rate=1.0,
    )


# ---------------------------------------------------------------------------
# 4. Digital ELN
# ---------------------------------------------------------------------------

def digital_eln(
    spot: float,
    rate: float,
    div_yield: float,
    vol: float,
    T: float,
    barrier: float,
    coupon_if_above: float,
    notional: float = 1000.0,
) -> ELNResult:
    """Digital (binary) ELN.

    Pays ``coupon_if_above * notional`` at maturity if the index closes above
    ``barrier``.  Principal is always returned (capital protected).

    Structure::

        Price = ZCB + coupon_if_above * notional * digital_call(barrier)

    Args:
        spot:           Current index level.
        rate:           Continuously compounded risk-free rate.
        div_yield:      Continuous dividend yield.
        vol:            Index implied volatility.
        T:              Time to maturity in years.
        barrier:        Barrier level (absolute, e.g. 1.05 * spot).
        coupon_if_above: Enhanced coupon rate paid if index > barrier (e.g. 0.08).
        notional:       Face value.

    Returns:
        :class:`ELNResult` with decomposed price components.
    """
    df = _df(rate, T)
    bond_floor = notional * df

    dig = _digital_call_bs(spot, rate, div_yield, vol, T, barrier)
    coupon_pv = coupon_if_above * notional * dig
    option_component = coupon_pv

    price = bond_floor + option_component

    return ELNResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_component=float(option_component),
        coupon_value=float(coupon_pv),
        max_loss=0.0,
        participation_rate=0.0,
    )


# ---------------------------------------------------------------------------
# 5. Twin-Win ELN
# ---------------------------------------------------------------------------

def twin_win_eln(
    spot: float,
    rate: float,
    div_yield: float,
    vol: float,
    T: float,
    barrier: float,
    participation: float = 1.0,
    notional: float = 1000.0,
) -> ELNResult:
    """Twin-win ELN: profits from both upside and downside moves.

    Gains participation on |return| unless a lower barrier is breached, at
    which point the product converts to a direct index tracker (losses flow
    through).

    Approximate closed-form structure (barrier approximated as European)::

        Price = ZCB
              + participation * call@spot         (upside)
              + participation * put@spot           (downside flip)
              - participation * put@barrier        (knock-out adjustment)

    Args:
        spot:          Current index level.
        rate:          Continuously compounded risk-free rate.
        div_yield:     Continuous dividend yield.
        vol:           Index implied volatility.
        T:             Time to maturity in years.
        barrier:       Lower barrier level (absolute, e.g. 0.70 * spot).
        participation: Participation on both sides (default 1.0).
        notional:      Face value.

    Returns:
        :class:`ELNResult` with decomposed price components.
    """
    df = _df(rate, T)
    bond_floor = notional * df

    call_atm = _bs_call(spot, rate, div_yield, vol, T, spot)
    put_atm = _bs_put(spot, rate, div_yield, vol, T, spot)
    put_bar = _bs_put(spot, rate, div_yield, vol, T, barrier)

    option_component = participation * (call_atm + put_atm - put_bar) * notional / spot
    price = bond_floor + option_component
    max_loss = (1.0 - barrier / spot) * notional

    return ELNResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_component=float(option_component),
        coupon_value=0.0,
        max_loss=float(max_loss),
        participation_rate=float(participation),
    )


# ---------------------------------------------------------------------------
# 6. Worst-of basket ELN (Monte Carlo)
# ---------------------------------------------------------------------------

def worst_of_eln(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    rate: float,
    div_yields: list[float],
    T: float,
    barrier: float,
    coupon: float,
    notional: float = 1000.0,
    n_paths: int = 50_000,
    seed: int = 42,
) -> ELNResult:
    """Worst-of basket ELN priced by Monte Carlo.

    Pays ``coupon * notional`` at maturity unless ANY underlying closes below
    its initial level times ``barrier``.  Principal is returned in all cases
    (capital protected structure).

    Uses a Cholesky decomposition of the correlation matrix to generate
    correlated log-normal paths.

    Args:
        spots:        List of current spot levels (length N).
        vols:         List of implied volatilities (length N).
        correlations: NxN correlation matrix (list of lists).
        rate:         Continuously compounded risk-free rate.
        div_yields:   List of continuous dividend yields (length N).
        T:            Time to maturity in years.
        barrier:      Barrier as fraction of each initial spot (e.g. 0.80).
        coupon:       Coupon paid if no barrier breach (e.g. 0.07 for 7%).
        notional:     Face value.
        n_paths:      Number of Monte Carlo paths (default 50 000).
        seed:         Random seed for reproducibility.

    Returns:
        :class:`ELNResult` with Monte Carlo price and components.
    """
    n = len(spots)
    df = _df(rate, T)
    bond_floor = notional * df

    # Cholesky decomposition (lower triangular)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = correlations[i][i] - s
                L[i][j] = math.sqrt(max(val, 0.0))
            else:
                L[i][j] = (correlations[i][j] - s) / L[j][j] if L[j][j] > 1e-12 else 0.0

    # Drift and diffusion per asset
    drifts = [(rate - div_yields[i] - 0.5 * vols[i] ** 2) * T for i in range(n)]
    sigmas = [vols[i] * math.sqrt(T) for i in range(n)]

    rng = np.random.default_rng(seed)

    # Cholesky matrix as numpy array
    L_arr = np.array(L)  # (n, n) lower triangular

    # Vectorised MC: generate all paths at once
    z = rng.standard_normal((n_paths, n))  # independent normals
    w = z @ L_arr.T  # correlated normals: (n_paths, n)

    # Terminal asset prices for each path
    drifts_arr = np.array(drifts)   # (n,)
    sigmas_arr = np.array(sigmas)   # (n,)
    spots_arr = np.array(spots)     # (n,)

    s_T = spots_arr * np.exp(drifts_arr + sigmas_arr * w)  # (n_paths, n)

    # Check if any asset breached the barrier
    barrier_levels = barrier * spots_arr  # (n,)
    breached = np.any(s_T < barrier_levels, axis=1)  # (n_paths,)
    prob_coupon = float(np.mean(~breached))
    coupon_pv = coupon * notional * df * prob_coupon
    option_component = coupon_pv

    price = bond_floor + option_component
    max_loss = 0.0  # capital protected

    return ELNResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_component=float(option_component),
        coupon_value=float(coupon_pv),
        max_loss=float(max_loss),
        participation_rate=float(prob_coupon),
    )
