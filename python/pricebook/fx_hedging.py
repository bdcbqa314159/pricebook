"""FX hedging structures and extended barriers.

* :func:`window_barrier_option` — barrier active only during a window.
* :func:`fader_option` — barrier that phases in/out gradually.
* :func:`participating_forward` — floor + leveraged upside participation.
* :func:`seagull` — risk reversal + cap (3-strike structure).
* :func:`ratio_forward` — long 1 forward + short N forwards at better strike.
* :func:`knock_in_reverse` — reverse convertible with knock-in barrier.

References:
    Wystup (2017). FX Options and Structured Products, 2nd ed., Ch. 2-3.
    Clark (2011). FX Option Pricing, Ch. 5-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


@dataclass
class WindowBarrierResult:
    price: float
    knockout_probability: float
    window_start: float
    window_end: float
    barrier: float
    def to_dict(self) -> dict:
        return vars(self)


def window_barrier_option(
    spot: float, strike: float, barrier: float,
    rate_dom: float, rate_for: float, vol: float, T: float,
    window_start: float, window_end: float,
    is_up: bool = True, is_knock_out: bool = True, is_call: bool = True,
    n_paths: int = 20_000, n_steps: int = 252, seed: int | None = 42,
) -> WindowBarrierResult:
    """Barrier active only during [window_start, window_end]."""
    from pricebook.mc_migrate import gbm_paths
    drift = rate_dom - rate_for
    paths = gbm_paths(spot, drift, vol, T, n_steps, n_paths, seed or 42)
    dt = T / n_steps
    df = math.exp(-rate_dom * T)
    s0 = max(0, int(window_start / dt))
    s1 = min(n_steps, int(window_end / dt))
    w = paths[:, s0:s1 + 1]
    hit = np.any(w >= barrier, axis=1) if is_up else np.any(w <= barrier, axis=1)
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - strike, 0.0) if is_call else np.maximum(strike - S_T, 0.0)
    payoff = payoff * (~hit) if is_knock_out else payoff * hit
    return WindowBarrierResult(df * float(payoff.mean()), float(hit.mean()),
                                window_start, window_end, barrier)


@dataclass
class FaderResult:
    price: float
    average_fading_factor: float
    n_observations: int
    def to_dict(self) -> dict:
        return vars(self)


def fader_option(
    spot: float, strike: float, barrier: float,
    rate_dom: float, rate_for: float, vol: float, T: float,
    n_observations: int = 12, is_up: bool = True, is_call: bool = True,
    n_paths: int = 20_000, seed: int | None = 42,
) -> FaderResult:
    """Fader: payoff = vanilla x (fraction of observations NOT breached)."""
    from pricebook.mc_migrate import gbm_paths
    paths = gbm_paths(spot, rate_dom - rate_for, vol, T, n_observations, n_paths, seed or 42)
    df = math.exp(-rate_dom * T)
    m = paths[:, 1:]
    not_hit = (m < barrier) if is_up else (m > barrier)
    fading = not_hit.sum(axis=1) / n_observations
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - strike, 0.0) if is_call else np.maximum(strike - S_T, 0.0)
    return FaderResult(df * float((payoff * fading).mean()), float(fading.mean()), n_observations)


@dataclass
class ParticipatingForwardResult:
    price: float
    floor_rate: float
    participation_rate: float
    forward_rate: float
    zero_cost: bool
    def to_dict(self) -> dict:
        return vars(self)


def participating_forward(
    spot: float, rate_dom: float, rate_for: float, vol: float, T: float,
    floor_rate: float | None = None, participation: float | None = None,
    notional: float = 1_000_000,
) -> ParticipatingForwardResult:
    """Participating forward: guaranteed floor + partial upside. Zero-cost solve."""
    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)
    if floor_rate is None and participation is None:
        floor_rate = F
    if participation is None:
        p = black76_price(F, floor_rate, vol, T, df, OptionType.PUT)
        c = black76_price(F, floor_rate, vol, T, df, OptionType.CALL)
        participation = p / c if c > 1e-10 else 0.0
        zc = True
    elif floor_rate is None:
        from pricebook.solvers import brentq
        def obj(K):
            return (black76_price(F, K, vol, T, df, OptionType.PUT)
                    - participation * black76_price(F, K, vol, T, df, OptionType.CALL))
        floor_rate = brentq(obj, spot * 0.5, spot * 1.5)
        zc = True
    else:
        zc = False
    put_pv = black76_price(F, floor_rate, vol, T, df, OptionType.PUT) * notional
    call_pv = black76_price(F, floor_rate, vol, T, df, OptionType.CALL) * notional
    return ParticipatingForwardResult(float(put_pv - participation * call_pv),
                                       float(floor_rate), float(participation), float(F), zc)


@dataclass
class SeagullResult:
    price: float
    put_strike: float
    call_strike_low: float
    call_strike_high: float
    max_gain: float
    forward_rate: float
    def to_dict(self) -> dict:
        return vars(self)


def seagull(
    spot: float, rate_dom: float, rate_for: float, vol: float, T: float,
    put_strike: float, call_strike_low: float, call_strike_high: float,
    notional: float = 1_000_000,
) -> SeagullResult:
    """Seagull: long put + short put (lower) + short call = zero-cost collar variant."""
    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)
    lp = black76_price(F, put_strike, vol, T, df, OptionType.PUT)
    sp = black76_price(F, call_strike_low, vol, T, df, OptionType.PUT)
    sc = black76_price(F, call_strike_high, vol, T, df, OptionType.CALL)
    return SeagullResult(float((lp - sp - sc) * notional), put_strike,
                          call_strike_low, call_strike_high, float(max(call_strike_high - F, 0)), float(F))


# ---------------------------------------------------------------------------
# Ratio forward
# ---------------------------------------------------------------------------

@dataclass
class RatioForwardResult:
    """Ratio forward result."""
    price: float
    enhanced_strike: float
    ratio: float
    forward_rate: float
    max_loss_unlimited: bool

    def to_dict(self) -> dict:
        return vars(self)


def ratio_forward(
    spot: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    ratio: float = 2.0,
    enhanced_strike: float | None = None,
    notional: float = 1_000_000,
) -> RatioForwardResult:
    """Ratio forward: long 1 put + short N calls at enhanced strike.

    The hedger buys protection at a better strike than the outright forward,
    but sells N times the upside. Zero-cost structure.

    If spot finishes above enhanced_strike, the hedger delivers N × notional
    at the enhanced strike — uncapped downside on the short leg.

    Args:
        ratio: leverage ratio (e.g. 2.0 = short 2 calls per 1 put).
        enhanced_strike: strike for both legs. If None, solve for zero-cost.
    """
    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)

    if enhanced_strike is None:
        # Solve for zero-cost: put(K) = ratio × call(K)
        from pricebook.solvers import brentq

        def obj(K):
            return (black76_price(F, K, vol, T, df, OptionType.PUT)
                    - ratio * black76_price(F, K, vol, T, df, OptionType.CALL))
        enhanced_strike = brentq(obj, F * 0.8, F * 1.2)

    put_pv = black76_price(F, enhanced_strike, vol, T, df, OptionType.PUT) * notional
    call_pv = black76_price(F, enhanced_strike, vol, T, df, OptionType.CALL) * notional * ratio
    price = float(put_pv - call_pv)

    return RatioForwardResult(
        price=price,
        enhanced_strike=float(enhanced_strike),
        ratio=ratio,
        forward_rate=float(F),
        max_loss_unlimited=True,
    )


# ---------------------------------------------------------------------------
# Knock-in reverse convertible
# ---------------------------------------------------------------------------

@dataclass
class KnockInReverseResult:
    """Knock-in reverse convertible result."""
    price: float
    enhanced_yield: float
    barrier: float
    knock_in_probability: float
    expected_loss: float

    def to_dict(self) -> dict:
        return vars(self)


def knock_in_reverse(
    spot: float,
    barrier: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    coupon_rate: float,
    notional: float = 1_000_000,
    n_paths: int = 20_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> KnockInReverseResult:
    """Knock-in reverse convertible: enhanced yield + conditional principal risk.

    Investor receives above-market coupon. At maturity:
    - If spot never breached barrier (downside): full principal returned.
    - If barrier was breached: principal converted at initial spot
      (investor bears the loss).

    Structure: bond + short knock-in put.

    Args:
        barrier: knock-in barrier (below spot, e.g. 0.85 × spot).
        coupon_rate: enhanced annual coupon rate.
    """
    from pricebook.mc_migrate import gbm_paths

    drift = rate_dom - rate_for
    paths = gbm_paths(spot, drift, vol, T, n_steps, n_paths, seed or 42)
    df = math.exp(-rate_dom * T)

    # Check barrier breach
    knocked_in = np.any(paths[:, 1:] <= barrier, axis=1)
    ki_prob = float(knocked_in.mean())

    S_T = paths[:, -1]

    # Payoff: coupon always paid. Principal:
    # - No knock-in: get notional back
    # - Knock-in: get notional × (S_T / spot), capped at notional
    coupon_pv = notional * coupon_rate * T * df
    principal = np.where(
        knocked_in,
        notional * np.minimum(S_T / spot, 1.0),
        notional,
    )
    principal_pv = df * principal

    total_pv = coupon_pv + float(principal_pv.mean())

    # Expected loss from knock-in
    loss = np.where(knocked_in, notional * np.maximum(1 - S_T / spot, 0), 0)
    expected_loss = df * float(loss.mean())

    # Enhanced yield vs risk-free
    enhanced_yield = coupon_rate - (rate_dom * T) / T  # excess over risk-free

    return KnockInReverseResult(
        price=float(total_pv),
        enhanced_yield=float(enhanced_yield),
        barrier=float(barrier),
        knock_in_probability=ki_prob,
        expected_loss=float(expected_loss),
    )
