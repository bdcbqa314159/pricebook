"""Crypto structured products: dual investment, shark fin, accumulator.

Popular on centralised venues (Binance Earn, OKX, Bybit).

* :func:`dual_investment` — buy low / sell high with enhanced yield.
* :func:`crypto_shark_fin` — capped upside with principal protection.
* :func:`crypto_accumulator` — daily accumulation at discount.
* :func:`crypto_snowball` — autocall with coupon and knock-in.

References:
    Binance, *Dual Investment Product Guide*.
    OKX, *Structured Products Specification*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import OptionType, black76_price, _norm_cdf


@dataclass
class DualInvestmentResult:
    """Dual investment result."""
    enhanced_yield_apy: float
    base_yield_apy: float
    settlement_price: float     # strike
    exercise_probability: float # prob of conversion
    settlement_currency: str    # what you get if exercised
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


def dual_investment(
    spot: float,
    strike: float,
    vol: float,
    T: float,
    notional: float = 10_000.0,
    base_yield: float = 0.05,
    direction: str = "buy_low",
) -> DualInvestmentResult:
    """Dual investment: enhanced yield by selling an option.

    "Buy Low": deposit USDT, sell a put. If BTC < strike at expiry,
    you buy BTC at strike (at a discount). Otherwise, earn enhanced yield.

    "Sell High": deposit BTC, sell a call. If BTC > strike at expiry,
    you sell BTC at strike (at a premium). Otherwise, keep BTC + yield.

    Enhanced yield = base_yield + option_premium / notional × (365/T).

    Args:
        spot: current BTC price.
        strike: conversion price.
        vol: implied volatility.
        T: term in years.
        notional: deposit amount.
        base_yield: base deposit rate.
        direction: "buy_low" (put) or "sell_high" (call).
    """
    fwd = spot
    df = 1.0  # crypto: r ≈ 0

    if direction == "buy_low":
        otype = OptionType.PUT
        premium = black76_price(fwd, strike, vol, T, df, otype)
        settlement_ccy = "BTC"
    else:
        otype = OptionType.CALL
        premium = black76_price(fwd, strike, vol, T, df, otype)
        settlement_ccy = "USDT"

    # Enhanced yield
    premium_pct = premium / spot
    enhanced_apy = base_yield + premium_pct / T

    # Exercise probability (N(d2) for put, 1-N(d2) for call)
    if T > 0 and vol > 0:
        d2 = (math.log(fwd / strike) - 0.5 * vol**2 * T) / (vol * math.sqrt(T))
        if direction == "buy_low":
            exercise_prob = _norm_cdf(-d2)
        else:
            exercise_prob = 1 - _norm_cdf(d2)
    else:
        exercise_prob = 0

    return DualInvestmentResult(
        enhanced_yield_apy=enhanced_apy * 100,
        base_yield_apy=base_yield * 100,
        settlement_price=strike,
        exercise_probability=exercise_prob,
        settlement_currency=settlement_ccy,
        notional=notional,
    )


@dataclass
class CryptoSharkFinResult:
    """Crypto shark fin note result."""
    price: float
    max_return_pct: float
    knock_out_barrier: float
    participation: float
    protection_pct: float

    def to_dict(self) -> dict:
        return vars(self)


def crypto_shark_fin(
    spot: float,
    vol: float,
    T: float,
    barrier: float,
    participation: float = 1.0,
    protection: float = 1.0,
    notional: float = 10_000.0,
    is_bullish: bool = True,
) -> CryptoSharkFinResult:
    """Crypto shark fin: capped upside with barrier knock-out.

    Bullish: earns if BTC goes up, but capped if barrier hit.
    Bearish: earns if BTC goes down, capped if barrier hit.

    Payoff (bullish):
    - If S_T < barrier: participation × (S_T/S_0 − 1) × notional
    - If S_T ≥ barrier: fixed rebate (cap)
    - Principal protected at protection level.

    Args:
        barrier: knock-out barrier level.
        participation: upside participation rate.
        protection: principal protection (1.0 = 100%).
    """
    fwd = spot
    df = 1.0

    if is_bullish:
        # Bull call spread + knockout
        call_atm = black76_price(fwd, spot, vol, T, df, OptionType.CALL)
        call_barrier = black76_price(fwd, barrier, vol, T, df, OptionType.CALL)
        spread = call_atm - call_barrier
        price = protection * notional + participation * spread / spot * notional
    else:
        put_atm = black76_price(fwd, spot, vol, T, df, OptionType.PUT)
        put_barrier = black76_price(fwd, barrier, vol, T, df, OptionType.PUT)
        spread = put_atm - put_barrier
        price = protection * notional + participation * spread / spot * notional

    max_return = abs(barrier / spot - 1) * participation * 100

    return CryptoSharkFinResult(
        price=price,
        max_return_pct=max_return,
        knock_out_barrier=barrier,
        participation=participation,
        protection_pct=protection * 100,
    )


@dataclass
class CryptoAccumulatorResult:
    """Crypto accumulator (DCA with leverage) result."""
    expected_quantity: float
    expected_avg_price: float
    max_loss_pct: float
    ko_probability: float
    n_observations: int

    def to_dict(self) -> dict:
        return vars(self)


def crypto_accumulator(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    T: float,
    daily_amount: float = 100.0,
    leverage: float = 2.0,
    n_sims: int = 10_000,
    seed: int = 42,
) -> CryptoAccumulatorResult:
    """Crypto accumulator: buy daily at discount, knock-out on rally.

    Each day: buy crypto at strike (below spot) if spot < barrier.
    If spot ≥ barrier: knocked out (stop buying).
    If spot < strike: buy at leverage × daily_amount (forced buying).

    Popular on OKX, Binance.

    Args:
        strike: discounted buy price.
        barrier: knock-out level (above spot).
        leverage: multiplier when below strike.
    """
    rng = np.random.default_rng(seed)
    n_days = int(T * 365)
    dt = 1 / 365

    total_qty = 0.0
    total_cost = 0.0
    ko_count = 0

    for _ in range(n_sims):
        S = spot
        sim_qty = 0.0
        sim_cost = 0.0
        knocked_out = False

        for _ in range(n_days):
            S *= math.exp(-0.5 * vol**2 * dt + vol * math.sqrt(dt) * rng.standard_normal())

            if S >= barrier:
                knocked_out = True
                break

            if S < strike:
                buy_qty = daily_amount * leverage / strike
            else:
                buy_qty = daily_amount / strike

            sim_qty += buy_qty
            sim_cost += buy_qty * strike

        total_qty += sim_qty
        total_cost += sim_cost
        if knocked_out:
            ko_count += 1

    avg_qty = total_qty / n_sims
    avg_cost = total_cost / n_sims
    avg_price = avg_cost / avg_qty if avg_qty > 0 else 0
    ko_prob = ko_count / n_sims

    max_loss = (strike / spot - 1) * leverage * 100 if spot > 0 else 0

    return CryptoAccumulatorResult(
        expected_quantity=avg_qty,
        expected_avg_price=avg_price,
        max_loss_pct=abs(max_loss),
        ko_probability=ko_prob,
        n_observations=n_days,
    )


@dataclass
class CryptoSnowballResult:
    """Crypto snowball (autocall with knock-in) result."""
    price: float
    autocall_probability: float
    knock_in_probability: float
    expected_coupon: float
    expected_life_days: float

    def to_dict(self) -> dict:
        return vars(self)


def crypto_snowball(
    spot: float,
    vol: float,
    T: float,
    autocall_barrier: float = 1.0,
    knock_in_barrier: float = 0.70,
    coupon_rate: float = 0.30,
    observation_freq_days: int = 7,
    n_sims: int = 20_000,
    seed: int = 42,
) -> CryptoSnowballResult:
    """Crypto snowball: autocall with coupon and downside knock-in.

    Weekly observations:
    - If S ≥ autocall_barrier × S₀: redeem at par + coupon.
    - If S ever < knock_in_barrier × S₀: lose principal protection.
    - At maturity: if not knocked in, return par + coupon. If knocked in, bear loss.

    Args:
        autocall_barrier: fraction of spot for autocall (1.0 = at-the-money).
        knock_in_barrier: fraction of spot for knock-in (0.70 = -30%).
        coupon_rate: annualised coupon.
        observation_freq_days: days between observations.
    """
    rng = np.random.default_rng(seed)
    n_obs = int(T * 365 / observation_freq_days)
    dt = observation_freq_days / 365

    total_price = 0.0
    ac_count = 0
    ki_count = 0
    total_coupon = 0.0
    total_life = 0.0

    for _ in range(n_sims):
        S = spot
        knocked_in = False
        autocalled = False

        for obs in range(1, n_obs + 1):
            S *= math.exp(-0.5 * vol**2 * dt + vol * math.sqrt(dt) * rng.standard_normal())

            if S < knock_in_barrier * spot:
                knocked_in = True

            if S >= autocall_barrier * spot:
                coupon = coupon_rate * (obs * dt)
                total_price += 1 + coupon
                total_coupon += coupon
                total_life += obs * observation_freq_days
                autocalled = True
                ac_count += 1
                break

        if not autocalled:
            total_life += T * 365
            if knocked_in:
                # Bear loss
                loss = min(S / spot, 1.0)
                total_price += loss
                ki_count += 1
            else:
                total_price += 1 + coupon_rate * T
                total_coupon += coupon_rate * T

    price = total_price / n_sims
    return CryptoSnowballResult(
        price=price,
        autocall_probability=ac_count / n_sims,
        knock_in_probability=ki_count / n_sims,
        expected_coupon=total_coupon / n_sims,
        expected_life_days=total_life / n_sims,
    )
