"""American option pricing with discrete dividends.

Optimal early exercise around ex-dividend dates:
- Tree-based pricing with ex-dates as explicit nodes
- Roll-Geske-Whaley closed-form for single dividend
- Exercise boundary computation

    from pricebook.options.american_dividend import (
        american_with_dividends, roll_geske_whaley,
        exercise_boundary_around_exdate,
    )

References:
    Roll (1977). An Analytic Valuation Formula for Unprotected American
        Call Options on Stocks with Known Dividends.
    Geske (1979). A Note on an Analytical Valuation Formula for Unprotected
        American Call Options on Stocks with Known Dividends.
    Whaley (1981). On the Valuation of American Call Options on Stocks
        with Known Dividends.
    Hull (2022). Options, Futures, and Other Derivatives, Ch. 21.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.stats import norm

from pricebook.models.black76 import OptionType, black76_price


@dataclass
class AmericanDivResult:
    """American option pricing result with discrete dividends."""
    price: float
    european_price: float
    early_exercise_premium: float
    n_steps: int
    exercise_boundary: list[float] | None   # critical spot per step

    def to_dict(self) -> dict:
        return {
            "price": self.price, "european_price": self.european_price,
            "early_exercise_premium": self.early_exercise_premium,
            "n_steps": self.n_steps,
        }


@dataclass
class RGWResult:
    """Roll-Geske-Whaley result."""
    price: float
    european_price: float
    early_exercise_premium: float
    critical_spot: float    # S* at which exercise is optimal

    def to_dict(self) -> dict:
        return vars(self)


def american_with_dividends(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    dividends: list[tuple[float, float]],
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 500,
) -> AmericanDivResult:
    """Price American option with discrete dividends via binomial tree.

    Inserts ex-dividend dates as explicit tree nodes. At each ex-date,
    the spot drops by the dividend amount.

    Args:
        spot: current spot price.
        strike: option strike.
        rate: risk-free rate.
        vol: volatility.
        T: time to expiry in years.
        dividends: list of (time_years, amount) tuples.
        option_type: CALL or PUT.
        n_steps: number of tree steps.

    Returns:
        AmericanDivResult with price, European price, and early exercise premium.
    """
    dt = T / n_steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(rate * dt) - d) / (u - d)
    df_step = math.exp(-rate * dt)

    # Map dividends to nearest step
    div_at_step = {}
    for t_div, amount in dividends:
        step = round(t_div / dt)
        step = max(1, min(step, n_steps - 1))
        div_at_step[step] = div_at_step.get(step, 0.0) + amount

    # Build terminal spots
    spots = np.zeros(n_steps + 1)
    spots[0] = spot * u**n_steps
    for j in range(1, n_steps + 1):
        spots[j] = spots[j - 1] * d / u  # spots[j] = S0 * u^(n-2j)

    # Apply dividends forward to get adjusted terminal spots
    # Actually, easier to backward-induct with dividend at the step
    # Reset: build spot tree during backward pass
    # Use 1D array backward induction

    # Terminal payoff
    if option_type == OptionType.CALL:
        values = np.maximum(spots - strike, 0.0)
    else:
        values = np.maximum(strike - spots, 0.0)

    # European values (no early exercise)
    eu_values = values.copy()

    exercise_boundary = [0.0] * (n_steps + 1)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        # Spots at step i
        step_spots = np.zeros(i + 1)
        step_spots[0] = spot * u**i
        for j in range(1, i + 1):
            step_spots[j] = step_spots[j - 1] * d / u

        # Apply dividend at this step (spot drops)
        if i in div_at_step:
            step_spots = np.maximum(step_spots - div_at_step[i], 0.01)

        # Continuation value
        cont = df_step * (p * values[:i + 1] + (1 - p) * values[1:i + 2])
        eu_cont = df_step * (p * eu_values[:i + 1] + (1 - p) * eu_values[1:i + 2])

        # Intrinsic value
        if option_type == OptionType.CALL:
            intrinsic = np.maximum(step_spots - strike, 0.0)
        else:
            intrinsic = np.maximum(strike - step_spots, 0.0)

        # American: max(intrinsic, continuation)
        values = np.maximum(intrinsic, cont)
        eu_values = eu_cont  # no early exercise for European

        # Exercise boundary: find critical spot
        exercised = intrinsic > cont
        if np.any(exercised):
            idx = np.where(exercised)[0]
            exercise_boundary[i] = float(step_spots[idx[0]])

    am_price = float(values[0])
    eu_price = float(eu_values[0])

    return AmericanDivResult(
        price=am_price,
        european_price=eu_price,
        early_exercise_premium=max(am_price - eu_price, 0.0),
        n_steps=n_steps,
        exercise_boundary=[b for b in exercise_boundary if b > 0],
    )


def roll_geske_whaley(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    div_amount: float,
    div_time: float,
) -> RGWResult:
    """Roll-Geske-Whaley formula for American call with single dividend.

    Closed-form approximation. The call may be exercised just before
    the ex-dividend date if the dividend exceeds the time value.

    Args:
        spot: current spot.
        strike: option strike.
        rate: risk-free rate.
        vol: volatility.
        T: time to expiry.
        div_amount: discrete dividend amount.
        div_time: time to ex-date (in years, div_time < T).

    Returns:
        RGWResult with price, European price, and critical spot S*.
    """
    if div_time >= T:
        # Dividend after expiry → pure European
        fwd = spot * math.exp(rate * T)
        df = math.exp(-rate * T)
        eu = black76_price(fwd, strike, vol, T, df, OptionType.CALL)
        return RGWResult(eu, eu, 0.0, float("inf"))

    # Adjusted spot: S' = S - PV(div)
    pv_div = div_amount * math.exp(-rate * div_time)
    s_adj = spot - pv_div

    if s_adj <= 0:
        return RGWResult(0.0, 0.0, 0.0, 0.0)

    # European price on adjusted spot
    fwd = s_adj * math.exp(rate * T)
    df = math.exp(-rate * T)
    eu_price = black76_price(fwd, strike, vol, T, df, OptionType.CALL)

    # Critical spot S*: exercise if S* - K > BS(S*, K, T-t_d)
    # Newton's method to find S*
    s_star = strike + div_amount  # initial guess
    for _ in range(50):
        fwd_star = s_star * math.exp(rate * (T - div_time))
        df_star = math.exp(-rate * (T - div_time))
        bs_val = black76_price(fwd_star, strike, vol, T - div_time, df_star, OptionType.CALL)
        f_val = s_star - strike - bs_val
        # Derivative: 1 - delta_BS
        d1 = (math.log(fwd_star / strike) + 0.5 * vol**2 * (T - div_time)) / (vol * math.sqrt(T - div_time))
        delta_bs = norm.cdf(d1)
        f_prime = 1 - delta_bs
        if abs(f_prime) < 1e-10:
            break
        s_star -= f_val / f_prime
        s_star = max(s_star, strike * 0.5)

    # If spot < S* at ex-date → don't exercise → European price
    # If spot > S* → exercise → S - K
    # RGW adjustment: add early exercise premium
    if spot - pv_div > s_star:
        # Likely to exercise
        am_price = max(eu_price, spot - pv_div - strike)
    else:
        # Compute compound option value
        # Simplified: add a premium based on probability of exercise
        t1 = div_time
        d1_ex = (math.log(s_adj / s_star) + (rate + 0.5 * vol**2) * t1) / (vol * math.sqrt(t1))
        prob_exercise = norm.cdf(d1_ex)
        exercise_value = (s_adj * math.exp(rate * t1) - strike) * math.exp(-rate * t1)
        premium = max(prob_exercise * max(exercise_value - eu_price, 0), 0)
        am_price = eu_price + premium

    return RGWResult(
        price=am_price,
        european_price=eu_price,
        early_exercise_premium=max(am_price - eu_price, 0.0),
        critical_spot=s_star,
    )


def exercise_boundary_around_exdate(
    strike: float,
    rate: float,
    vol: float,
    T: float,
    div_amount: float,
    div_time: float,
    spot_range: list[float] | None = None,
) -> list[dict]:
    """Compute exercise decision across spot levels at the ex-date.

    For each spot, compare:
    - Exercise value: S - K
    - Continuation value: BS(S - D, K, T - t_d)

    Args:
        spot_range: list of spot levels to evaluate.

    Returns:
        List of dicts with spot, exercise_value, continuation_value, optimal_action.
    """
    if spot_range is None:
        spot_range = [strike * m for m in np.arange(0.8, 1.3, 0.02)]

    tau = T - div_time
    df = math.exp(-rate * tau)

    results = []
    for S in spot_range:
        exercise_val = max(S - strike, 0.0)
        s_ex = S - div_amount  # post-dividend spot
        if s_ex > 0 and tau > 0:
            fwd = s_ex * math.exp(rate * tau)
            cont_val = black76_price(fwd, strike, vol, tau, df, OptionType.CALL)
        else:
            cont_val = 0.0

        results.append({
            "spot": S,
            "exercise_value": exercise_val,
            "continuation_value": cont_val,
            "optimal": "exercise" if exercise_val > cont_val else "hold",
        })

    return results
