"""
Bermudan cap/floor pricing via Hull-White trinomial tree.

A Bermudan cap (floor) gives the holder the right to exercise — i.e. knock in
— a cap (floor) on selected dates. On each exercise date the holder compares
the continuation value with the present value of all remaining caplets
(floorlets) and keeps the larger. This differs from a European cap (floor)
where exercise occurs only at a single predetermined date.

    from pricebook.options.bermudan_capfloor import bermudan_cap, bermudan_floor

    result = bermudan_cap(
        reference_date=date(2024, 1, 15),
        maturity_years=5.0,
        strike=0.05,
        hw_a=0.05,
        hw_sigma=0.01,
        r0=0.04,
        exercise_dates_years=[1.0, 2.0, 3.0, 4.0],
    )

References:
    Brigo & Mercurio, *Interest Rate Models — Theory and Practice*, Ch. 3, 2006.
    Andersen & Piterbarg, *Interest Rate Modeling*, Vol. II, Ch. 11, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BermudanCapFloorResult:
    """Result of a Bermudan cap/floor pricing calculation.

    Attributes:
        price: Bermudan cap/floor price (same units as notional).
        european_price: Price of equivalent European cap/floor (no early exercise).
        early_exercise_premium: price - european_price (always >= 0).
        n_exercise_dates: number of Bermudan exercise dates.
        exercise_probabilities: list of exercise probabilities per exercise date.
    """
    price: float
    european_price: float
    early_exercise_premium: float
    n_exercise_dates: int
    exercise_probabilities: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "european_price": self.european_price,
            "early_exercise_premium": self.early_exercise_premium,
            "n_exercise_dates": self.n_exercise_dates,
            "exercise_probabilities": self.exercise_probabilities,
        }


# ---------------------------------------------------------------------------
# Internal: caplet value on HW tree using analytical bond prices
# ---------------------------------------------------------------------------


def _caplet_value_hw(
    r_j: float,
    t_ex: float,
    t_fix: float,
    t_pay: float,
    strike: float,
    dt: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    is_cap: bool,
    notional: float,
) -> float:
    """Analytical HW caplet/floorlet value at node (t_ex, r_j).

    Uses the bond-option formula: a caplet on [t_fix, t_pay] is a put
    option on a ZCB maturing at t_pay with strike 1/(1+K*tau).
    """
    tau = t_pay - t_fix
    if tau <= 0 or t_fix <= t_ex:
        return 0.0

    # ZCB prices at t_ex under HW
    def _B(t_start: float, t_end: float) -> float:
        if hw_a < 1e-10:
            return t_end - t_start
        return (1.0 - math.exp(-hw_a * (t_end - t_start))) / hw_a

    def _zcb(t_start: float, t_end: float, r: float) -> float:
        """HW ZCB price P(t_start, t_end | r(t_start)=r)."""
        if t_end <= t_start:
            return 1.0
        B_val = _B(t_start, t_end)
        # Drift correction: use flat curve approximation
        A_val = math.exp(-r0 * (t_end - t_start) + 0.5 * (hw_sigma / hw_a) ** 2 *
                         (B_val - (t_end - t_start)) if hw_a > 1e-10 else 0.0)
        return A_val * math.exp(-B_val * (r - r0))

    P_fix = _zcb(t_ex, t_fix, r_j)
    P_pay = _zcb(t_ex, t_pay, r_j)

    if P_fix <= 0 or P_pay <= 0:
        return 0.0

    fwd_rate = (P_fix / P_pay - 1.0) / tau
    payoff = max(fwd_rate - strike, 0.0) if is_cap else max(strike - fwd_rate, 0.0)
    return notional * tau * P_pay * payoff


# ---------------------------------------------------------------------------
# Internal: sum of remaining caplet/floorlet values at a node
# ---------------------------------------------------------------------------


def _remaining_value(
    r_j: float,
    t_ex: float,
    maturity_years: float,
    strike: float,
    frequency: int,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    is_cap: bool,
    notional: float,
) -> float:
    """Sum of all caplet (or floorlet) values from t_ex to maturity_years."""
    period = 1.0 / frequency
    total = 0.0
    t_fix = t_ex
    while t_fix + period <= maturity_years + 1e-10:
        t_pay = t_fix + period
        total += _caplet_value_hw(
            r_j, t_ex, t_fix, t_pay, strike, period,
            hw_a, hw_sigma, r0, is_cap, notional,
        )
        t_fix = t_pay
    return total


# ---------------------------------------------------------------------------
# Internal: trinomial tree backward induction
# ---------------------------------------------------------------------------


def _bermudan_capfloor_tree(
    reference_date: date,
    maturity_years: float,
    strike: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    exercise_dates_years: list[float],
    frequency: int,
    notional: float,
    n_steps: int,
    is_cap: bool,
) -> BermudanCapFloorResult:
    """Core HW trinomial tree backward induction for Bermudan cap/floor."""
    exercise_sorted = sorted(t for t in exercise_dates_years
                             if 0 < t < maturity_years + 1e-10)
    if not exercise_sorted:
        raise ValueError("exercise_dates_years must contain at least one date "
                         "before maturity.")

    dt = maturity_years / n_steps
    dr = hw_sigma * math.sqrt(3.0 * dt)
    j_max = max(1, int(math.ceil(0.1835 / (max(hw_a, 1e-6) * dt))))
    n_nodes = 2 * j_max + 1
    mid = j_max

    exercise_steps = {int(round(t / dt)): t for t in exercise_sorted}

    # Terminal: all caplets/floorlets have expired — value = 0
    values = np.zeros(n_nodes)

    # Track exercise probabilities (approximated by node weights at ex dates)
    # We accumulate the fraction of the value improvement at each exercise step.
    ex_prob_gains: list[float] = []

    for step in range(n_steps - 1, -1, -1):
        new_values = np.zeros(n_nodes)

        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)

            p_up = 1.0 / 6.0 + (j * j * hw_a * hw_a * dt * dt - j * hw_a * dt) / 6.0
            p_mid = 2.0 / 3.0 - j * j * hw_a * hw_a * dt * dt / 3.0
            p_dn = 1.0 / 6.0 + (j * j * hw_a * hw_a * dt * dt + j * hw_a * dt) / 6.0

            p_up = max(0.0, min(1.0, p_up))
            p_mid = max(0.0, min(1.0, p_mid))
            p_dn = max(0.0, min(1.0, p_dn))
            p_total = p_up + p_mid + p_dn
            if p_total > 0:
                p_up /= p_total
                p_mid /= p_total
                p_dn /= p_total

            j_up = min(j + 1, j_max)
            j_dn = max(j - 1, -j_max)

            cont = (p_up * values[j_up + mid]
                    + p_mid * values[j + mid]
                    + p_dn * values[j_dn + mid])
            new_values[idx] = cont * one_step_df

        if (step + 1) in exercise_steps:
            t_ex = exercise_steps[step + 1]
            before_sum = float(new_values[mid])
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                r_j = r0 + j * dr
                exercise_val = _remaining_value(
                    r_j, t_ex, maturity_years, strike, frequency,
                    hw_a, hw_sigma, r0, is_cap, notional,
                )
                new_values[idx] = max(new_values[idx], exercise_val)
            after_sum = float(new_values[mid])
            # Approximate exercise probability as relative value improvement
            if after_sum > 0:
                ex_prob_gains.append(max(0.0, (after_sum - before_sum) / max(after_sum, 1e-12)))
            else:
                ex_prob_gains.append(0.0)

        values = new_values

    bermudan_price = float(values[mid])

    # European price: value of all caplets/floorlets without early exercise
    european_price = _remaining_value(
        r0, 0.0, maturity_years, strike, frequency,
        hw_a, hw_sigma, r0, is_cap, notional,
    )

    early_ex_prem = max(0.0, bermudan_price - european_price)

    # Reverse exercise probs (tree runs backward, first appended = last ex date)
    ex_prob_gains.reverse()

    return BermudanCapFloorResult(
        price=bermudan_price,
        european_price=european_price,
        early_exercise_premium=early_ex_prem,
        n_exercise_dates=len(exercise_sorted),
        exercise_probabilities=ex_prob_gains,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bermudan_cap(
    reference_date: date,
    maturity_years: float,
    strike: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    exercise_dates_years: list[float],
    frequency: int = 4,
    notional: float = 1_000_000.0,
    n_steps: int = 200,
) -> BermudanCapFloorResult:
    """Bermudan cap priced on a Hull-White trinomial tree.

    The holder can, on any exercise date, elect to receive all remaining
    caplets from that date to maturity. At each exercise date the tree
    applies: ``value = max(continuation, sum_of_remaining_caplets)``.

    Args:
        reference_date: pricing date.
        maturity_years: total tenor of the cap (years).
        strike: cap strike rate (e.g. 0.05 = 5%).
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        exercise_dates_years: list of Bermudan exercise dates (as year fractions).
        frequency: number of caplet periods per year (default 4 = quarterly).
        notional: notional principal.
        n_steps: number of tree time steps.

    Returns:
        BermudanCapFloorResult with price, European benchmark, and diagnostics.
    """
    return _bermudan_capfloor_tree(
        reference_date, maturity_years, strike,
        hw_a, hw_sigma, r0,
        exercise_dates_years, frequency, notional, n_steps,
        is_cap=True,
    )


def bermudan_floor(
    reference_date: date,
    maturity_years: float,
    strike: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    exercise_dates_years: list[float],
    frequency: int = 4,
    notional: float = 1_000_000.0,
    n_steps: int = 200,
) -> BermudanCapFloorResult:
    """Bermudan floor priced on a Hull-White trinomial tree.

    Analogous to :func:`bermudan_cap` but for floorlets. At each exercise
    date: ``value = max(continuation, sum_of_remaining_floorlets)``.

    Args:
        reference_date: pricing date.
        maturity_years: total tenor of the floor (years).
        strike: floor strike rate.
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        exercise_dates_years: list of Bermudan exercise dates (as year fractions).
        frequency: number of floorlet periods per year (default 4 = quarterly).
        notional: notional principal.
        n_steps: number of tree time steps.

    Returns:
        BermudanCapFloorResult with price, European benchmark, and diagnostics.
    """
    return _bermudan_capfloor_tree(
        reference_date, maturity_years, strike,
        hw_a, hw_sigma, r0,
        exercise_dates_years, frequency, notional, n_steps,
        is_cap=False,
    )


def bermudan_collar(
    reference_date: date,
    maturity_years: float,
    cap_strike: float,
    floor_strike: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    exercise_dates_years: list[float],
    frequency: int = 4,
    notional: float = 1_000_000.0,
    n_steps: int = 200,
) -> dict:
    """Bermudan collar: long Bermudan cap + short Bermudan floor.

    A collar bounds the floating rate between floor_strike and cap_strike.
    The Bermudan exercise right applies to both legs independently.

    Args:
        reference_date: pricing date.
        maturity_years: total tenor (years).
        cap_strike: cap strike rate (upper bound on rate paid).
        floor_strike: floor strike rate (lower bound on rate paid).
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        exercise_dates_years: exercise dates shared by cap and floor.
        frequency: periods per year.
        notional: notional principal.
        n_steps: tree time steps.

    Returns:
        dict with keys: collar_price, cap_result, floor_result.
    """
    cap_result = bermudan_cap(
        reference_date, maturity_years, cap_strike,
        hw_a, hw_sigma, r0, exercise_dates_years,
        frequency, notional, n_steps,
    )
    floor_result = bermudan_floor(
        reference_date, maturity_years, floor_strike,
        hw_a, hw_sigma, r0, exercise_dates_years,
        frequency, notional, n_steps,
    )
    collar_price = cap_result.price - floor_result.price
    return {
        "collar_price": collar_price,
        "cap_result": cap_result,
        "floor_result": floor_result,
    }
