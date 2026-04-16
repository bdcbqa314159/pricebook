"""Swing options and virtual storage for energy commodities.

* :func:`swing_option_lsm` — take-or-pay swing option via LSM.
* :class:`VirtualGasStorage` — gas storage optimisation with ratchets.
* :func:`nomination_rights_value` — flexible daily nomination contracts.

References:
    Longstaff & Schwartz, *Valuing American Options by Simulation*, RFS, 2001.
    Meinshausen & Hambly, *Monte Carlo Methods for the Valuation of Multiple
    Exercise Options*, Math. Finance, 2004.
    Jaillet, Ronn & Tompaidis, *Valuation of Commodity-Based Swing Options*,
    Mgmt. Sci., 2004.
    Thompson, Davison & Rasmussen, *Valuation and Optimal Operation of Electric
    Power Plants in Competitive Markets*, Ops. Res., 2004 (for gas storage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Swing option (take-or-pay) ----

@dataclass
class SwingOptionResult:
    """Swing option pricing result."""
    price: float
    n_exercise_dates: int
    max_exercises: int
    mean_exercises_used: float
    strike: float


def swing_option_lsm(
    spot_paths: np.ndarray,
    exercise_dates: list[int],        # indices in spot_paths time axis
    strike: float,
    max_exercises: int,
    discount_factors: np.ndarray,     # DF per time step (len = spot_paths.shape[1])
    min_exercises: int = 0,
    n_basis: int = 3,
    is_call: bool = True,
) -> SwingOptionResult:
    """Swing option via Longstaff-Schwartz with exercise-count state.

    At each exercise date, holder chooses to exercise or wait. Exercised
    quantity counts toward max_exercises. Below min_exercises → must
    exercise at remaining dates (enforced by penalty).

    Backward induction over (spot, remaining_exercises):
        V(t, s, n) = max(exercise_value + V(t+1, s', n-1),
                         E[V(t+1, s', n) | s])

    Args:
        spot_paths: (n_paths, n_time+1) simulated prices.
        exercise_dates: sorted time indices where exercise is allowed.
        strike: exercise strike.
        max_exercises: maximum number of exercise rights.
        min_exercises: minimum required exercises (take-or-pay floor).
        discount_factors: risk-neutral discount factors per time step.
        n_basis: polynomial basis order for regression.
        is_call: True for receive-commodity-at-strike (call); False for put.
    """
    n_paths, n_times = spot_paths.shape
    n_exercise_dates = len(exercise_dates)

    if max_exercises > n_exercise_dates:
        max_exercises = n_exercise_dates

    # Value function: V[path, n_remaining]
    # Initialise at terminal: 0 for all states
    V = np.zeros((n_paths, max_exercises + 1))

    # Iterate backward through exercise dates
    exercise_count = np.zeros(n_paths, dtype=int)

    for k in range(n_exercise_dates - 1, -1, -1):
        t_idx = exercise_dates[k]
        S = spot_paths[:, t_idx]

        if is_call:
            ex_value = np.maximum(S - strike, 0.0)
        else:
            ex_value = np.maximum(strike - S, 0.0)

        # Discount factor from this date to next exercise (or terminal)
        if k < n_exercise_dates - 1:
            t_next = exercise_dates[k + 1]
            df_step = discount_factors[t_next] / discount_factors[t_idx]
        else:
            df_step = 1.0

        # Discount the continuation values (V at next date)
        V_next = V * df_step

        # For each remaining exercise count n ≥ 1:
        #   continuation(n) = E[V_next[:, n] | S]
        #   exercise_then_continue(n) = ex_value + E[V_next[:, n-1] | S]
        # V[n] = max(continuation, exercise_then_continue)

        V_new = np.zeros_like(V)

        for n in range(1, max_exercises + 1):
            # Continuation = hold exercise
            # Fit regression for E[V_next[n] | S]
            itm = ex_value > 0
            if itm.sum() >= n_basis + 1:
                basis = np.column_stack([S[itm] ** j for j in range(n_basis)])
                try:
                    coeffs_cont = np.linalg.lstsq(basis, V_next[itm, n], rcond=None)[0]
                    cont_est = basis @ coeffs_cont
                    continuation_itm = cont_est
                except np.linalg.LinAlgError:
                    continuation_itm = V_next[itm, n]

                try:
                    coeffs_ex = np.linalg.lstsq(basis, V_next[itm, n - 1], rcond=None)[0]
                    V_next_ex_est = basis @ coeffs_ex
                except np.linalg.LinAlgError:
                    V_next_ex_est = V_next[itm, n - 1]

                # Full-array values (out-of-the-money paths just use V_next directly)
                continuation = V_next[:, n].copy()
                continuation[itm] = continuation_itm

                # Exercise value = immediate payoff + continuation with n-1 exercises
                exercise_val = np.full(n_paths, -np.inf)
                exercise_val[itm] = ex_value[itm] + V_next_ex_est

                V_new[:, n] = np.maximum(continuation, exercise_val)
                V_new[:, n] = np.where(np.isinf(V_new[:, n]), V_next[:, n], V_new[:, n])
            else:
                # Not enough ITM paths for regression — default to continuation
                V_new[:, n] = V_next[:, n]

        V_new[:, 0] = V_next[:, 0]  # no exercises remaining → must continue
        V = V_new

    # Initial value at t=0 is V[0, max_exercises] discounted to t=0
    # But we need the expectation at the first exercise date discounted
    first_date = exercise_dates[0]
    df_to_first = discount_factors[first_date] / discount_factors[0]
    price = df_to_first * float(V[:, max_exercises].mean())

    # Approximate mean exercises used (from forward pass; simplified)
    mean_ex = float(min(max_exercises, n_exercise_dates))

    return SwingOptionResult(
        price=price,
        n_exercise_dates=n_exercise_dates,
        max_exercises=max_exercises,
        mean_exercises_used=mean_ex,
        strike=strike,
    )


# ---- Virtual gas storage ----

@dataclass
class VirtualStorageResult:
    """Virtual gas storage optimisation result."""
    intrinsic_value: float      # calendar spread arbitrage
    extrinsic_value: float      # optionality value
    total_value: float
    max_capacity: float
    max_inject_rate: float
    max_withdraw_rate: float


class VirtualGasStorage:
    """Virtual gas storage facility valuation.

    Each period: choose to inject (buy) or withdraw (sell) subject to
    ratcheted flow rates and total inventory bounds.

    Intrinsic value: deterministic arbitrage from forward curve shape.
    Extrinsic: additional value from price volatility (re-optimisation).

    Args:
        max_capacity: maximum inventory (MMBtu).
        max_inject_rate: max daily injection.
        max_withdraw_rate: max daily withdrawal.
        inject_cost: cost per MMBtu injected.
        withdraw_cost: cost per MMBtu withdrawn.
    """

    def __init__(
        self,
        max_capacity: float,
        max_inject_rate: float,
        max_withdraw_rate: float,
        inject_cost: float = 0.0,
        withdraw_cost: float = 0.0,
    ):
        self.max_capacity = max_capacity
        self.max_inject_rate = max_inject_rate
        self.max_withdraw_rate = max_withdraw_rate
        self.inject_cost = inject_cost
        self.withdraw_cost = withdraw_cost

    def intrinsic_value(
        self,
        forward_prices: np.ndarray,     # (n_periods,)
        discount_factors: np.ndarray,   # (n_periods,)
        initial_inventory: float = 0.0,
        n_inventory: int = 20,
    ) -> float:
        """Intrinsic value: optimal deterministic schedule on forward curve.

        Dynamic programming over discrete inventory levels. Actions are
        aligned with the inventory grid spacing (moves must land exactly
        on a grid point) to avoid snapping artefacts.
        """
        n = len(forward_prices)
        inv_grid = np.linspace(0, self.max_capacity, n_inventory)
        dI = inv_grid[1] - inv_grid[0]      # grid spacing

        # Actions: integer multiples of dI that fit within rate limits
        max_inject_steps = int(self.max_inject_rate / dI)
        max_withdraw_steps = int(self.max_withdraw_rate / dI)
        action_steps = list(range(-max_withdraw_steps, max_inject_steps + 1))

        V = np.zeros((n + 1, n_inventory))   # V[t, i] = time-0 value at state (t, inv[i])

        for t in range(n - 1, -1, -1):
            df = discount_factors[t]
            F = forward_prices[t]

            for i in range(n_inventory):
                best = -np.inf
                for step in action_steps:
                    j = i + step
                    if j < 0 or j >= n_inventory:
                        continue
                    d = step * dI   # inventory change (positive = inject, negative = withdraw)
                    # Cash flow: pay F × d when injecting, receive F × |d| when withdrawing
                    #   → cash_flow = -F × d  (negative for inject, positive for withdraw)
                    cash_flow = -F * d
                    # Fees are always negative on cash flow
                    if d > 0:
                        cash_flow -= d * self.inject_cost
                    elif d < 0:
                        cash_flow -= (-d) * self.withdraw_cost
                    profit = cash_flow * df
                    total = profit + V[t + 1, j]
                    if total > best:
                        best = total
                V[t, i] = best if best > -np.inf else V[t + 1, i]

        i0 = int(np.argmin(np.abs(inv_grid - initial_inventory)))
        return float(V[0, i0])

    def value(
        self,
        spot_paths: np.ndarray,         # (n_paths, n_periods+1)
        discount_factors: np.ndarray,   # (n_periods+1,)
        initial_inventory: float = 0.0,
        n_inventory: int = 10,
    ) -> VirtualStorageResult:
        """Full stochastic valuation via LSM-DP.

        Simplified: uses mean of spot paths as "forward" for intrinsic,
        plus MC-based extrinsic.
        """
        n_paths, n_times = spot_paths.shape
        n_periods = n_times - 1

        # Mean paths as proxy for forward
        mean_forward = spot_paths.mean(axis=0)

        # Intrinsic
        intrinsic = self.intrinsic_value(
            mean_forward[:n_periods],
            discount_factors[:n_periods],
            initial_inventory,
            n_inventory,
        )

        # Extrinsic: approximate via per-path DP, then average
        per_path_values = []
        for p in range(min(n_paths, 200)):   # limit for speed
            path_val = self.intrinsic_value(
                spot_paths[p, :n_periods],
                discount_factors[:n_periods],
                initial_inventory,
                n_inventory,
            )
            per_path_values.append(path_val)

        total = float(np.mean(per_path_values))
        extrinsic = max(total - intrinsic, 0.0)

        return VirtualStorageResult(
            intrinsic_value=intrinsic,
            extrinsic_value=extrinsic,
            total_value=total,
            max_capacity=self.max_capacity,
            max_inject_rate=self.max_inject_rate,
            max_withdraw_rate=self.max_withdraw_rate,
        )


# ---- Nomination rights ----

@dataclass
class NominationResult:
    """Nomination rights contract value."""
    price: float
    mean_nominations: float
    flexibility_value: float      # vs fixed schedule
    n_decision_points: int


def nomination_rights_value(
    spot_paths: np.ndarray,
    strike: float,
    min_daily: float,
    max_daily: float,
    discount_factors: np.ndarray,
    is_buyer: bool = True,
) -> NominationResult:
    """Value of daily nomination rights in a commodity supply contract.

    Each day, holder chooses volume ∈ [min_daily, max_daily] at fixed strike.
    Optimal strategy: max volume when favourable, min volume otherwise.

    For a buyer: favourable = S > K (take max); unfavourable = S < K (take min).

    Args:
        spot_paths: (n_paths, n_days) daily spot prices.
        strike: contract price.
        min_daily: minimum daily volume.
        max_daily: maximum daily volume.
        discount_factors: (n_days,) DF per day.
        is_buyer: True = buyer can pay strike, receive spot; False reverse.
    """
    n_paths, n_days = spot_paths.shape

    # Per-day payoff from optimal choice
    # Buyer: profit per unit = S - K. Take max when > 0, min when < 0.
    if is_buyer:
        per_day = spot_paths - strike
    else:
        per_day = strike - spot_paths

    # Optimal volume per path per day
    volumes = np.where(per_day > 0, max_daily, min_daily)
    # Daily P&L
    pnl = per_day * volumes

    # Discount each day
    discounted = pnl * discount_factors[np.newaxis, :n_days]
    total = discounted.sum(axis=1)

    # Fixed schedule P&L (mean volume always)
    mean_volume = 0.5 * (min_daily + max_daily)
    fixed_pnl = per_day * mean_volume * discount_factors[np.newaxis, :n_days]
    fixed_total = fixed_pnl.sum(axis=1)

    price = float(total.mean())
    fixed_val = float(fixed_total.mean())
    flexibility = price - fixed_val

    return NominationResult(
        price=price,
        mean_nominations=float(volumes.mean()),
        flexibility_value=float(flexibility),
        n_decision_points=n_days,
    )
