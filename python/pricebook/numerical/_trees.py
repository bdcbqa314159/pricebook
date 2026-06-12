"""Unified tree framework — binomial and trinomial lattice pricing.

Class-based solver with method selection, exercise types, barriers,
discrete dividends, Greeks from tree nodes, and convergence analysis.

    from pricebook.numerical._trees import (
        TreeSolver, TreeMethod, ExerciseType, TreeResult,
        solve_tree, solve_tree_2d,
    )

    # One-liner: CRR European call
    result = solve_tree(100, 100, 0.04, 0.25, 1.0)

    # Configurable: LR American put with barrier
    solver = TreeSolver(TreeMethod.LR, n_steps=201, exercise=ExerciseType.AMERICAN,
                         barrier_type=BarrierType.DOWN_OUT, barrier_level=80)
    result = solver.solve(100, 100, 0.04, 0.25, 1.0, is_call=False)

References:
    Cox, Ross & Rubinstein (1979). Option Pricing: A Simplified Approach.
    Leisen & Reimer (1996). Binomial Models for Option Valuation.
    Tian (1993). A Modified Lattice Approach to Option Pricing. JFQA.
    Kamrad & Ritchken (1991). Multinomial Approximations. Management Science.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════


class TreeMethod(Enum):
    CRR = "crr"                  # Cox-Ross-Rubinstein
    JR = "jr"                    # Jarrow-Rudd (equal prob)
    LR = "lr"                    # Leisen-Reimer (fast convergence)
    TRINOMIAL = "trinomial"      # Kamrad-Ritchken
    TIAN = "tian"                # Tian moment-matching


class ExerciseType(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"


class BarrierType(Enum):
    UP_OUT = "up_out"
    DOWN_OUT = "down_out"
    UP_IN = "up_in"
    DOWN_IN = "down_in"


# ═══════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════


@dataclass
class TreeResult:
    """Tree pricing result with Greeks."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float | None = None
    method: str = ""
    n_steps: int = 0
    exercise: str = ""
    convergence: dict | None = None
    node_prices: np.ndarray | None = None
    spot_tree: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "price": self.price, "delta": self.delta, "gamma": self.gamma,
            "theta": self.theta, "vega": self.vega,
            "method": self.method, "n_steps": self.n_steps, "exercise": self.exercise,
        }


# ═══════════════════════════════════════════════════════════════
# Parameter functions
# ═══════════════════════════════════════════════════════════════


def _crr_params(r, q, vol, dt):
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    return u, d, max(0, min(1, p)), disc


def _jr_params(r, q, vol, dt):
    drift = (r - q - 0.5 * vol**2) * dt
    u = math.exp(drift + vol * math.sqrt(dt))
    d = math.exp(drift - vol * math.sqrt(dt))
    disc = math.exp(-r * dt)
    return u, d, 0.5, disc


def _tian_params(r, q, vol, dt):
    """Tian (1993) moment-matching binomial parameters.

    Fix T1.5: pre-fix `V` was computed as `M² × (exp(σ²·dt) − 1)` — the
    *variance* of the lognormal — and the u/d formula was missing the V
    multiplier. The discriminant `V² + 2V − 3` was nearly always negative
    for typical parameters, so the function silently fell through to CRR
    via the "very small dt × vol" branch — meaning the Tian method had
    been running CRR for every call. Now matches the Tian (1993) JFQA
    derivation:

        M = exp((r − q)·dt)
        V = exp(σ²·dt)
        u = M·V/2 · (V + 1 + √(V² + 2V − 3))
        d = M·V/2 · (V + 1 − √(V² + 2V − 3))
    """
    M = math.exp((r - q) * dt)
    V = math.exp(vol ** 2 * dt)
    disc_arg = V ** 2 + 2 * V - 3
    if disc_arg < 0:
        # Degenerate (V < 1, ie dt × σ² very near zero): fall back to CRR.
        return _crr_params(r, q, vol, dt)
    u = 0.5 * M * V * (V + 1 + math.sqrt(disc_arg))
    d = 0.5 * M * V * (V + 1 - math.sqrt(disc_arg))
    disc = math.exp(-r * dt)
    p = (M - d) / (u - d) if abs(u - d) > 1e-15 else 0.5
    return u, d, max(0, min(1, p)), disc


def _peizer_pratt(z, n):
    if abs(z) < 1e-10:
        return 0.5
    m = n + 1.0/3 + 0.1/(n + 1)
    return 0.5 + math.copysign(
        math.sqrt(0.25 - 0.25 * math.exp(-(z / m)**2 * (n + 1.0/6))),
        z,
    )


def _lr_params(spot, strike, r, q, vol, T, n_steps):
    if n_steps % 2 == 0:
        n_steps += 1
    dt = T / n_steps
    d1 = (math.log(spot / strike) + (r - q + 0.5*vol**2)*T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    p_prime = _peizer_pratt(d1, n_steps)
    p = _peizer_pratt(d2, n_steps)
    ern = math.exp((r - q) * dt)
    u = ern * p_prime / p
    d = (ern - p * u) / (1.0 - p)
    disc = math.exp(-r * dt)
    return u, d, max(0.001, min(0.999, p)), disc, n_steps


def _trinomial_params(r, q, vol, dt, lam=None):
    if lam is None:
        lam = math.sqrt(1.5)
    u = math.exp(lam * vol * math.sqrt(dt))
    d = 1.0 / u
    nu = (r - q - 0.5*vol**2) * math.sqrt(dt) / (lam * vol) if vol > 0 else 0
    p_u = 1.0 / (2 * lam**2) + nu / 2
    p_d = 1.0 / (2 * lam**2) - nu / 2
    p_m = 1.0 - p_u - p_d
    disc = math.exp(-r * dt)
    p_u = max(0, min(1, p_u))
    p_d = max(0, min(1, p_d))
    p_m = max(0, 1 - p_u - p_d)
    return u, d, p_u, p_m, p_d, disc


# ═══════════════════════════════════════════════════════════════
# TreeSolver
# ═══════════════════════════════════════════════════════════════


class TreeSolver:
    """Configurable lattice solver for option pricing.

    Args:
        method: tree construction method.
        n_steps: number of time steps.
        exercise: exercise type.
        exercise_dates: for Bermudan, list of step indices where exercise is allowed.
        barrier_type: optional barrier.
        barrier_level: barrier price level.
        dividends: list of (step_index, amount) for discrete dividends.
        store_tree: if True, store full node prices in result.
    """

    def __init__(
        self,
        method: TreeMethod = TreeMethod.CRR,
        n_steps: int = 200,
        exercise: ExerciseType = ExerciseType.EUROPEAN,
        exercise_dates: list[int] | None = None,
        barrier_type: BarrierType | None = None,
        barrier_level: float | None = None,
        dividends: list[tuple[int, float]] | None = None,
        store_tree: bool = False,
    ):
        self.method = method
        self.n_steps = n_steps
        self.exercise = exercise
        self.exercise_dates = set(exercise_dates or [])
        self.barrier_type = barrier_type
        self.barrier_level = barrier_level
        self.dividends = dict(dividends or [])
        self.store_tree = store_tree
        self._computing_vega = False

    def solve(
        self,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        T: float,
        payoff: callable | None = None,
        is_call: bool = True,
        div_yield: float = 0.0,
    ) -> TreeResult:
        """Price an option via tree."""
        if self.method == TreeMethod.TRINOMIAL:
            return self._solve_trinomial(spot, strike, rate, vol, T, payoff, is_call, div_yield)
        else:
            return self._solve_binomial(spot, strike, rate, vol, T, payoff, is_call, div_yield)

    def _solve_binomial(self, spot, strike, rate, vol, T, payoff_fn, is_call, div_yield):
        n = self.n_steps
        dt = T / n

        # Get parameters
        if self.method == TreeMethod.LR:
            u, d, p, disc, n = _lr_params(spot, strike, rate, div_yield, vol, T, n)
            dt = T / n
        elif self.method == TreeMethod.JR:
            u, d, p, disc = _jr_params(rate, div_yield, vol, dt)
        elif self.method == TreeMethod.TIAN:
            u, d, p, disc = _tian_params(rate, div_yield, vol, dt)
        else:
            u, d, p, disc = _crr_params(rate, div_yield, vol, dt)

        # Build spot tree at maturity (index 0 = all down, index n = all up)
        S = np.array([spot * d**(n - j) * u**j for j in range(n + 1)])

        # Apply discrete dividends (shift spots)
        for step, amount in self.dividends.items():
            if step <= n:
                # Approximate: reduce all spots proportionally
                S = np.maximum(S - amount, 0.01)

        # Terminal payoff
        if payoff_fn is not None:
            V = np.array([payoff_fn(s) for s in S])
        elif is_call:
            V = np.maximum(S - strike, 0.0)
        else:
            V = np.maximum(strike - S, 0.0)

        # Store for tree inspection
        stored = [V.copy()] if self.store_tree else None

        # Snapshots for Greeks (steps n, n-1, n-2)
        V_snapshots = {n: V.copy()}

        # Backward induction (index 0 = min S, index i = more up-moves)
        for step in range(n - 1, -1, -1):
            V = disc * (p * V[1:] + (1 - p) * V[:-1])

            # Spot prices at this step (same convention: 0 = all down)
            S_step = np.array([spot * d**(step - j) * u**j for j in range(step + 1)])

            # Exercise
            if self._should_exercise(step):
                if payoff_fn is not None:
                    intrinsic = np.array([payoff_fn(s) for s in S_step])
                elif is_call:
                    intrinsic = np.maximum(S_step - strike, 0.0)
                else:
                    intrinsic = np.maximum(strike - S_step, 0.0)
                V = np.maximum(V, intrinsic)

            # Barrier
            if self.barrier_type is not None and self.barrier_level is not None:
                V = self._apply_barrier(V, S_step)

            # Store snapshots for Greeks
            if step <= 2:
                V_snapshots[step] = V.copy()

            if self.store_tree and stored is not None:
                stored.append(V.copy())

        price = float(V[0])

        # Greeks from tree nodes
        delta, gamma, theta = self._extract_greeks_binomial(
            V_snapshots, spot, u, d, dt, n)

        # Vega via bump (skip if already in vega computation)
        vega = None
        if not self._computing_vega:
            vega = self._compute_vega(spot, strike, rate, vol, T, payoff_fn, is_call, div_yield, price)

        return TreeResult(
            price=price, delta=delta, gamma=gamma, theta=theta, vega=vega,
            method=self.method.value, n_steps=n, exercise=self.exercise.value,
            node_prices=stored[-1] if stored else None,
        )

    def _solve_trinomial(self, spot, strike, rate, vol, T, payoff_fn, is_call, div_yield):
        n = self.n_steps
        dt = T / n
        u, d, p_u, p_m, p_d, disc = _trinomial_params(rate, div_yield, vol, dt)

        # Terminal: 2n+1 nodes
        S = np.array([spot * u**(n - i) for i in range(2 * n + 1)])
        # Adjust: node j corresponds to S × u^(n-j)
        S = np.array([spot * u**j for j in range(n, -n - 1, -1)])

        if payoff_fn is not None:
            V = np.array([payoff_fn(s) for s in S])
        elif is_call:
            V = np.maximum(S - strike, 0.0)
        else:
            V = np.maximum(strike - S, 0.0)

        V_snapshots = {n: V.copy()}

        for step in range(n - 1, -1, -1):
            n_nodes = 2 * step + 1
            V_new = np.zeros(n_nodes)
            for j in range(n_nodes):
                V_new[j] = disc * (p_u * V[j] + p_m * V[j + 1] + p_d * V[j + 2])

            S_step = np.array([spot * u**j for j in range(step, -step - 1, -1)])

            if self._should_exercise(step):
                if payoff_fn is not None:
                    intrinsic = np.array([payoff_fn(s) for s in S_step])
                elif is_call:
                    intrinsic = np.maximum(S_step - strike, 0.0)
                else:
                    intrinsic = np.maximum(strike - S_step, 0.0)
                V_new = np.maximum(V_new, intrinsic)

            if self.barrier_type is not None and self.barrier_level is not None:
                V_new = self._apply_barrier(V_new, S_step)

            V = V_new
            if step <= 2:
                V_snapshots[step] = V.copy()

        price = float(V[0])
        delta, gamma, theta = self._extract_greeks_trinomial(
            V_snapshots, spot, u, dt, n)
        vega = None
        if not self._computing_vega:
            vega = self._compute_vega(spot, strike, rate, vol, T, payoff_fn, is_call, div_yield, price)

        return TreeResult(
            price=price, delta=delta, gamma=gamma, theta=theta, vega=vega,
            method=self.method.value, n_steps=n, exercise=self.exercise.value,
        )

    def _should_exercise(self, step):
        if self.exercise == ExerciseType.AMERICAN:
            return True
        if self.exercise == ExerciseType.BERMUDAN:
            return step in self.exercise_dates
        return False

    def _apply_barrier(self, V, S_step):
        if self.barrier_type == BarrierType.DOWN_OUT:
            V[S_step <= self.barrier_level] = 0.0
        elif self.barrier_type == BarrierType.UP_OUT:
            V[S_step >= self.barrier_level] = 0.0
        elif self.barrier_type == BarrierType.DOWN_IN:
            # Knock-in: price = vanilla - knock-out (via in-out parity later)
            pass  # complex — for now, only knock-out supported
        elif self.barrier_type == BarrierType.UP_IN:
            pass
        return V

    def _extract_greeks_binomial(self, snaps, spot, u, d, dt, n):
        delta = gamma = theta = 0.0
        # Convention: index 0 = all down (min S), index n = all up (max S)
        if 1 in snaps and len(snaps[1]) >= 2:
            S_d = spot * d
            S_u = spot * u
            V_d = snaps[1][0]   # index 0 = down
            V_u = snaps[1][1]   # index 1 = up
            delta = (V_u - V_d) / (S_u - S_d) if S_u != S_d else 0.0

        if 2 in snaps and len(snaps[2]) >= 3:
            S_dd = spot * d**2
            S_ud = spot * u * d
            S_uu = spot * u**2
            V_dd = snaps[2][0]
            V_ud = snaps[2][1]
            V_uu = snaps[2][2]
            delta_u = (V_uu - V_ud) / (S_uu - S_ud) if S_uu != S_ud else 0.0
            delta_d = (V_ud - V_dd) / (S_ud - S_dd) if S_ud != S_dd else 0.0
            gamma = (delta_u - delta_d) / (0.5 * (S_uu - S_dd)) if S_uu != S_dd else 0.0
            theta = (V_ud - snaps[0][0]) / (2 * dt) if 0 in snaps else 0.0

        return float(delta), float(gamma), float(theta)

    def _extract_greeks_trinomial(self, snaps, spot, u, dt, n):
        delta = gamma = theta = 0.0
        if 1 in snaps and len(snaps[1]) >= 3:
            S_u = spot * u
            S_m = spot
            S_d = spot / u
            V_u = snaps[1][0]
            V_d = snaps[1][2]
            delta = (V_u - V_d) / (S_u - S_d) if S_u != S_d else 0.0

        if 2 in snaps and len(snaps[2]) >= 5:
            S_uu = spot * u**2
            S_dd = spot / u**2
            V_uu = snaps[2][0]
            V_mm = snaps[2][2]
            V_dd = snaps[2][4]
            gamma = (V_uu - 2*V_mm + V_dd) / ((spot * (u - 1/u))**2) if u != 1 else 0.0
            theta = (V_mm - snaps[0][0]) / (2 * dt) if 0 in snaps else 0.0

        return float(delta), float(gamma), float(theta)

    def _compute_vega(self, spot, strike, rate, vol, T, payoff_fn, is_call, div_yield, base_price):
        bump = 0.01
        self._computing_vega = True
        saved_store = self.store_tree
        self.store_tree = False
        r_up = self.solve(spot, strike, rate, vol + bump, T, payoff_fn, is_call, div_yield)
        self.store_tree = saved_store
        self._computing_vega = False
        return (r_up.price - base_price) / bump

    def convergence_analysis(
        self,
        spot, strike, rate, vol, T,
        n_steps_list: list[int] | None = None,
        **kwargs,
    ) -> dict:
        """Run at multiple N and Richardson-extrapolate."""
        if n_steps_list is None:
            n_steps_list = [50, 100, 200, 400]

        prices = []
        original_n = self.n_steps
        for n in n_steps_list:
            self.n_steps = n
            r = self.solve(spot, strike, rate, vol, T, **kwargs)
            prices.append(r.price)
        self.n_steps = original_n

        # Richardson: P* = (4P(2N) - P(N)) / 3
        richardson = None
        if len(prices) >= 2:
            richardson = (4 * prices[-1] - prices[-2]) / 3

        return {
            "n_steps": n_steps_list,
            "prices": prices,
            "richardson": richardson,
            "estimated_error": abs(prices[-1] - prices[-2]) if len(prices) >= 2 else None,
        }


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════


def solve_tree(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    method: TreeMethod = TreeMethod.CRR,
    n_steps: int = 200,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    is_call: bool = True,
    payoff: callable | None = None,
    div_yield: float = 0.0,
    barrier_type: BarrierType | None = None,
    barrier_level: float | None = None,
    dividends: list[tuple[int, float]] | None = None,
    store_tree: bool = False,
) -> TreeResult:
    """One-liner tree pricing."""
    solver = TreeSolver(method, n_steps, exercise,
                         barrier_type=barrier_type, barrier_level=barrier_level,
                         dividends=dividends, store_tree=store_tree)
    return solver.solve(spot, strike, rate, vol, T, payoff, is_call, div_yield)


def solve_tree_2d(
    S1: float, S2: float, strike: float,
    rate: float, vol1: float, vol2: float, rho: float, T: float,
    n_steps: int = 50,
    payoff: callable | None = None,
    payoff_type: str = "spread_call",
    div_yield1: float = 0.0, div_yield2: float = 0.0,
    is_american: bool = False,
) -> TreeResult:
    """Two-asset binomial tree (Rubinstein 1994)."""
    dt = T / n_steps
    u1 = math.exp(vol1 * math.sqrt(dt))
    d1 = 1.0 / u1
    u2 = math.exp(vol2 * math.sqrt(dt))
    d2 = 1.0 / u2

    mu1 = (rate - div_yield1 - 0.5 * vol1**2) * dt
    mu2 = (rate - div_yield2 - 0.5 * vol2**2) * dt
    sqrt_dt = math.sqrt(dt)

    p_uu = max(0, 0.25 * (1 + rho + mu1/(vol1*sqrt_dt) + mu2/(vol2*sqrt_dt)))
    p_ud = max(0, 0.25 * (1 - rho + mu1/(vol1*sqrt_dt) - mu2/(vol2*sqrt_dt)))
    p_du = max(0, 0.25 * (1 - rho - mu1/(vol1*sqrt_dt) + mu2/(vol2*sqrt_dt)))
    p_dd = max(0, 1.0 - p_uu - p_ud - p_du)
    p_total = p_uu + p_ud + p_du + p_dd
    if p_total > 0:
        p_uu /= p_total; p_ud /= p_total; p_du /= p_total; p_dd /= p_total

    df = math.exp(-rate * dt)
    n = n_steps

    # Terminal
    values = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            s1 = S1 * u1**(n-i) * d1**i
            s2 = S2 * u2**(n-j) * d2**j
            if payoff is not None:
                values[i, j] = payoff(s1, s2)
            elif payoff_type == "spread_call":
                values[i, j] = max(s1 - s2 - strike, 0)
            elif payoff_type == "spread_put":
                values[i, j] = max(strike - (s1 - s2), 0)
            elif payoff_type == "best_of_call":
                values[i, j] = max(max(s1, s2) - strike, 0)
            elif payoff_type == "worst_of_call":
                values[i, j] = max(min(s1, s2) - strike, 0)

    for step in range(n - 1, -1, -1):
        new_v = np.zeros((step + 1, step + 1))
        for i in range(step + 1):
            for j in range(step + 1):
                new_v[i, j] = df * (p_uu*values[i,j] + p_ud*values[i,j+1]
                                     + p_du*values[i+1,j] + p_dd*values[i+1,j+1])
                if is_american:
                    s1 = S1 * u1**(step-i) * d1**i
                    s2 = S2 * u2**(step-j) * d2**j
                    if payoff is not None:
                        new_v[i,j] = max(new_v[i,j], payoff(s1, s2))
                    elif payoff_type == "spread_call":
                        new_v[i,j] = max(new_v[i,j], max(s1-s2-strike, 0))
        values = new_v

    return TreeResult(price=float(values[0, 0]), delta=0.0, gamma=0.0, theta=0.0,
                       method="binomial_2d", n_steps=n_steps, exercise="european" if not is_american else "american")


