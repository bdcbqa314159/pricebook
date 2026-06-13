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
    """Trinomial-tree parameters.

    Fix T2.9: pre-fix the probability clamp was `p_u = max(0, min(1, p_u))`,
    `p_d = max(0, min(1, p_d))`, `p_m = max(0, 1 − p_u − p_d)`.  When the raw
    formula produced negative `p_u` or `p_d` (large drift relative to
    volatility), clamping shifted mass into `p_m` without renormalising —
    the resulting triple still summed to 1 but the *moments* (drift, variance)
    were broken, breaking the risk-neutral measure silently.

    Post-fix: clamp each leg to ≥ 0, then renormalise the triple so that the
    sum is 1 and the moments stay as close to risk-neutral as possible given
    the clamp.
    """
    if lam is None:
        lam = math.sqrt(1.5)
    u = math.exp(lam * vol * math.sqrt(dt))
    d = 1.0 / u
    nu = (r - q - 0.5*vol**2) * math.sqrt(dt) / (lam * vol) if vol > 0 else 0
    p_u_raw = 1.0 / (2 * lam**2) + nu / 2
    p_d_raw = 1.0 / (2 * lam**2) - nu / 2
    p_m_raw = 1.0 - p_u_raw - p_d_raw
    disc = math.exp(-r * dt)

    # Clamp each leg ≥ 0, then renormalise.  Mirrors the renormalisation
    # already used in `_2d_trinomial_params` for the 2D tree below.
    p_u = max(0.0, p_u_raw)
    p_d = max(0.0, p_d_raw)
    p_m = max(0.0, p_m_raw)
    total = p_u + p_d + p_m
    if total > 0:
        p_u /= total
        p_d /= total
        p_m /= total
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
        # Fix T2.8: knock-in barriers via in-out parity.  Pre-fix
        # `_apply_barrier` had `pass` for DOWN_IN / UP_IN, so the barrier
        # condition was checked but never enforced — knock-ins were silently
        # priced as the corresponding vanilla.  The parity
        #     V_knock_in + V_knock_out = V_vanilla
        # gives V_knock_in = V_vanilla − V_knock_out (under the same expiry
        # and exercise convention).  Run the knock-out and vanilla pricers
        # and return the difference, with Greeks combined by linearity.
        if self.barrier_type in (BarrierType.DOWN_IN, BarrierType.UP_IN):
            ko_type = (BarrierType.DOWN_OUT
                       if self.barrier_type == BarrierType.DOWN_IN
                       else BarrierType.UP_OUT)
            divs = [(s, a) for s, a in self.dividends.items()]
            exer_dates = list(self.exercise_dates)
            ko_solver = TreeSolver(
                method=self.method, n_steps=self.n_steps,
                exercise=self.exercise, exercise_dates=exer_dates,
                barrier_type=ko_type, barrier_level=self.barrier_level,
                dividends=divs, store_tree=False,
            )
            vanilla_solver = TreeSolver(
                method=self.method, n_steps=self.n_steps,
                exercise=self.exercise, exercise_dates=exer_dates,
                dividends=divs, store_tree=False,
            )
            ko = ko_solver.solve(spot, strike, rate, vol, T, payoff, is_call, div_yield)
            van = vanilla_solver.solve(spot, strike, rate, vol, T, payoff, is_call, div_yield)
            vega = None
            if van.vega is not None and ko.vega is not None:
                vega = van.vega - ko.vega
            return TreeResult(
                price=van.price - ko.price,
                delta=van.delta - ko.delta,
                gamma=van.gamma - ko.gamma,
                theta=van.theta - ko.theta,
                vega=vega,
                method=self.method.value, n_steps=self.n_steps,
                exercise=self.exercise.value,
            )

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

        # Cumulative-dividend helper. At step `s`, the underlying has paid all
        # dividends scheduled at-or-before `s` (escrowed-dividend convention).
        # Fix T1.6 (pre-fix the dividends were applied only to the TERMINAL
        # grid, leaving intermediate-step `S_step` unchanged — so early-
        # exercise / barrier checks at step `s` saw the pre-dividend spot
        # even when dividends had been paid by then).
        def _cum_div_through(s: int) -> float:
            return sum(amt for step, amt in self.dividends.items() if step <= s)

        def _spot_at_step(s: int) -> np.ndarray:
            grid = np.array([spot * d ** (s - j) * u ** j for j in range(s + 1)])
            cum = _cum_div_through(s)
            if cum > 0:
                grid = np.maximum(grid - cum, 0.01)
            return grid

        # Build spot tree at maturity (index 0 = all down, index n = all up),
        # with all dividends through maturity subtracted.
        S = _spot_at_step(n)

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

            # Spot prices at this step (same convention: 0 = all down),
            # WITH dividends paid at-or-before `step` subtracted.
            S_step = _spot_at_step(step)

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

        # Fix T1.6: pre-fix the trinomial branch ignored `self.dividends`
        # entirely — `_solve_binomial` at least applied them at the terminal,
        # but the trinomial branch had no dividend handling whatsoever.
        # Apply the same escrowed-dividend convention here.
        def _cum_div_through(s: int) -> float:
            return sum(amt for step, amt in self.dividends.items() if step <= s)

        def _spot_at_step(s: int) -> np.ndarray:
            grid = np.array([spot * u ** j for j in range(s, -s - 1, -1)])
            cum = _cum_div_through(s)
            if cum > 0:
                grid = np.maximum(grid - cum, 0.01)
            return grid

        # Terminal: 2n+1 nodes, escrowed for all dividends through maturity.
        S = _spot_at_step(n)

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

            # Escrowed spot at this step (dividends at-or-before `step` subtracted).
            S_step = _spot_at_step(step)

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
        """Vega via 1-vol-point bump.

        Fix T4-TR2: pre-fix this routine set ``self._computing_vega = True``
        and ``self.store_tree = False`` and only restored them on the happy
        path.  If the inner ``self.solve(...)`` raised, the solver was left
        in a "computing vega" state with no snapshots, silently breaking
        every subsequent ``solve()`` call until reconstruction.  Same
        shape as the MCEngine.greek exception-safety bug fixed in v0.946.
        Now wrapped in try/finally.
        """
        bump = 0.01
        saved_computing = self._computing_vega
        saved_store = self.store_tree
        self._computing_vega = True
        self.store_tree = False
        try:
            r_up = self.solve(spot, strike, rate, vol + bump, T, payoff_fn, is_call, div_yield)
        finally:
            self.store_tree = saved_store
            self._computing_vega = saved_computing
        return (r_up.price - base_price) / bump

    def convergence_analysis(
        self,
        spot, strike, rate, vol, T,
        n_steps_list: list[int] | None = None,
        **kwargs,
    ) -> dict:
        """Run at multiple N and extrapolate.

        The extrapolated value depends on the method's convergence order:

        - **LR** (Leisen-Reimer): smooth O(1/N²) convergence — Richardson
          ``P* = (4·P(2N) - P(N)) / 3`` cancels the leading-order error.
        - **CRR / JR**: oscillatory O(1/N) convergence (sawtooth between
          odd and even N).  Richardson assumes O(1/N²) and OVER-amplifies
          the oscillation; instead, the standard remedy is to AVERAGE
          adjacent N values to suppress the parity oscillation.  Here we
          report a simple average of the last two grid points along with
          a note explaining the choice.
        - **Tian, Trinomial**: behave similarly to CRR/JR for vanilla
          payoffs (O(1/N) with oscillation) — same average-based smoothing.

        Fix T4-TR1: pre-fix this routine unconditionally applied the
        Richardson formula regardless of the method's convergence order,
        producing meaningless "extrapolated" values for CRR/JR/TIAN/
        TRINOMIAL.  It also did not validate that ``n_steps_list[-1]``
        was twice ``n_steps_list[-2]`` — without that, the "4·P(2N) − P(N)"
        weighting has no theoretical basis at all.  Now we validate and
        select the appropriate extrapolation per method.
        """
        if n_steps_list is None:
            n_steps_list = [50, 100, 200, 400]

        if len(n_steps_list) < 2:
            raise ValueError(
                "convergence_analysis requires at least 2 grid sizes "
                f"(got n_steps_list={n_steps_list})."
            )
        for a, b in zip(n_steps_list, n_steps_list[1:]):
            if b <= a:
                raise ValueError(
                    "n_steps_list must be strictly increasing "
                    f"(got {n_steps_list})."
                )

        prices = []
        original_n = self.n_steps
        try:
            for n in n_steps_list:
                self.n_steps = n
                r = self.solve(spot, strike, rate, vol, T, **kwargs)
                prices.append(r.price)
        finally:
            self.n_steps = original_n

        N_prev, N_last = n_steps_list[-2], n_steps_list[-1]
        doubling = (N_last == 2 * N_prev)

        # Legacy `richardson` key: literal Richardson formula on the last
        # two prices.  Theoretically valid only for smooth O(1/N²) methods
        # (LR) with doubling grids; reported regardless so existing callers
        # see no break.
        richardson_legacy = (4.0 * prices[-1] - prices[-2]) / 3.0

        # `extrapolated`: the choice this routine endorses for the given
        # method.  This is the value a calibration / model-validation
        # consumer should trust.
        if self.method == TreeMethod.LR and doubling:
            extrapolated = richardson_legacy
            extrapolation_method = "richardson_O(1/N^2)"
        elif self.method == TreeMethod.LR:
            extrapolated = prices[-1]
            extrapolation_method = "no_extrapolation_non_doubling"
        else:
            # CRR / JR / Tian / Trinomial: oscillatory O(1/N).  Average
            # cancels the parity sawtooth; Richardson formula is not
            # theoretically justified here.
            extrapolated = 0.5 * (prices[-1] + prices[-2])
            extrapolation_method = "average_O(1/N)_oscillation_suppression"

        return {
            "n_steps": n_steps_list,
            "prices": prices,
            "richardson": richardson_legacy,
            "extrapolated": extrapolated,
            "extrapolation_method": extrapolation_method,
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
    exercise_dates: list[int] | None = None,
) -> TreeResult:
    """One-liner tree pricing.

    Fix T3.10: pre-fix this convenience wrapper did not accept
    ``exercise_dates``.  A caller passing ``exercise=BERMUDAN`` with the
    intended exercise schedule got a TreeSolver built with empty
    ``exercise_dates``, and ``_should_exercise`` returned False at every
    step for BERMUDAN — silently degrading Bermudan options to European.
    """
    solver = TreeSolver(method, n_steps, exercise,
                         exercise_dates=exercise_dates,
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

    # Fix T3.9: pre-fix the American projection only handled "spread_call".
    # spread_put, best_of_call, worst_of_call, and custom payoff callables
    # were silently treated as European inside the backward iteration.
    def _intrinsic_2d(s1: float, s2: float) -> float:
        if payoff is not None:
            return payoff(s1, s2)
        if payoff_type == "spread_call":
            return max(s1 - s2 - strike, 0.0)
        if payoff_type == "spread_put":
            return max(strike - (s1 - s2), 0.0)
        if payoff_type == "best_of_call":
            return max(max(s1, s2) - strike, 0.0)
        if payoff_type == "worst_of_call":
            return max(min(s1, s2) - strike, 0.0)
        return 0.0

    values_step1 = None  # for T3.8 Greeks readout
    for step in range(n - 1, -1, -1):
        new_v = np.zeros((step + 1, step + 1))
        for i in range(step + 1):
            for j in range(step + 1):
                new_v[i, j] = df * (p_uu*values[i,j] + p_ud*values[i,j+1]
                                     + p_du*values[i+1,j] + p_dd*values[i+1,j+1])
                if is_american:
                    s1 = S1 * u1**(step-i) * d1**i
                    s2 = S2 * u2**(step-j) * d2**j
                    new_v[i, j] = max(new_v[i, j], _intrinsic_2d(s1, s2))
        # Capture the step-1 grid AFTER it has been built (i.e. when we are
        # about to step from 1 down to 0, `new_v` is the step-0 grid and
        # `values` is still the step-1 grid).  Save before overwriting.
        if step == 0:
            values_step1 = values
        values = new_v

    # Fix T3.8: pre-fix this returned delta=gamma=theta=0 always.  Extract
    # delta1 and delta2 from the step-1 grid (4 nodes), averaging out the
    # spot direction we are NOT differentiating with respect to.
    # Gamma and theta on a 2-asset recombining tree are noisy without
    # additional refinement (the step-1 grid has no centred node at (S1, S2)
    # — only 4 shifted nodes — so straight ∂V/∂t mixes spot-diffusion
    # convexity with time decay). Left at 0 here; computed via bump+reprice
    # if needed.
    delta1, delta2 = 0.0, 0.0
    if values_step1 is not None and values_step1.shape == (2, 2):
        s1_u, s1_d = S1 * u1, S1 * d1
        s2_u, s2_d = S2 * u2, S2 * d2
        # delta1 = ∂V/∂S1 from averaging over the S2 direction at step 1.
        v_at_S1u = 0.5 * (values_step1[0, 0] + values_step1[0, 1])
        v_at_S1d = 0.5 * (values_step1[1, 0] + values_step1[1, 1])
        delta1 = (v_at_S1u - v_at_S1d) / (s1_u - s1_d)
        v_at_S2u = 0.5 * (values_step1[0, 0] + values_step1[1, 0])
        v_at_S2d = 0.5 * (values_step1[0, 1] + values_step1[1, 1])
        delta2 = (v_at_S2u - v_at_S2d) / (s2_u - s2_d)

    return TreeResult(
        price=float(values[0, 0]),
        delta=float(delta1), gamma=0.0, theta=0.0, vega=None,
        method="binomial_2d", n_steps=n_steps,
        exercise="american" if is_american else "european",
        node_prices=np.array([float(delta1), float(delta2)]),
    )


