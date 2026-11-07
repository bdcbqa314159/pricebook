"""The Monte Carlo path engine: LMM forwards evolved and averaged (L4). JAECKEL-2002 eq 13.2.

Evolves the forward curve over a time grid in **logarithmic coordinates** — Jäckel's words for
eq 12.34 are *"the Euler scheme in logarithmic coordinates"*, with the `√Δt` and `Δt` scalings
absorbed into `A` and `C` — and averages a payoff into an estimate with its standard error:

    f_i(t+Δt) = f_i(t)·exp( µ_i(f(t))·Δt − ½c_ii + Σ_j a_ij z_j )                      (13.2)
    c_ij      = σ_i σ_j ρ_ij Δt                                                        (13.3)
    A         = pseudo_square_root(C, m)                                               (12.4/12.7)

The drift is **recomputed from the current state at every step**, because eq 12.8 is state-dependent
(*"clearly state-dependent and thus indirectly stochastic"*). Freezing it at `f(0)` would be a
different model, and the oracle that catches that is the martingale property, not the price.

**L4 and extracted now because there are two consumers, not one.** `models/black_mc.py` (tranche 15)
and this share the evolve-and-average shape — rule-of-two, the same timing as Black in tranche 9 and
the Hull-White core in tranche 10. Tranche 15 deliberately declined an L4 engine because a single
consumer made it speculative surface (§6); the second consumer is what earns it.

**`black_mc` is NOT retrofitted onto this, and the reason is measured rather than asserted.** It draws
the terminal value in one step from a total variance; this engine walks a grid accumulating per-step
covariances. The two orders of floating-point operations differ, so the refactor cannot be
bit-identical — and charter §4 says a pure refactor is proved by an exact-arithmetic identity plus
bounded agreement, never by a claim. The identity holds (one lognormal step with variance `V²` equals
`n` steps of `V²/n`), but the ULP gap is real, so `black_mc` stays as it is. Collapsing them would
buy a line of shared code at the cost of a tranche-15 oracle's bit-exactness.

**Why the caplet anchor is sharp, which is the one thing worth understanding here.** Choose the
numéraire so the target forward has exactly zero drift (`N = i+1`, eq 12.8's middle branch). Its step
then reduces to `exp(−½c_ii + Σ_j a_ij z_j)`, an **exact** lognormal martingale at any step count:
the per-step variances sum to `σ_i²T_i` and the `−½c` terms sum to match. So that forward carries
**no discretisation bias at all**, and the only gap to `black_price` is Monte Carlo error — which is
what lets oracle 1 be a sharp `|MC − Black| < k·SE`. The engine's own oracle asserts this by showing
one step and thirty-two give the same answer.

Provenance:
  stock:  parked/ng/.../swaption_mc.py, models/mc_processes.py — SHAPE and edge cases only, via
          docs/stock/mc_base_inventory.md; never imported (§5)
  source: JAECKEL-2002 (docs/references.md) §13.2 eq 13.2 and 13.3 (the log-Euler step and the step
          covariance) and §12.5 eq 12.32/12.34 (the SDE and the log-coordinate reading), all read in
          `-raw` and `-layout`, which agree
  oracle: tests/L4/test_mc_path_engine.py — a caplet repricing to OUR `black_price` within four
          MEASURED standard errors at three strikes, with a non-vacuous-band companion and a
          strike-monotonicity check; the target forward a martingale under its own measure, with the
          companion that the OTHER forwards are not (so the drift is no no-op) and that the drift's
          SIGN matches eq 12.8 either side of the numéraire; the −½ rate; the step-count
          independence that proves the exactness claimed above; determinism; and positivity, which
          log-Euler guarantees where plain Euler does not
  crosscheck: quantlib LiborForwardModel / MarketModelEvolver — deferred to the optional QuantLib tier
  slice:  tranche 16 — the LIBOR market model by Monte Carlo
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pricebook.foundation.brownian_bridge import BrownianBridge
from pricebook.foundation.distributions import inverse_norm_cdf
from pricebook.foundation.linalg import pseudo_square_root
from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.sobol import SOBOL_MAX_DIMENSIONS, SobolSequence
from pricebook.foundation.variates import standard_normals
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel


@dataclass(frozen=True)
class PathGrid:
    """The time discretisation: a `horizon` reached in `steps` equal steps.

    Separate from `MonteCarloConfig` (paths and seed) because the two answer different questions —
    how finely time is cut, versus how many times the experiment is run — and the oracles vary them
    independently: the step count must NOT move a driftless forward, while the path count must move
    the standard error at the −½ rate."""

    horizon: float
    steps: int

    def __post_init__(self) -> None:
        if self.horizon <= 0.0:
            raise ValueError(f"a path grid needs a positive horizon, got {self.horizon}")
        if self.steps < 1:
            raise ValueError(f"a path grid needs at least one step, got {self.steps}")

    @property
    def dt(self) -> float:
        return self.horizon / self.steps


class VariateSource(Enum):
    """Where the simulation's randomness comes from — pseudo-random, or a Sobol' sequence."""

    PSEUDO_RANDOM = "pseudo_random"
    SOBOL = "sobol"


class PathConstruction(Enum):
    """How a draw is mapped onto time — in time order, or by the Brownian bridge's refinement."""

    INCREMENTAL = "incremental"
    BRIDGE = "bridge"


@dataclass(frozen=True)
class Sampling:
    """How the simulation spends its randomness: the source, the time mapping, and the mirror.

    **A bundle because §7's five-argument limit forced one, and that is the right answer rather than a
    workaround.** Three sampling choices now have real consumers — Sobol (tranche 20b), the Brownian
    bridge (21), and antithetic mirroring (19) — and each is a decision about *the draws*, not about
    the model or the payoff. Grouping them is §7's prescribed response to the limit ("group into a
    frozen value object"), and it keeps the engine's signature at four arguments however many sampling
    techniques the arc adds.

    **The default is exactly tranche 16's behaviour**, so every existing caller is unchanged: pseudo-
    random draws mapped in time order with no mirror. Nothing here is speculative — each field is the
    switch some landed tranche needed.

    **Why `SOBOL` and `BRIDGE` belong together.** Either is defensible alone and neither is much use
    alone: Sobol's advantage washes out by `d ≈ 24` (tranche 20b measured it) and an LMM path is
    `steps × forwards` dimensions, so on its own Sobol buys little there; the bridge reorders the
    *same* Gaussian and so changes nothing at all under pseudo-random draws. Together they are the
    point — the bridge puts the path's coarse shape on the dimensions Sobol is good at."""

    source: VariateSource = VariateSource.PSEUDO_RANDOM
    construction: PathConstruction = PathConstruction.INCREMENTAL
    mirrored: bool = False


@dataclass(frozen=True)
class ObservationSchedule:
    """The times a simulation must report its state at, and how finely to step between them.

    **The generalisation of `PathGrid`, pulled by the Bermudan (tranche 18).** A European payoff
    reads one date, so `PathGrid(horizon, steps)` served tranches 16 and 17; a Bermudan evaluates its
    exercise decision at *every* exercise date, so it needs the state at each. Rule-of-two fires here
    — the swaption reads the endpoint, the Bermudan reads the path — which is exactly the second
    consumer tranche 17 deferred this widening to rather than building it speculatively (§6).

    `steps_per_segment` applies between consecutive observations, so the discretisation is uniform
    *within* a segment and the observation times themselves may be unevenly spaced — which they are,
    since exercise dates follow the tenor's accruals. Jäckel does the same (§12.9: *"10 steps to the
    first exercise decision at t = 1, and then one in between each subsequent exercise
    opportunities"*), though he distributes them unevenly; one count for all segments is the
    simplification, and it is what makes the reduction to `PathGrid` exact."""

    times: tuple[float, ...]
    steps_per_segment: int

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("an observation schedule needs at least one time")
        if self.times[0] <= 0.0:
            raise ValueError(f"an observation time must be positive, got {self.times[0]}")
        for earlier, later in zip(self.times, self.times[1:]):
            if later <= earlier:
                raise ValueError(
                    f"observation times must be strictly increasing, got {earlier} then {later}"
                )
        if self.steps_per_segment < 1:
            raise ValueError(
                f"a schedule needs at least one step per segment, got {self.steps_per_segment}"
            )


def _step_factorisation(
    model: LiborMarketModel, dt: float
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """`A` and the diagonal of `C` for one step — eq 13.3 then 12.4/12.7.

    Computed once per simulation rather than per path or per step: with constant volatilities the
    step covariance is the same at every step, and a factorisation per path would be the same
    arithmetic many thousands of times. The *drift* still varies per path and per step; only this
    does not."""
    sigma = model.tenor.volatilities
    n = len(sigma)
    covariance = [
        [
            sigma[i] * sigma[j] * model.correlation.matrix[i][j] * dt
            for j in range(n)
        ]
        for i in range(n)
    ]
    diagonal = tuple(covariance[i][i] for i in range(n))
    # A zero-volatility forward makes its row of C identically zero, which eq 12.7 cannot rescale;
    # that is a degenerate model rather than a numerical problem, so it is refused loudly.
    for i, c in enumerate(diagonal):
        if c <= 0.0:
            raise ValueError(
                f"forward {i} has zero volatility, so its step covariance cannot be factorised; "
                f"a zero-vol forward is deterministic and does not need simulating"
            )
    return pseudo_square_root(covariance, factors=n), diagonal


def simulate_forward_paths(
    model: LiborMarketModel,
    schedule: ObservationSchedule,
    config: MonteCarloConfig,
    sampling: Sampling = Sampling(),
) -> list[tuple[tuple[float, ...], ...]]:
    """The forward curve at each of `schedule.times`, one tuple of states per path — eq 13.2.

    **The one definition of the evolution; `simulate_terminal_forwards` is now a thin last-slice read
    of this.** The loop is unchanged from tranche 16's: the same vectorised log-Euler step, the same
    single random stream consumed in the same step-major order, so a schedule of one segment
    reproduces the old terminal result **bit-for-bit**. Recording the intermediate states does not
    alter how the final one is computed, which is why §4's stronger proof (bit-identity) is owed here
    rather than the exact-arithmetic-identity substitute — no operation is reordered.

    **`sampling` decides where the randomness comes from and how it is mapped onto time**, defaulting
    to tranche 16's pseudo-random, time-ordered, unmirrored behaviour so existing callers are
    unchanged. `VariateSource.SOBOL` draws from the low-discrepancy sequence through the inverse CDF
    (never Box-Muller — it is not monotone, which would destroy the equidistribution), and
    `PathConstruction.BRIDGE` reorders each factor's draws across time by the Brownian bridge so the
    first, best-equidistributed Sobol dimensions carry the path's coarse shape.

    **`sampling.mirrored=True` is tranche 19's antithetic mode (§10.1):** the same seed, the same stream, every
    normal negated — *"if a single evaluation driven by a Gaussian variate vector draw `z_i` is given
    by `v_i = v(z_i)`, we also use `ṽ_i = v(−z_i)`."* One flag rather than a second function, because
    it is the *same* simulation with one sign: a caller pairs the two runs and averages, and the
    pairing plus the standard error is `foundation.mc_estimator.antithetic_estimate`'s business. The
    negation is applied to the normals, not the uniforms — mirroring `u` would give `1−u`, which is
    the antithetic *uniform*, a different and weaker technique for a Gaussian-driven model.

    **The factorisation is recomputed per segment**, because `dt` differs between segments when the
    observation times are unevenly spaced; within a segment it is constant, so it is computed once
    there rather than per step or per path.

    **Eq 12.8 still has exactly one definition.** The drift is `ratio(state) @ M.T`, where `M` is the
    model's own `drift_matrix()` (12.8's constant part) and `ratio` its `state_ratios` (the
    state-dependent part) — the same two pieces the scalar `drift_at` composes. The engine does the
    multiplication; it does not restate the formula, which §4 forbids."""
    n = len(model.tenor)
    total_steps = len(schedule.times) * schedule.steps_per_segment

    drift_matrix = np.array(model.drift_matrix(), dtype=float)
    tau = np.array(model.tenor.accruals, dtype=float)
    sigma = np.array(model.tenor.volatilities, dtype=float)

    # One stream for the whole simulation, consumed in a fixed step-major order, so the same seed
    # reproduces the same paths exactly (§3/§9) however the array is reshaped or segmented.
    normals = _draw(sampling, config, total_steps, n)
    if sampling.mirrored:
        normals = -normals

    forwards = np.tile(np.array(model.tenor.forwards, dtype=float), (config.paths, 1))
    observed: list[np.ndarray] = []
    step = 0
    previous = 0.0
    for time in schedule.times:
        dt = (time - previous) / schedule.steps_per_segment
        factorisation, diagonal = _step_factorisation(model, dt)
        a = np.array(factorisation, dtype=float)             # (n, n) step pseudo-square-root
        half_variance = 0.5 * np.array(diagonal, dtype=float)
        for _ in range(schedule.steps_per_segment):
            scaled = forwards * tau
            ratios = scaled / (1.0 + scaled) * sigma         # `state_ratios`, vectorised
            drift = ratios @ drift_matrix.T                  # eq 12.8 for every path at once
            forwards = forwards * np.exp(
                drift * dt - half_variance + normals[step] @ a.T
            )
            step += 1
        # `forwards` is REASSIGNED each step, never mutated, so this records the state as it was.
        observed.append(forwards)
        previous = time

    return [
        tuple(tuple(float(x) for x in state[path]) for state in observed)
        for path in range(config.paths)
    ]


def _draw(
    sampling: Sampling, config: MonteCarloConfig, total_steps: int, factors: int
) -> np.ndarray:
    """The `(steps, paths, factors)` normals block, per the sampling choice.

    **Pseudo-random** consumes one stream in a fixed step-major order, so a seed reproduces a run
    exactly however the array is reshaped (§3/§9). **Sobol** is per-path instead: each path is one
    point of the sequence, with `steps × factors` coordinates, because that is what a low-discrepancy
    sequence means — the *points* fill the cube evenly, so a path must be a point, not a slice of one
    long stream.

    **The bridge reorders within a factor, across time.** Each factor's `steps` coordinates are mapped
    through `BrownianBridge.increments`, which is a reordering of the same Gaussian: the increments
    stay i.i.d. standard normal (oracled exactly at L0) while *which* coordinate carries the coarse
    shape changes. Applied per factor rather than across the whole draw, because it is the time
    dimension the bridge is about; mixing factors into one bridge would be constructing a path through
    a space that has no time order."""
    if sampling.source is VariateSource.PSEUDO_RANDOM:
        needed = config.paths * total_steps * factors
        uniforms = UniformSource(seed=config.seed).uniforms(needed + needed % 2)
        block = np.array(standard_normals(uniforms)[:needed], dtype=float).reshape(
            total_steps, config.paths, factors
        )
    else:
        dimensions = total_steps * factors
        if dimensions > SOBOL_MAX_DIMENSIONS:
            raise ValueError(
                f"a Sobol-driven path needs {total_steps} x {factors} = {dimensions} dimensions, "
                f"beyond the {SOBOL_MAX_DIMENSIONS} our table covers; use fewer steps or factors, or "
                f"extend the table (docs/to_go_deeper.md)"
            )
        points = SobolSequence(dimensions=dimensions).uniforms(config.paths)
        # Coordinate `k*steps + s` is factor `k` at step `s` — factor-major, so a factor's time
        # series is contiguous and the bridge can reorder it.
        drawn = np.array(
            [[inverse_norm_cdf(coordinate) for coordinate in point] for point in points],
            dtype=float,
        )
        block = np.empty((total_steps, config.paths, factors), dtype=float)
        for k in range(factors):
            block[:, :, k] = drawn[:, k * total_steps : (k + 1) * total_steps].T

    if sampling.construction is PathConstruction.BRIDGE:
        if total_steps < 2:
            raise ValueError(
                "the Brownian bridge needs at least two steps to have an interior to subdivide; a "
                "one-step path has nothing to reorder"
            )
        # One matmul over every path and factor, not a Python loop per path. The bridge is linear,
        # so its precalculated matrix (§10.8.3: *"all of the weighting coefficients can be
        # precalculated"*) applies to the whole block at once along the time axis.
        weights = np.array(
            BrownianBridge(steps=total_steps, dt=1.0).increment_matrix(), dtype=float
        )
        block = np.tensordot(weights, block, axes=([1], [0]))
    return block


def simulate_terminal_forwards(
    model: LiborMarketModel, grid: PathGrid, config: MonteCarloConfig
) -> list[tuple[float, ...]]:
    """The forward curve at `grid.horizon`, one tuple per path — the last slice of a one-segment path.

    Kept as its own name because that is what tranches 16 and 17 ask for and a terminal payoff should
    not have to index a path it never wanted. It delegates rather than duplicating the loop: one
    definition of eq 13.2, and the delegation is bit-identical because a single segment over
    `[0, horizon]` with `grid.steps` steps *is* the old computation, operation for operation."""
    schedule = ObservationSchedule(times=(grid.horizon,), steps_per_segment=grid.steps)
    return [path[-1] for path in simulate_forward_paths(model, schedule, config)]


def mc_caplet_price(
    model: LiborMarketModel,
    index: int,
    strike: float,
    grid: PathGrid,
    config: MonteCarloConfig,
) -> MonteCarloEstimate:
    """`E[max(f_index(T) − strike, 0)]` by simulation, undiscounted and per unit of accrual.

    Undiscounted and accrual-free so it is directly comparable with `black_price`, which returns the
    same bare expectation — the discounting and the `τ` are the caller's, exactly as every other
    pricer here multiplies by its own `df`."""
    if not 0 <= index < len(model.tenor):
        raise IndexError(
            f"forward index must lie in [0, {len(model.tenor)}), got {index}"
        )
    paths = simulate_terminal_forwards(model, grid, config)
    return estimate([max(path[index] - strike, 0.0) for path in paths])
