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

import numpy as np

from pricebook.foundation.linalg import pseudo_square_root
from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
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
            sigma[i] * sigma[j] * (1.0 if i == j else model.correlation) * dt
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


def simulate_terminal_forwards(
    model: LiborMarketModel, grid: PathGrid, config: MonteCarloConfig
) -> list[tuple[float, ...]]:
    """The forward curve at `grid.horizon`, one tuple per path — eq 13.2 stepped along the grid.

    **Vectorised over paths, and that was the `slow` gate's doing.** The scalar version evolved one
    path at a time: 20,000 paths x 8 steps x 4 forwards of Python arithmetic took the oracle file to
    8.4s locally, which is ~35s on CI against a 10s ceiling. Vectorising is the remedy the gate
    itself prescribes, and the honest one — the alternative was cutting path counts until the oracles
    lost their power.

    **Eq 12.8 still has exactly one definition.** The drift is `ratio(state) @ M.T`, where `M` is the
    model's own `drift_matrix()` (12.8's constant part) and `ratio` its `state_ratios` (the
    state-dependent part) — the same two pieces the scalar `drift_at` composes. The engine does the
    multiplication; it does not restate the formula, which §4 forbids."""
    n = len(model.tenor)
    factorisation, diagonal = _step_factorisation(model, grid.dt)

    a = np.array(factorisation, dtype=float)                 # (n, n) step pseudo-square-root
    drift_matrix = np.array(model.drift_matrix(), dtype=float)
    tau = np.array(model.tenor.accruals, dtype=float)
    sigma = np.array(model.tenor.volatilities, dtype=float)
    half_variance = 0.5 * np.array(diagonal, dtype=float)

    # One stream for the whole simulation, consumed in a fixed order, so the same seed reproduces
    # the same paths exactly (§3/§9) however the array is reshaped.
    needed = config.paths * grid.steps * n
    uniforms = UniformSource(seed=config.seed).uniforms(needed + needed % 2)
    normals = np.array(standard_normals(uniforms)[:needed], dtype=float).reshape(
        grid.steps, config.paths, n
    )

    forwards = np.tile(np.array(model.tenor.forwards, dtype=float), (config.paths, 1))
    for step in range(grid.steps):
        scaled = forwards * tau
        ratios = scaled / (1.0 + scaled) * sigma          # `state_ratios`, vectorised
        drift = ratios @ drift_matrix.T                   # eq 12.8 for every path at once
        forwards = forwards * np.exp(
            drift * grid.dt - half_variance + normals[step] @ a.T
        )
    return [tuple(float(x) for x in row) for row in forwards]


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
