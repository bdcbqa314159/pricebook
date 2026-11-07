"""The Brownian bridge path construction (L0). JAECKEL-2002 §10.8.3, appendix §10.9.2.

The incremental construction spends its variates in time order, so every one contributes equally to
the path's shape. The bridge spends them in order of **importance**: `z_1` fixes the terminal value,
`z_2` the midpoint given it, and each later variate only refines a smaller interval. Given `W_ti` and
`W_tk` with `t_i < t_j < t_k`, the conditional law of `W_tj` is Gaussian with

    E[W_tj] = (t_k − t_j)/(t_k − t_i) · W_ti  +  (t_j − t_i)/(t_k − t_i) · W_tk        (10.33)
    V[W_tj] = (t_j − t_i)(t_k − t_j)/(t_k − t_i)                                        (10.34)

**It produces the same process, so on its own it changes nothing.** The path is a standard Wiener path
either way and the increments are i.i.d. either way — `A Aᵀ = min(t_i,t_j)` exactly, which is this
module's anchor. What changes is *which variate carries which part of the variance*, and that matters
only in company: paired with a low-discrepancy sequence whose early dimensions are better
equidistributed than its later ones (tranche 20b measured the advantage washing out by `d ≈ 24`), the
bridge puts the coarse shape of the path on the dimensions Sobol is good at. That is the reason it is
built, and why tranche 20b deliberately did **not** wire Sobol into the 48-dimensional LMM path first.

**The order, for an arbitrary number of steps.** *"The Brownian bridge is particularly easy to
construct if the number of steps is a power of two ... In general, we can construct the Brownian bridge
by always subdividing the interval from the last point that was set to the next one that is already
set, and halve this interval into two parts containing approximately the same number of points."* So
the plan is a breadth-first subdivision, which gives midpoint-first refinement at every scale and
degrades gracefully when a gap has an odd number of points.

**A source misprint in the appendix, faithful in both renderings.** §10.9.2's eq 10.76 prints
`W_tk = W_tj + √(t_k − t_i) z`, which contradicts eq 10.77's own second equality three lines later:
`z := (W_tk − W_ti)/√(t_k − t_i)` forces `W_ti`, and 10.76 is introduced as a rewrite of 10.75, whose
expansion begins at `W_ti`. With `W_tj` the first increment is counted twice. Both `-raw` and
`-layout` print it identically, so transcription independence cannot see it (§4) — the internal
inconsistency is what does. It does not touch 10.33/10.34, which are what this module implements and
which were re-derived here independently rather than read from the typo'd sketch:
`E[W_tj] = W_ti + (t_j−t_i)/(t_k−t_i)·(W_tk − W_ti)` rearranges to 10.33, and the variance follows.

Provenance:
  stock:  _stock/quarry/pricebook/numerical/_mc.py — read for SHAPE only; the weights are read from
          the source (§5)
  source: JAECKEL-2002 (docs/references.md) §10.8.3 eqs 10.33/10.34 (the conditional mean and
          variance) and the prose subdivision rule for a non-power-of-two step count; appendix
          §10.9.2 as the derivation, which contains a misprint at eq 10.76 (recorded above). Both
          renderings read; they agree on 10.33/10.34 and agree on the misprint
  oracle: tests/L0/test_brownian_bridge.py — the EXACT Wiener covariance identity `A Aᵀ = min(t_i,t_j)`
          with `A` recovered from unit basis vectors (the construction is linear, so there is no
          estimator and no tolerance to choose); the increment map's Gram matrix exactly the identity;
          the non-vacuity companion that the terminal value depends on the FIRST variate alone and the
          map is not the incremental one; the plan asserted midpoint-first and well-bracketed,
          including the source's own non-power-of-two example (14 steps, figure 10.4); eq 10.34
          pointwise, peaking at the midpoint and vanishing at the brackets
  crosscheck: quantlib BrownianBridge — deferred to the optional QuantLib tier
  slice:  tranche 21 — path construction / the Brownian bridge
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def conditional_variance(left: float, target: float, right: float) -> float:
    """Eq 10.34 — `(t_j − t_i)(t_k − t_j)/(t_k − t_i)`.

    Zero at either bracket (the bridge knows the endpoints exactly) and greatest at the midpoint,
    which is *why* the midpoint is where the next variate buys the most and therefore why the
    construction order is what it is."""
    if right <= left:
        raise ValueError(f"the bracket must satisfy left < right, got {left} and {right}")
    return (target - left) * (right - target) / (right - left)


@dataclass(frozen=True)
class BrownianBridge:
    """A bridge plan over `steps` equal intervals of length `dt`, times `t_i = i·dt` for `i = 1..steps`.

    Frozen and stateless: the plan depends only on the grid, so it is derived on demand rather than
    cached — a `steps`-length plan is a few dozen integers and the simulation that consumes it is
    orders of magnitude more work. The origin `t_0 = 0` is fixed at `W = 0` and is not a point of the
    grid; it is the left bracket of the earliest interval."""

    steps: int
    dt: float

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError(
                f"a bridge needs at least two points to have an interior to subdivide, got "
                f"{self.steps}"
            )
        if self.dt <= 0.0:
            raise ValueError(f"a time step must be positive, got {self.dt}")

    def plan(self) -> tuple[tuple[int, int, int], ...]:
        """`(target, left, right)` in construction order — the terminal first, then midpoints.

        Breadth-first subdivision, which is the source's general rule: each gap is halved into two
        parts of approximately equal size, and taking the gaps in discovery order refines every scale
        before descending to the next. For a power-of-two step count this is the clean binary
        subdivision the text calls *"particularly easy"*; for 14 it is the worked example of figure
        10.4.

        The terminal entry's brackets are `(0, steps)` — degenerate, since the terminal is drawn
        unconditionally (`W_tn = √t_n z_1`) rather than from a bracket. It carries them so that every
        entry has the same shape and `path` needs no special case beyond the first."""
        plan: list[tuple[int, int, int]] = [(self.steps, 0, self.steps)]
        gaps: list[tuple[int, int]] = [(0, self.steps)]
        while gaps:
            low, high = gaps.pop(0)
            if high - low < 2:
                continue
            middle = (low + high) // 2
            plan.append((middle, low, high))
            gaps.append((low, middle))
            gaps.append((middle, high))
        return tuple(plan)

    def path(self, variates) -> tuple[float, ...]:
        """`W_t1 … W_tn` from `steps` standard normal variates — eqs 10.33 and 10.34.

        The first variate sets the terminal value; each later one places a point inside a bracket of
        already-placed points, at the conditional mean plus the conditional standard deviation times
        the variate."""
        self._check(variates)
        horizon = self.steps * self.dt
        # Index 0 is the origin, fixed at zero; indices 1..steps are the grid.
        values = [0.0] * (self.steps + 1)

        plan = self.plan()
        values[self.steps] = math.sqrt(horizon) * variates[0]
        for index, (target, left, right) in enumerate(plan[1:], start=1):
            t_left, t_target, t_right = left * self.dt, target * self.dt, right * self.dt
            weight = (t_target - t_left) / (t_right - t_left)
            mean = (1.0 - weight) * values[left] + weight * values[right]
            spread = math.sqrt(conditional_variance(t_left, t_target, t_right))
            values[target] = mean + spread * variates[index]
        return tuple(values[1:])

    def increments(self, variates) -> tuple[float, ...]:
        """The per-step unit normals a simulation consumes — `ΔW/√dt` from the bridge path.

        This is the form the engines want: they step a process with one standard normal per step, so
        the bridge is a drop-in **reordering** of those draws rather than a change to the process.
        The increments of a Wiener path are independent, so these are i.i.d. standard normal — the
        oracle asserts that as an exact identity on the map, not as a sampled property."""
        values = self.path(variates)
        scale = 1.0 / math.sqrt(self.dt)
        previous = 0.0
        out = []
        for value in values:
            out.append((value - previous) * scale)
            previous = value
        return tuple(out)

    def increment_matrix(self) -> tuple[tuple[float, ...], ...]:
        """`B` such that `increments = B z` — the construction's weights, precalculated.

        *"Since all of the weighting coefficients can be precalculated, we only need to carry out
        three multiplications and two additions for each point"* (§10.8.3). The construction is
        linear, so the whole of it collapses to one matrix, recovered by running it on the unit basis
        vectors — which is not a trick but the definition of a linear map's matrix.

        **It exists because a caller applying the bridge per path in Python is the cost the `slow`
        gate's own docstring warns about.** With the matrix, a simulation reorders every path and
        factor in one `tensordot`; without it, tranche 21's oracles spent most of their time in a
        Python loop. The oracle that recovers `B` the same way is not circular: it checks `B Bᵀ = I`,
        a property of the matrix, not the route by which it was obtained."""
        size = self.steps
        columns = [
            self.increments([1.0 if i == k else 0.0 for i in range(size)]) for k in range(size)
        ]
        return tuple(tuple(columns[k][i] for k in range(size)) for i in range(size))

    def _check(self, variates) -> None:
        if len(variates) != self.steps:
            raise ValueError(
                f"a bridge over {self.steps} points needs exactly {self.steps} variates, got "
                f"{len(variates)}"
            )
        for value in variates:
            if not math.isfinite(value):
                raise ValueError(f"a variate is not finite ({value}); refusing to build a path")
