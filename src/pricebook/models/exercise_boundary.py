"""The parametric exercise boundary and its training objective (L3). JAECKEL-2002 §12.7-12.8.

Jäckel's exercise-decision function, in the two coordinates §12.6 settles on after looking at the
geometry of the exercise domain — the forward rate `f_i(t_i)` and the next coterminal swap rate
`s_{i+1}(t_i)`, *"using the projection onto the first forward rate `f_i` and the forward swap rate
`s_{i+1}` starting from the reset time of `f_{i+1}` going out to the terminal maturity results in a
reasonable separation of exercise and non-exercise decisions with very little overlap"*:

    E_i(f(t_i)) = [ f_i(t_i) − p_i1 · s_{i+1}(0)/(s_{i+1}(t_i) + p_i2) − p_i3 ] · ω      (12.38)

`ω = +1` for a payer, `−1` for a receiver; **exercise iff `E_i > 0`**. Three free coefficients per
exercise slice, refitted at each. The final slice is pinned rather than fitted:

    p_(n−1)1 = 0 ,  p_(n−1)2 = 0 ,  p_(n−1)3 = K                                          (12.39)

and the training objective maximised at each earlier slice is

    U_i(p_i) = (1/N) Σ_j [ I_ji  if E_i(f_ji; p_i) > 0  else  v_j ]                       (12.41)

with `I_ji` the residual intrinsic value on path `j` at `t_i` and `v_j` the path's value if no
exercise has occurred up to and including `t_i`.

**THE SIGN OF `p_i3` IS RULED, NOT TRANSCRIBED — the source is internally inconsistent.** Both
renderings of 12.38 print `+ p_i3` (`-layout` places the `±1` brace after that term; `-raw`
disambiguates that the brace multiplies the whole bracket, and on *that* the two agree). But 12.39
pins the last slice to `(0, 0, K)`, and the text states exactly what it must mean: *"at the very last
exercise opportunity we have exact knowledge if exercise is optimal, namely when the residual swap is
in the money."* At `t_(n−1)` the residual swap is a single period, so its swap rate **is** `f_(n−1)`,
and in-the-money for a payer is `f_(n−1) > K`. The printed `+K` yields `f + K > 0`, true for every
positive rate — it would exercise a deep out-of-the-money payer. Only `− p_i3` reproduces the stated
condition, so 12.39 rules 12.38.

Charter §4 prescribes exactly this: where two steps of one source disagree, the reader rules by the
finance — by dimension, by the standard convention, or by what the downstream equations use — and
pins the choice with an oracle. Here 12.39 wins because it carries *exact financial content*, while
12.38's sign convention is arbitrary. **And that arbitrariness is the interesting half: at every
slice except the last the sign cannot matter**, because `p_i3` is a free parameter the optimiser fits,
so it converges to `−p_i3` instead and the two conventions describe the same family of boundaries.
The discrepancy is decisive *only* at the pinned slice — which is both why it is easy to miss and why
it is exactly testable.

Provenance:
  stock:  parked/ng/.../swaption_mc.py and the legacy options/bermudan_* — read for SHAPE only; the
          coordinates, the function and the objective are read from the source (§5)
  source: JAECKEL-2002 (docs/references.md) §12.6 (the coordinate pair and eq 12.37's correlation),
          §12.7 eqs 12.38/12.39 (the decision function and the pinned final slice) and §12.8 eqs
          12.40/12.41 (the training objective), read in `-raw` and `-layout`; the two agree on the
          bracket structure and DISAGREE with 12.39 on the sign of `p_i3`, ruled above
  oracle: tests/L3/test_exercise_boundary.py — the final slice reproduces `ω(f − K) > 0` exactly at
          seven forwards and both directions (the ruling), with the printed variant asserted to FAIL
          it so the ruling is proved necessary; the sign's immateriality at a trained slice; the
          pointwise transcription of 12.38; the hyperbolic shape in `s_{i+1}`; the `p_i1 = 0`
          reduction that lets the final slice ignore a coordinate that does not exist; and eq 12.41's
          selection with its exact attainable ceiling `mean(max(I_j, v_j))`
  crosscheck: none — a parametrisation choice, not a market convention a library would share
  slice:  tranche 18 — the Bermudan swaption under the LMM
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryParams:
    """`p_i1`, `p_i2`, `p_i3` of eq 12.38 for one exercise slice.

    Named for their roles rather than their subscripts: `slope` weights the swap-rate ratio,
    `offset` shifts its denominator (what makes the function hyperbolic rather than a simple
    reciprocal), and `level` is the threshold the forward is compared against."""

    slope: float
    offset: float
    level: float

    @classmethod
    def final(cls, strike: float) -> BoundaryParams:
        """Eq 12.39 — the last exercise opportunity, where no optimisation is needed.

        `(0, 0, K)` makes `E_(n−1) = ω(f_(n−1) − K)`: exercise exactly when the one-period residual
        swap is in the money, which is known exactly rather than fitted. The `slope = 0` also
        removes the `s_{i+1}` coordinate, which matters because at the final slice there is no
        coterminal swap beyond terminal maturity for it to refer to."""
        return cls(slope=0.0, offset=0.0, level=strike)


@dataclass(frozen=True)
class BoundaryState:
    """The two coordinates eq 12.38 reads, plus the `s_{i+1}(0)` its numerator needs.

    `initial_swap_rate` is `s_{i+1}(0)` — fixed at inception, not simulated — and it sits here with
    the evolved coordinates because eq 12.38 needs all three together and a caller who passed the
    evolved and initial swap rates the wrong way round would get a plausible wrong decision."""

    forward: float
    swap_rate: float
    initial_swap_rate: float


@dataclass(frozen=True)
class TrainingSlice:
    """One exercise date's training data: the state, intrinsic and continuation value per path.

    Eq 12.40's stored evolutions, reduced to exactly what eq 12.41 reads. Bundled because the three
    are indexed together by path and a length mismatch would silently train on misaligned data —
    checked once here instead of at every use."""

    states: tuple[BoundaryState, ...]
    intrinsics: tuple[float, ...]
    continuations: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.states) == len(self.intrinsics) == len(self.continuations)):
            raise ValueError(
                f"states, intrinsics and continuations must have the same length, got "
                f"{len(self.states)}, {len(self.intrinsics)}, {len(self.continuations)}"
            )

    def __len__(self) -> int:
        return len(self.states)


def exercise_value(params: BoundaryParams, state: BoundaryState, omega: float) -> float:
    """`E_i` — eq 12.38, with `p_i3` entering NEGATIVELY as eq 12.39 rules (see the module note)."""
    denominator = state.swap_rate + params.offset
    if params.slope != 0.0 and denominator == 0.0:
        # `inf` here would decide every path the same way and look like a boundary rather than a
        # bug — the non-finite-poisoning failure this project has catalogued. Refused loudly.
        raise ValueError(
            f"the exercise boundary's denominator s_{{i+1}} + p_i2 vanishes "
            f"({state.swap_rate} + {params.offset}); the decision variable would be infinite"
        )
    ratio = 0.0 if params.slope == 0.0 else params.slope * state.initial_swap_rate / denominator
    return (state.forward - ratio - params.level) * omega


def exercise_now(params: BoundaryParams, state: BoundaryState, omega: float) -> bool:
    """*"The parametric exercise decision given an evolved yield curve is then simply to exercise if
    `E_i > 0`."* Strictly greater: at `E_i = 0` the holder is indifferent and does not exercise."""
    return exercise_value(params, state, omega) > 0.0


def policy_value(params: BoundaryParams, slice_: TrainingSlice, omega: float) -> float:
    """`U_i(p_i)` — eq 12.41: the average over paths of intrinsic-if-exercising, else continuation.

    **Piecewise constant in `params`**, which is a property of the objective and not an artefact:
    it changes only when some path's decision flips, so it has zero gradient almost everywhere.
    That is why the optimiser choice matters and is recorded where the training happens."""
    return math.fsum(
        intrinsic if exercise_now(params, state, omega) else continuation
        for state, intrinsic, continuation in zip(
            slice_.states, slice_.intrinsics, slice_.continuations, strict=True
        )
    ) / len(slice_)


def roll_continuations(
    params: BoundaryParams, slice_: TrainingSlice, omega: float
) -> tuple[float, ...]:
    """§12.8 step 1 — carry each path's value back through one exercise date.

    *"For each path, if `(E_{i+1}(f_ji) > 0)` and `(I_j(i+1) > 0)`, re-assign `v_j := I_j(i+1)`, else
    leave `v_j` unchanged."* `slice_` holds the **next** date's states and intrinsics together with
    the incoming `v`; the result is the outgoing `v`.

    **It overwrites rather than maximises, and that is correct only because the induction runs
    backwards.** Going backwards, the last write to `v_j` is the *earliest* date at which the policy
    exercises — which is the date the holder actually stops at. Run forwards, the same overwrite
    would keep the latest exercise instead, and the continuation values would be wrong: this is the
    function that makes the direction load-bearing, so it is named and oracled rather than inlined."""
    return tuple(
        intrinsic if (exercise_now(params, state, omega) and intrinsic > 0.0) else carried
        for state, intrinsic, carried in zip(
            slice_.states, slice_.intrinsics, slice_.continuations, strict=True
        )
    )


def train_slice(slice_: TrainingSlice, omega: float, strike: float) -> BoundaryParams:
    """Maximise eq 12.41 over `(p_i1, p_i2, p_i3)` for one exercise slice.

    **The optimiser is NELDER-MEAD, not Jäckel's BFGS, and the reason is a property of his own
    objective.** §12.8 says *"one can use the Broyden-Fletcher-Goldfarb-Shanno multi-dimensional
    variable metric method"* — but eq 12.41 is an average of an **indicator-selected** quantity, so it
    is piecewise constant in `p`: it changes only when some path's decision flips, and its gradient is
    exactly zero everywhere else. A quasi-Newton method built on gradients has nothing to descend; with
    finite differences it moves only when the step happens to be large enough to flip a path, which
    makes progress an artefact of the step size. Nelder-Mead is derivative-free and is the appropriate
    instrument for a piecewise-constant objective. The deviation is measured, not assumed — see the
    tranche doc.

    **The start is the European boundary**, `(0, 0, K)`: exercise iff in the money. That is a real,
    sensible policy, so training can only improve on it and the resulting lower bound is never worse
    than the European — which is what oracle 2's early-exercise premium needs in order to be a
    statement about *early exercise* rather than about optimiser luck."""
    from scipy.optimize import minimize

    def negated(vector) -> float:
        params = BoundaryParams(float(vector[0]), float(vector[1]), float(vector[2]))
        try:
            return -policy_value(params, slice_, omega)
        except ValueError:
            # A vanishing denominator is not a value, so it must not be ranked against one.
            return math.inf

    start = (0.0, 0.0, strike)
    fitted = minimize(negated, start, method="Nelder-Mead")
    best = BoundaryParams(
        float(fitted.x[0]), float(fitted.x[1]), float(fitted.x[2])
    )
    # The start is a valid policy, so never return something worse than it. Nelder-Mead on a
    # piecewise-constant objective can terminate on a simplex that has drifted off a plateau.
    opening = BoundaryParams(*start)
    if policy_value(best, slice_, omega) < policy_value(opening, slice_, omega):
        return opening
    return best
