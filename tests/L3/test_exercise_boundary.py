"""The parametric exercise boundary (L3). JAECKEL-2002 §12.7-12.8, eqs 12.38/12.39/12.41.

Jäckel's exercise-decision function, in the two coordinates §12.6 settles on — the forward rate
`f_i(t_i)` and the next coterminal swap rate `s_{i+1}(t_i)`:

    E_i(f(t_i)) = [ f_i(t_i) − p_i1 · s_{i+1}(0)/(s_{i+1}(t_i) + p_i2) − p_i3 ] · ω      (12.38)

with `ω = +1` for a payer and `−1` for a receiver, and the decision *exercise iff `E_i > 0`*. Three
free coefficients per exercise slice, retrained at each. At the final slice they are pinned, not
fitted:

    p_(n−1)1 = 0 ,  p_(n−1)2 = 0 ,  p_(n−1)3 = K                                          (12.39)

**That third term's sign is RULED, not transcribed, and the source is internally inconsistent.** Both
renderings of 12.38 print `+ p_i3` (the `-layout` pass puts the ±1 brace after it; `-raw` disambiguates
that the brace multiplies the whole bracket). But 12.39 pins the last slice to `(0, 0, K)` and the
text states exactly what that must mean — *"at the very last exercise opportunity we have exact
knowledge if exercise is optimal, namely when the residual swap is in the money"*. At `t_(n−1)` the
residual swap is a single period, so its swap rate **is** `f_(n−1)`, and in-the-money for a payer is
`f_(n−1) > K`. The printed `+K` gives `f + K > 0`, which is true for every positive rate — it
exercises a deep out-of-the-money payer. Only `− p_i3` reproduces the stated condition.

So §4's discrepancy rule applies: two steps of one source disagree, and the reader rules by the
finance rather than deferring. The ruling is forced by 12.39, which carries exact financial content,
over 12.38, whose sign convention is arbitrary — and that last point is the interesting half:

**for every slice except the last, the sign cannot matter**, because `p_i3` is a free parameter the
optimiser fits, so it simply converges to `−p_i3` instead. The discrepancy is decisive *only* at the
pinned slice. That is what the oracles below pin, and they assert the printed variant FAILS, so the
ruling is proved necessary rather than assumed.
"""

import math

import pytest

from pricebook.models.exercise_boundary import (
    BoundaryParams,
    BoundaryState,
    TrainingSlice,
    exercise_now,
    exercise_value,
    policy_value,
)
from pricebook.products.swaption import SwaptionKind

STRIKE = 0.040


def _state(forward: float, swap_rate: float, initial: float = 0.042) -> BoundaryState:
    return BoundaryState(forward=forward, swap_rate=swap_rate, initial_swap_rate=initial)


# ============================================================================================
# THE RULING — eq 12.39 pins the last slice, and only one sign of p_i3 reproduces the finance.
# ============================================================================================


@pytest.mark.parametrize("forward", [0.030, 0.038, 0.0399, 0.040, 0.0401, 0.042, 0.050])
@pytest.mark.parametrize("kind", [SwaptionKind.PAYER, SwaptionKind.RECEIVER])
def test_the_FINAL_slice_exercises_EXACTLY_when_the_residual_swap_is_IN_THE_MONEY(forward, kind):
    """Eq 12.39's parameters must reproduce `ω(f − K) > 0` exactly — the one case with no freedom.

    This is the requirement oracle that rules the §4 discrepancy. It is derived from the finance
    (Jäckel's own sentence about the last exercise opportunity), so it survives the transcription
    error it is diagnosing."""
    params = BoundaryParams.final(STRIKE)
    assert (params.slope, params.offset, params.level) == (0.0, 0.0, STRIKE)
    in_the_money = kind.omega * (forward - STRIKE) > 0.0
    assert exercise_now(params, _state(forward, forward), kind.omega) is in_the_money


@pytest.mark.parametrize("forward", [0.030, 0.038])
def test_the_PRINTED_sign_of_p3_FAILS_the_final_slice_so_the_ruling_is_NECESSARY(forward):
    """The companion that makes the ruling a finding rather than a preference.

    With `+ p_i3` as printed, a payer's `E = f + K` is positive for any positive rate, so a deep
    out-of-the-money payer exercises. Asserted here against the same two out-of-the-money forwards
    the oracle above requires to be *held*, so the two together say: this sign, not that one, and
    here is the case that separates them."""
    printed = exercise_value(BoundaryParams.final(STRIKE), _state(forward, forward), 1.0)
    as_printed_variant = forward + STRIKE  # `+ p_i3`, the rendering both passes agree on
    assert as_printed_variant > 0.0, "the printed variant exercises OTM — the defect"
    assert printed < 0.0, "the ruled form must NOT exercise an out-of-the-money payer"
    assert printed == pytest.approx(forward - STRIKE, rel=1e-15)


def test_the_sign_of_p3_is_IMMATERIAL_at_a_TRAINED_slice():
    """Why the discrepancy is decisive only at the pinned slice — and worth asserting, because it
    bounds how much the ruling can matter.

    At a trained slice `p_i3` is free, so the two conventions describe the *same family* of
    boundaries: whatever decision `+p` reaches with level `L`, `−p` reaches with level `−L`. The
    oracle shows the decision sets coincide under that relabelling, which is the precise sense in
    which the source's inconsistency is harmless everywhere except at 12.39."""
    level = 0.0035
    for forward in (0.030, 0.038, 0.045):
        ruled = exercise_value(BoundaryParams(0.5, 0.01, level), _state(forward, 0.041), 1.0)
        relabelled = exercise_value(BoundaryParams(0.5, 0.01, -level), _state(forward, 0.041), 1.0)
        assert ruled == pytest.approx(relabelled - 2.0 * level, rel=1e-15)


# ============================================================================================
# THE POINTWISE TRANSCRIPTION of eq 12.38, and its hyperbolic shape in s_{i+1}.
# ============================================================================================


@pytest.mark.parametrize(
    ("slope", "offset", "level"), [(0.5, 0.01, 0.002), (1.5, 0.005, -0.001), (0.2, 0.02, 0.0)]
)
@pytest.mark.parametrize("kind", [SwaptionKind.PAYER, SwaptionKind.RECEIVER])
def test_eq_12_38_matches_its_POINTWISE_transcription(slope, offset, level, kind):
    """The load-bearing formula's **value**, transcribed independently — T17's lesson: a product's
    value, not its factors' presence. `p_i1` multiplies a *ratio* of swap rates, so a mutation
    dropping the `s_{i+1}(0)` numerator or inverting the ratio is a magnitude error that a
    sign-and-presence check would miss."""
    for forward, swap_rate, initial in ((0.035, 0.041, 0.042), (0.050, 0.038, 0.042)):
        expected = (forward - slope * initial / (swap_rate + offset) - level) * kind.omega
        got = exercise_value(
            BoundaryParams(slope, offset, level), _state(forward, swap_rate, initial), kind.omega
        )
        assert got == pytest.approx(expected, rel=1e-14)


def test_the_boundary_is_HYPERBOLIC_in_the_next_swap_rate():
    """*"This function is hyperbolic in `s_{i+1}`"* — a structural requirement, so it survives a
    transcription error in the coefficients. With `p_i1 > 0` the subtracted term falls as
    `s_{i+1}` rises, so `E` rises monotonically: a steeper residual curve makes a payer MORE likely
    to exercise, which is the financial content and the reason this coordinate pair was chosen."""
    params = BoundaryParams(0.5, 0.01, 0.002)
    values = [exercise_value(params, _state(0.040, s), 1.0) for s in (0.030, 0.035, 0.040, 0.045)]
    assert values == sorted(values)
    assert values[0] != values[-1], "a flat response would not be hyperbolic in anything"


def test_a_ZERO_slope_removes_the_swap_rate_coordinate_entirely():
    """An exact reduction, and the one 12.39 relies on: at `p_i1 = 0` the `s_{i+1}` term vanishes,
    which is what lets the final slice ignore a coordinate that does not exist (there is no
    `s_n` beyond terminal maturity). Must hold even where the ratio itself would be undefined."""
    params = BoundaryParams(0.0, 0.0, STRIKE)
    assert exercise_value(params, _state(0.045, 0.0, 0.0), 1.0) == pytest.approx(0.045 - STRIKE)


def test_a_boundary_whose_denominator_VANISHES_is_refused():
    """`s_{i+1}(t_i) + p_i2 = 0` is a division by zero that would otherwise return `inf` and
    silently exercise everything — refused loudly, since a non-finite decision variable is the
    numerical-cast-away failure this project has catalogued."""
    with pytest.raises(ValueError, match="denominator"):
        exercise_value(BoundaryParams(0.5, -0.040, 0.0), _state(0.040, 0.040), 1.0)


# ============================================================================================
# EQ 12.41 — the training objective.
# ============================================================================================


def test_eq_12_41_selects_the_INTRINSIC_where_it_exercises_and_the_CONTINUATION_otherwise():
    """`U_i(p) = (1/N) Σ_j [ I_ji if E_i(f_ji; p) > 0 else v_j ]` — eq 12.41, pointwise.

    The objective is an *average of a selection*, which is why it is piecewise constant in `p`: it
    changes only when a path's decision flips. Transcribed here against a hand-built three-path
    slice so the selection itself is pinned, not merely the average."""
    states = (_state(0.050, 0.041), _state(0.030, 0.041), _state(0.045, 0.041))
    intrinsics = (0.010, 0.002, 0.005)
    continuations = (0.004, 0.006, 0.003)
    slice_ = TrainingSlice(states, intrinsics, continuations)

    # `p_i3 = 0.040` with no swap-rate term: exercise iff f > 0.040 -> paths 0 and 2.
    params = BoundaryParams(0.0, 0.0, 0.040)
    expected = (intrinsics[0] + continuations[1] + intrinsics[2]) / 3.0
    assert policy_value(params, slice_, 1.0) == pytest.approx(expected, rel=1e-15)

    # A level above every forward exercises nothing, so the objective IS the continuation average.
    never = BoundaryParams(0.0, 0.0, 1.0)
    assert policy_value(never, slice_, 1.0) == pytest.approx(math.fsum(continuations) / 3.0)
    # And one below every forward exercises everything.
    always = BoundaryParams(0.0, 0.0, -1.0)
    assert policy_value(always, slice_, 1.0) == pytest.approx(math.fsum(intrinsics) / 3.0)


def test_the_objective_is_MAXIMISED_by_exercising_only_where_intrinsic_beats_continuation():
    """A requirement oracle on the objective rather than on a number: the best achievable value of
    eq 12.41 over *any* decision rule is `mean(max(I_j, v_j))`, attained by the rule that exercises
    exactly where the intrinsic wins. No parametric boundary can beat it, and the optimiser's job is
    to approach it — so this is the yardstick the training is measured against, and it is exact."""
    states = (_state(0.050, 0.041), _state(0.030, 0.041), _state(0.045, 0.041))
    intrinsics = (0.010, 0.002, 0.005)
    continuations = (0.004, 0.006, 0.003)
    slice_ = TrainingSlice(states, intrinsics, continuations)
    ceiling = math.fsum(max(i, v) for i, v in zip(intrinsics, continuations)) / 3.0

    for params in (
        BoundaryParams(0.0, 0.0, 0.040),
        BoundaryParams(0.0, 0.0, 1.0),
        BoundaryParams(0.0, 0.0, -1.0),
        BoundaryParams(0.5, 0.01, 0.002),
    ):
        assert policy_value(params, slice_, 1.0) <= ceiling + 1e-15
    # Attained, so the ceiling is reachable rather than a vacuous upper bound.
    assert policy_value(BoundaryParams(0.0, 0.0, 0.040), slice_, 1.0) == pytest.approx(ceiling)


def test_a_RAGGED_training_slice_is_refused():
    with pytest.raises(ValueError, match="same length"):
        TrainingSlice((_state(0.04, 0.04),), (0.1, 0.2), (0.0,))
