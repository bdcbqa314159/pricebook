"""The Brownian bridge path construction (L0). JAECKEL-2002 §10.8.3, appendix §10.9.2.

The bridge sets the path's *coarse shape* from the first few Gaussian variates and fills in detail
with the later ones: `z_1` fixes the terminal value `W_tn = √t_n · z_1`, the next fixes the midpoint
conditional on it, and so on. Given `W_ti` and `W_tk` with `t_i < t_j < t_k`, the conditional law of
`W_tj` is Gaussian with

    E[W_tj] = (t_k − t_j)/(t_k − t_i) · W_ti  +  (t_j − t_i)/(t_k − t_i) · W_tk        (10.33)
    V[W_tj] = (t_j − t_i)(t_k − t_j)/(t_k − t_i)                                        (10.34)

**The anchor here is EXACT, with no Monte Carlo in it at all** — which is better than the spec's
"realised covariance to tolerance", and is why no tolerance had to be chosen. The bridge is a *linear*
map of the input normals, so its matrix `A` is recovered exactly by feeding unit basis vectors, and the
claim "same Gaussian, reordered" becomes the algebraic identity

    A Aᵀ = M     where   M_ij = min(t_i, t_j)   — the Wiener covariance

checked to machine precision. A realised covariance over sampled paths would have had to be banded by
the estimator's own error; the linear-map form has no estimator.

**A source misprint, faithful in both renderings.** Appendix §10.9.2's eq 10.76 prints
`W_tk = W_tj + √(t_k − t_i) z`, which contradicts eq 10.77's own second equality three lines later —
`z := (W_tk − W_ti)/√(t_k − t_i)` forces `W_ti`, and 10.76 is introduced as a rewrite of 10.75 whose
expansion starts from `W_ti`. With `W_tj` the first increment is counted twice. Both `-raw` and
`-layout` print it identically, so transcription independence is powerless (§4) — the internal
inconsistency is what catches it. It does **not** touch 10.33/10.34, which are the formulas built here
and which were re-derived independently rather than taken from the typo'd sketch.
"""

import math

import pytest

from pricebook.foundation.brownian_bridge import BrownianBridge, conditional_variance


def _linear_map(bridge: BrownianBridge) -> list[list[float]]:
    """`A` such that `path = A z` — recovered exactly, because the construction IS linear.

    Column `k` is the path built from the `k`th unit basis vector. No sampling, no estimator: this is
    the construction's matrix, read off."""
    size = bridge.steps
    columns = []
    for k in range(size):
        basis = [1.0 if i == k else 0.0 for i in range(size)]
        columns.append(bridge.path(basis))
    return [[columns[k][i] for k in range(size)] for i in range(size)]


# ============================================================================================
# THE EXACT ANCHOR — the bridge is the same Gaussian, reordered.
# ============================================================================================


@pytest.mark.parametrize("steps", [2, 3, 4, 5, 8, 14, 16])
def test_T21_the_bridge_has_EXACTLY_the_WIENER_covariance(steps):
    """`A Aᵀ = min(t_i, t_j)` to machine precision — the exact anchor.

    This is the whole correctness claim in one identity: whatever order the bridge consumes its
    variates in, the path it produces is a standard Wiener path on the grid. A wrong weight in 10.33
    or a wrong variance in 10.34 changes `A` and breaks it.

    `steps = 14` is in the sweep because it is the size of the source's own worked example (figure
    10.4) and is **not** a power of two — the case the text singles out as needing the general
    "halve into two parts containing approximately the same number of points" rule rather than the
    easy binary subdivision."""
    bridge = BrownianBridge(steps=steps, dt=0.25)
    a = _linear_map(bridge)
    times = [(i + 1) * 0.25 for i in range(steps)]

    for i in range(steps):
        for j in range(steps):
            gram = math.fsum(a[i][k] * a[j][k] for k in range(steps))
            assert gram == pytest.approx(min(times[i], times[j]), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("steps", [4, 8, 14])
def test_T21_the_INCREMENTS_are_independent_and_unit_variance(steps):
    """The form the simulation actually consumes, and the same identity one level down.

    The engine wants per-step unit normals, so the bridge's product is `ΔW/√dt`. "Same Gaussian,
    reordered" means those increments are still **i.i.d. standard normal** — the Gram matrix of the
    increment map is exactly the identity. If it were not, the bridge would be changing the process
    rather than the order in which randomness is spent."""
    bridge = BrownianBridge(steps=steps, dt=0.25)
    a = bridge.increment_matrix()          # the SHIPPED precalculated weights, not a local copy

    # And the matrix must reproduce the step-by-step construction on an arbitrary vector, or the
    # precalculation is a second implementation rather than the same one factored out.
    sample = [0.4, -1.1, 0.7, 0.2, -0.9, 1.3, -0.3, 0.6, 1.9, -1.4, 0.5, -0.8, 0.1, 2.1][:steps]
    direct = bridge.increments(sample)
    composed = tuple(
        math.fsum(a[i][k] * sample[k] for k in range(steps)) for i in range(steps)
    )
    assert composed == pytest.approx(direct, rel=1e-12, abs=1e-14)

    for i in range(steps):
        for j in range(steps):
            gram = math.fsum(a[i][k] * a[j][k] for k in range(steps))
            assert gram == pytest.approx(1.0 if i == j else 0.0, rel=1e-12, abs=1e-13)


def test_T21_the_bridge_is_NOT_the_INCREMENTAL_construction():
    """The mandatory non-vacuity companion: a bridge that was the identity reordering would satisfy
    every covariance identity above and achieve nothing.

    Two things are asserted. The **terminal** value must be driven entirely by the *first* variate
    (`W_tn = √t_n z_1`, §10.8.3's opening move) — under the incremental construction it is driven
    equally by all of them. And the map must genuinely differ from the incremental one, which is
    lower-triangular with equal weights."""
    steps, dt = 8, 0.25
    bridge = BrownianBridge(steps=steps, dt=dt)
    a = _linear_map(bridge)

    terminal = a[steps - 1]
    assert terminal[0] == pytest.approx(math.sqrt(steps * dt), rel=1e-14)
    assert all(abs(weight) < 1e-14 for weight in terminal[1:]), (
        "the terminal value must depend on the FIRST variate alone — that is the bridge's premise"
    )

    # The incremental construction: W_i = sqrt(dt) * (z_1 + ... + z_i), i.e. lower triangular.
    incremental = [
        [math.sqrt(dt) if k <= i else 0.0 for k in range(steps)] for i in range(steps)
    ]
    assert a != incremental


@pytest.mark.parametrize("steps", [4, 7, 8, 14, 16])
def test_T21_the_construction_order_is_MIDPOINT_FIRST(steps):
    """§10.8.3's refinement: terminal, then *"always subdividing the interval from the last point that
    was set to the next one that is already set, and halve this interval into two parts containing
    approximately the same number of points."*

    Asserted structurally rather than against a transcribed table, because the source states the rule
    as prose and illustrates it with a figure (10.4) — a figure is not a transcribable value (§4's
    no-number-fitted-to-a-figure rule). What *is* checkable: every point is placed exactly once, the
    terminal goes first, and each later point is strictly inside a bracket of already-placed points."""
    plan = BrownianBridge(steps=steps, dt=1.0).plan()
    assert len(plan) == steps, "one variate per point, no more and no fewer"

    targets = [target for target, _, _ in plan]
    assert sorted(targets) == list(range(1, steps + 1)), "every point placed exactly once"
    assert targets[0] == steps, "the terminal point is set first, from the first variate"

    placed = {0, steps}                      # the origin is known, and the terminal was just set
    for target, left, right in plan[1:]:
        assert left in placed and right in placed, f"point {target} bracketed by unplaced points"
        assert left < target < right, f"point {target} is not strictly inside its bracket"
        placed.add(target)


@pytest.mark.parametrize("steps", [8, 14, 16])
def test_T21_the_plan_spends_variates_on_the_WIDEST_remaining_gap_first(steps):
    """The property that makes the order *midpoint-first at every scale* — and the one a mutation
    found missing.

    Switching the subdivision from breadth-first to depth-first **survived the whole suite**: it
    still places every point exactly once, still brackets each by already-placed points, and still
    produces a valid Wiener path, so the covariance identity cannot see it. But it descends one
    branch to the bottom before starting the next, so the *second* variate would shape a small early
    region instead of the global midpoint — which defeats the entire purpose, since the point is to
    put the path's coarse structure on Sobol's best dimensions.

    That is tranche 18's lesson again, in a new place: **a property of the output cannot see the
    algorithm that produced it.** The distinguishing property is the bracket *width*: refining
    coarsest-first means the widths are non-increasing through the plan. Depth-first takes a width-2
    gap before a width-4 one and fails here."""
    plan = BrownianBridge(steps=steps, dt=1.0).plan()
    widths = [right - left for _, left, right in plan[1:]]

    # **Non-increasing up to a slack of ONE, not strictly** — and the slack is the source's own
    # "approximately the same number of points". When a gap has an odd width its halves differ by
    # one, so within a single level of refinement a width-3 bracket can precede a width-4 one: at 14
    # steps the widths run 14, 7, 7, 3, 4, ... A first version of this oracle demanded strict
    # monotonicity and failed at exactly that point — over-specifying the source's rule rather than
    # catching a defect. The slack-1 form still rejects depth-first, which returns to a width-7 gap
    # after width-2 ones.
    for earlier, later in zip(widths, widths[1:]):
        assert earlier >= later - 1, (
            f"bracket widths {widths} are not coarsest-first — the refinement descends before it has "
            f"finished the level above"
        )
    # Non-vacuity: there must BE several scales, or the property is trivially satisfied.
    assert len(set(widths)) > 1


# ============================================================================================
# §10.9.2's CONDITIONAL VARIANCE — eq 10.34, pointwise and as a shape.
# ============================================================================================


def test_T21_eq_10_34_matches_its_POINTWISE_transcription():
    """`V = (t_j − t_i)(t_k − t_j)/(t_k − t_i)`, transcribed here independently of the module."""
    for left, target, right in ((0.0, 1.0, 4.0), (0.5, 2.0, 2.5), (1.0, 3.0, 7.0)):
        expected = (target - left) * (right - target) / (right - left)
        assert conditional_variance(left, target, right) == pytest.approx(expected, rel=1e-15)


def test_T21_the_conditional_variance_PEAKS_at_the_MIDPOINT_and_VANISHES_at_the_brackets():
    """A requirement oracle from the finance, so it survives a transcription error in eq 10.34.

    The bridge knows the endpoints exactly, so uncertainty must be zero there and greatest in the
    middle — which is also *why* the midpoint is the informative place to put the next variate, and
    therefore why the construction order is what it is. A formula that peaked elsewhere would still
    produce a valid-looking path and a wrong allocation of variance."""
    left, right = 0.0, 1.0
    assert conditional_variance(left, left, right) == 0.0
    assert conditional_variance(left, right, right) == 0.0

    values = [conditional_variance(left, t / 100.0, right) for t in range(1, 100)]
    assert max(values) == pytest.approx(conditional_variance(left, 0.5, right), rel=1e-15)
    assert conditional_variance(0.0, 0.5, 1.0) == pytest.approx(0.25, rel=1e-15)
    # Unimodal: rising to the midpoint, falling after.
    peak = values.index(max(values))
    assert values[:peak] == sorted(values[:peak])
    assert values[peak:] == sorted(values[peak:], reverse=True)


# ============================================================================================
# CONTRACT — determinism and the refusals.
# ============================================================================================


def test_T21_the_path_is_a_PURE_FUNCTION_of_the_variates_and_the_grid():
    bridge = BrownianBridge(steps=6, dt=0.5)
    draws = [0.3, -1.2, 0.8, 0.1, -0.4, 1.7]
    assert bridge.path(draws) == bridge.path(draws)
    assert bridge.path(draws) != bridge.path([-value for value in draws])


def test_T21_a_BAD_grid_or_a_RAGGED_variate_vector_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        BrownianBridge(steps=1, dt=0.5)
    with pytest.raises(ValueError, match="positive"):
        BrownianBridge(steps=4, dt=0.0)
    with pytest.raises(ValueError, match="exactly 4"):
        BrownianBridge(steps=4, dt=0.5).path([0.1, 0.2])
    with pytest.raises(ValueError, match="not finite"):
        BrownianBridge(steps=2, dt=0.5).path([0.1, float("nan")])
