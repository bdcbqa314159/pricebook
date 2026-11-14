"""Gauss-Hermite expectation under a standard normal (L0).

`standard_normal_expectation(f)` computes `E[f(Z)]` for `Z ~ N(0,1)`. Market Model I's year-on-year
value (BERHAN eq 3.1.11) is exactly such an expectation, and it has no closed form.

**Gauss-Hermite rather than an adaptive integrator, deliberately.** `scipy.integrate.quad` is
adaptive: it returns a different node set per integrand, carries its own error estimate, and can
shift between scipy versions -- none of which belongs under a pinned reproducibility contract. A
fixed Gauss-Hermite rule is deterministic, identical on both CI operating systems, and has no
tolerance knob to turn until something passes. Its accuracy is instead a property the oracles
measure: convergence in the node count, against integrals whose answers are known in closed form.

**The oracles test THIS adapter, never numpy's.** `E[1] = 1`, `E[Z] = 0`, `E[Z^2] = 1`,
`E[e^Z] = e^{1/2}` -- moments known exactly, so an error in the weights, the sqrt(2) scaling or the
1/sqrt(pi) normalisation shows immediately. Those three constants are exactly what a Gauss-Hermite
transcription gets wrong.
"""

import math

import pytest

from pricebook.foundation.quadrature import standard_normal_expectation


def test_the_expectation_of_a_CONSTANT_is_the_constant():
    """The normalisation: if the weights do not sum to 1, this fails and nothing else needs to."""
    assert standard_normal_expectation(lambda z: 1.0) == pytest.approx(1.0, abs=1e-14)
    assert standard_normal_expectation(lambda z: 7.5) == pytest.approx(7.5, abs=1e-13)


def test_the_ODD_moments_vanish():
    assert standard_normal_expectation(lambda z: z) == pytest.approx(0.0, abs=1e-14)
    assert standard_normal_expectation(lambda z: z**3) == pytest.approx(0.0, abs=1e-13)


@pytest.mark.parametrize(("power", "expected"), [(2, 1.0), (4, 3.0), (6, 15.0)])
def test_the_EVEN_moments_are_the_double_factorials(power, expected):
    """`E[Z^2k] = (2k-1)!!` — 1, 3, 15. Catches a wrong `sqrt(2)` scaling, which would rescale every
    moment by a power of 2 and is invisible on the constant alone."""
    assert standard_normal_expectation(lambda z: z**power) == pytest.approx(expected, rel=1e-12)


def test_the_MGF_is_exp_half():
    """`E[e^Z] = e^{1/2}` — the moment-generating function at 1. A transcendental target, so it
    exercises the rule rather than reproducing a polynomial the rule integrates exactly."""
    assert standard_normal_expectation(math.exp) == pytest.approx(math.exp(0.5), rel=1e-12)


@pytest.mark.parametrize("t", [-2.0, -0.5, 0.5, 2.0])
def test_the_MGF_matches_at_several_arguments(t):
    assert standard_normal_expectation(lambda z: math.exp(t * z)) == pytest.approx(
        math.exp(0.5 * t * t), rel=1e-9
    )


def test_MORE_nodes_do_not_move_a_converged_answer():
    """Convergence, which is how a fixed rule states its accuracy instead of carrying a tolerance."""
    coarse = standard_normal_expectation(math.exp, nodes=24)
    fine = standard_normal_expectation(math.exp, nodes=96)
    assert coarse == pytest.approx(fine, rel=1e-12)
    assert fine == pytest.approx(math.exp(0.5), rel=1e-13)


def test_TOO_FEW_nodes_is_measurably_worse_so_the_node_count_matters():
    """The companion: without this, the convergence test above could pass on a rule whose node
    count is ignored entirely."""
    crude = standard_normal_expectation(lambda z: math.exp(3.0 * z), nodes=4)
    good = standard_normal_expectation(lambda z: math.exp(3.0 * z), nodes=128)
    truth = math.exp(4.5)
    assert good == pytest.approx(truth, rel=1e-9)
    assert abs(crude / truth - 1.0) > 0.1, "4 nodes is not measurably worse — the count is inert"


def test_a_nonsensical_node_count_is_refused():
    with pytest.raises(ValueError, match="at least"):
        standard_normal_expectation(math.exp, nodes=0)


class TestAnUnbuildableRuleIsREFUSED:
    """Found at tranche 41 — above ~370 nodes the rule silently integrates to the wrong answer.

    `numpy.polynomial.hermite.hermgauss` loses its weights to underflow for large degree. Measured
    on numpy 2.4.3: correct to 370 nodes; at **371 every weight is `0.0`**, so `E[1]` comes back as
    `0.0` — finite, plausible, and 100% wrong, which is strictly worse than the `NaN` that appears
    from ~380. Either way a caller asking for a finer rule than the library can build was handed a
    number instead of a refusal, against §3.4.

    The guard asserts this module's **own** stated invariant (`E[1] = 1`, the normalisation its
    docstring names as an oracle) rather than a hard-coded 370, so the usable range can move with
    numpy without the guard moving.
    """

    def test_the_largest_WORKING_rule_still_normalises(self) -> None:
        assert standard_normal_expectation(lambda z: 1.0, nodes=370) == pytest.approx(1.0, abs=1e-13)

    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    def test_the_FIRST_BROKEN_rule_is_refused_rather_than_returning_ZERO(self) -> None:
        """371 is the dangerous one: its weights are all zero, so without the guard this returns
        `0.0` for every integrand — a silent wrong number, not a visible failure."""
        with pytest.raises(ValueError, match="does not normalise"):
            standard_normal_expectation(lambda z: 1.0, nodes=371)

    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
    def test_a_FAR_TOO_FINE_rule_is_refused_rather_than_returning_NaN(self) -> None:
        with pytest.raises(ValueError, match="does not normalise"):
            standard_normal_expectation(lambda z: 1.0, nodes=512)

    def test_THE_PREMISE_the_guard_is_not_simply_rejecting_everything_large(self) -> None:
        """A guard that refused every big rule would pass the two tests above while breaking the
        256-node rule tranche 41 actually needs."""
        assert standard_normal_expectation(lambda z: z * z, nodes=256) == pytest.approx(
            1.0, abs=1e-12)
