"""FVA — the funding valuation adjustment (L5), tranche 28: a fixed point with a closed form.

Crepey defines FVA as a **semilinear fixed point** (eq 19) and proves it has a unique closed form
(Prop 1, eq 17):

    FVA = gamma * (P - CVA - FVA)^+                       (19, the definition)
    FVA = gamma/(1+gamma) * (P - CVA)^+                   (17, the solution)

**The oracle structure, ruled (spec §2), and why it is in this order.** The closed form is what
production returns and is transcribed pointwise. But the *correctness anchor* is that the closed form
**satisfies its own defining equation** — §4's "a formula and its stated inverse must actually
invert". That check is derived from the finance, so it survives a transcription error, and it is
sharp against exactly the mistake the source's typography invites: `gamma/(1+gamma)` is printed as a
stacked fraction, so a `-layout` reflow splits the numerator from the denominator across two lines
(the hazard that produced the eq-25 KVA error), and the natural misreading `gamma*(P-CVA)^+`
**does not satisfy the fixed point**. This file asserts the correct form satisfies it *and* that the
misreading does not — the companion that proves the oracle has power rather than assuming it (§4,
the Box-Muller pattern).

The Picard iteration converging is kept as a **consistency guard only**: it shares `gamma` and
`P-CVA` with the closed form, so it telescopes.

**Both regimes of the `^+` are exercised.** A net payer (`P-CVA > 0`) has a funding cost; a net
receiver (`P-CVA <= 0`) has none, and FVA is *exactly* zero there. Testing only the first would make
the `^+` invisible; testing only the second would make the whole formula invisible (§4's degenerate
fixture).
"""

import math

import pytest
from scipy.optimize import brentq

from pricebook.risk.funding_valuation_adjustment import (
    funding_valuation_adjustment,
    fva_fixed_point_residual,
    iterate_fva,
)

# A net PAYER: `P - CVA` materially positive, so the `^+` is open and the formula is visible.
CLEAN_MARK = 1_000.0
CVA = 40.0
SPREAD = 0.03

# A net RECEIVER: `P - CVA` negative, so the `^+` binds and FVA is exactly zero.
RECEIVER_MARK = -250.0


def _reference_by_rootfind(clean_mark: float, cva: float, spread: float) -> float:
    """Solve the fixed point `x = gamma*(P-CVA-x)^+` numerically — an independent ROUTE.

    Different code from the closed form (a bracketed root-find on the defining equation rather than
    its algebraic solution), so agreement is evidence rather than restatement. The bracket is
    `[0, gamma*(P-CVA)]`: the map is a contraction into that interval, and its upper end is the
    un-discounted first iterate.
    """
    net = clean_mark - cva
    if net <= 0.0:
        return 0.0
    return brentq(lambda x: x - spread * max(net - x, 0.0), 0.0, spread * net, xtol=1e-15, rtol=1e-15)


class TestTheFixtureIsNotDegenerate:
    def test_the_net_payer_fixture_has_a_MATERIALLY_POSITIVE_net_mark(self) -> None:
        """Else FVA == 0 and the entire closed form is invisible."""
        assert CLEAN_MARK - CVA > 100.0

    def test_the_two_fixtures_sit_on_OPPOSITE_sides_of_the_positive_part(self) -> None:
        assert CLEAN_MARK - CVA > 0.0 and RECEIVER_MARK - CVA < 0.0

    def test_the_spread_is_NON_ZERO(self) -> None:
        """Else FVA == 0 for a different reason and the reduction oracle is vacuous."""
        assert SPREAD > 0.0


class TestThePointwiseTranscription:
    """Eq 17, read in BOTH renderings and reconciled: the coefficient is `gamma/(1+gamma)`."""

    def test_the_FVA_is_the_transcribed_closed_form(self) -> None:
        expected = SPREAD / (1.0 + SPREAD) * (CLEAN_MARK - CVA)
        assert funding_valuation_adjustment(CLEAN_MARK, CVA, SPREAD) == pytest.approx(expected, rel=1e-15)

    def test_it_agrees_with_an_INDEPENDENT_root_find_on_the_defining_equation(self) -> None:
        got = funding_valuation_adjustment(CLEAN_MARK, CVA, SPREAD)
        assert got == pytest.approx(_reference_by_rootfind(CLEAN_MARK, CVA, SPREAD), rel=1e-12)

    @pytest.mark.parametrize("spread", [0.001, 0.01, 0.05, 0.25, 0.9])
    def test_the_agreement_holds_across_the_spread_range(self, spread: float) -> None:
        got = funding_valuation_adjustment(CLEAN_MARK, CVA, spread)
        assert got == pytest.approx(_reference_by_rootfind(CLEAN_MARK, CVA, spread), rel=1e-12)


class TestTheDefiningEquationIsTheAnchor:
    """§4: a formula and its stated inverse must actually invert — and it survives a bad transcription."""

    @pytest.mark.parametrize("spread", [0.001, 0.03, 0.25, 0.9])
    def test_the_closed_form_SATISFIES_its_own_fixed_point(self, spread: float) -> None:
        fva = funding_valuation_adjustment(CLEAN_MARK, CVA, spread)
        residual = fva_fixed_point_residual(fva, CLEAN_MARK, CVA, spread)
        assert abs(residual) <= 1e-12 * max(fva, 1.0), f"residual {residual}"

    @pytest.mark.parametrize("spread", [0.03, 0.25, 0.9])
    def test_the_REFLOW_MISREADING_does_NOT_satisfy_it_so_the_anchor_has_power(self, spread: float) -> None:
        """`gamma*(P-CVA)^+` — what a `-layout` split of the stacked fraction invites.

        The companion that proves the anchor is not vacuous: it must REJECT the wrong form. Without
        this, "the closed form satisfies its fixed point" could be true of a formula family rather
        than of this formula.
        """
        wrong = spread * (CLEAN_MARK - CVA)
        residual = fva_fixed_point_residual(wrong, CLEAN_MARK, CVA, spread)
        assert abs(residual) > 1e-6 * wrong, f"the wrong form must fail the fixed point, residual {residual}"

    def test_the_residual_is_EXACTLY_zero_for_a_net_receiver(self) -> None:
        fva = funding_valuation_adjustment(RECEIVER_MARK, CVA, SPREAD)
        assert fva == 0.0
        assert fva_fixed_point_residual(fva, RECEIVER_MARK, CVA, SPREAD) == 0.0


class TestTheIterationIsAConsistencyGuard:
    """Telescoping (it shares `gamma` and `P-CVA` with the closed form), so a guard, not the anchor."""

    def test_the_picard_iteration_CONVERGES_to_the_closed_form(self) -> None:
        closed = funding_valuation_adjustment(CLEAN_MARK, CVA, SPREAD)
        path = iterate_fva(CLEAN_MARK, CVA, SPREAD, 60)
        assert path[0] == 0.0, "the iteration starts from FVA_0 = 0"
        assert path[-1] == pytest.approx(closed, rel=1e-14)

    def test_the_convergence_is_GEOMETRIC_at_the_contraction_factor(self) -> None:
        """The map's factor is `gamma/(1+gamma) < 1`; the error must fall at that ratio."""
        spread = 0.5
        closed = funding_valuation_adjustment(CLEAN_MARK, CVA, spread)
        path = iterate_fva(CLEAN_MARK, CVA, spread, 12)
        errors = [abs(x - closed) for x in path[:8]]
        ratios = [b / a for a, b in zip(errors, errors[1:]) if a > 0.0]
        expected = spread  # |e_{n+1}| / |e_n| = gamma while the positive part stays open
        assert all(abs(r - expected) < 1e-9 for r in ratios), f"ratios {ratios}"

    def test_the_iteration_is_EXACTLY_zero_for_a_net_receiver_too(self) -> None:
        assert set(iterate_fva(RECEIVER_MARK, CVA, SPREAD, 10)) == {0.0}


class TestTheRequirementOracles:
    def test_a_ZERO_funding_spread_gives_a_ZERO_FVA(self) -> None:
        assert funding_valuation_adjustment(CLEAN_MARK, CVA, 0.0) == 0.0

    def test_the_FVA_rises_with_the_funding_spread(self) -> None:
        values = [funding_valuation_adjustment(CLEAN_MARK, CVA, g) for g in (0.01, 0.05, 0.2, 0.6)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_the_FVA_rises_with_the_net_mark_and_FALLS_with_the_CVA(self) -> None:
        base = funding_valuation_adjustment(CLEAN_MARK, CVA, SPREAD)
        assert funding_valuation_adjustment(CLEAN_MARK + 500.0, CVA, SPREAD) > base
        assert funding_valuation_adjustment(CLEAN_MARK, CVA + 200.0, SPREAD) < base

    def test_the_FVA_never_exceeds_the_funded_amount(self) -> None:
        """`gamma/(1+gamma) < 1`, so the adjustment cannot exceed what is being funded."""
        for spread in (0.01, 0.5, 5.0):
            fva = funding_valuation_adjustment(CLEAN_MARK, CVA, spread)
            assert 0.0 <= fva < CLEAN_MARK - CVA

    def test_the_FVA_is_never_negative_whatever_the_inputs(self) -> None:
        for mark in (-1000.0, -1.0, 0.0, 41.0, 5000.0):
            assert funding_valuation_adjustment(mark, CVA, SPREAD) >= 0.0


class TestTheGuardsAreReachable:
    def test_a_NEGATIVE_funding_spread_is_refused(self) -> None:
        """Crepey: *"gamma (a probability) is nonnegative"* — the contraction argument needs it."""
        with pytest.raises(ValueError, match="funding spread"):
            funding_valuation_adjustment(CLEAN_MARK, CVA, -0.01)

    def test_a_NEGATIVE_cva_is_refused(self) -> None:
        with pytest.raises(ValueError, match="CVA"):
            funding_valuation_adjustment(CLEAN_MARK, -1.0, SPREAD)

    def test_a_NON_POSITIVE_iteration_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            iterate_fva(CLEAN_MARK, CVA, SPREAD, 0)

    def test_a_NON_FINITE_mark_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            funding_valuation_adjustment(math.inf, CVA, SPREAD)
