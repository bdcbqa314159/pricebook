"""Wrong-way risk: equity CVA with correlated counterparty default (L3/L5) — T36b. **Pre-registered.**

Brockhaus 14.5.1 applies 13.1.2's apparatus to *counterparty* default. The stock does **not** default
here — `S_T = F·e^{σ_S X − σ_S²/2}` — the counterparty does, and the risky value of a positive payoff
is `E[f(S_T)·1{τ>T}] = E[f(S_T)e^{−H_T}]`. The integrated hazard is approximated by a truncated
Gaussian `H_T ≈ (a_T + σ_H Y)⁺`, with `Y` correlated to the equity's `X` at `ρ`, and `a_T` solved from
the survival probability by eq 13.4.

    CVA = V̂ − V,   V̂ = BSCall(F,K,σ_S),   V = the four-term formula of 14.5.1

**The `ρ → 0` reduction is exact, and exact for a reason worth stating.** At zero correlation the
first two terms sum to `F·Φ(d₊)·[Φ(−a_T/σ_H) + e^{−a_T+σ_H²/2}Φ(a_T/σ_H−σ_H)]`, and the bracket **is
eq 13.4** — it is `p(T)`. So `V = p(T)·V̂` and `CVA = (1−p(T))·V̂`, which is tranche 36a's
independent-default CVA exactly. Two things make this bit-identical rather than close: `Φ₂` is exact
at `ρ = 0` by construction (its correlation integral is empty), and the `a_T` solve is the inverse of
the very identity the reduction collapses through — so a solver error and a formula error cannot
cancel.

**The wrong-way DIRECTION is derived, never intuited (the T35 lesson, built in).** "Wrong-way risk
makes CVA bigger" is a real-world heuristic whose sign depends on the exposure's direction and on
which way `ρ` is referenced. Here `ρ` is the correlation between the equity's `X` and the hazard's
`Y`, and `H_T` is *increasing* in `Y` — so **positive `ρ` means default becomes more likely exactly
when the stock (and hence a call's exposure) is high**, which is the wrong-way case for a long call.
The convention is stated so no later reader re-imports the intuition, and the monotonicity is read
off the model rather than assumed.

**The consumer asserts it stays inside `Φ₂`'s validity regime** (`Φ₂ ≳ 1e-8` and `|ρ| ≲ 0.99`),
rather than trusting the primitive silently — the obligation the primitive's own docstring states.
"""

import math

import pytest

from pricebook.foundation.bivariate_normal import bivariate_normal_cdf
from pricebook.models.black import black_price
from pricebook.products.equity_option import EquityOptionKind
from pricebook.risk.wrong_way_risk import (
    GaussianHazard,
    risky_call_value,
    solve_truncation_level,
    survival_from_truncation,
    wrong_way_cva,
)

FORWARD, STRIKE = 107.8, 100.0
SIGMA_S = 0.30 * math.sqrt(3.0)     # total (horizon) vols, as 14.5.1 writes them
SIGMA_H = 0.45
SURVIVAL = 0.88
RHOS = (-0.8, -0.4, 0.0, 0.4, 0.8)


def _default_free() -> float:
    return black_price(FORWARD, STRIKE, SIGMA_S * SIGMA_S, EquityOptionKind.CALL)


class TestTheTruncationSolveEq134:
    """`p(T) = Φ(−a/σ_H) + e^{−a+σ_H²/2}·Φ(a/σ_H − σ_H)`, inverted for `a`."""

    @pytest.mark.parametrize("survival", [0.5, 0.7, 0.88, 0.97, 0.995])
    @pytest.mark.parametrize("sigma_h", [0.15, 0.45, 1.2])
    def test_the_solve_INVERTS_its_own_forward_formula(self, survival: float, sigma_h: float) -> None:
        """A formula and its stated inverse must actually invert (§4) — the T8 instrument."""
        level = solve_truncation_level(survival, sigma_h)
        assert survival_from_truncation(level, sigma_h) == pytest.approx(survival, rel=1e-12)

    @pytest.mark.parametrize("sigma_h", [0.15, 0.45, 1.2])
    def test_the_forward_formula_is_DECREASING_in_the_level(self, sigma_h: float) -> None:
        """The book says so, and it is what makes the inverse well-posed."""
        values = [survival_from_truncation(a, sigma_h) for a in (-1.0, 0.0, 1.0, 2.5)]
        assert all(b < a for a, b in zip(values, values[1:])), values

    @pytest.mark.parametrize("sigma_h,level", [(0.1, 2.0), (0.05, 1.5), (0.2, 3.0)])
    def test_the_books_own_APPROXIMATION_checks_13_4_independently(
        self, sigma_h: float, level: float
    ) -> None:
        """The book's derivation step: `σ_H ≪ a_T ⇒ p ≈ e^{−a_T+σ_H²/2}`.

        **The relation symbol is dropped by `pdftotext` in both renderings** — the text reads
        "If σ_H ␣␣ a_T" — so which way it points had to be settled by the finance, not the glyph.
        It is `≪`: measured, `σ_H ≪ a_T` reproduces the exact `p` to six digits (0.136014 vs
        0.136014), while reading it as `≫` gives **p = 7.03**, which is not a probability. Same
        shape as eq 11.1's and 13.4's missing `Φ`, one level down: this time the casualty was a
        *relation operator*, and the requirement `p ≤ 1` is what decided it.

        With the direction settled it is a genuine independent check on 13.4 — a limit of the
        source's own derivation, which §4 rates above a second rendering of the same line.
        """
        assert sigma_h < level, "the premise: this is the σ_H ≪ a_T regime"
        assert survival_from_truncation(level, sigma_h) == pytest.approx(
            math.exp(-level + sigma_h * sigma_h / 2.0), rel=1e-5
        )

    def test_the_OTHER_reading_of_the_dropped_symbol_is_refuted_by_the_finance(self) -> None:
        """The companion that makes the test above a decision rather than a guess: in the `σ_H ≫
        a_T` regime the same expression exceeds 1, so it cannot be a survival probability."""
        assert math.exp(-0.05 + 2.0 * 2.0 / 2.0) > 1.0
        assert survival_from_truncation(0.05, 2.0) <= 1.0

    def test_a_CERTAIN_survival_needs_a_very_negative_level(self) -> None:
        assert solve_truncation_level(0.999, 0.45) < solve_truncation_level(0.5, 0.45)


class TestTheExactReductionToTrancheThirtySixA:
    """`ρ = 0` ⇒ `CVA = (1−p)·V̂`, T36a exactly — and exact, not approximate."""

    @pytest.mark.parametrize("sigma_h", [0.15, 0.45, 1.2])
    def test_at_ZERO_correlation_the_risky_value_is_p_times_the_default_free(self, sigma_h: float) -> None:
        hazard = GaussianHazard(SURVIVAL, sigma_h, 0.0)
        assert risky_call_value(FORWARD, STRIKE, SIGMA_S, hazard) == pytest.approx(
            SURVIVAL * _default_free(), rel=1e-12
        )

    @pytest.mark.parametrize("sigma_h", [0.15, 0.45, 1.2])
    def test_at_ZERO_correlation_the_CVA_is_TRANCHE_36As(self, sigma_h: float) -> None:
        hazard = GaussianHazard(SURVIVAL, sigma_h, 0.0)
        expected = (1.0 - SURVIVAL) * _default_free()
        assert wrong_way_cva(FORWARD, STRIKE, SIGMA_S, hazard) == pytest.approx(expected, rel=1e-12)

    def test_THE_COMPANION_a_live_correlation_moves_the_CVA_MATERIALLY(self) -> None:
        """The reduction is satisfied by a no-op that ignores `ρ`, so the general case is asserted
        non-degenerate — and in both directions, since the sign is the whole content."""
        independent = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, 0.0))
        wrong_way = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, 0.6))
        right_way = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, -0.6))
        assert abs(wrong_way / independent - 1.0) > 0.03
        assert abs(right_way / independent - 1.0) > 0.03


class TestTheWrongWayDirectionIsDerivedNotIntuited:
    """The T35 lesson applied before the fact: the sign comes from the model, and is stated."""

    def test_the_convention_is_stated_and_POSITIVE_rho_is_the_WRONG_way_for_a_long_call(self) -> None:
        """`H_T ≈ (a_T + σ_H Y)⁺` is **increasing** in `Y`, and `Y` correlates with the equity's `X`
        at `ρ`. So `ρ > 0` means the counterparty is more likely to default exactly when the stock
        is high — i.e. when a long call's exposure is largest. That is wrong-way risk for this
        exposure, and it must raise the CVA. The direction is read from the model's own
        construction, not from the phrase.
        """
        independent = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, 0.0))
        wrong_way = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, 0.7))
        right_way = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, -0.7))
        assert wrong_way > independent > right_way

    def test_the_CVA_is_MONOTONE_in_the_correlation(self) -> None:
        values = [
            wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, r))
            for r in RHOS
        ]
        assert all(b > a for a, b in zip(values, values[1:])), values


class TestTheRequirementOracles:
    @pytest.mark.parametrize("rho", RHOS)
    def test_the_CVA_is_non_negative_and_bounded_by_the_full_option_value(self, rho: float) -> None:
        cva = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, rho))
        assert 0.0 <= cva <= _default_free() + 1e-12

    @pytest.mark.parametrize("rho", RHOS)
    def test_a_CERTAIN_survival_gives_a_ZERO_cva(self, rho: float) -> None:
        cva = wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(1.0 - 1e-12, SIGMA_H, rho))
        assert cva == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize("rho", RHOS)
    def test_the_CVA_RISES_as_survival_falls(self, rho: float) -> None:
        values = [
            wrong_way_cva(FORWARD, STRIKE, SIGMA_S, GaussianHazard(p, SIGMA_H, rho))
            for p in (0.97, 0.88, 0.7, 0.5)
        ]
        assert all(b > a for a, b in zip(values, values[1:])), values

    def test_the_risky_value_never_EXCEEDS_the_default_free_value(self) -> None:
        for rho in RHOS:
            hazard = GaussianHazard(SURVIVAL, SIGMA_H, rho)
            assert risky_call_value(FORWARD, STRIKE, SIGMA_S, hazard) <= _default_free() + 1e-12


class TestTheConsumerStaysInsidePhi2sValidityRegime:
    """The obligation `bivariate_normal.py` states: `Φ₂ ≳ 1e-8` and `|ρ| ≲ 0.99`."""

    @pytest.mark.parametrize("rho", RHOS)
    def test_every_PHI2_call_this_formula_makes_lands_above_the_value_floor(self, rho: float) -> None:
        hazard = GaussianHazard(SURVIVAL, SIGMA_H, rho)
        for first, second, correlation in hazard.phi2_arguments(FORWARD, STRIKE, SIGMA_S):
            value = bivariate_normal_cdf(first, second, correlation)
            assert value > 1e-8, f"Φ₂({first},{second};{correlation}) = {value} is below the floor"
            assert abs(correlation) <= 0.99

    def test_a_correlation_OUTSIDE_the_primitives_regime_is_REFUSED(self) -> None:
        """Rather than silently returning a number the primitive cannot support."""
        with pytest.raises(ValueError, match="0.99"):
            GaussianHazard(SURVIVAL, SIGMA_H, 0.995)


class TestTheGuardsAreReachable:
    def test_a_survival_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="survival"):
            GaussianHazard(1.5, SIGMA_H, 0.0)

    def test_a_NON_POSITIVE_hazard_volatility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            GaussianHazard(SURVIVAL, 0.0, 0.0)
