"""Regression for L2 Wave-2 audit — `TreeSolver.convergence_analysis`
returned a Richardson-only extrapolation without method awareness or
input validation.

Pre-fix:
- The returned `richardson` was the only extrapolation, computed as
  ``(4·P(2N) - P(N)) / 3``.  This formula assumes smooth O(1/N²) error
  (true for LR), but CRR/JR/Tian binomial trees have OSCILLATORY O(1/N)
  convergence (parity sawtooth between odd/even N) — Richardson is not
  theoretically justified here.
- ``n_steps_list[-1] = 2 · n_steps_list[-2]`` was not validated.
- An empty/singleton ``n_steps_list`` silently returned ``richardson=None``.

Post-fix:
- `richardson` key is preserved as the literal Richardson formula
  (backwards-compatibility).
- New `extrapolated` key holds the method-appropriate extrapolation:
    - LR + doubling: Richardson (genuinely O(1/N²) → O(1/N⁴)).
    - LR + non-doubling: fall back to last price (formula not valid).
    - CRR/JR/Tian/Trinomial: average of last two prices (cancels parity
      oscillation).
- New `extrapolation_method` key documents the choice.
- Length < 2 or non-monotonic list raises ``ValueError`` upfront.
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._trees import (
    ExerciseType,
    TreeMethod,
    TreeSolver,
)


# ATM European call: spot=100, strike=100, r=0.05, vol=0.20, T=1.
# Black-Scholes price ≈ 10.4506.
PARAMS = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0)
BS_REF = 10.4506


class TestInputValidation:
    def test_empty_n_steps_list_raises(self):
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=50)
        with pytest.raises(ValueError, match="at least 2 grid sizes"):
            solver.convergence_analysis(**PARAMS, n_steps_list=[100])

    def test_non_monotonic_raises(self):
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=50)
        with pytest.raises(ValueError, match="strictly increasing"):
            solver.convergence_analysis(**PARAMS, n_steps_list=[100, 200, 150])


class TestMethodAwareExtrapolation:
    def test_lr_uses_richardson_when_doubling(self):
        """LR has smooth O(1/N²) error — Richardson applies."""
        solver = TreeSolver(method=TreeMethod.LR, n_steps=50,
                            exercise=ExerciseType.EUROPEAN)
        result = solver.convergence_analysis(
            **PARAMS, n_steps_list=[51, 101, 201, 401],  # LR needs odd N
        )
        # 401 = 2·201 - 1, close to doubling; treat as doubling
        # Actually 401 is NOT exactly 2·201. The selector requires exact
        # doubling.  Use clean doubling instead.
        result = solver.convergence_analysis(
            **PARAMS, n_steps_list=[50, 100, 200, 400],
        )
        # Even with LR's odd-N preference, the formula should fire when
        # n_steps_list[-1] == 2 · n_steps_list[-2].
        assert result["extrapolation_method"] == "richardson_O(1/N^2)"
        assert result["extrapolated"] == pytest.approx(BS_REF, abs=0.02)

    def test_lr_falls_back_when_non_doubling(self):
        solver = TreeSolver(method=TreeMethod.LR, n_steps=50,
                            exercise=ExerciseType.EUROPEAN)
        result = solver.convergence_analysis(
            **PARAMS, n_steps_list=[50, 100, 150, 200],  # 200 ≠ 2·150
        )
        assert result["extrapolation_method"] == "no_extrapolation_non_doubling"
        assert result["extrapolated"] == result["prices"][-1]
        # The legacy `richardson` field still contains the Richardson formula
        # (not the fallback) — backwards compat.
        formula = (4 * result["prices"][-1] - result["prices"][-2]) / 3
        assert result["richardson"] == pytest.approx(formula)

    def test_crr_uses_average_not_richardson(self):
        """CRR has oscillatory O(1/N) — Richardson over-amplifies; average
        suppresses the sawtooth."""
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=50,
                            exercise=ExerciseType.EUROPEAN)
        result = solver.convergence_analysis(
            **PARAMS, n_steps_list=[50, 100, 200, 400],
        )
        assert result["extrapolation_method"] == "average_O(1/N)_oscillation_suppression"
        # The averaged value should be the mean of the last two prices.
        expected = 0.5 * (result["prices"][-1] + result["prices"][-2])
        assert result["extrapolated"] == pytest.approx(expected)
        # And it should be reasonably close to BS.
        assert result["extrapolated"] == pytest.approx(BS_REF, abs=0.2)

    def test_jr_uses_average(self):
        solver = TreeSolver(method=TreeMethod.JR, n_steps=50,
                            exercise=ExerciseType.EUROPEAN)
        result = solver.convergence_analysis(
            **PARAMS, n_steps_list=[50, 100, 200, 400],
        )
        assert result["extrapolation_method"] == "average_O(1/N)_oscillation_suppression"


class TestResultShape:
    def test_result_contains_both_legacy_and_new_keys(self):
        """Backwards-compatibility: the legacy key 'richardson' is still
        present (holding the literal Richardson formula).  New keys
        `extrapolated` and `extrapolation_method` carry the method-aware
        choice."""
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=50,
                            exercise=ExerciseType.EUROPEAN)
        result = solver.convergence_analysis(
            **PARAMS, n_steps_list=[50, 100, 200, 400],
        )
        assert "richardson" in result
        assert "extrapolated" in result
        assert "extrapolation_method" in result
        # For CRR, the literal Richardson formula and the method-aware
        # average DIFFER in general.
        formula = (4 * result["prices"][-1] - result["prices"][-2]) / 3
        avg = 0.5 * (result["prices"][-1] + result["prices"][-2])
        assert result["richardson"] == pytest.approx(formula)
        assert result["extrapolated"] == pytest.approx(avg)
