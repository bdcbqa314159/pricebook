"""Regression for L1 A.2 B1 — global_solver detects duplicate maturities.

Pre-fix, two instruments at the same maturity silently overwrote each other
in the residual vector. The Newton system was under-determined and the curve
silently failed to reprice the dropped instrument.

Post-fix, duplicate maturities raise `ValueError` with a clear diagnostic.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.curves.global_solver import global_bootstrap


REF = date(2024, 1, 1)
ONE_Y = date(2025, 1, 1)


class TestGlobalSolverDuplicateMaturity:
    def test_deposit_swap_collision_raises(self):
        """A 1Y deposit AND a 1Y swap → duplicate maturity → ValueError."""
        with pytest.raises(ValueError, match="Duplicate maturity"):
            global_bootstrap(
                REF,
                deposits=[(ONE_Y, 0.05)],
                swaps=[(ONE_Y, 0.04)],
            )

    def test_duplicate_deposit_raises(self):
        with pytest.raises(ValueError, match="Duplicate maturity"):
            global_bootstrap(
                REF,
                deposits=[(ONE_Y, 0.05), (ONE_Y, 0.06)],
                swaps=[],
            )

    def test_duplicate_swap_raises(self):
        two_y = date(2026, 1, 1)
        with pytest.raises(ValueError, match="Duplicate maturity"):
            global_bootstrap(
                REF,
                deposits=[(ONE_Y, 0.05)],
                swaps=[(two_y, 0.04), (two_y, 0.05)],
            )

    def test_error_message_identifies_conflict(self):
        """The error message must identify which instrument was already there."""
        with pytest.raises(ValueError, match="'deposit'"):
            global_bootstrap(
                REF,
                deposits=[(ONE_Y, 0.05)],
                swaps=[(ONE_Y, 0.04)],
            )

    def test_no_collision_succeeds(self):
        """Distinct maturities → bootstrap works as before."""
        deposits = [(date(2024, 4, 1), 0.05)]
        swaps = [(date(2025, 1, 1), 0.04), (date(2026, 1, 1), 0.045)]
        curve = global_bootstrap(REF, deposits, swaps)
        # Round-trip: deposit reprices.
        import math
        from pricebook.core.day_count import year_fraction, DayCountConvention
        for mat, rate in deposits:
            t = year_fraction(REF, mat, DayCountConvention.ACT_360)
            implied = (1.0 / curve.df(mat) - 1.0) / t
            assert implied == pytest.approx(rate, abs=1e-6)
