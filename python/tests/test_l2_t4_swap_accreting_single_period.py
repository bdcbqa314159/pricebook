"""Regression for L2 Wave-2 audit — `InterestRateSwap.accreting` silently
dropped `final_notional` for single-period schedules.

Pre-fix the per-period notional was

    initial + (final - initial) · i / max(n - 1, 1)

For n=1, the divisor is `max(0, 1) = 1` and the loop yields only i=0, so
the result is `initial + 0 = initial` — `final_notional` is completely
ignored.  A user calling
``accreting(initial=500_000, final=1_000_000)`` on a single-period schedule
silently got ``[500_000]``, with no accretion and no warning.

Post-fix the n=1 case uses the average ``(initial + final) / 2`` — the
unique value that honours both endpoint inputs and is symmetric in
``initial`` / ``final``.  Multi-period behaviour is unchanged.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.schedule import Frequency
from pricebook.fixed_income.swap import InterestRateSwap


REF = date(2024, 1, 1)


class TestAccretingSinglePeriod:
    def test_single_period_uses_average_of_endpoints(self):
        swap = InterestRateSwap.accreting(
            start=REF, end=date(2024, 7, 1),
            fixed_rate=0.04,
            initial_notional=500_000.0, final_notional=1_000_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        assert swap.notional_schedule == [750_000.0]

    def test_single_period_with_equal_endpoints(self):
        swap = InterestRateSwap.accreting(
            start=REF, end=date(2024, 7, 1),
            fixed_rate=0.04,
            initial_notional=1_000_000.0, final_notional=1_000_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        assert swap.notional_schedule == [1_000_000.0]

    def test_final_notional_is_honoured(self):
        """Pre-fix this assertion would have failed (final silently ignored)."""
        swap = InterestRateSwap.accreting(
            start=REF, end=date(2024, 7, 1),
            fixed_rate=0.04,
            initial_notional=500_000.0, final_notional=1_500_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        # Average of 500_000 and 1_500_000 is 1_000_000.
        # Pre-fix would have been 500_000 (final silently dropped).
        assert swap.notional_schedule[0] == pytest.approx(1_000_000.0)
        assert swap.notional_schedule[0] != 500_000.0


class TestAccretingMultiPeriodUnchanged:
    def test_endpoints_match_inputs(self):
        """Multi-period behaviour unchanged: first period = initial, last = final."""
        swap = InterestRateSwap.accreting(
            start=REF, end=date(2029, 1, 1),
            fixed_rate=0.04,
            initial_notional=500_000.0, final_notional=1_000_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        n = len(swap.notional_schedule)
        assert swap.notional_schedule[0] == pytest.approx(500_000.0)
        assert swap.notional_schedule[-1] == pytest.approx(1_000_000.0)
        # Linear ramp: differences are constant.
        diffs = [
            swap.notional_schedule[i + 1] - swap.notional_schedule[i]
            for i in range(n - 1)
        ]
        for d in diffs:
            assert d == pytest.approx(diffs[0], abs=1e-6)

    def test_monotonic_increase(self):
        swap = InterestRateSwap.accreting(
            start=REF, end=date(2029, 1, 1),
            fixed_rate=0.04,
            initial_notional=500_000.0, final_notional=2_000_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        for a, b in zip(swap.notional_schedule, swap.notional_schedule[1:]):
            assert b > a
