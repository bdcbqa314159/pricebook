"""Tests for schedule generation."""

import pytest
from datetime import date

from pricebook.schedule import (
    Frequency,
    StubType,
    generate_schedule,
)
from pricebook.calendar import USSettlementCalendar, BusinessDayConvention


class TestBasicSchedule:
    """Schedule generation without calendar adjustment."""

    def test_quarterly_exact(self):
        # 1 year, quarterly = 5 dates (start + 4 periods)
        sched = generate_schedule(date(2024, 1, 15), date(2025, 1, 15), Frequency.QUARTERLY)
        assert sched == [
            date(2024, 1, 15),
            date(2024, 4, 15),
            date(2024, 7, 15),
            date(2024, 10, 15),
            date(2025, 1, 15),
        ]

    def test_semi_annual_exact(self):
        sched = generate_schedule(date(2024, 1, 15), date(2025, 1, 15), Frequency.SEMI_ANNUAL)
        assert sched == [
            date(2024, 1, 15),
            date(2024, 7, 15),
            date(2025, 1, 15),
        ]

    def test_annual_exact(self):
        sched = generate_schedule(date(2024, 3, 1), date(2026, 3, 1), Frequency.ANNUAL)
        assert sched == [
            date(2024, 3, 1),
            date(2025, 3, 1),
            date(2026, 3, 1),
        ]

    def test_monthly_exact(self):
        sched = generate_schedule(date(2024, 1, 15), date(2024, 4, 15), Frequency.MONTHLY)
        assert sched == [
            date(2024, 1, 15),
            date(2024, 2, 15),
            date(2024, 3, 15),
            date(2024, 4, 15),
        ]


class TestStubs:
    """Front stub handling."""

    def test_short_front_stub(self):
        # 13 months quarterly: short first period
        sched = generate_schedule(
            date(2024, 1, 15), date(2025, 2, 15),
            Frequency.QUARTERLY, stub=StubType.SHORT_FRONT,
        )
        # Rolls backward from end: Feb 15, Nov 15, Aug 15, May 15 -> start is Jan 15
        # Short stub: Jan 15 to Feb 15 (1 month)
        assert sched[0] == date(2024, 1, 15)
        assert sched[-1] == date(2025, 2, 15)
        assert len(sched) == 6  # start + 4 regular + end

    def test_long_front_stub(self):
        # Same dates, long front stub merges the short first period
        sched = generate_schedule(
            date(2024, 1, 15), date(2025, 2, 15),
            Frequency.QUARTERLY, stub=StubType.LONG_FRONT,
        )
        assert sched[0] == date(2024, 1, 15)
        assert sched[-1] == date(2025, 2, 15)
        # First period should be longer (merged), so fewer dates
        assert len(sched) == 5

    def test_no_stub_when_exact(self):
        # Exact 1Y quarterly — same result regardless of stub type
        for stub in StubType:
            sched = generate_schedule(
                date(2024, 1, 15), date(2025, 1, 15),
                Frequency.QUARTERLY, stub=stub,
            )
            assert len(sched) == 5


class TestEndOfMonth:
    """End-of-month rule."""

    def test_eom_preserved(self):
        # Start on Jan 31 (EOM), quarterly with eom=True
        sched = generate_schedule(
            date(2024, 1, 31), date(2025, 1, 31),
            Frequency.QUARTERLY, eom=True,
        )
        # All dates should be end of month
        assert sched == [
            date(2024, 1, 31),
            date(2024, 4, 30),
            date(2024, 7, 31),
            date(2024, 10, 31),
            date(2025, 1, 31),
        ]

    def test_eom_not_applied_to_mid_month(self):
        # Start on Jan 15 — EOM rule should not affect it
        sched = generate_schedule(
            date(2024, 1, 15), date(2025, 1, 15),
            Frequency.QUARTERLY, eom=True,
        )
        assert sched[1] == date(2024, 4, 15)

    def test_eom_disabled(self):
        # Start on Jan 31 with eom=False
        sched = generate_schedule(
            date(2024, 1, 31), date(2025, 1, 31),
            Frequency.QUARTERLY, eom=False,
        )
        # April has 30 days, so Jan 31 + 3 months = Apr 30 (dateutil default)
        # But without EOM rule, it's just regular month addition
        assert sched[0] == date(2024, 1, 31)


class TestCalendarAdjustment:
    """Schedule with business day adjustment."""

    def test_weekend_adjusted(self):
        nyc = USSettlementCalendar()
        # Jan 15 2023 is a Sunday
        sched = generate_schedule(
            date(2023, 1, 15), date(2023, 7, 15),
            Frequency.QUARTERLY,
            calendar=nyc,
            convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        )
        # Jan 15 (Sun) -> Mon Jan 16 (but that's MLK Day) -> Tue Jan 17
        assert sched[0] == date(2023, 1, 17)

    def test_holiday_adjusted(self):
        nyc = USSettlementCalendar()
        # Create schedule where a date lands on Christmas
        sched = generate_schedule(
            date(2024, 6, 25), date(2024, 12, 25),
            Frequency.SEMI_ANNUAL,
            calendar=nyc,
            convention=BusinessDayConvention.FOLLOWING,
        )
        # Dec 25 2024 is Wednesday (Christmas) -> Dec 26
        assert sched[-1] == date(2024, 12, 26)


class TestValidation:
    """Input validation."""

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError):
            generate_schedule(date(2024, 1, 1), date(2024, 1, 1), Frequency.QUARTERLY)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            generate_schedule(date(2025, 1, 1), date(2024, 1, 1), Frequency.QUARTERLY)
