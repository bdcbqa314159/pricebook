"""Tests for settlement-lag interpreted as business days (fix C.7 B1).

Pre-fix all settlement helpers (`cash_settlement`, `cds_settlement_*`,
`option_settlement_*`, `futures_settlement_physical`) used calendar days,
so a Friday trade settling T+2 landed on Sunday (a non-business day) — wrong
for FX spot, equity option physical settlement, US Treasury delivery.

Post-fix all six routes go through `add_business_days(d, lag, calendar)`.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.calendar import USSettlementCalendar
from pricebook.core.settlement import (
    cash_settlement,
    cds_settlement_cash,
    cds_settlement_physical,
    futures_settlement_physical,
    option_settlement_cash,
    option_settlement_physical,
)


# ============================================================
# Friday + T+N business days → skip weekend
# ============================================================

class TestFridayWeekendSkip:
    FRI = date(2025, 8, 1)        # Friday
    MON = date(2025, 8, 4)        # Monday
    TUE = date(2025, 8, 5)        # Tuesday

    def test_cash_settlement_t2_skips_weekend(self):
        r = cash_settlement(pv=100.0, exercise_date=self.FRI, lag_days=2)
        # PRE-FIX: 2025-08-03 (Sunday) — WRONG.
        # POST-FIX: 2025-08-05 (Tuesday).
        assert r.settlement_date == self.TUE

    def test_option_cash_t1_skips_weekend(self):
        r = option_settlement_cash(
            spot=100, strike=95, is_call=True, contracts=10,
            exercise_date=self.FRI, lag_days=1,
        )
        # PRE-FIX: 2025-08-02 (Saturday). POST-FIX: 2025-08-04 (Monday).
        assert r.settlement_date == self.MON

    def test_option_physical_t2_skips_weekend(self):
        r = option_settlement_physical(
            spot=100, strike=95, is_call=True, contracts=10,
            exercise_date=self.FRI, lag_days=2,
        )
        assert r.settlement_date == self.TUE

    def test_futures_physical_t3_skips_weekend(self):
        # Friday + T+3 BUS = Wednesday (skip Sat+Sun).
        r = futures_settlement_physical(
            entry_price=100, invoice_price=101, contracts=1, multiplier=1000,
            expiry=self.FRI, delivery_lag=3,
        )
        assert r.settlement_date == date(2025, 8, 6)   # Wednesday

    def test_cds_cash_settle_skips_weekend(self):
        r = cds_settlement_cash(
            notional=1_000_000, recovery=0.4, event_date=self.FRI, lag_days=5,
        )
        # Friday + T+5 business = next Friday Aug 8.
        assert r.settlement_date == date(2025, 8, 8)


# ============================================================
# Calendar honoured when provided
# ============================================================

class TestCalendarHoliday:
    NYC = USSettlementCalendar()

    def test_july_4_friday_holiday_pushes_settlement(self):
        # July 3 2025 = Thursday. T+1 with US calendar should skip Friday
        # July 4 (Independence Day) and land on Monday July 7.
        thu = date(2025, 7, 3)
        r = option_settlement_cash(
            spot=100, strike=95, is_call=True, contracts=10,
            exercise_date=thu, lag_days=1, calendar=self.NYC,
        )
        # Without calendar this would have been Friday July 4 (holiday).
        # With calendar — Monday July 7.
        assert r.settlement_date == date(2025, 7, 7)


# ============================================================
# T+0 / mid-week sanity
# ============================================================

class TestRegularCases:
    def test_t0_returns_same_date(self):
        d = date(2025, 8, 1)
        r = cash_settlement(pv=100.0, exercise_date=d, lag_days=0)
        assert r.settlement_date == d

    def test_mid_week_t1_no_skip(self):
        # Wednesday + T+1 = Thursday.
        wed = date(2025, 7, 30)
        r = cash_settlement(pv=100.0, exercise_date=wed, lag_days=1)
        assert r.settlement_date == date(2025, 7, 31)

    def test_cds_physical_30bd_lag(self):
        # T+30 business days from Wed Jul 30 2025 → 30 business days later.
        # 30 BD ≈ 42 calendar days. Wed Jul 30 + 30 BD = Wed Sep 10 2025.
        wed = date(2025, 7, 30)
        r = cds_settlement_physical(
            notional=1_000_000, recovery=0.4, event_date=wed, lag_days=30,
        )
        # Verify it's a business day (not Sat/Sun).
        assert r.settlement_date.weekday() < 5
        # Exact value: walk 30 BD from Wed Jul 30 → Wed Sep 10
        assert r.settlement_date == date(2025, 9, 10)
