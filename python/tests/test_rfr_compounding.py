"""Tests for RFR compounding conventions — full ISDA mechanics."""

import pytest
from datetime import date, timedelta

from pricebook.fixed_income.rfr_compounding import (
    RFRAccrualConfig, compound_rfr_full, compound_rfr_from_curve,
    rfr_accrual_schedule, get_rfr_config, list_rfr_configs,
    SOFR_CONFIG, ESTR_CONFIG, SONIA_CONFIG, TONA_CONFIG,
    SARON_CONFIG, CORRA_CONFIG, CDI_CONFIG,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import USSettlementCalendar


REF = date(2024, 1, 15)


def _make_fixings(start, n_days, rate):
    """Helper: create flat fixings dict."""
    return {start + timedelta(days=i): rate for i in range(n_days)}


class TestRFRConfigRegistry:
    def test_list_configs(self):
        configs = list_rfr_configs()
        assert len(configs) == 12
        assert "SOFR" in configs
        assert "ESTR" in configs
        assert "SONIA" in configs
        assert "CDI" in configs

    def test_get_sofr(self):
        cfg = get_rfr_config("SOFR")
        assert cfg.currency == "USD"
        assert cfg.observation_shift == 2
        assert cfg.day_count.value == "ACT/360"

    def test_get_estr(self):
        cfg = get_rfr_config("ESTR")
        assert cfg.currency == "EUR"
        assert cfg.observation_shift == 2

    def test_get_sonia(self):
        cfg = get_rfr_config("SONIA")
        assert cfg.currency == "GBP"
        assert cfg.lookback_days == 5
        assert cfg.observation_shift == 0

    def test_get_tona(self):
        cfg = get_rfr_config("TONA")
        assert cfg.currency == "JPY"

    def test_get_cdi(self):
        cfg = get_rfr_config("CDI")
        assert cfg.currency == "BRL"
        assert cfg.day_count.value == "BUS/252"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown RFR"):
            get_rfr_config("FAKE")

    def test_frozen(self):
        with pytest.raises(Exception):
            SOFR_CONFIG.name = "WRONG"

    def test_to_dict(self):
        d = SOFR_CONFIG.to_dict()
        assert d["name"] == "SOFR"
        assert d["day_count"] == "ACT/360"


class TestAccrualSchedule:
    def test_basic_schedule(self):
        start = date(2024, 7, 1)  # Monday
        end = date(2024, 7, 5)    # Friday
        schedule = rfr_accrual_schedule(start, end, SOFR_CONFIG)
        assert len(schedule) == 4  # Mon, Tue, Wed, Thu

    def test_observation_shift(self):
        """SOFR: observation shifted back 2 business days."""
        start = date(2024, 7, 3)  # Wednesday
        end = date(2024, 7, 5)    # Friday
        schedule = rfr_accrual_schedule(start, end, SOFR_CONFIG)
        assert len(schedule) == 2
        # Wed accrual → Mon observation (2 bd back)
        assert schedule[0].accrual_date == date(2024, 7, 3)
        assert schedule[0].observation_date == date(2024, 7, 1)

    def test_sonia_lookback(self):
        """SONIA: lookback 5 business days (no observation shift)."""
        start = date(2024, 7, 8)  # Monday
        end = date(2024, 7, 9)    # Tuesday
        schedule = rfr_accrual_schedule(start, end, SONIA_CONFIG)
        assert len(schedule) == 1
        # Mon accrual → 5 bd lookback = previous Mon (Jul 1)
        assert schedule[0].observation_date == date(2024, 7, 1)

    def test_weekend_weight(self):
        """Friday should have weight=3 (covers Sat+Sun)."""
        start = date(2024, 7, 5)   # Friday
        end = date(2024, 7, 8)     # Monday
        schedule = rfr_accrual_schedule(start, end, SOFR_CONFIG)
        assert len(schedule) == 1
        assert schedule[0].weight_days == 3  # Fri→Mon

    def test_empty_for_same_dates(self):
        schedule = rfr_accrual_schedule(REF, REF, SOFR_CONFIG)
        assert len(schedule) == 0


class TestCompoundRFRFull:
    def test_flat_rate(self):
        """Flat SOFR = 5% over 1 week → compounded ≈ 5%."""
        start = date(2024, 7, 1)  # Monday
        end = date(2024, 7, 8)    # next Monday
        fixings = _make_fixings(date(2024, 6, 20), 30, 0.05)
        rate = compound_rfr_full(fixings, start, end, SOFR_CONFIG)
        assert abs(rate - 0.05) < 0.001

    def test_zero_rate(self):
        start = date(2024, 7, 1)
        end = date(2024, 7, 5)
        fixings = _make_fixings(date(2024, 6, 20), 30, 0.0)
        rate = compound_rfr_full(fixings, start, end, SOFR_CONFIG)
        assert rate == 0.0

    def test_estr_flat_rate(self):
        """ESTR at 3.5% → compounded ≈ 3.5%."""
        start = date(2024, 7, 1)
        end = date(2024, 7, 8)
        fixings = _make_fixings(date(2024, 6, 20), 30, 0.035)
        rate = compound_rfr_full(fixings, start, end, ESTR_CONFIG)
        assert abs(rate - 0.035) < 0.001

    def test_varying_rates(self):
        """Varying rates should compound correctly."""
        start = date(2024, 7, 1)
        end = date(2024, 7, 4)  # 3 business days
        fixings = {
            date(2024, 6, 27): 0.04,  # obs for Jul 1 (shifted 2 back)
            date(2024, 6, 28): 0.05,  # obs for Jul 2
            date(2024, 7, 1): 0.06,   # obs for Jul 3
        }
        rate = compound_rfr_full(fixings, start, end, SOFR_CONFIG)
        assert rate > 0


class TestCompoundFromCurve:
    def test_flat_curve_matches(self):
        """Flat curve at 4% → compounded rate ≈ 4%."""
        curve = DiscountCurve.flat(REF, 0.04)
        start = date(2024, 2, 1)
        end = date(2024, 2, 8)
        rate = compound_rfr_from_curve(curve, start, end, SOFR_CONFIG)
        assert abs(rate - 0.04) < 0.005

    def test_different_currencies(self):
        """Should work for any RFR config."""
        curve = DiscountCurve.flat(REF, 0.035)
        start = date(2024, 2, 1)
        end = date(2024, 2, 8)
        for cfg in [SOFR_CONFIG, ESTR_CONFIG, SONIA_CONFIG, TONA_CONFIG]:
            rate = compound_rfr_from_curve(curve, start, end, cfg)
            assert abs(rate - 0.035) < 0.01

    def test_with_calendar(self):
        """Using a real calendar should handle holidays."""
        cal = USSettlementCalendar()
        curve = DiscountCurve.flat(date(2024, 3, 1), 0.05)
        # Normal week with no holidays
        start = date(2024, 3, 4)
        end = date(2024, 3, 11)
        rate = compound_rfr_from_curve(curve, start, end, SOFR_CONFIG, cal)
        assert abs(rate - 0.05) < 0.01


class TestLockoutAndCutoff:
    def test_lockout(self):
        """Lockout: last 2 days freeze the rate."""
        cfg = RFRAccrualConfig(
            "TEST", "USD", SOFR_CONFIG.day_count,
            observation_shift=0, lookback_days=0, lockout_days=2,
            rate_cutoff_days=0, payment_delay=0, fixing_lag=0,
        )
        start = date(2024, 7, 1)  # Mon
        end = date(2024, 7, 5)    # Fri (4 bd: Mon-Thu)
        schedule = rfr_accrual_schedule(start, end, cfg)
        # Last 2 days should have same observation as the 2nd-to-last non-lockout day
        assert schedule[-1].observation_date == schedule[-2].observation_date

    def test_rate_cutoff(self):
        """Rate cut-off: last 2 days use the cut-off date's rate."""
        cfg = RFRAccrualConfig(
            "TEST", "USD", SOFR_CONFIG.day_count,
            observation_shift=0, lookback_days=0, lockout_days=0,
            rate_cutoff_days=2, payment_delay=0, fixing_lag=0,
        )
        start = date(2024, 7, 1)
        end = date(2024, 7, 5)
        schedule = rfr_accrual_schedule(start, end, cfg)
        # Last 2 should use same observation as the cutoff point
        assert schedule[-1].observation_date == schedule[-2].observation_date
