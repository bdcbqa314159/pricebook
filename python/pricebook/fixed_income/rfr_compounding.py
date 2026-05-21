"""RFR compounding conventions — full ISDA mechanics for all major currencies.

Production-grade backward-looking compounded rate calculation with:
- Observation shift (SOFR, ESTR: 2 days)
- Lookback (without observation shift)
- Lockout (last N days use frozen rate)
- Rate cut-off (last N days excluded from compounding)
- Payment delay

    from pricebook.fixed_income.rfr_compounding import (
        RFRAccrualConfig, compound_rfr_full, rfr_accrual_schedule,
        SOFR_CONFIG, ESTR_CONFIG, SONIA_CONFIG,
    )

References:
    ISDA (2021). IBOR Fallbacks Protocol.
    ARRC (2020). SOFR Conventions for OIS.
    BOE (2020). SONIA Conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from pricebook.core.day_count import DayCountConvention, year_fraction

if TYPE_CHECKING:
    from pricebook.core.calendar import Calendar


@dataclass(frozen=True)
class RFRAccrualConfig:
    """RFR accrual conventions per ISDA definitions."""
    name: str
    currency: str
    day_count: DayCountConvention
    observation_shift: int       # business days: observation dates shift back
    lookback_days: int           # business days: lookback without shift (alternative to obs shift)
    lockout_days: int            # business days: last N days freeze the rate
    rate_cutoff_days: int        # business days: last N days excluded from compounding
    payment_delay: int           # business days after period end
    fixing_lag: int              # business days: rate published T+fixing_lag

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()
                if k not in ("day_count",)} | {"day_count": self.day_count.value}


# ═══════════════════════════════════════════════════════════════
# G10 RFR Conventions
# ═══════════════════════════════════════════════════════════════

SOFR_CONFIG = RFRAccrualConfig(
    "SOFR", "USD", DayCountConvention.ACT_360,
    observation_shift=2, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=2, fixing_lag=1)

ESTR_CONFIG = RFRAccrualConfig(
    "ESTR", "EUR", DayCountConvention.ACT_360,
    observation_shift=2, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=2, fixing_lag=1)

SONIA_CONFIG = RFRAccrualConfig(
    "SONIA", "GBP", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=5, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=0)

TONA_CONFIG = RFRAccrualConfig(
    "TONA", "JPY", DayCountConvention.ACT_365_FIXED,
    observation_shift=2, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=2, fixing_lag=1)

SARON_CONFIG = RFRAccrualConfig(
    "SARON", "CHF", DayCountConvention.ACT_360,
    observation_shift=2, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=2, fixing_lag=0)

CORRA_CONFIG = RFRAccrualConfig(
    "CORRA", "CAD", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=2, fixing_lag=1)

AONIA_CONFIG = RFRAccrualConfig(
    "AONIA", "AUD", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=0)

# ═══════════════════════════════════════════════════════════════
# EM RFR Conventions
# ═══════════════════════════════════════════════════════════════

CDI_CONFIG = RFRAccrualConfig(
    "CDI", "BRL", DayCountConvention.BUS_252,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=0)

KOFR_CONFIG = RFRAccrualConfig(
    "KOFR", "KRW", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=1)

SORA_CONFIG = RFRAccrualConfig(
    "SORA", "SGD", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=0)

HONIA_CONFIG = RFRAccrualConfig(
    "HONIA", "HKD", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=0)

THOR_CONFIG = RFRAccrualConfig(
    "THOR", "THB", DayCountConvention.ACT_365_FIXED,
    observation_shift=0, lookback_days=0, lockout_days=0,
    rate_cutoff_days=0, payment_delay=0, fixing_lag=0)


_CONFIG_REGISTRY: dict[str, RFRAccrualConfig] = {
    "SOFR": SOFR_CONFIG, "ESTR": ESTR_CONFIG, "SONIA": SONIA_CONFIG,
    "TONA": TONA_CONFIG, "SARON": SARON_CONFIG, "CORRA": CORRA_CONFIG,
    "AONIA": AONIA_CONFIG, "CDI": CDI_CONFIG, "KOFR": KOFR_CONFIG,
    "SORA": SORA_CONFIG, "HONIA": HONIA_CONFIG, "THOR": THOR_CONFIG,
}


def get_rfr_config(name: str) -> RFRAccrualConfig:
    """Look up RFR config by name (e.g. 'SOFR', 'ESTR', 'SONIA')."""
    key = name.upper()
    cfg = _CONFIG_REGISTRY.get(key)
    if cfg is None:
        available = sorted(_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown RFR {key!r}. Available: {available}")
    return cfg


def list_rfr_configs() -> list[str]:
    """Return sorted list of available RFR config names."""
    return sorted(_CONFIG_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════
# Accrual schedule
# ═══════════════════════════════════════════════════════════════


@dataclass
class RFRAccrualPeriod:
    """Single day in an RFR accrual schedule."""
    accrual_date: date           # the date being accrued
    observation_date: date       # the date whose fixing is used
    weight_days: int             # number of calendar days this rate applies
    year_fraction: float         # day count fraction for this day


def rfr_accrual_schedule(
    start: date,
    end: date,
    config: RFRAccrualConfig,
    calendar: Calendar | None = None,
) -> list[RFRAccrualPeriod]:
    """Generate the full RFR accrual schedule for a period.

    Maps each business day in [start, end) to its observation date
    (shifted back by observation_shift or lookback_days), and computes
    the weight (number of calendar days until next business day).

    Args:
        start: accrual period start (inclusive).
        end: accrual period end (exclusive).
        config: RFR conventions.
        calendar: business day calendar. If None, weekends only.
    """
    if start >= end:
        return []

    def is_bd(d: date) -> bool:
        if calendar is not None:
            return calendar.is_business_day(d)
        return d.weekday() < 5

    def add_bd(d: date, n: int) -> date:
        if calendar is not None:
            return calendar.add_business_days(d, n)
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        current = d
        while remaining > 0:
            current += timedelta(days=step)
            if current.weekday() < 5:
                remaining -= 1
        return current

    # Build list of business days in accrual period
    business_days = []
    current = start
    while current < end:
        if is_bd(current):
            business_days.append(current)
        current += timedelta(days=1)

    if not business_days:
        return []

    dc = config.day_count
    schedule = []

    for i, bd in enumerate(business_days):
        # Observation date: shifted back
        if config.observation_shift > 0:
            obs_date = add_bd(bd, -config.observation_shift)
        elif config.lookback_days > 0:
            obs_date = add_bd(bd, -config.lookback_days)
        else:
            obs_date = bd

        # Rate cut-off: last N days use the rate from cut-off date
        if config.rate_cutoff_days > 0:
            n_remaining = len(business_days) - 1 - i
            if n_remaining < config.rate_cutoff_days:
                cutoff_idx = len(business_days) - 1 - config.rate_cutoff_days
                if cutoff_idx >= 0:
                    obs_date = business_days[cutoff_idx]

        # Lockout: last N days use frozen rate
        if config.lockout_days > 0:
            n_remaining = len(business_days) - 1 - i
            if n_remaining < config.lockout_days:
                lockout_idx = len(business_days) - 1 - config.lockout_days
                if lockout_idx >= 0:
                    obs_date = business_days[lockout_idx]

        # Weight: calendar days until next business day (or end)
        if i + 1 < len(business_days):
            next_bd = business_days[i + 1]
        else:
            next_bd = end
        weight = (next_bd - bd).days

        # Year fraction for this weight
        yf = weight / (360.0 if dc == DayCountConvention.ACT_360 else 365.0)

        schedule.append(RFRAccrualPeriod(
            accrual_date=bd,
            observation_date=obs_date,
            weight_days=weight,
            year_fraction=yf,
        ))

    return schedule


# ═══════════════════════════════════════════════════════════════
# Full compounding
# ═══════════════════════════════════════════════════════════════


def compound_rfr_full(
    fixings: dict[date, float],
    start: date,
    end: date,
    config: RFRAccrualConfig,
    calendar: Calendar | None = None,
) -> float:
    """Compound RFR with full ISDA mechanics.

    Args:
        fixings: {date: rate} — historical fixing values.
        start: accrual start.
        end: accrual end.
        config: RFR conventions.
        calendar: business day calendar.

    Returns:
        Annualised compounded rate for the period.
    """
    schedule = rfr_accrual_schedule(start, end, config, calendar)
    if not schedule:
        return 0.0

    product = 1.0
    total_yf = 0.0

    for period in schedule:
        rate = fixings.get(period.observation_date, 0.0)
        product *= (1.0 + rate * period.year_fraction)
        total_yf += period.year_fraction

    if total_yf <= 0:
        return 0.0

    return (product - 1.0) / total_yf


def compound_rfr_from_curve(
    curve: 'DiscountCurve',
    start: date,
    end: date,
    config: RFRAccrualConfig,
    calendar: Calendar | None = None,
) -> float:
    """Compound RFR from a discount curve (forward-looking, for pricing).

    Extracts the implied overnight forward rate from the curve for each
    business day in the accrual period, then compounds with full ISDA mechanics.
    """
    from pricebook.core.discount_curve import DiscountCurve

    schedule = rfr_accrual_schedule(start, end, config, calendar)
    if not schedule:
        return 0.0

    product = 1.0
    total_yf = 0.0

    for period in schedule:
        d1 = period.accrual_date
        d2 = d1 + timedelta(days=period.weight_days)
        # Implied forward rate from curve
        df1 = curve.df(d1)
        df2 = curve.df(d2)
        if df2 > 0 and period.year_fraction > 0:
            fwd_rate = (df1 / df2 - 1.0) / period.year_fraction
        else:
            fwd_rate = 0.0
        product *= (1.0 + fwd_rate * period.year_fraction)
        total_yf += period.year_fraction

    if total_yf <= 0:
        return 0.0

    return (product - 1.0) / total_yf
