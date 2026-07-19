"""L0 foundation — the public API surface (the vocabulary every upper layer speaks).

`__all__` is the *contract*: it is the set of names the market-data, model, product and
trade layers may depend on. Anything not listed is an implementation detail and may move.
Defined explicitly here (before consumers freeze it for us) per the foundation audit §3.1.
"""

from __future__ import annotations

from pricebook_ng.foundation.calendars import (
    BusinessDayConvention,
    Calendar,
    Coverage,
    DayType,
    HolidaySet,
    JointCalendar,
    Observance,
    Rule,
    Weekend,
    christmas_boxing,
    day_after_thanksgiving,
    easter,
    fixed,
    gregorian_easter,
    mexico_inauguration,
    midsummer_eve,
    monday,
    nth,
    orthodox,
    orthodox_easter,
    victoria_day,
)
from pricebook_ng.foundation.cashflow import SCHEMA_VERSION, Cashflow, Leg
from pricebook_ng.foundation.day_count import (
    Accrual,
    CouponPeriod,
    DayCountConvention,
    business_days_between,
    year_fraction,
)
from pricebook_ng.foundation.distributions import norm_cdf, norm_pdf, norm_ppf
from pricebook_ng.foundation.interpolation import (
    Extrapolation,
    ExtrapolationEnds,
    Interpolation,
    interpolate,
)
from pricebook_ng.foundation.market_calendars import (
    calendar_for_currency,
    get_calendar,
    list_calendars,
)
from pricebook_ng.foundation.money import (
    CURRENCIES,
    UNITS,
    Currency,
    CurrencyPair,
    Money,
    Quantity,
    Unit,
    currency,
    list_currencies,
    list_units,
    register_currency,
    register_unit,
    unit,
)
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.rate_basis import Compounding, convert_rate, growth_factor
from pricebook_ng.foundation.rate_index import (
    AccrualConvention,
    AccrualMethod,
    FixingHistory,
    FixingRule,
    IndexId,
    ObservationStyle,
    RateIndex,
    RfrConvention,
    accrued_rate,
    get_rate_index,
    list_rate_indices,
)
from pricebook_ng.foundation.results import DiscountBasis, PricingFailure, PricingResult
from pricebook_ng.foundation.schedule import (
    Frequency,
    RollConvention,
    RollRule,
    Schedule,
    ScheduleTerms,
    StubType,
    build_schedule,
    cds_roll_date,
    imm_date,
    next_cds_roll,
    next_imm,
)
from pricebook_ng.foundation.settlement import Delivery, SettlementTerms, SettlementType, settlement_date
from pricebook_ng.foundation.solvers import brent, least_squares, newton, secant
from pricebook_ng.foundation.tenor import Tenor, TenorUnit
from pricebook_ng.foundation.underlying import AssetClass, Underlying

__all__ = [
    # time & tenor
    "Tenor", "TenorUnit",
    # calendars (the rule DSL + types)
    "Calendar", "JointCalendar", "HolidaySet", "Weekend", "Observance", "DayType",
    "Coverage", "BusinessDayConvention", "Rule",
    "fixed", "easter", "orthodox", "nth", "monday", "christmas_boxing", "victoria_day",
    "midsummer_eve", "mexico_inauguration", "day_after_thanksgiving",
    "gregorian_easter", "orthodox_easter",
    "get_calendar", "calendar_for_currency", "list_calendars",
    # day counts & accrual
    "DayCountConvention", "year_fraction", "CouponPeriod", "Accrual", "business_days_between",
    # schedules
    "Frequency", "StubType", "RollRule", "RollConvention", "ScheduleTerms", "Schedule",
    "build_schedule", "imm_date", "next_imm", "cds_roll_date", "next_cds_roll",
    # money & quantity
    "Currency", "Unit", "Money", "Quantity", "CurrencyPair",
    "register_currency", "currency", "list_currencies", "CURRENCIES",
    "register_unit", "unit", "list_units", "UNITS",
    # rate basis
    "Compounding", "growth_factor", "convert_rate",
    # index identity
    "RateIndex", "IndexId", "AccrualConvention", "FixingRule", "RfrConvention",
    "AccrualMethod", "ObservationStyle", "FixingHistory", "accrued_rate",
    "get_rate_index", "list_rate_indices",
    # underlying
    "Underlying", "AssetClass",
    # instrument atoms
    "Cashflow", "Leg", "SCHEMA_VERSION",
    # settlement
    "SettlementType", "Delivery", "SettlementTerms", "settlement_date",
    # numerics
    "NumericalConfig", "PricingResult", "PricingFailure", "DiscountBasis",
    "norm_pdf", "norm_cdf", "norm_ppf",
    "brent", "newton", "secant", "least_squares",
    "Interpolation", "Extrapolation", "ExtrapolationEnds", "interpolate",
]
