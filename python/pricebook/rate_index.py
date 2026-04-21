"""Rate index registry: SOFR, EURIBOR, SONIA, ESTR, TONA and others.

Each RateIndex defines the fixing lag, day count, compounding method,
and calendar used by a specific interest rate benchmark.

    from pricebook.rate_index import get_rate_index, RateIndex

    sofr = get_rate_index("SOFR")
    assert sofr.fixing_lag == 0
    assert sofr.day_count == DayCountConvention.ACT_360
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pricebook.day_count import DayCountConvention


class CompoundingMethod(Enum):
    """How overnight rates are compounded over a period."""
    COMPOUNDED = "compounded"     # Standard: ∏(1 + r_i × δ_i) - 1
    AVERAGED = "averaged"         # Simple average of daily rates
    FLAT = "flat"                 # Single fixing at period start (IBOR-style)


@dataclass(frozen=True)
class RateIndex:
    """A benchmark interest rate index."""

    name: str
    currency: str
    day_count: DayCountConvention
    fixing_lag: int                          # Business days before accrual start
    compounding: CompoundingMethod
    observation_shift: int                   # Calendar days of observation shift
    payment_delay: int                       # Calendar days after accrual end
    tenor_months: int | None                 # None for overnight, 1/3/6 for term
    is_overnight: bool
    administrator: str


# ---- Registry ----

_REGISTRY: dict[str, RateIndex] = {}


def _register(index: RateIndex) -> RateIndex:
    _REGISTRY[index.name] = index
    return index


# Overnight RFR indices
SOFR = _register(RateIndex(
    name="SOFR", currency="USD",
    day_count=DayCountConvention.ACT_360,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=2, payment_delay=2,
    tenor_months=None, is_overnight=True,
    administrator="FRBNY",
))

ESTR = _register(RateIndex(
    name="ESTR", currency="EUR",
    day_count=DayCountConvention.ACT_360,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=2, payment_delay=2,
    tenor_months=None, is_overnight=True,
    administrator="ECB",
))

SONIA = _register(RateIndex(
    name="SONIA", currency="GBP",
    day_count=DayCountConvention.ACT_365_FIXED,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=0, payment_delay=0,
    tenor_months=None, is_overnight=True,
    administrator="BOE",
))

TONA = _register(RateIndex(
    name="TONA", currency="JPY",
    day_count=DayCountConvention.ACT_365_FIXED,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=2, payment_delay=2,
    tenor_months=None, is_overnight=True,
    administrator="BOJ",
))

SARON = _register(RateIndex(
    name="SARON", currency="CHF",
    day_count=DayCountConvention.ACT_360,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=2, payment_delay=2,
    tenor_months=None, is_overnight=True,
    administrator="SIX",
))

CORRA = _register(RateIndex(
    name="CORRA", currency="CAD",
    day_count=DayCountConvention.ACT_365_FIXED,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=2, payment_delay=2,
    tenor_months=None, is_overnight=True,
    administrator="BOC",
))

AONIA = _register(RateIndex(
    name="AONIA", currency="AUD",
    day_count=DayCountConvention.ACT_365_FIXED,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=0, payment_delay=0,
    tenor_months=None, is_overnight=True,
    administrator="RBA",
))

NZOCR = _register(RateIndex(
    name="NZOCR", currency="NZD",
    day_count=DayCountConvention.ACT_365_FIXED,
    fixing_lag=0, compounding=CompoundingMethod.COMPOUNDED,
    observation_shift=0, payment_delay=0,
    tenor_months=None, is_overnight=True,
    administrator="RBNZ",
))

# Term IBOR indices (legacy, still used for some products)
EURIBOR_3M = _register(RateIndex(
    name="EURIBOR_3M", currency="EUR",
    day_count=DayCountConvention.ACT_360,
    fixing_lag=2, compounding=CompoundingMethod.FLAT,
    observation_shift=0, payment_delay=0,
    tenor_months=3, is_overnight=False,
    administrator="EMMI",
))

EURIBOR_6M = _register(RateIndex(
    name="EURIBOR_6M", currency="EUR",
    day_count=DayCountConvention.ACT_360,
    fixing_lag=2, compounding=CompoundingMethod.FLAT,
    observation_shift=0, payment_delay=0,
    tenor_months=6, is_overnight=False,
    administrator="EMMI",
))

TIBOR_3M = _register(RateIndex(
    name="TIBOR_3M", currency="JPY",
    day_count=DayCountConvention.ACT_365_FIXED,
    fixing_lag=2, compounding=CompoundingMethod.FLAT,
    observation_shift=0, payment_delay=0,
    tenor_months=3, is_overnight=False,
    administrator="JBATA",
))


def get_rate_index(name: str) -> RateIndex:
    """Look up a rate index by name. Raises ValueError if not found."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown rate index '{name}'. Available: {available}")
    return _REGISTRY[name]


def all_rate_indices() -> list[RateIndex]:
    """Return all registered rate indices, sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda x: x.name)


def overnight_indices() -> list[RateIndex]:
    """Return all overnight RFR indices."""
    return [idx for idx in all_rate_indices() if idx.is_overnight]


def indices_for_currency(currency: str) -> list[RateIndex]:
    """Return all indices for a given currency."""
    return [idx for idx in all_rate_indices() if idx.currency == currency]
