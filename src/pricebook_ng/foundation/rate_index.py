"""Interest-rate index identity — decomposed, carrying its own calendar (L0).

Gate audit F2/F3: the earlier `RateIndex` was a flat 12-field bag that **inferred its
calendar from currency** — the exact quarry flaw #14/#16 ratified as fixed (SOFR fixes on
SIFMA, not generic USD; two USD indices can sit on different calendars). Fixed here: the
index carries its own `RollRule` (hence calendar), and it **decomposes** by convention
family (no `fields-exempt`). It satisfies the `Underlying` protocol (`name`,
`asset_class`), so it is the rates sibling of the general identity concept.

`accrued_rate` is the one realized-rate function; branching only on `AccrualMethod`
(and forward/backward). Registry by explicit construction — no import-time JSON reload.

Provenance:
  quarry: python/pricebook/core/rate_index.py
  source: ISDA 2006/2021; RFR compounding (ARRC/ISDA); IBOR fallbacks; gate audit F2/F3
  oracle: SOFR-on-SIFMA ≠ a same-currency index on another calendar; compounded RFR;
          lookback ≠ observation-shift; forward ≠ backward; fallback = base + spread
  slice:  index-rework (Topic 0 gate rework, F2/F3/F4)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from types import MappingProxyType

from pricebook_ng.foundation.calendars import CalendarProtocol
from pricebook_ng.foundation.day_count import Accrual, DayCountConvention
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.schedule import BusinessDayConvention, RollRule
from pricebook_ng.foundation.tenor import Tenor, TenorUnit
from pricebook_ng.foundation.underlying import AssetClass


class AccrualMethod(Enum):
    COMPOUNDED = "compounded"  # money-market ∏(1 + r_i·δ_i) − 1 (RFR)
    EXPONENTIAL = "exponential"  # Brazilian ∏(1 + r_i)^(1/basis) − CDI/SELIC
    AVERAGED = "averaged"  # weighted average of the daily rates
    FLAT = "flat"  # a single fixing for the period (IBOR / term RFR)


class ObservationStyle(Enum):
    BACKWARD_LOOKING = "backward"  # compounded in arrears (RFR)
    FORWARD_LOOKING = "forward"  # fixed at the start for the period (IBOR, Term SOFR)


@dataclass(frozen=True)
class IndexId:
    name: str
    currency: Currency
    tenor: Tenor  # `Tenor(1, DAY)` for overnight


@dataclass(frozen=True)
class AccrualConvention:
    """How the period accrues: the day count and the roll rule — **which carries the
    calendar** (F2). No currency inference anywhere."""

    day_count: DayCountConvention
    roll: RollRule


@dataclass(frozen=True)
class FixingRule:
    observation_style: ObservationStyle
    compounding: AccrualMethod
    fixing_lag: int


@dataclass(frozen=True)
class RfrConvention:
    observation_shift: int = 0
    lookback: int = 0
    lockout: int = 0
    payment_delay: int = 0

    @classmethod
    def none(cls) -> RfrConvention:
        """An IBOR / term rate has no RFR mechanics — the distinction is visible, not five zeros."""
        return cls()


@dataclass(frozen=True)
class RateIndex:
    """An interest-rate index identity (5 sub-parts, no exemption). Satisfies `Underlying`."""

    id: IndexId
    accrual: AccrualConvention
    fixing: FixingRule
    rfr: RfrConvention
    spread_adjustment: float = (
        0.0  # ISDA fallback credit spread, added to the realized rate
    )

    @property
    def name(self) -> str:
        return self.id.name

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.RATES


@dataclass(frozen=True)
class FixingHistory:
    """Published fixings keyed by index name then date — generic over index (rates now;
    inflation levels / FX fixings / equity observations later)."""

    fixings: Mapping[str, Mapping[date, float]]

    def rate(self, index_name: str, on: date) -> float:
        by_index = self.fixings.get(index_name)
        if by_index is None or on not in by_index:
            raise ValueError(f"no fixing for {index_name!r} on {on}")
        return by_index[on]


_ANNUAL_BASIS: dict[DayCountConvention, int] = {DayCountConvention.BUS_252: 252}


def _annual_basis(day_count: DayCountConvention) -> int:
    basis = _ANNUAL_BASIS.get(day_count)
    if basis is None:
        raise ValueError(
            f"{day_count.value} is not a business-day-counted convention (no annual basis)"
        )
    return basis


def exponential_growth(
    rate: float, business_days: int, day_count: DayCountConvention
) -> float:
    """Brazilian compound **growth factor** for a SINGLE fixed rate over `business_days`:
    `(1 + rate) ** (business_days / basis)` (basis from the day-count). The fixed-`r`
    case (LTN/NTN-F); the daily series `r_i` is `accrued_rate`."""
    return (1.0 + rate) ** (business_days / _annual_basis(day_count))


def _denominator(day_count: DayCountConvention) -> float:
    return 365.0 if day_count is DayCountConvention.ACT_365_FIXED else 360.0


def _overnight_days(
    start: date, end: date, calendar: CalendarProtocol
) -> list[tuple[date, int]]:
    days: list[date] = []
    d = start
    while d < end:
        if calendar.is_business_day(d):
            days.append(d)
        d += timedelta(days=1)
    out: list[tuple[date, int]] = []
    for i, b in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else end
        out.append((b, (nxt - b).days))
    return out


def accrued_rate(index: RateIndex, accrual: Accrual, fixings: FixingHistory) -> float:
    """The realized rate of `index` over `accrual`, plus its `spread_adjustment`. The
    business-day calendar is **the index's own** (`index.accrual.roll.calendar`) — never
    inferred from currency (F2)."""
    calendar = index.accrual.roll.calendar
    if calendar is None:
        raise ValueError(f"index {index.name!r} has no calendar on its roll rule")
    day_count = index.accrual.day_count

    if index.fixing.compounding is AccrualMethod.FLAT:
        fixing_date = calendar.add_business_days(
            accrual.start, -index.fixing.fixing_lag
        )
        return fixings.rate(index.name, fixing_date) + index.spread_adjustment

    rfr = index.rfr
    if rfr.observation_shift:
        window = _overnight_days(
            calendar.add_business_days(accrual.start, -rfr.observation_shift),
            calendar.add_business_days(accrual.end, -rfr.observation_shift),
            calendar,
        )
        rate_dates = [b for b, _ in window]
    else:
        window = _overnight_days(accrual.start, accrual.end, calendar)
        rate_dates = [calendar.add_business_days(b, -rfr.lookback) for b, _ in window]

    if not window:
        raise ValueError(
            f"{index.name}: accrual [{accrual.start}, {accrual.end}) contains no business day "
            f"on {calendar.identity} — a degenerate rate window (S14)"
        )

    if rfr.lockout:
        frozen = len(window) - 1 - rfr.lockout
        rate_dates = [rate_dates[min(i, frozen)] for i in range(len(window))]

    if index.fixing.compounding is AccrualMethod.EXPONENTIAL:
        basis = _annual_basis(day_count)
        factor = 1.0
        for rate_date in rate_dates:
            factor *= (1.0 + fixings.rate(index.name, rate_date)) ** (1.0 / basis)
        realized = factor ** (basis / len(window)) - 1.0
        return realized + index.spread_adjustment

    denom = _denominator(day_count)
    factor, weighted, total = 1.0, 0.0, 0
    for (_, weight), rate_date in zip(window, rate_dates):
        r = fixings.rate(index.name, rate_date)
        factor *= 1.0 + r * (weight / denom)
        weighted += r * weight
        total += weight

    if index.fixing.compounding is AccrualMethod.COMPOUNDED:
        # Normalise over the SAME window as the numerator: total compounding days / basis.
        # Under observation_shift the window is the shifted (observation) period, and its
        # day-count fraction must follow it — not the interest period (F2). For lookback the
        # two coincide, so this is a no-op there and the SOFR 1e-12 oracle is unchanged.
        realized = (factor - 1.0) / (total / denom)
    else:  # AVERAGED
        realized = weighted / total
    return realized + index.spread_adjustment


# ── registry: explicit construction, index carries its own calendar ──────────────
_registry: dict[
    str, RateIndex
] = {}  # mutable backing; frozen into `_REGISTRY` below (A2)
_MOD_FOL = BusinessDayConvention.MODIFIED_FOLLOWING


def register_rate_index(index: RateIndex) -> RateIndex:
    """Register a rate index (open registry — new indices are declarations; audit 3.3). A
    conflicting name **raises** (no silent overwrite of the 28-index table)."""
    if index.name in _registry:
        raise ValueError(f"rate index {index.name!r} is already registered")
    _registry[index.name] = index
    return index


_register = register_rate_index  # the declarations below


@contextmanager
def temporary_rate_index(index: RateIndex) -> Iterator[RateIndex]:
    """Register a rate index for a `with` block, then remove it — registry isolation for tests."""
    register_rate_index(index)
    try:
        yield index
    finally:
        del _registry[index.name]


def _accrual(cal_id: str, dc: DayCountConvention) -> AccrualConvention:
    return AccrualConvention(dc, RollRule(get_calendar(cal_id), _MOD_FOL, eom=False))


def _on(
    name: str, ccy: Currency, cal_id: str, dc: DayCountConvention, rfr: RfrConvention
) -> RateIndex:
    """A backward-looking compounded overnight RFR (the RFR knobs bundled in `rfr`)."""
    return RateIndex(
        IndexId(name, ccy, Tenor(1, TenorUnit.DAY)),
        _accrual(cal_id, dc),
        FixingRule(
            ObservationStyle.BACKWARD_LOOKING, AccrualMethod.COMPOUNDED, fixing_lag=0
        ),
        rfr,
    )


def _term(
    name: str, ccy: Currency, cal_id: str, tenor: str, dc: DayCountConvention
) -> RateIndex:
    return RateIndex(
        IndexId(name, ccy, Tenor.parse(tenor)),
        _accrual(cal_id, dc),
        FixingRule(ObservationStyle.FORWARD_LOOKING, AccrualMethod.FLAT, fixing_lag=2),
        RfrConvention.none(),
    )


_DC = DayCountConvention
# ISDA-standard SOFR OIS is plain compounded-in-arrears over the calculation period — NO
# observation shift, NO lookback — with a 2-business-day payment offset (audit 2.1). The
# loan/FRN `observation_shift=2` convention lives on the LIBOR-fallback index below, not here.
SOFR = _register(
    _on(
        "SOFR",
        Currency.USD,
        "US_GOVERNMENT_SECURITIES",
        _DC.ACT_360,
        RfrConvention(payment_delay=2),
    )
)
SONIA = _register(
    _on("SONIA", Currency.GBP, "LONDON", _DC.ACT_365_FIXED, RfrConvention())
)
ESTR = _register(_on("ESTR", Currency.EUR, "TARGET", _DC.ACT_360, RfrConvention()))
TONA = _register(_on("TONA", Currency.JPY, "TOKYO", _DC.ACT_365_FIXED, RfrConvention()))
SARON = _register(_on("SARON", Currency.CHF, "ZURICH", _DC.ACT_360, RfrConvention()))
# Brazilian CDI — exponential BUS/252 (built inline: the one non-compounded overnight)
CDI = _register(
    RateIndex(
        IndexId("CDI", Currency.BRL, Tenor(1, TenorUnit.DAY)),
        _accrual("SAO_PAULO", _DC.BUS_252),
        FixingRule(
            ObservationStyle.BACKWARD_LOOKING, AccrualMethod.EXPONENTIAL, fixing_lag=0
        ),
        RfrConvention(),
    )
)
EURIBOR_3M = _register(_term("EURIBOR_3M", Currency.EUR, "TARGET", "3M", _DC.ACT_360))
TERM_SOFR_3M = _register(
    _term("TERM_SOFR_3M", Currency.USD, "US_GOVERNMENT_SECURITIES", "3M", _DC.ACT_360)
)
USD_LIBOR_3M_FALLBACK = _register(
    RateIndex(
        IndexId("USD_LIBOR_3M_FALLBACK", Currency.USD, Tenor.parse("3M")),
        AccrualConvention(
            _DC.ACT_360, RollRule(get_calendar("US_GOVERNMENT_SECURITIES"), _MOD_FOL, eom=False)
        ),
        FixingRule(
            ObservationStyle.BACKWARD_LOOKING, AccrualMethod.COMPOUNDED, fixing_lag=0
        ),
        RfrConvention(observation_shift=2, payment_delay=2),
        spread_adjustment=0.0026161,
    )
)

# Closed registry — all indices are declared above by explicit construction (S5: no
# import-time I/O). The frozen view cannot be rebound or mutated (A2 — the quarry's bug was a
# JSON load that replaced the whole registry dict, dropping 27 of 28 indices).
_REGISTRY: Mapping[str, RateIndex] = MappingProxyType(_registry)


def get_rate_index(name: str) -> RateIndex:
    index = _REGISTRY.get(name)
    if index is None:
        raise ValueError(f"no rate index {name!r}. Known: {sorted(_REGISTRY)}")
    return index


def list_rate_indices() -> list[str]:
    return sorted(_REGISTRY)
