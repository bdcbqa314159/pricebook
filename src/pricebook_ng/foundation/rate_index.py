"""Interest-rate index identity + fixings + the accrued-rate primitive (L0).

The declarative index identity: **a new index is a DECLARATION, never a code change.**
`RateIndex` is the first instance of the shared concept and covers *all* rate kinds —
backward-looking compounded RFR (SOFR/SONIA/ESTR) and forward-looking term/IBOR
(EURIBOR, Term SOFR) via `observation_style` — plus `spread_adjustment` for ISDA
fallbacks (RFR + credit spread). Sibling identities (inflation level, FX fixing,
equity/commodity observation) follow the same pattern in later topics; `FixingHistory`
is already generic over index so it can hold their fixings too.

`accrued_rate` is the one generic realized-rate function; the only branching is on
`CompoundingMethod` (and the forward/backward split). Content mined from
`core/rate_index.py`; the registry is built by **explicit construction** — the quarry
rebound `_REGISTRY` from a JSON load at import, where one bad row dropped the other 27.

Provenance:
  quarry: python/pricebook/core/rate_index.py
  source: ISDA 2006 / 2021 definitions; RFR compounding (ARRC/ISDA); IBOR fallbacks
  oracle: compounded RFR vs hand series; lookback≠observation-shift; forward≠backward;
          fallback = base + spread
  slice:  index-identity (Topic 0 S5)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from pricebook_ng.foundation.cashflow import Accrual
from pricebook_ng.foundation.day_count import DayCountConvention
from pricebook_ng.foundation.market_calendars import calendar_for_currency
from pricebook_ng.foundation.money import Currency


class CompoundingMethod(Enum):
    COMPOUNDED = "compounded"    # money-market ∏(1 + r_i·δ_i) − 1 (RFR, δ = days/360)
    EXPONENTIAL = "exponential"  # Brazilian ∏(1 + r_i)^(1/252) − i.e. (1+r)^(bd/252); CDI/SELIC
    AVERAGED = "averaged"        # weighted average of the daily rates
    FLAT = "flat"               # a single fixing for the whole period (IBOR / term RFR)


class ObservationStyle(Enum):
    BACKWARD_LOOKING = "backward"   # compounded/averaged in arrears over the period (RFR)
    FORWARD_LOOKING = "forward"     # fixed at the start for the period ahead (IBOR, Term SOFR)


@dataclass(frozen=True)
class RateIndex:
    """A benchmark interest-rate index identity. Wide by design — an identity record of
    conventions, not a value with behaviour."""
    # fields-exempt: index-identity aggregate — cross-asset convention record (redesign/16 §2.4)

    name: str
    currency: Currency
    tenor: str                          # "ON" for overnight, else "1M"/"3M"/"6M"
    day_count: DayCountConvention
    fixing_lag: int                     # business days before accrual start (forward fixings)
    observation_shift: int              # shift the whole observation window back k business days
    lookback: int                       # shift only the rate observation back k business days
    lockout: int                        # freeze the rate for the last k business days
    payment_delay: int                  # business days after accrual end
    compounding: CompoundingMethod
    observation_style: ObservationStyle
    spread_adjustment: float = 0.0      # ISDA fallback credit spread, added to the realized rate


@dataclass(frozen=True)
class FixingHistory:
    """Published fixings keyed by index name then date — generic over index, so it holds
    rates, and later inflation levels / FX fixings / equity observations."""

    fixings: Mapping[str, Mapping[date, float]]

    def rate(self, index_name: str, on: date) -> float:
        by_index = self.fixings.get(index_name)
        if by_index is None or on not in by_index:
            raise ValueError(f"no fixing for {index_name!r} on {on}")
        return by_index[on]


def _denominator(day_count: DayCountConvention) -> float:
    return 365.0 if day_count is DayCountConvention.ACT_365_FIXED else 360.0


# Business days per year for a business-day-counted convention. Generalises the
# Brazilian 252: another basis (BUS/250, …) is a new day-count entry here, not a literal.
_ANNUAL_BASIS: dict[DayCountConvention, int] = {DayCountConvention.BUS_252: 252}


def _annual_basis(day_count: DayCountConvention) -> int:
    basis = _ANNUAL_BASIS.get(day_count)
    if basis is None:
        raise ValueError(f"{day_count.value} is not a business-day-counted convention (no annual basis)")
    return basis


def exponential_growth(rate: float, business_days: int, day_count: DayCountConvention) -> float:
    """The Brazilian compound **growth factor** for a SINGLE fixed `rate` over
    `business_days` days: ``(1 + rate) ** (business_days / basis)`` (basis from the
    day-count — BUS/252 → 252). This is the `r` case (fixed LTN/NTN-F coupons); the daily
    floating series `r_i` (CDI in arrears) is `accrued_rate`, which returns a *rate*."""
    return (1.0 + rate) ** (business_days / _annual_basis(day_count))


def _overnight_days(start: date, end: date, calendar) -> list[tuple[date, int]]:
    """Business days in ``[start, end)`` with each one's day-count weight (calendar days
    until the next business day, the last one running to `end`)."""
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
    """The realized rate of `index` over the `accrual` period from `fixings`, plus the
    index's `spread_adjustment`. Forward-looking / FLAT → the single fixing at the start;
    backward-looking → compounded or averaged over the observation window."""
    calendar = calendar_for_currency(index.currency.value)

    if index.compounding is CompoundingMethod.FLAT:
        fixing_date = calendar.add_business_days(accrual.start, -index.fixing_lag)
        return fixings.rate(index.name, fixing_date) + index.spread_adjustment

    # backward-looking: build the observation window and its weights
    if index.observation_shift:
        window = _overnight_days(
            calendar.add_business_days(accrual.start, -index.observation_shift),
            calendar.add_business_days(accrual.end, -index.observation_shift),
            calendar,
        )
        rate_dates = [b for b, _ in window]                     # rate read at the shifted date
    else:
        window = _overnight_days(accrual.start, accrual.end, calendar)
        rate_dates = [calendar.add_business_days(b, -index.lookback) for b, _ in window]

    if index.lockout:
        frozen = len(window) - 1 - index.lockout
        rate_dates = [rate_dates[min(i, frozen)] for i in range(len(window))]

    if index.compounding is CompoundingMethod.EXPONENTIAL:
        # Brazilian BUS/basis: compound the daily SERIES r_i, each over one business day
        # (1/basis of a year), then re-annualise over the business-day count. A flat series
        # collapses to the single-rate `exponential_growth` form; a flat rate returns itself.
        basis = _annual_basis(index.day_count)
        factor = 1.0
        for rate_date in rate_dates:
            factor *= (1.0 + fixings.rate(index.name, rate_date)) ** (1.0 / basis)
        realized = factor ** (basis / len(window)) - 1.0
        return realized + index.spread_adjustment

    denom = _denominator(index.day_count)
    factor, weighted, total = 1.0, 0.0, 0
    for (_, weight), rate_date in zip(window, rate_dates):
        r = fixings.rate(index.name, rate_date)
        factor *= 1.0 + r * (weight / denom)
        weighted += r * weight
        total += weight

    if index.compounding is CompoundingMethod.COMPOUNDED:
        realized = (factor - 1.0) / accrual.year_fraction()
    else:  # AVERAGED
        realized = weighted / total
    return realized + index.spread_adjustment


# ── registry: explicit construction, no import-time I/O ──────────────────────────
_REGISTRY: dict[str, RateIndex] = {}


def _register(index: RateIndex) -> RateIndex:
    _REGISTRY[index.name] = index
    return index


def _overnight(name: str, ccy: Currency, dc: DayCountConvention, obs_shift: int, pay_delay: int) -> RateIndex:
    return RateIndex(
        name=name, currency=ccy, tenor="ON", day_count=dc, fixing_lag=0,
        observation_shift=obs_shift, lookback=0, lockout=0, payment_delay=pay_delay,
        compounding=CompoundingMethod.COMPOUNDED,
        observation_style=ObservationStyle.BACKWARD_LOOKING,
    )


def _term(name: str, ccy: Currency, tenor: str, dc: DayCountConvention, style: ObservationStyle) -> RateIndex:
    return RateIndex(
        name=name, currency=ccy, tenor=tenor, day_count=dc, fixing_lag=2,
        observation_shift=0, lookback=0, lockout=0, payment_delay=0,
        compounding=CompoundingMethod.FLAT, observation_style=style,
    )


SOFR = _register(_overnight("SOFR", Currency.USD, DayCountConvention.ACT_360, 2, 2))
SONIA = _register(_overnight("SONIA", Currency.GBP, DayCountConvention.ACT_365_FIXED, 0, 0))
ESTR = _register(_overnight("ESTR", Currency.EUR, DayCountConvention.ACT_360, 0, 0))
TONA = _register(_overnight("TONA", Currency.JPY, DayCountConvention.ACT_365_FIXED, 0, 0))
SARON = _register(_overnight("SARON", Currency.CHF, DayCountConvention.ACT_360, 0, 0))

# Brazilian overnight — exponential BUS/252 compounding (CDI/SELIC)
CDI = _register(RateIndex(
    name="CDI", currency=Currency.BRL, tenor="ON", day_count=DayCountConvention.BUS_252,
    fixing_lag=0, observation_shift=0, lookback=0, lockout=0, payment_delay=0,
    compounding=CompoundingMethod.EXPONENTIAL, observation_style=ObservationStyle.BACKWARD_LOOKING,
))

EURIBOR_3M = _register(_term("EURIBOR_3M", Currency.EUR, "3M", DayCountConvention.ACT_360,
                             ObservationStyle.FORWARD_LOOKING))
TERM_SOFR_3M = _register(_term("TERM_SOFR_3M", Currency.USD, "3M", DayCountConvention.ACT_360,
                               ObservationStyle.FORWARD_LOOKING))

# an ISDA IBOR fallback: compounded SOFR + the fixed credit adjustment spread (3M)
USD_LIBOR_3M_FALLBACK = _register(RateIndex(
    name="USD_LIBOR_3M_FALLBACK", currency=Currency.USD, tenor="3M",
    day_count=DayCountConvention.ACT_360, fixing_lag=0, observation_shift=2, lookback=0,
    lockout=0, payment_delay=2, compounding=CompoundingMethod.COMPOUNDED,
    observation_style=ObservationStyle.BACKWARD_LOOKING, spread_adjustment=0.0026161,
))


def get_rate_index(name: str) -> RateIndex:
    index = _REGISTRY.get(name)
    if index is None:
        raise ValueError(f"no rate index {name!r}. Known: {sorted(_REGISTRY)}")
    return index


def list_rate_indices() -> list[str]:
    return sorted(_REGISTRY)
