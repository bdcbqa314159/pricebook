"""First-class IBOR projection curve with tenor-specific conventions.

An IBORCurve wraps a DiscountCurve with the metadata needed for
correct IBOR swap pricing: float/fixed frequencies, day counts,
fixing lag, and a reference to the OIS discount curve used in calibration.

    from pricebook.ibor_curve import IBORCurve, bootstrap_ibor
    from pricebook.ibor_curve import EURIBOR_3M_CONVENTIONS

    ois = bootstrap_ois(ref, ois_rates)
    ibor = bootstrap_ibor(ref, EURIBOR_3M_CONVENTIONS, ois,
                          deposits=deps, swaps=swaps)
    fwd = ibor.forward_rate(d1, d2)

References:
    Ametrano & Bianchetti, *Everything You Always Wanted to Know
    About Multiple Interest Rate Curve Bootstrapping*, 2013.
    Henrard, M., *Interest Rate Modelling in the Multi-Curve
    Framework*, Palgrave Macmillan, 2014, Ch. 2-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.bootstrap import bootstrap_forward_curve
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.rate_index import (
    RateIndex, EURIBOR_3M, EURIBOR_6M, TIBOR_3M,
)
from pricebook.schedule import Frequency


# ---- Conventions ----

@dataclass(frozen=True)
class IBORConventions:
    """Tenor-specific swap conventions for an IBOR index.

    Encapsulates everything needed to bootstrap and reprice IBOR swaps:
    the index itself, plus the market conventions for fixed and floating legs.

    Args:
        index: the RateIndex (EURIBOR_3M, TIBOR_3M, etc.).
        float_frequency: floating leg payment frequency (QUARTERLY for 3M).
        float_day_count: floating leg day count (ACT/360 for EURIBOR).
        fixed_frequency: fixed leg payment frequency (ANNUAL for EUR).
        fixed_day_count: fixed leg day count (30/360 for EUR).
        spot_lag: settlement lag in business days (2 for most).
    """
    index: RateIndex
    float_frequency: Frequency
    float_day_count: DayCountConvention
    fixed_frequency: Frequency
    fixed_day_count: DayCountConvention
    spot_lag: int = 2

    @property
    def tenor_months(self) -> int:
        return self.index.tenor_months or 0

    @property
    def currency(self) -> str:
        return self.index.currency

    @property
    def name(self) -> str:
        return self.index.name


# Pre-built convention sets for standard IBOR indices

EURIBOR_3M_CONVENTIONS = IBORConventions(
    index=EURIBOR_3M,
    float_frequency=Frequency.QUARTERLY,
    float_day_count=DayCountConvention.ACT_360,
    fixed_frequency=Frequency.ANNUAL,
    fixed_day_count=DayCountConvention.THIRTY_360,
    spot_lag=2,
)

EURIBOR_6M_CONVENTIONS = IBORConventions(
    index=EURIBOR_6M,
    float_frequency=Frequency.SEMI_ANNUAL,
    float_day_count=DayCountConvention.ACT_360,
    fixed_frequency=Frequency.ANNUAL,
    fixed_day_count=DayCountConvention.THIRTY_360,
    spot_lag=2,
)

TIBOR_3M_CONVENTIONS = IBORConventions(
    index=TIBOR_3M,
    float_frequency=Frequency.QUARTERLY,
    float_day_count=DayCountConvention.ACT_365_FIXED,
    fixed_frequency=Frequency.SEMI_ANNUAL,
    fixed_day_count=DayCountConvention.ACT_365_FIXED,
    spot_lag=2,
)


# ---- Conventions registry ----

_CONVENTIONS_REGISTRY: dict[str, IBORConventions] = {
    "EURIBOR_3M": EURIBOR_3M_CONVENTIONS,
    "EURIBOR_6M": EURIBOR_6M_CONVENTIONS,
    "TIBOR_3M": TIBOR_3M_CONVENTIONS,
}


def get_conventions(name: str) -> IBORConventions:
    """Look up IBORConventions by name. Raises KeyError if not found."""
    if name not in _CONVENTIONS_REGISTRY:
        available = ", ".join(sorted(_CONVENTIONS_REGISTRY.keys()))
        raise KeyError(f"Unknown conventions '{name}'. Available: {available}")
    return _CONVENTIONS_REGISTRY[name]


def register_conventions(name: str, conventions: IBORConventions) -> None:
    """Register custom IBORConventions for serialisation."""
    _CONVENTIONS_REGISTRY[name] = conventions


# ---- IBORCurve ----

class IBORCurve:
    """First-class IBOR projection curve.

    Wraps a DiscountCurve with convention metadata and a reference to the
    OIS discount curve used during calibration. Provides IBOR-specific
    methods: fixing() for a single date, forward_rate() for a period.

    The underlying DiscountCurve stores pseudo-discount factors whose
    ratios produce the correct IBOR forward rates under dual-curve
    pricing (Henrard 2014 §2.3).
    """

    def __init__(
        self,
        projection_curve: DiscountCurve,
        conventions: IBORConventions,
        discount_curve: DiscountCurve | None = None,
    ):
        self._projection = projection_curve
        self.conventions = conventions
        self._discount = discount_curve

    @property
    def reference_date(self) -> date:
        return self._projection.reference_date

    @property
    def index(self) -> RateIndex:
        return self.conventions.index

    @property
    def tenor_months(self) -> int:
        return self.conventions.tenor_months

    @property
    def projection_curve(self) -> DiscountCurve:
        """The underlying DiscountCurve for forward rate projection."""
        return self._projection

    @property
    def discount_curve(self) -> DiscountCurve | None:
        """The OIS discount curve used during calibration (if available)."""
        return self._discount

    def df(self, d: date) -> float:
        """Pseudo-discount factor from the projection curve."""
        return self._projection.df(d)

    def forward_rate(self, start: date, end: date) -> float:
        """Simply-compounded forward IBOR rate between start and end.

        F(t1, t2) = (df(t1) / df(t2) - 1) / tau

        where tau uses the floating leg day count convention.
        """
        df1 = self._projection.df(start)
        df2 = self._projection.df(end)
        tau = year_fraction(start, end, self.conventions.float_day_count)
        if tau < 1e-10:
            return 0.0
        return (df1 / df2 - 1.0) / tau

    def fixing(self, fixing_date: date) -> float:
        """IBOR fixing for a specific date.

        Computes the forward rate from fixing_date to
        fixing_date + tenor months, using exact month arithmetic.
        """
        from dateutil.relativedelta import relativedelta

        tenor = self.tenor_months
        if tenor <= 0:
            raise ValueError(f"Cannot compute fixing for overnight index {self.index.name}")
        end = fixing_date + relativedelta(months=tenor)
        return self.forward_rate(fixing_date, end)

    def zero_rate(self, d: date) -> float:
        """Continuously compounded zero rate from the projection curve."""
        return self._projection.zero_rate(d)

    def bumped(self, shift: float) -> IBORCurve:
        """Return a parallel-bumped IBORCurve."""
        return IBORCurve(
            self._projection.bumped(shift),
            self.conventions,
            self._discount,
        )


# ---- Bootstrap ----

def bootstrap_ibor(
    reference_date: date,
    conventions: IBORConventions,
    discount_curve: DiscountCurve,
    deposits: list[tuple[date, float]] | None = None,
    fras: list[tuple[date, date, float]] | None = None,
    futures: list[tuple[date, date, float]] | None = None,
    swaps: list[tuple[date, float]] | None = None,
    hw_convexity_a: float = 0.0,
    hw_convexity_sigma: float = 0.0,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> IBORCurve:
    """Bootstrap an IBOR projection curve using OIS discounting.

    Thin wrapper around bootstrap_forward_curve() from bootstrap.py,
    passing the correct conventions from IBORConventions.

    The resulting curve's forward rates, when used to price the input
    IBOR swaps with OIS discounting, reprice them at par within
    round-trip tolerance (1e-6).

    Args:
        reference_date: curve date.
        conventions: IBORConventions defining the index and swap structure.
        discount_curve: OIS curve for discounting all cashflows.
        deposits: short-end deposit rates [(maturity, rate)].
        fras: FRA rates [(start, end, rate)].
        futures: IR futures [(start, end, price_or_rate)].
        swaps: IBOR swap par rates [(maturity, rate)], sorted by maturity.
        hw_convexity_a: Hull-White mean reversion for futures convexity.
        hw_convexity_sigma: Hull-White vol for futures convexity.
        interpolation: curve interpolation method.
        calendar: business day calendar.
        convention: business day convention.

    Returns:
        An IBORCurve wrapping the bootstrapped projection curve.
    """
    if swaps is None:
        swaps = []
    if not swaps and not deposits and not fras and not futures:
        raise ValueError("At least one instrument required for bootstrapping")

    projection = bootstrap_forward_curve(
        reference_date=reference_date,
        swaps=swaps,
        discount_curve=discount_curve,
        deposits=deposits,
        fras=fras,
        futures=futures,
        deposit_day_count=conventions.float_day_count,
        fixed_day_count=conventions.fixed_day_count,
        float_day_count=conventions.float_day_count,
        fixed_frequency=conventions.fixed_frequency,
        float_frequency=conventions.float_frequency,
        interpolation=interpolation,
        calendar=calendar,
        convention=convention,
        hw_convexity_a=hw_convexity_a,
        hw_convexity_sigma=hw_convexity_sigma,
    )

    return IBORCurve(projection, conventions, discount_curve)

from pricebook.serialisable import _register

IBORCurve._SERIAL_TYPE = "ibor_curve"

def _ibor_to_dict(self):
    d = {"type": "ibor_curve", "params": {
        "conventions_name": self.conventions.name,
        "projection_curve": self._projection.to_dict(),
    }}
    if self._discount is not None:
        d["params"]["discount_curve"] = self._discount.to_dict()
    return d

@classmethod
def _ibor_from_dict(cls, d):
    from pricebook.serialisable import from_dict as _fd
    p = d["params"]
    conventions = get_conventions(p["conventions_name"])
    proj = _fd(p["projection_curve"])
    disc = _fd(p["discount_curve"]) if "discount_curve" in p else None
    return cls(proj, conventions, disc)

IBORCurve.to_dict = _ibor_to_dict
IBORCurve.from_dict = _ibor_from_dict
_register(IBORCurve)
