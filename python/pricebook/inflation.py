"""
Inflation curve and instruments.

CPI curve: maps dates to expected CPI index levels.
    CPI(T) = CPI(0) * (1 + breakeven)^T

Zero-coupon inflation swap:
    Fixed leg pays (1+K)^T - 1, inflation leg pays CPI(T)/CPI(0) - 1.

Year-on-year inflation swap:
    Each period pays CPI(t_i)/CPI(t_{i-1}) - 1.

Inflation-linked bond (linker):
    Notional and coupons indexed to CPI.
"""

from __future__ import annotations

import math
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod, create_interpolator
from pricebook.schedule import Frequency, generate_schedule
from pricebook.solvers import brentq


class CPICurve:
    """Forward CPI index curve.

    Args:
        reference_date: valuation date.
        base_cpi: CPI index level at reference date.
        dates: pillar dates.
        cpi_levels: expected CPI at each pillar date.
        day_count: for year fraction computation.
    """

    def __init__(
        self,
        reference_date: date,
        base_cpi: float,
        dates: list[date],
        cpi_levels: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(dates) != len(cpi_levels):
            raise ValueError("dates and cpi_levels must have the same length")
        if base_cpi <= 0:
            raise ValueError(f"base_cpi must be positive, got {base_cpi}")

        self.reference_date = reference_date
        self.base_cpi = base_cpi
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]
        log_ratios = [math.log(c / base_cpi) for c in cpi_levels]

        if len(times) == 1:
            self._single_rate = log_ratios[0] / times[0] if times[0] > 0 else 0.0
            self._interp = None
        else:
            self._single_rate = None
            self._interp = create_interpolator(
                InterpolationMethod.LINEAR,
                np.array(times), np.array(log_ratios),
            )

    def cpi(self, d: date) -> float:
        """Expected CPI index at date d."""
        t = year_fraction(self.reference_date, d, self.day_count)
        if t <= 0:
            return self.base_cpi
        if self._single_rate is not None:
            return self.base_cpi * math.exp(self._single_rate * t)
        return self.base_cpi * math.exp(float(self._interp(t)))

    def breakeven_rate(self, d: date) -> float:
        """Annualised breakeven inflation rate to date d.

        Defined as: (CPI(T)/CPI(0))^(1/T) - 1
        """
        t = year_fraction(self.reference_date, d, self.day_count)
        if t <= 0:
            return 0.0
        ratio = self.cpi(d) / self.base_cpi
        return ratio ** (1.0 / t) - 1.0

    @staticmethod
    def from_breakevens(
        reference_date: date,
        base_cpi: float,
        dates: list[date],
        breakeven_rates: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ) -> "CPICurve":
        """Build a CPI curve from breakeven inflation rates."""
        cpi_levels = []
        for d, r in zip(dates, breakeven_rates):
            t = year_fraction(reference_date, d, day_count)
            cpi_levels.append(base_cpi * (1 + r) ** t)
        return CPICurve(reference_date, base_cpi, dates, cpi_levels, day_count)


# ---------------------------------------------------------------------------
# Zero-coupon inflation swap
# ---------------------------------------------------------------------------

def zc_inflation_swap_pv(
    fixed_rate: float,
    discount_curve: DiscountCurve,
    cpi_curve: CPICurve,
    maturity: date,
    notional: float = 1_000_000.0,
    T: float | None = None,
) -> float:
    """PV of a zero-coupon inflation swap (receiver inflation).

    PV = notional * df(T) * [CPI(T)/CPI(0) - (1+K)^T]

    T is computed from the curve's reference date if not provided.
    """
    if T is None:
        T = year_fraction(discount_curve.reference_date, maturity,
                          DayCountConvention.ACT_365_FIXED)
    df = discount_curve.df(maturity)
    cpi_ratio = cpi_curve.cpi(maturity) / cpi_curve.base_cpi
    return notional * df * (cpi_ratio - (1 + fixed_rate) ** T)


def zc_inflation_par_rate(
    discount_curve: DiscountCurve,
    cpi_curve: CPICurve,
    maturity: date,
    T: float | None = None,
) -> float:
    """Par rate of a ZC inflation swap: K such that PV = 0.

    K = (CPI(T)/CPI(0))^(1/T) - 1
    """
    if T is None:
        T = year_fraction(discount_curve.reference_date, maturity,
                          DayCountConvention.ACT_365_FIXED)
    cpi_ratio = cpi_curve.cpi(maturity) / cpi_curve.base_cpi
    return cpi_ratio ** (1.0 / T) - 1.0


# ---------------------------------------------------------------------------
# Year-on-year inflation swap
# ---------------------------------------------------------------------------

def yoy_inflation_swap_pv(
    fixed_rate: float,
    discount_curve: DiscountCurve,
    cpi_curve: CPICurve,
    start: date,
    end: date,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.ANNUAL,
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
) -> float:
    """PV of a year-on-year inflation swap (receiver inflation).

    Each period: notional * df(t_i) * [CPI(t_i)/CPI(t_{i-1}) - 1 - K*yf]
    """
    schedule = generate_schedule(start, end, frequency)
    pv = 0.0
    for i in range(1, len(schedule)):
        t_prev, t_curr = schedule[i - 1], schedule[i]
        df = discount_curve.df(t_curr)
        yf = year_fraction(t_prev, t_curr, day_count)
        cpi_ratio = cpi_curve.cpi(t_curr) / cpi_curve.cpi(t_prev)
        pv += notional * df * ((cpi_ratio - 1) - fixed_rate * yf)
    return pv


def yoy_inflation_par_rate(
    discount_curve: DiscountCurve,
    cpi_curve: CPICurve,
    start: date,
    end: date,
    frequency: Frequency = Frequency.ANNUAL,
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
) -> float:
    """Par rate of a YoY inflation swap: fixed rate that makes PV = 0."""
    schedule = generate_schedule(start, end, frequency)
    num = 0.0
    den = 0.0
    for i in range(1, len(schedule)):
        t_prev, t_curr = schedule[i - 1], schedule[i]
        df = discount_curve.df(t_curr)
        yf = year_fraction(t_prev, t_curr, day_count)
        cpi_ratio = cpi_curve.cpi(t_curr) / cpi_curve.cpi(t_prev)
        num += df * (cpi_ratio - 1)
        den += df * yf
    return num / den if abs(den) > 1e-15 else 0.0


# ---------------------------------------------------------------------------
# Inflation-linked bond
# ---------------------------------------------------------------------------

class InflationLinkedBond:
    """Inflation-linked bond (linker).

    Principal indexed to CPI: notional * CPI(T) / CPI(base).
    Coupons: coupon_rate * indexed_notional * year_frac.

    Args:
        start: issue date.
        end: maturity date.
        coupon_rate: real coupon rate.
        base_cpi_value: CPI at issue (for indexation ratio).
        notional: face value.
        frequency: coupon frequency.
        day_count: day count for accrual.
        cpi_lag_months: CPI publication lag (3 for US TIPS, 8 for UK ILG).
    """

    def __init__(
        self,
        start: date,
        end: date,
        coupon_rate: float,
        base_cpi_value: float,
        notional: float = 100.0,
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        cpi_lag_months: int = 3,
    ):
        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.base_cpi_value = base_cpi_value
        self.notional = notional
        self.frequency = frequency
        self.day_count = day_count
        self.cpi_lag_months = cpi_lag_months
        self.schedule = generate_schedule(start, end, frequency)

    def _lagged_date(self, d: date) -> date:
        """Apply CPI lag: the CPI used for date d is from d - lag months."""
        return d - relativedelta(months=self.cpi_lag_months)

    def _future_periods(self, settlement: date) -> list[tuple[date, date]]:
        """Return periods where payment_date > settlement."""
        return [
            (self.schedule[i - 1], self.schedule[i])
            for i in range(1, len(self.schedule))
            if self.schedule[i] > settlement
        ]

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        cpi_curve: CPICurve,
        settlement: date | None = None,
    ) -> float:
        """Full price per 100 face: PV of future indexed coupons + principal."""
        settle = settlement if settlement is not None else discount_curve.reference_date
        pv = 0.0
        for t_start, t_end in self._future_periods(settle):
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            index_ratio = cpi_curve.cpi(self._lagged_date(t_end)) / self.base_cpi_value
            pv += self.notional * self.coupon_rate * yf * index_ratio * df

        # Indexed principal at maturity
        index_ratio_mat = cpi_curve.cpi(self._lagged_date(self.end)) / self.base_cpi_value
        pv += self.notional * index_ratio_mat * discount_curve.df(self.end)

        return pv / self.notional * 100.0

    def real_yield(
        self,
        market_price: float,
        cpi_curve: CPICurve,
        settlement: date | None = None,
    ) -> float:
        """Real yield from settlement: constant real rate that matches price."""
        settle = settlement if settlement is not None else self.start

        def objective(y: float) -> float:
            pv = 0.0
            for t_start, t_end in self._future_periods(settle):
                yf = year_fraction(t_start, t_end, self.day_count)
                T_i = year_fraction(settle, t_end, self.day_count)
                index_ratio = cpi_curve.cpi(self._lagged_date(t_end)) / self.base_cpi_value
                df = math.exp(-y * T_i)
                pv += self.notional * self.coupon_rate * yf * index_ratio * df

            T_mat = year_fraction(settle, self.end, self.day_count)
            index_ratio_mat = cpi_curve.cpi(self._lagged_date(self.end)) / self.base_cpi_value
            pv += self.notional * index_ratio_mat * math.exp(-y * T_mat)

            return pv / self.notional * 100.0 - market_price

        return brentq(objective, -0.10, 0.30)

    def ie01(
        self,
        discount_curve: DiscountCurve,
        cpi_curve: CPICurve,
        shift_bps: float = 1.0,
        settlement: date | None = None,
    ) -> float:
        """IE01: price change for a 1bp parallel shift in breakeven inflation.

        Bumps CPI curve by creating a new curve with shifted breakevens.
        """
        settle = settlement if settlement is not None else discount_curve.reference_date
        price_base = self.dirty_price(discount_curve, cpi_curve, settle)

        # Bump breakevens by shift_bps
        shift = shift_bps / 10000.0
        ref = cpi_curve.reference_date
        pillar_dates = [d for d in [self.end] if d > ref]  # simplified: use maturity
        be = cpi_curve.breakeven_rate(self.end)
        bumped_cpi = CPICurve.from_breakevens(
            ref, cpi_curve.base_cpi, pillar_dates, [be + shift], cpi_curve.day_count,
        )
        price_bumped = self.dirty_price(discount_curve, bumped_cpi, settle)
        return price_bumped - price_base


# ---------------------------------------------------------------------------
# Inflation curve bootstrap
# ---------------------------------------------------------------------------

def bootstrap_cpi_curve(
    reference_date: date,
    base_cpi: float,
    zc_swap_quotes: list[tuple[date, float]],
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
) -> CPICurve:
    """Bootstrap a CPI curve from zero-coupon inflation swap rates.

    Args:
        reference_date: valuation date.
        base_cpi: CPI index at reference date.
        zc_swap_quotes: list of (maturity_date, par_rate) pairs, sorted.
    """
    dates = []
    cpi_levels = []
    for mat, rate in sorted(zc_swap_quotes, key=lambda x: x[0]):
        T = year_fraction(reference_date, mat, day_count)
        if T <= 0:
            continue
        cpi_T = base_cpi * (1 + rate) ** T
        dates.append(mat)
        cpi_levels.append(cpi_T)

    return CPICurve(reference_date, base_cpi, dates, cpi_levels, day_count)
