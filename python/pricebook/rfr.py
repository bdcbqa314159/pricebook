"""
Risk-Free Rate (RFR) compounding and IBOR fallback framework.

Modern multi-curve: RFR (SOFR, ESTER, SONIA) as the base, IBOR as
RFR + deterministic or stochastic spread.

    from pricebook.rfr import compound_rfr, RFRCurve, IBORFallback

    rate = compound_rfr(daily_rates, day_fracs)  # backward-looking
    rfr = RFRCurve(ois_curve)
    ibor = IBORFallback(rfr_curve, spread_curve)
"""

from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod, create_interpolator
from pricebook.schedule import Frequency
from pricebook.special_process import OUProcess


def compound_rfr(
    daily_rates: list[float],
    day_fracs: list[float],
    lockout_days: int = 0,
) -> float:
    """Backward-looking compounded RFR.

    rate = prod(1 + r_i * d_i) - 1, annualised by dividing by sum(d_i).

    Args:
        daily_rates: overnight rate for each business day.
        day_fracs: year fraction for each day (e.g. 1/360 for ACT/360).
        lockout_days: if > 0, last N days use the rate from day (N-lockout).
    """
    if not daily_rates:
        return 0.0

    rates = list(daily_rates)
    if lockout_days > 0 and len(rates) > lockout_days:
        lockout_rate = rates[-lockout_days - 1]
        for i in range(len(rates) - lockout_days, len(rates)):
            rates[i] = lockout_rate

    product = 1.0
    for r, d in zip(rates, day_fracs):
        product *= (1.0 + r * d)

    total_yf = sum(day_fracs)
    if total_yf <= 0:
        return 0.0

    return (product - 1.0) / total_yf


def compound_rfr_from_curve(
    curve: DiscountCurve,
    start: date,
    end: date,
) -> float:
    """Compounded RFR from a discount curve (exact, no daily simulation).

    Equivalent to the simply-compounded forward rate: (df_start/df_end - 1) / yf.
    """
    return curve.forward_rate(start, end)


@dataclass
class SpreadCurve:
    """Deterministic spread term structure: spread(T).

    Used for IBOR = RFR + spread.
    """

    reference_date: date
    dates: list[date]
    spreads: list[float]
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED

    def __post_init__(self):
        if len(self.dates) != len(self.spreads):
            raise ValueError("dates and spreads must have same length")
        times = [year_fraction(self.reference_date, d, self.day_count) for d in self.dates]
        if len(times) == 1:
            self._single = self.spreads[0]
            self._interp = None
        else:
            self._single = None
            self._interp = create_interpolator(
                InterpolationMethod.LINEAR,
                np.array(times), np.array(self.spreads),
            )

    def spread(self, d: date) -> float:
        """Spread at date d."""
        if self._single is not None:
            return self._single
        t = year_fraction(self.reference_date, d, self.day_count)
        return float(self._interp(max(t, 0.0)))


class IBORProjection:
    """IBOR rate = RFR forward rate + deterministic spread.

    Args:
        rfr_curve: OIS/RFR discount curve.
        spread_curve: deterministic IBOR-RFR spread curve.
    """

    def __init__(self, rfr_curve: DiscountCurve, spread_curve: SpreadCurve):
        self.rfr_curve = rfr_curve
        self.spread_curve = spread_curve

    def forward_rate(self, start: date, end: date) -> float:
        """IBOR forward rate = RFR forward + spread."""
        rfr_fwd = self.rfr_curve.forward_rate(start, end)
        mid = date.fromordinal(
            (start.toordinal() + end.toordinal()) // 2,
        )
        spread = self.spread_curve.spread(mid)
        return rfr_fwd + spread


def bootstrap_spread_curve(
    reference_date: date,
    ibor_swap_rates: list[tuple[date, float]],
    ois_curve: DiscountCurve,
    float_frequency: Frequency = Frequency.QUARTERLY,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_frequency: Frequency = Frequency.ANNUAL,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
) -> SpreadCurve:
    """Bootstrap the IBOR-RFR spread curve from IBOR swap rates.

    Given: IBOR swap par rates + OIS discount curve.
    For each maturity, solve for the spread s such that an IBOR swap with
    floating rate = OIS_forward + s reprices at the quoted par rate.

    Uses iterative root-finding (brentq) at each maturity.

    Args:
        ibor_swap_rates: [(maturity, par_rate)] sorted by maturity.
        ois_curve: OIS discount/projection curve.
        float_frequency: floating leg frequency (QUARTERLY for 3M IBOR).
        float_day_count: floating leg day count.
        fixed_frequency: fixed leg frequency.
        fixed_day_count: fixed leg day count.
    """
    from pricebook.solvers import brentq as _brentq
    from pricebook.schedule import generate_schedule

    dates = []
    spreads = []

    for mat, ibor_rate in sorted(ibor_swap_rates, key=lambda x: x[0]):
        fixed_sched = generate_schedule(reference_date, mat, fixed_frequency)
        float_sched = generate_schedule(reference_date, mat, float_frequency)

        # PV of fixed leg (discounted on OIS)
        pv_fixed = sum(
            ibor_rate * year_fraction(fixed_sched[i-1], fixed_sched[i], fixed_day_count)
            * ois_curve.df(fixed_sched[i])
            for i in range(1, len(fixed_sched))
        )

        def objective(s: float, _float_sched=float_sched) -> float:
            pv_float = 0.0
            for i in range(1, len(_float_sched)):
                d1, d2 = _float_sched[i - 1], _float_sched[i]
                ois_fwd = ois_curve.forward_rate(d1, d2)
                yf = year_fraction(d1, d2, float_day_count)
                pv_float += (ois_fwd + s) * yf * ois_curve.df(d2)
            return pv_fixed - pv_float

        spread = _brentq(objective, -0.05, 0.05)
        dates.append(mat)
        spreads.append(spread)

    return SpreadCurve(reference_date, dates, spreads, day_count)


class StochasticBasis:
    """Stochastic IBOR-RFR basis spread via Ornstein-Uhlenbeck.

    ds = -a*(s - s_bar)*dt + sigma_s*dW

    For joint simulation with rates: pass correlated BM increments.
    """

    def __init__(
        self,
        mean_spread: float,
        mean_reversion: float,
        vol: float,
        seed: int | None = 42,
    ):
        self.mean_spread = mean_spread
        self.mean_reversion = mean_reversion
        self.vol = vol
        self._ou = OUProcess(
            a=mean_reversion, mu=mean_spread, sigma=vol, seed=seed,
        )

    def simulate(
        self, s0: float, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Simulate spread paths. Shape: (n_paths, n_steps + 1)."""
        return self._ou.sample(x0=s0, T=T, n_steps=n_steps, n_paths=n_paths)

    def stationary_mean(self) -> float:
        return self.mean_spread

    def stationary_std(self) -> float:
        return math.sqrt(self._ou.stationary_variance())


@dataclass
class FallbackConfig:
    """IBOR fallback configuration.

    spread_adjustment: ISDA-published median spread (e.g. 26.161bp for USD LIBOR 3M).
    cessation_date: date when IBOR ceased publication.
    """

    spread_adjustment: float
    cessation_date: date


def ibor_fallback_rate(
    rfr_compounded: float,
    config: FallbackConfig,
) -> float:
    """Compute IBOR fallback rate.

    Post-cessation: IBOR = compounded RFR + ISDA spread adjustment.
    """
    return rfr_compounded + config.spread_adjustment

from pricebook.serialisable import _register

SpreadCurve._SERIAL_TYPE = "spread_curve"

def _spread_to_dict(self):
    return {"type": "spread_curve", "params": {
        "reference_date": self.reference_date.isoformat(),
        "dates": [d.isoformat() for d in self.dates],
        "spreads": [float(s) for s in self.spreads],
        "day_count": self.day_count.value,
    }}

@classmethod
def _spread_from_dict(cls, d):
    from datetime import date as _d
    p = d["params"]
    return cls(reference_date=_d.fromisoformat(p["reference_date"]),
               dates=[_d.fromisoformat(s) for s in p["dates"]],
               spreads=p["spreads"],
               day_count=DayCountConvention(p.get("day_count", "ACT_365_FIXED")))

SpreadCurve.to_dict = _spread_to_dict
SpreadCurve.from_dict = _spread_from_dict
_register(SpreadCurve)
