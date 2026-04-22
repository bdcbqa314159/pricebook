"""FX forward curve: term structure of forward rates and points.

Stores the full forward point curve for a currency pair, derived from
the spot rate and two discount curves via CIP.

    from pricebook.fx_forward_curve import FXForwardCurve

    curve = FXForwardCurve.from_curves("EUR/USD", 1.10, ref, eur_curve, usd_curve, tenors)

References:
    Wystup, *FX Options and Structured Products*, Wiley, 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod, create_interpolator


class FXForwardCurve:
    """Term structure of FX forward rates.

    Args:
        pair: currency pair string.
        reference_date: valuation date.
        spot: spot FX rate.
        dates: tenor dates.
        forwards: forward rates at each tenor.
    """

    def __init__(
        self,
        pair: str,
        reference_date: date,
        spot: float,
        dates: list[date],
        forwards: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(dates) != len(forwards):
            raise ValueError("dates and forwards must have the same length")

        self.pair = pair
        self.reference_date = reference_date
        self.spot = spot
        self._dates = list(dates)
        self._forwards = list(forwards)
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]
        if len(times) == 1:
            self._interp = None
            self._single = forwards[0]
        else:
            self._interp = create_interpolator(
                InterpolationMethod.LINEAR, np.array(times), np.array(forwards),
            )
            self._single = None

    @classmethod
    def from_curves(
        cls,
        pair: str,
        spot: float,
        reference_date: date,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
        tenor_dates: list[date],
    ) -> "FXForwardCurve":
        """Build forward curve from spot + two discount curves via CIP."""
        forwards = [
            spot * base_curve.df(d) / quote_curve.df(d)
            for d in tenor_dates
        ]
        return cls(pair, reference_date, spot, tenor_dates, forwards)

    def forward(self, d: date) -> float:
        """Interpolated forward rate at date d."""
        if self._single is not None:
            return self._single
        t = year_fraction(self.reference_date, d, self.day_count)
        t = max(t, 0.0)
        return float(self._interp(t))

    def forward_points(self, d: date) -> float:
        """Forward points at date d: F(d) - S."""
        return self.forward(d) - self.spot

    def forward_points_curve(self) -> list[tuple[date, float]]:
        """Full forward points curve."""
        return [(d, f - self.spot) for d, f in zip(self._dates, self._forwards)]

    def implied_basis(
        self,
        d: date,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """Implied basis at tenor d: difference between market and CIP forward."""
        cip_fwd = self.spot * base_curve.df(d) / quote_curve.df(d)
        market_fwd = self.forward(d)
        return market_fwd - cip_fwd
