"""Survival curve for credit modeling."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.interpolation import (
    InterpolationMethod,
    create_interpolator,
    Interpolator,
)


class SurvivalCurve:
    """
    A survival curve maps dates to survival probabilities.

    Q(t) = probability of no default by time t.
    Under piecewise constant hazard rates: Q(t) = exp(-∫h(s)ds).

    Internally the curve stores survival probabilities at pillar dates and
    interpolates in log-space (piecewise constant hazard rates between pillars).

    Provides:
        - survival(date) -> Q(t), the survival probability
        - hazard_rate(date) -> instantaneous hazard rate at t
        - default_prob(d1, d2) -> probability of default between d1 and d2
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        survival_probs: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    ):
        if len(dates) != len(survival_probs):
            raise ValueError("dates and survival_probs must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")
        for sp in survival_probs:
            if sp <= 0 or sp > 1:
                raise ValueError(f"survival probabilities must be in (0, 1], got {sp}")

        self.reference_date = reference_date
        self.day_count = day_count
        self._interp_method = interpolation
        self._pillar_dates = list(dates)

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        # Prepend t=0, Q=1 if not already present
        if times[0] > 0:
            times = [0.0] + times
            survival_probs = [1.0] + list(survival_probs)

        self._times = np.array(times)
        self._survs = np.array(survival_probs)
        self._interpolator: Interpolator = create_interpolator(
            interpolation, self._times, self._survs
        )

    @classmethod
    def flat(cls, reference_date: date, hazard_rate: float, tenors: list[int] | None = None) -> "SurvivalCurve":
        """Build a flat survival curve at a constant hazard rate."""
        from dateutil.relativedelta import relativedelta
        if tenors is None:
            tenors = [1, 2, 3, 5, 7, 10]
        dates = [reference_date + relativedelta(years=t) for t in tenors]
        # Use actual year fractions so the curve is truly flat
        day_count = DayCountConvention.ACT_365_FIXED
        actual_times = [year_fraction(reference_date, d, day_count) for d in dates]
        survs = [math.exp(-hazard_rate * t) for t in actual_times]
        return cls(reference_date, dates, survs)

    def _time(self, d: date) -> float:
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def survival(self, d: date) -> float:
        """Survival probability Q(t). Returns 1.0 for d <= reference_date."""
        t = self._time(d)
        if t <= 0:
            return 1.0
        return self._interpolator(t)

    def hazard_rate(self, d: date) -> float:
        """
        Piecewise constant hazard rate at date d.

        h(t) = -d/dt ln(Q(t)). For piecewise constant hazard between pillars:
        h = -ln(Q(t2)/Q(t1)) / (t2 - t1)
        """
        t = self._time(d)
        if t <= 0:
            # Return short-end hazard rate (first segment)
            if len(self._times) >= 2 and self._times[1] > 0:
                q1 = float(self._survs[1])
                if q1 > 0:
                    return -math.log(q1) / float(self._times[1])
            return 0.0
        # Find the segment
        idx = int(np.searchsorted(self._times, t)) - 1
        idx = max(0, min(idx, len(self._times) - 2))
        t1, t2 = self._times[idx], self._times[idx + 1]
        q1, q2 = self._survs[idx], self._survs[idx + 1]
        if t2 <= t1 or q2 <= 0 or q1 <= 0:
            return 0.0
        return -math.log(q2 / q1) / (t2 - t1)

    def default_prob(self, d1: date, d2: date) -> float:
        """Probability of default between d1 and d2: Q(d1) - Q(d2)."""
        if d1 >= d2:
            raise ValueError(f"d1 ({d1}) must be before d2 ({d2})")
        return self.survival(d1) - self.survival(d2)

    def forward_hazard(self, d1: date, d2: date) -> float:
        """Forward hazard rate between d1 and d2.

        h(d1, d2) = -ln(Q(d2) / Q(d1)) / τ(d1, d2)

        This is the hazard rate that would apply if the entity is known
        to have survived to d1.
        """
        if d1 >= d2:
            raise ValueError(f"d1 ({d1}) must be before d2 ({d2})")
        q1 = self.survival(d1)
        q2 = self.survival(d2)
        if q1 <= 0 or q2 <= 0:
            return 0.0
        tau = year_fraction(d1, d2, self.day_count)
        if tau <= 0:
            return 0.0
        return -math.log(q2 / q1) / tau

    def forward_survival(self, d1: date, d2: date) -> float:
        """Conditional survival probability: Q(d2 | alive at d1).

        Q(d1, d2) = Q(d2) / Q(d1)

        Probability of surviving to d2 given survival to d1.
        """
        q1 = self.survival(d1)
        if q1 <= 0:
            return 0.0
        return self.survival(d2) / q1

    def marginal_default_density(self, d: date) -> float:
        """Instantaneous default density at date d.

        f(t) = h(t) × Q(t)

        where h(t) is the hazard rate and Q(t) is survival probability.
        This is the unconditional probability density of defaulting at time t.
        """
        return self.hazard_rate(d) * self.survival(d)

    def term_structure(self) -> list[dict]:
        """Extract the full hazard rate term structure at each pillar.

        Returns list of {date, survival, hazard_rate, default_prob_1y}.
        """
        result = []
        for i, d in enumerate(self._pillar_dates):
            from datetime import timedelta
            d_1y = d + timedelta(days=365)
            result.append({
                "date": d.isoformat(),
                "survival": self.survival(d),
                "hazard_rate": self.hazard_rate(d),
                "default_prob_1y": self.default_prob(d, d_1y) if d_1y <= self._pillar_dates[-1] + timedelta(days=365) else 0.0,
            })
        return result

    def pillar_hazards(self) -> list[tuple[float, float]]:
        """Extract (time, piecewise_constant_hazard) pairs at each pillar.

        Returns list suitable for HWHazardRate/CIRPlusPlus market_hazards parameter.
        h_i = -ln(Q_i / Q_{i-1}) / (t_i - t_{i-1}) for each segment.
        """
        result = []
        for i in range(1, len(self._times)):
            t_prev, t_curr = float(self._times[i - 1]), float(self._times[i])
            q_prev, q_curr = float(self._survs[i - 1]), float(self._survs[i])
            dt = t_curr - t_prev
            if dt > 0 and q_prev > 0 and q_curr > 0:
                h = -math.log(q_curr / q_prev) / dt
                result.append((t_curr, max(h, 0.0)))
        return result

    def bumped_at(self, pillar_idx: int, shift: float) -> SurvivalCurve:
        """Return a new curve with one pillar's hazard rate bumped.

        Public API for per-pillar sensitivity (previously internal in credit_risk.py).
        """
        from pricebook.credit_risk import _bump_survival_curve_at
        return _bump_survival_curve_at(self, pillar_idx, shift)

    def bumped(self, shift: float) -> SurvivalCurve:
        """Return a new curve with all hazard rates parallel-shifted."""
        from pricebook.credit_risk import _bump_survival_curve
        return _bump_survival_curve(self, shift)

from pricebook.serialisable import _register

SurvivalCurve._SERIAL_TYPE = "survival_curve"

def _sc_to_dict(self):
    pillar_survs = [float(s) for t, s in zip(self._times, self._survs) if t > 0]
    return {"type": "survival_curve", "params": {
        "reference_date": self.reference_date.isoformat(),
        "dates": [d.isoformat() for d in self._pillar_dates],
        "survival_probs": pillar_survs, "day_count": self.day_count.value,
        "interpolation": self._interp_method.value if hasattr(self, '_interp_method') else "log_linear",
    }}

@classmethod
def _sc_from_dict(cls, d):
    from datetime import date as _d
    from pricebook.interpolation import InterpolationMethod
    p = d["params"]
    interp = InterpolationMethod(p["interpolation"]) if "interpolation" in p else InterpolationMethod.LOG_LINEAR
    return cls(reference_date=_d.fromisoformat(p["reference_date"]),
               dates=[_d.fromisoformat(s) for s in p["dates"]],
               survival_probs=p["survival_probs"],
               day_count=DayCountConvention(p["day_count"]),
               interpolation=interp)

SurvivalCurve.to_dict = _sc_to_dict
SurvivalCurve.from_dict = _sc_from_dict
_register(SurvivalCurve)
