"""Credit-Risk-Adjusted (CRA) discount curve.

Embeds survival probability into the discount curve, producing
risky discount factors and risky forward rates.

    from pricebook.cra_curve import CRADiscountCurve

    cra = CRADiscountCurve(ois_curve, survival_curve, recovery=0.4)
    risky_df = cra.df(maturity)            # D(T) × [Q(T) + R(1-Q(T))]
    risky_fwd = cra.forward_rate(d1, d2)   # differs from risk-free!

References:
    Pucci (2014). CMT Convexity. Eq 7: D̂_tT = D_tT × e^{Γ_t - Γ_T}.
    Schönbucher (2003). Credit Derivatives Pricing Models. Ch. 3.
    Duffie & Singleton (1999). Modeling Term Structures of Defaultable Bonds.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import _register


class CRADiscountCurve:
    """Credit-risk-adjusted discount curve.

    Produces risky discount factors:
        D̂(T) = D(T) × [Q(T) + R × (1 - Q(T))]

    where D(T) is risk-free, Q(T) is survival, R is recovery.

    Without recovery (R=0): D̂(T) = D(T) × Q(T) (pure defaultable).

    The risky forward rate under this measure differs from risk-free:
        F̂(t1, t2) = (D̂(t1)/D̂(t2) - 1) / τ

    This captures the credit adjustment to floating rate projection.

    Args:
        risk_free_curve: OIS/risk-free discount curve.
        survival_curve: issuer's survival curve.
        recovery: recovery rate (0 = zero-recovery, 0.4 = standard).
    """

    _SERIAL_TYPE = "cra_curve"

    def __init__(
        self,
        risk_free_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        recovery: float = 0.0,
    ):
        if not 0.0 <= recovery <= 1.0:
            raise ValueError(f"recovery must be in [0, 1], got {recovery}")
        self._rf = risk_free_curve
        self._sc = survival_curve
        self.recovery = recovery

    @property
    def reference_date(self) -> date:
        return self._rf.reference_date

    @property
    def risk_free_curve(self) -> DiscountCurve:
        return self._rf

    @property
    def survival_curve(self) -> SurvivalCurve:
        return self._sc

    def df(self, d: date) -> float:
        """Risky discount factor: D(T) × [Q(T) + R(1-Q(T))].

        Duffie-Singleton (1999): defaultable zero coupon bond price
        under recovery-of-market-value assumption.
        """
        rf_df = self._rf.df(d)
        q = self._sc.survival(d)
        return rf_df * (q + self.recovery * (1 - q))

    def forward_rate(self, d1: date, d2: date) -> float:
        """Risky forward rate between d1 and d2.

        F̂(t1, t2) = (D̂(t1)/D̂(t2) - 1) / τ

        Note: this is NOT risk_free_forward + spread.
        When survival decreases fast, the risky forward can be
        lower than risk-free (counterintuitive but correct).
        """
        df1 = self.df(d1)
        df2 = self.df(d2)
        tau = year_fraction(d1, d2, DayCountConvention.ACT_365_FIXED)
        if tau < 1e-10 or df2 < 1e-15:
            return 0.0
        return (df1 / df2 - 1.0) / tau

    def zero_rate(self, d: date) -> float:
        """Risky zero rate: -ln(D̂(T)) / T."""
        t = year_fraction(self.reference_date, d, DayCountConvention.ACT_365_FIXED)
        if t <= 0:
            return 0.0
        dhat = self.df(d)
        if dhat <= 0:
            return 0.0
        return -math.log(dhat) / t

    def credit_adjustment(self, d: date) -> float:
        """Credit spread embedded in the CRA curve at date d.

        Difference between risky zero rate and risk-free zero rate.
        """
        return self.zero_rate(d) - self._rf.zero_rate(d)

    def as_discount_curve(self) -> DiscountCurve:
        """Convert to a plain DiscountCurve for use in standard pricers.

        Samples at standard tenors.
        """
        ref = self.reference_date
        tenors_days = [30, 91, 182, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300]
        dates = [ref + timedelta(days=d) for d in tenors_days]
        dfs = [self.df(d) for d in dates]
        return DiscountCurve(ref, dates, dfs)

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "risk_free_curve": self._rf.to_dict(),
            "survival_curve": self._sc.to_dict(),
            "recovery": self.recovery,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> CRADiscountCurve:
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(
            risk_free_curve=_fd(p["risk_free_curve"]),
            survival_curve=_fd(p["survival_curve"]),
            recovery=p.get("recovery", 0.0),
        )


_register(CRADiscountCurve)
