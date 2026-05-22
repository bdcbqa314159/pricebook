"""Real-time curve bumping framework with Jacobian caching.

Efficient curve perturbation for production risk systems.

    from pricebook.curves.curve_bumper import (
        CurveBumper, InstrumentRiskReport,
    )

References:
    Henrard (2014). Interest Rate Modelling in the Multi-Curve Framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


@dataclass
class InstrumentRiskReport:
    """Risk report for a single instrument."""
    instrument_id: str
    parallel_dv01: float
    key_rate_dv01s: dict[float, float]
    convexity: float

    def to_dict(self) -> dict:
        return vars(self)


class CurveBumper:
    """Efficient curve bumper with Jacobian-based fast repricing.

    Pre-computes the Jacobian ∂PV/∂z_i once, then applies arbitrary
    perturbation vectors via matrix multiplication.

    Args:
        base_curve: the base discount curve.
        pricer: callable(DiscountCurve) → float (returns PV).
        bump_size: finite-diff bump for Jacobian (default 0.5bp).
    """

    def __init__(
        self,
        base_curve: DiscountCurve,
        pricer: callable,
        bump_size: float = 0.00005,
    ):
        self.base_curve = base_curve
        self.pricer = pricer
        self.bump_size = bump_size
        self._base_pv = pricer(base_curve)
        self._jacobian: np.ndarray | None = None
        self._pillar_times: np.ndarray | None = None

    @property
    def base_pv(self) -> float:
        return self._base_pv

    @property
    def n_pillars(self) -> int:
        return len(self.base_curve.pillar_dates)

    def _ensure_jacobian(self) -> None:
        """Compute Jacobian if not cached."""
        if self._jacobian is not None:
            return

        ref = self.base_curve.reference_date
        n = self.n_pillars
        jac = np.zeros(n)
        self._pillar_times = np.array([
            (d - ref).days / 365.0 for d in self.base_curve.pillar_dates
        ])

        for i in range(n):
            bumped = self.base_curve.bumped_at(i, self.bump_size)
            pv_bumped = self.pricer(bumped)
            jac[i] = (pv_bumped - self._base_pv) / self.bump_size

        self._jacobian = jac

    def bump_and_reprice(self, shifts: np.ndarray) -> float:
        """Fast reprice via Jacobian: PV ≈ base_PV + J · Δz.

        Args:
            shifts: zero rate shift at each pillar (in decimal, e.g. 0.0001 = 1bp).
        """
        self._ensure_jacobian()
        return self._base_pv + float(self._jacobian @ shifts)

    def full_rebuild_and_reprice(self, shifts: np.ndarray) -> float:
        """Exact reprice by rebuilding the curve (slow but exact)."""
        ref = self.base_curve.reference_date
        new_dfs = []
        for i, d in enumerate(self.base_curve.pillar_dates):
            df = self.base_curve.df(d)
            t = (d - ref).days / 365.0
            new_dfs.append(df * math.exp(-shifts[i] * t) if t > 0 else df)
        new_curve = DiscountCurve(ref, self.base_curve.pillar_dates, new_dfs)
        return self.pricer(new_curve)

    def parallel_dv01(self) -> float:
        """DV01 for a 1bp parallel shift."""
        self._ensure_jacobian()
        shift_1bp = np.full(self.n_pillars, 0.0001)
        return float(self._jacobian @ shift_1bp)

    def key_rate_dv01s(self, tenors: list[float] | None = None) -> dict[float, float]:
        """Key-rate DV01s via Jacobian (fast)."""
        self._ensure_jacobian()
        if tenors is None:
            tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]

        results = {}
        for tenor in tenors:
            shifts = np.zeros(self.n_pillars)
            for i, t in enumerate(self._pillar_times):
                if abs(t - tenor) < 1.0:
                    w = max(0, 1.0 - abs(t - tenor))
                    shifts[i] = 0.0001 * w
            results[tenor] = float(self._jacobian @ shifts)
        return results

    def cross_gamma(self, i: int, j: int) -> float:
        """Cross gamma between pillars i and j."""
        bump = self.bump_size
        base = self._base_pv

        c_i = self.base_curve.bumped_at(i, bump)
        c_j = self.base_curve.bumped_at(j, bump)

        # Build double-bumped curve
        ref = self.base_curve.reference_date
        dfs_ij = []
        for k, d in enumerate(self.base_curve.pillar_dates):
            df = self.base_curve.df(d)
            t = (d - ref).days / 365.0
            s = bump if k == i else 0.0
            s += bump if k == j else 0.0
            dfs_ij.append(df * math.exp(-s * t) if t > 0 else df)
        c_ij = DiscountCurve(ref, self.base_curve.pillar_dates, dfs_ij)

        pv_i = self.pricer(c_i)
        pv_j = self.pricer(c_j)
        pv_ij = self.pricer(c_ij)

        return (pv_ij - pv_i - pv_j + base) / (bump ** 2)

    def risk_report(self, instrument_id: str = "") -> InstrumentRiskReport:
        """Full risk report for the instrument."""
        self._ensure_jacobian()
        pdv01 = self.parallel_dv01()
        kr = self.key_rate_dv01s()

        # Convexity: parallel 2nd derivative
        shift = 0.0001
        pv_up = self.full_rebuild_and_reprice(np.full(self.n_pillars, shift))
        pv_dn = self.full_rebuild_and_reprice(np.full(self.n_pillars, -shift))
        convexity = (pv_up - 2 * self._base_pv + pv_dn) / (shift ** 2)

        return InstrumentRiskReport(
            instrument_id=instrument_id,
            parallel_dv01=pdv01,
            key_rate_dv01s=kr,
            convexity=convexity,
        )
