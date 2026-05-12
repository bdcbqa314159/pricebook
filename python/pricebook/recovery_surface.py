"""Recovery surface: R(seniority, tenor) with term structure and market-implied recovery.

A recovery surface captures how recovery varies across:
- Seniority: senior secured > senior unsecured > subordinated > mezzanine
- Tenor: short-dated defaults typically have higher recovery than long-dated
- Credit quality: distressed names show recovery declining with tenor

    from pricebook.recovery_surface import RecoverySurface, implied_recovery

    # From Moody's seniority table
    surface = RecoverySurface.from_seniority_table()
    R = surface.recovery("senior_unsecured", 5.0)

    # From market CDS (two-recovery method)
    R_implied = implied_recovery(cds_spreads_senior, cds_spreads_sub,
                                  discount_curve, reference_date)

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives, Ch. 4.
    Moody's (2022). Annual Default Study.
    Altman, Resti, Sironi (2004). Default Recovery Rates in Credit Risk Modelling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---------------------------------------------------------------------------
# Seniority ordering and base recovery table (Moody's 2022)
# ---------------------------------------------------------------------------

SENIORITY_ORDER = [
    "senior_secured",
    "1L",
    "senior_unsecured",
    "senior",
    "2L",
    "mezzanine",
    "sub",
]

# (mean_recovery, std, typical_tenor_slope)
# tenor_slope: recovery change per year of maturity (negative = declines with tenor)
SENIORITY_TABLE = {
    "senior_secured":   (0.65, 0.22, -0.005),
    "1L":               (0.77, 0.20, -0.003),
    "senior_unsecured": (0.45, 0.22, -0.008),
    "senior":           (0.45, 0.22, -0.008),
    "2L":               (0.43, 0.25, -0.012),
    "mezzanine":        (0.35, 0.22, -0.015),
    "sub":              (0.28, 0.20, -0.018),
}


# ---------------------------------------------------------------------------
# RecoverySurface
# ---------------------------------------------------------------------------

@dataclass
class RecoverySurfacePoint:
    """One point on the recovery surface."""
    seniority: str
    tenor: float
    recovery: float
    std: float


class RecoverySurface:
    """2D recovery surface: R(seniority, tenor).

    Interpolates recovery rate across seniority classes and maturity tenors.
    Accounts for the empirical observation that recovery declines with tenor
    for distressed names (longer time to default → more asset erosion).

    Args:
        seniorities: list of seniority labels.
        tenors: list of tenor points (years).
        recoveries: 2D array (n_seniorities × n_tenors).
        stds: 2D array of recovery volatilities (same shape).
    """

    def __init__(
        self,
        seniorities: list[str],
        tenors: list[float],
        recoveries: np.ndarray,
        stds: np.ndarray | None = None,
    ):
        self.seniorities = list(seniorities)
        self.tenors = np.array(tenors, dtype=float)
        self.recoveries = np.array(recoveries, dtype=float)
        if stds is not None:
            self.stds = np.array(stds, dtype=float)
        else:
            self.stds = np.full_like(self.recoveries, 0.20)

        self._sen_idx = {s: i for i, s in enumerate(self.seniorities)}

    def recovery(self, seniority: str, tenor: float) -> float:
        """Interpolated recovery at (seniority, tenor)."""
        if seniority not in self._sen_idx:
            raise ValueError(f"Unknown seniority '{seniority}'. "
                             f"Available: {self.seniorities}")
        i = self._sen_idx[seniority]
        return float(np.interp(tenor, self.tenors, self.recoveries[i]))

    def std(self, seniority: str, tenor: float) -> float:
        """Interpolated recovery volatility at (seniority, tenor)."""
        i = self._sen_idx[seniority]
        return float(np.interp(tenor, self.tenors, self.stds[i]))

    def recovery_vector(self, seniority: str) -> tuple[np.ndarray, np.ndarray]:
        """(tenors, recoveries) for a given seniority — for plotting."""
        i = self._sen_idx[seniority]
        return self.tenors.copy(), self.recoveries[i].copy()

    def all_points(self) -> list[RecoverySurfacePoint]:
        """Flatten the surface to a list of points."""
        points = []
        for i, sen in enumerate(self.seniorities):
            for j, t in enumerate(self.tenors):
                points.append(RecoverySurfacePoint(
                    sen, float(t), float(self.recoveries[i, j]),
                    float(self.stds[i, j])))
        return points

    def to_dict(self) -> dict:
        return {
            "seniorities": self.seniorities,
            "tenors": self.tenors.tolist(),
            "recoveries": self.recoveries.tolist(),
            "stds": self.stds.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RecoverySurface:
        return cls(
            d["seniorities"], d["tenors"],
            np.array(d["recoveries"]), np.array(d["stds"]),
        )

    # ---- Factory methods ----

    @classmethod
    def from_seniority_table(
        cls,
        tenors: list[float] | None = None,
        table: dict | None = None,
    ) -> RecoverySurface:
        """Build from the standard seniority recovery table.

        Recovery at each tenor: R(T) = R_base + slope × (T - 5)
        where T=5 is the reference tenor.

        Args:
            tenors: maturity points. Default: [1, 3, 5, 7, 10].
            table: override seniority table {name: (mean, std, slope)}.
        """
        if tenors is None:
            tenors = [1.0, 3.0, 5.0, 7.0, 10.0]
        if table is None:
            table = SENIORITY_TABLE

        seniorities = [s for s in SENIORITY_ORDER if s in table]
        n_sen = len(seniorities)
        n_ten = len(tenors)

        recoveries = np.zeros((n_sen, n_ten))
        stds = np.zeros((n_sen, n_ten))

        for i, sen in enumerate(seniorities):
            base, std, slope = table[sen]
            for j, t in enumerate(tenors):
                r = base + slope * (t - 5.0)
                recoveries[i, j] = max(0.05, min(0.95, r))
                stds[i, j] = std

        return cls(seniorities, tenors, recoveries, stds)

    @classmethod
    def flat(cls, recovery: float, std: float = 0.20) -> RecoverySurface:
        """Single-point flat surface (all seniorities, all tenors)."""
        seniorities = list(SENIORITY_TABLE.keys())
        tenors = [1.0, 5.0, 10.0]
        n = len(seniorities)
        recoveries = np.full((n, 3), recovery)
        stds_arr = np.full((n, 3), std)
        return cls(seniorities, tenors, recoveries, stds_arr)


# ---------------------------------------------------------------------------
# Market-implied recovery (two-recovery method)
# ---------------------------------------------------------------------------

@dataclass
class ImpliedRecoveryResult:
    """Market-implied recovery from CDS spread differentials."""
    implied_recovery: float
    senior_hazard: float
    sub_hazard: float
    tenor: int
    method: str

    def to_dict(self) -> dict:
        return {
            "implied_recovery": self.implied_recovery,
            "senior_hazard": self.senior_hazard,
            "sub_hazard": self.sub_hazard,
            "tenor": self.tenor,
            "method": self.method,
        }


def implied_recovery(
    senior_spreads: dict[int, float],
    sub_spreads: dict[int, float],
    discount_curve: DiscountCurve,
    reference_date: date,
    senior_recovery: float = 0.45,
    sub_recovery: float = 0.25,
) -> list[ImpliedRecoveryResult]:
    """Implied recovery from senior vs subordinated CDS spreads.

    Method 1 (spread ratio): If both CDS reference the same entity:
        h = s_senior / (1 - R_senior) = s_sub / (1 - R_sub)
        → R_senior = 1 - s_senior × (1 - R_sub) / s_sub

    Method 2 (bootstrap): Bootstrap both curves independently,
        compare hazard rates (should be equal for same entity).

    Args:
        senior_spreads: {tenor: spread} for senior CDS.
        sub_spreads: {tenor: spread} for subordinated CDS.
        senior_recovery / sub_recovery: assumed recoveries.
    """
    from pricebook.cds_market import build_cds_curve
    from pricebook.day_count import DayCountConvention, year_fraction

    results = []
    common_tenors = sorted(set(senior_spreads.keys()) & set(sub_spreads.keys()))

    surv_senior = build_cds_curve(reference_date, senior_spreads, discount_curve,
                                   recovery=senior_recovery)
    surv_sub = build_cds_curve(reference_date, sub_spreads, discount_curve,
                                recovery=sub_recovery)

    for tenor in common_tenors:
        s_sen = senior_spreads[tenor]
        s_sub = sub_spreads[tenor]

        # Method 1: spread ratio
        if s_sub > 1e-8:
            R_implied = 1.0 - s_sen * (1.0 - sub_recovery) / s_sub
            R_implied = max(0.0, min(0.95, R_implied))
        else:
            R_implied = senior_recovery

        # Hazard rates from bootstrapped curves
        t = reference_date + relativedelta(years=tenor)
        T = year_fraction(reference_date, t, DayCountConvention.ACT_365_FIXED)
        q_sen = surv_senior.survival(t)
        q_sub = surv_sub.survival(t)
        h_sen = -math.log(max(q_sen, 1e-15)) / max(T, 1e-10)
        h_sub = -math.log(max(q_sub, 1e-15)) / max(T, 1e-10)

        results.append(ImpliedRecoveryResult(
            R_implied, h_sen, h_sub, tenor, "spread_ratio"))

    return results


# ---------------------------------------------------------------------------
# Recovery term structure from CDS term structure
# ---------------------------------------------------------------------------

@dataclass
class RecoveryTermPoint:
    """One point on the recovery term structure."""
    tenor: float
    recovery: float
    hazard: float
    spread: float


def recovery_term_structure(
    cds_spreads: dict[int, float],
    discount_curve: DiscountCurve,
    reference_date: date,
    base_recovery: float = 0.40,
    method: str = "flat",
) -> list[RecoveryTermPoint]:
    """Recovery term structure implied from CDS spreads.

    Method 'flat': same R at all tenors → different h per tenor.
    Method 'slope': R(T) = R_base + slope × (T - 5), calibrate slope
        so that forward hazard rates are smoother.

    For distressed names: short-dated CDS spreads imply higher survival,
    meaning the entity is more likely to survive short-term but less
    likely long-term. The recovery assumption affects this interpretation.

    Returns one point per CDS tenor.
    """
    from pricebook.cds_market import build_cds_curve
    from pricebook.day_count import DayCountConvention, year_fraction

    if method == "flat":
        surv = build_cds_curve(reference_date, cds_spreads, discount_curve,
                                recovery=base_recovery)
        points = []
        for tenor in sorted(cds_spreads.keys()):
            t = reference_date + relativedelta(years=tenor)
            T = year_fraction(reference_date, t, DayCountConvention.ACT_365_FIXED)
            q = surv.survival(t)
            h = -math.log(max(q, 1e-15)) / max(T, 1e-10)
            points.append(RecoveryTermPoint(float(tenor), base_recovery, h,
                                            cds_spreads[tenor]))
        return points

    # Method 'slope': calibrate tenor-dependent recovery
    # Use the approximation h ≈ s / (1-R) per tenor
    # and find R(T) that makes forward hazards smooth
    from pricebook.solvers import brentq

    tenors_sorted = sorted(cds_spreads.keys())
    points = []
    prev_h = None

    for tenor in tenors_sorted:
        s = cds_spreads[tenor]
        # Simple: h = s / (1-R), choose R so that h is close to prev_h
        if prev_h is not None:
            # Target: h ≈ prev_h (smooth forward hazard)
            # h = s / (1-R) → R = 1 - s/h
            R_target = 1.0 - s / prev_h if prev_h > 1e-8 else base_recovery
            R_target = max(0.05, min(0.90, R_target))
        else:
            R_target = base_recovery

        h = s / max(1.0 - R_target, 0.01)
        prev_h = h
        points.append(RecoveryTermPoint(float(tenor), R_target, h, s))

    return points
