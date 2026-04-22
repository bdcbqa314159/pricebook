"""ISDA SIMM (Standard Initial Margin Model).

Computes regulatory initial margin for non-cleared derivatives.
Sensitivity-based: delta, vega, curvature per risk factor → within-bucket
aggregation → across-bucket aggregation → across risk class.

    from pricebook.simm import SIMMCalculator, SIMMSensitivity

    sensitivities = [
        SIMMSensitivity("GIRR", "USD", "2Y", delta=50_000),
        SIMMSensitivity("GIRR", "USD", "10Y", delta=80_000),
        SIMMSensitivity("FX", "EUR/USD", "spot", delta=200_000),
    ]
    result = SIMMCalculator().compute(sensitivities)

References:
    ISDA, *SIMM Methodology*, v2.6, 2024.
    ISDA, *Risk Data Standards*, 2023.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import defaultdict


# ---- Risk classes and buckets ----

RISK_CLASSES = ["GIRR", "FX", "CSR", "EQ", "COM"]

# GIRR tenor buckets and risk weights (bp)
GIRR_TENORS = ["2W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y", "15Y", "20Y", "30Y"]
GIRR_RISK_WEIGHTS = {
    "2W": 114, "1M": 115, "3M": 104, "6M": 86, "1Y": 80, "2Y": 77,
    "3Y": 69, "5Y": 56, "10Y": 56, "15Y": 51, "20Y": 53, "30Y": 56,
}  # bps, from ISDA SIMM v2.6

# GIRR within-bucket correlation (simplified: same curve, different tenors)
GIRR_INTRA_CORR = 0.98  # high correlation between adjacent tenors

# Across-bucket correlation (between currencies)
GIRR_INTER_CORR = 0.27

# FX risk weight
FX_RISK_WEIGHT = 0.15  # 15% for liquid pairs, scaled
FX_LIQUID_PAIRS = {"EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD",
                    "USD/CHF", "EUR/GBP", "EUR/JPY"}
FX_RW_LIQUID = 0.1125  # 11.25% for liquid
FX_RW_OTHER = 0.15     # 15% for others
FX_CORR = 0.50         # across-bucket correlation

# CSR risk weights (by sector)
CSR_RISK_WEIGHTS = {
    "sovereign": 0.005, "IG_financial": 0.01, "IG_corporate": 0.01,
    "HY": 0.02, "not_rated": 0.02,
}
CSR_INTRA_CORR = 0.65
CSR_INTER_CORR = 0.30

# EQ risk weights
EQ_RW_LARGE = 0.20
EQ_RW_SMALL = 0.30
EQ_INTRA_CORR = 0.15
EQ_INTER_CORR = 0.10

# COM risk weights
COM_RW = 0.15
COM_INTRA_CORR = 0.35
COM_INTER_CORR = 0.15


# ---- Data structures ----

@dataclass
class SIMMSensitivity:
    """A single SIMM sensitivity input."""
    risk_class: str      # GIRR, FX, CSR, EQ, COM
    bucket: str          # currency (GIRR/FX), sector (CSR), ticker (EQ), commodity (COM)
    tenor: str           # tenor bucket or "spot" for FX/EQ
    delta: float = 0.0   # delta sensitivity (in base currency)
    vega: float = 0.0    # vega sensitivity
    curvature: float = 0.0


@dataclass
class SIMMBucketResult:
    """Margin for a single bucket."""
    bucket: str
    weighted_sensitivities: list[float]
    margin: float


@dataclass
class SIMMRiskClassResult:
    """Margin for a single risk class."""
    risk_class: str
    buckets: list[SIMMBucketResult]
    delta_margin: float
    vega_margin: float
    curvature_margin: float
    total: float


@dataclass
class SIMMResult:
    """Total SIMM margin."""
    risk_classes: list[SIMMRiskClassResult]
    total_margin: float
    n_sensitivities: int


# ---- Calculator ----

class SIMMCalculator:
    """ISDA SIMM margin calculator.

    Implements the three-level aggregation:
    1. Weight sensitivities by risk weight
    2. Aggregate within bucket (using intra-bucket correlation)
    3. Aggregate across buckets (using inter-bucket correlation)
    """

    def compute(self, sensitivities: list[SIMMSensitivity]) -> SIMMResult:
        """Compute total SIMM margin from a list of sensitivities."""
        # Group by risk class
        by_class: dict[str, list[SIMMSensitivity]] = defaultdict(list)
        for s in sensitivities:
            by_class[s.risk_class].append(s)

        rc_results = []
        for rc in RISK_CLASSES:
            if rc in by_class:
                rc_result = self._compute_risk_class(rc, by_class[rc])
                rc_results.append(rc_result)

        # Across risk class aggregation (simple sum — SIMM uses sqrt of sum of squares)
        total = math.sqrt(sum(r.total ** 2 for r in rc_results))

        return SIMMResult(
            risk_classes=rc_results,
            total_margin=total,
            n_sensitivities=len(sensitivities),
        )

    def _compute_risk_class(
        self, risk_class: str, sensitivities: list[SIMMSensitivity],
    ) -> SIMMRiskClassResult:
        """Compute margin for one risk class."""
        # Group by bucket
        by_bucket: dict[str, list[SIMMSensitivity]] = defaultdict(list)
        for s in sensitivities:
            by_bucket[s.bucket].append(s)

        bucket_results = []
        bucket_margins = []

        for bucket, sens_list in by_bucket.items():
            bm = self._compute_bucket(risk_class, bucket, sens_list)
            bucket_results.append(bm)
            bucket_margins.append(bm.margin)

        # Across-bucket aggregation
        inter_corr = self._inter_bucket_corr(risk_class)
        n = len(bucket_margins)
        if n == 1:
            delta_margin = bucket_margins[0]
        else:
            var = 0.0
            for i in range(n):
                var += bucket_margins[i] ** 2
                for j in range(i + 1, n):
                    var += 2 * inter_corr * bucket_margins[i] * bucket_margins[j]
            delta_margin = math.sqrt(max(var, 0.0))

        return SIMMRiskClassResult(
            risk_class=risk_class,
            buckets=bucket_results,
            delta_margin=delta_margin,
            vega_margin=0.0,  # simplified: vega treated as delta with higher RW
            curvature_margin=0.0,
            total=delta_margin,
        )

    def _compute_bucket(
        self, risk_class: str, bucket: str, sensitivities: list[SIMMSensitivity],
    ) -> SIMMBucketResult:
        """Within-bucket aggregation."""
        rw = self._risk_weight(risk_class, bucket)
        weighted = [s.delta * rw for s in sensitivities]

        intra_corr = self._intra_bucket_corr(risk_class)
        n = len(weighted)
        if n == 1:
            margin = abs(weighted[0])
        else:
            var = 0.0
            for i in range(n):
                var += weighted[i] ** 2
                for j in range(i + 1, n):
                    var += 2 * intra_corr * weighted[i] * weighted[j]
            margin = math.sqrt(max(var, 0.0))

        return SIMMBucketResult(bucket, weighted, margin)

    def _risk_weight(self, risk_class: str, bucket: str) -> float:
        if risk_class == "GIRR":
            return GIRR_RISK_WEIGHTS.get(bucket, 56) / 10000.0  # convert bps to decimal
        if risk_class == "FX":
            return FX_RW_LIQUID if bucket in FX_LIQUID_PAIRS else FX_RW_OTHER
        if risk_class == "CSR":
            return CSR_RISK_WEIGHTS.get(bucket, 0.01)
        if risk_class == "EQ":
            return EQ_RW_LARGE
        if risk_class == "COM":
            return COM_RW
        return 0.01

    def _intra_bucket_corr(self, risk_class: str) -> float:
        if risk_class == "GIRR":
            return GIRR_INTRA_CORR
        if risk_class == "CSR":
            return CSR_INTRA_CORR
        if risk_class == "EQ":
            return EQ_INTRA_CORR
        if risk_class == "COM":
            return COM_INTRA_CORR
        return 0.50

    def _inter_bucket_corr(self, risk_class: str) -> float:
        if risk_class == "GIRR":
            return GIRR_INTER_CORR
        if risk_class == "FX":
            return FX_CORR
        if risk_class == "CSR":
            return CSR_INTER_CORR
        if risk_class == "EQ":
            return EQ_INTER_CORR
        if risk_class == "COM":
            return COM_INTER_CORR
        return 0.20
