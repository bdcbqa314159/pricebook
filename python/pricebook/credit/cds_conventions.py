"""ISDA CDS conventions: Big Bang, standard coupons, IMM dates, indices.

* :func:`standard_cds_dates` — IMM-based premium payment dates.
* :func:`upfront_from_par_spread` — convert par spread to upfront at standard coupon.
* :func:`par_spread_from_upfront` — convert upfront to par spread.
* :class:`CDSIndexSpec` — CDX/iTraxx index specification.
* :func:`cds_index_roll_date` — next roll date for a given index.

References:
    ISDA, *2014 ISDA Credit Derivatives Definitions*.
    Markit, *CDS Big Bang Protocol*, 2009.
    O'Kane, *Modelling Single-Name and Multi-Name Credit Derivatives*, Wiley, 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


# ---- IMM dates for CDS ----

_IMM_MONTHS = [3, 6, 9, 12]
_IMM_DAY = 20


def next_imm_date(ref: date) -> date:
    """Next IMM date (20th of Mar, Jun, Sep, Dec) on or after ref."""
    for m in _IMM_MONTHS:
        candidate = date(ref.year, m, _IMM_DAY)
        if candidate >= ref:
            return candidate
    return date(ref.year + 1, 3, _IMM_DAY)


def standard_cds_dates(
    effective_date: date,
    maturity_years: int = 5,
    frequency_months: int = 3,
) -> list[date]:
    """Generate standard CDS premium payment dates (IMM-based).

    CDS effective dates snap to the previous IMM date.
    Premium payments every 3 months on IMM dates.

    Args:
        effective_date: trade date (snaps to previous IMM).
        maturity_years: standard tenors: 1, 3, 5, 7, 10.
        frequency_months: payment frequency (3 = quarterly, standard).
    """
    # Snap to previous IMM date
    prev_imm = effective_date
    for m in reversed(_IMM_MONTHS):
        candidate = date(effective_date.year, m, _IMM_DAY)
        if candidate <= effective_date:
            prev_imm = candidate
            break
    else:
        prev_imm = date(effective_date.year - 1, 12, _IMM_DAY)

    # Generate payment dates
    dates = []
    current = prev_imm
    end = date(prev_imm.year + maturity_years, prev_imm.month, _IMM_DAY)

    while current < end:
        month = current.month + frequency_months
        year = current.year
        while month > 12:
            month -= 12
            year += 1
        current = date(year, month, _IMM_DAY)
        dates.append(current)

    return dates


# ---- Standard coupons & upfront conversion ----

STANDARD_COUPONS_BPS = {
    "IG": 100,     # Investment grade: 100bp running
    "HY": 500,     # High yield: 500bp running
}

STANDARD_RECOVERY = {
    "IG": 0.40,    # 40% recovery for IG
    "HY": 0.25,    # 25% recovery for HY (ISDA convention for CDX.HY)
    "SENIOR": 0.40,
    "SUBORDINATED": 0.25,
}


@dataclass
class UpfrontResult:
    """CDS upfront conversion result."""
    upfront_pct: float          # upfront as % of notional (positive = protection buyer pays)
    par_spread_bps: float
    standard_coupon_bps: float
    risky_annuity: float        # PV01 of the CDS


def upfront_from_par_spread(
    par_spread_bps: float,
    standard_coupon_bps: float = 100,
    risky_annuity: float = 4.0,
) -> UpfrontResult:
    """Convert par spread to upfront at standard coupon.

    Upfront ≈ (par_spread − standard_coupon) × risky_annuity / 10000.

    The risky annuity (RPV01) is the PV of 1bp running over the CDS life.
    For a 5Y CDS with flat hazard, RPV01 ≈ 4.0-4.5.

    Args:
        par_spread_bps: market par CDS spread (e.g., 150bp).
        standard_coupon_bps: running coupon (100bp for IG, 500bp for HY).
        risky_annuity: PV01 of the CDS (from survival curve bootstrap).
    """
    upfront = (par_spread_bps - standard_coupon_bps) * risky_annuity / 10000
    return UpfrontResult(float(upfront), par_spread_bps, standard_coupon_bps, risky_annuity)


def par_spread_from_upfront(
    upfront_pct: float,
    standard_coupon_bps: float = 100,
    risky_annuity: float = 4.0,
) -> float:
    """Convert upfront to par spread.

    par_spread ≈ standard_coupon + upfront × 10000 / risky_annuity.
    """
    return standard_coupon_bps + upfront_pct * 10000 / max(risky_annuity, 1e-10)


# ---- CDS Index specs ----

@dataclass
class CDSIndexSpec:
    """Credit index specification."""
    name: str
    region: str                 # "NA", "EU", "ASIA"
    grade: str                  # "IG", "HY", "XOVER"
    n_names: int
    standard_coupon_bps: int
    standard_recovery: float
    roll_months: list[int]      # months when index rolls (typically Mar, Sep)
    tenor_years: int            # standard tenor (5Y for most)


_INDEX_SPECS = {
    "CDX.NA.IG": CDSIndexSpec("CDX.NA.IG", "NA", "IG", 125, 100, 0.40, [3, 9], 5),
    "CDX.NA.HY": CDSIndexSpec("CDX.NA.HY", "NA", "HY", 100, 500, 0.25, [3, 9], 5),
    "ITRAXX.EUR.IG": CDSIndexSpec("iTraxx Europe", "EU", "IG", 125, 100, 0.40, [3, 9], 5),
    "ITRAXX.EUR.XOVER": CDSIndexSpec("iTraxx Crossover", "EU", "XOVER", 75, 500, 0.40, [3, 9], 5),
    "ITRAXX.EUR.SENIOR": CDSIndexSpec("iTraxx Senior Fin", "EU", "IG", 30, 100, 0.40, [3, 9], 5),
}


def get_index_spec(name: str) -> CDSIndexSpec:
    """Look up CDS index specification by name."""
    key = name.upper().replace(" ", ".")
    if key in _INDEX_SPECS:
        return _INDEX_SPECS[key]
    # Fuzzy match
    for k, v in _INDEX_SPECS.items():
        if name.upper() in k:
            return v
    raise ValueError(f"Unknown index: {name}. Available: {list(_INDEX_SPECS.keys())}")


def cds_index_roll_date(index_name: str, ref: date) -> date:
    """Next roll date for a CDS index.

    CDX and iTraxx roll on the 20th of March and September.
    """
    spec = get_index_spec(index_name)
    for m in spec.roll_months:
        candidate = date(ref.year, m, _IMM_DAY)
        if candidate > ref:
            return candidate
    return date(ref.year + 1, spec.roll_months[0], _IMM_DAY)


# ---- Settlement conventions ----

@dataclass
class CDSSettlementConvention:
    """CDS settlement convention."""
    method: str                 # "auction", "physical", "cash"
    auction_timing_days: int    # days after credit event for auction
    accrued_on_default: bool    # whether accrued premium is paid on default


CDS_SETTLEMENT = CDSSettlementConvention(
    method="auction",
    auction_timing_days=30,
    accrued_on_default=True,    # ISDA 2009 Big Bang: accrued paid on default
)
