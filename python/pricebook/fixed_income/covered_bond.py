"""Covered bonds / Pfandbriefe (C5).

Dual-recourse bonds: investor has claim on both the issuer AND a
segregated cover pool. Lower spread than senior unsecured.

    from pricebook.fixed_income.covered_bond import (
        CoveredBond, price_covered_bond,
    )

References:
    ECBC (2023). European Covered Bond Fact Book.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.discount_curve import DiscountCurve


@dataclass
class CoverPool:
    """Cover pool characteristics."""
    total_assets: float          # cover pool nominal
    overcollateralisation: float # OC ratio (e.g. 1.15 = 15% OC)
    avg_ltv: float               # average loan-to-value
    pool_type: str               # "mortgage", "public_sector", "ship", "aircraft"
    country: str = ""

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CoveredBondResult:
    """Result of covered bond pricing."""
    price: float
    spread_bp: float             # spread over risk-free
    issuer_spread_bp: float      # senior unsecured spread
    cover_pool_benefit_bp: float # spread reduction from dual recourse
    oc_cushion_pct: float        # overcollateralisation buffer

    def to_dict(self) -> dict:
        return vars(self)


def price_covered_bond(
    coupon: float,
    maturity_years: float,
    discount_curve: DiscountCurve,
    issuer_spread_bp: float,
    cover_pool: CoverPool,
    face: float = 100.0,
    freq: int = 1,
) -> CoveredBondResult:
    """Price a covered bond.

    Covered bond spread = f(issuer_spread, OC, LTV, pool_type).
    Typically 50-80% tighter than senior unsecured due to dual recourse.

    The spread benefit depends on:
    1. Overcollateralisation (higher OC → lower spread)
    2. Cover pool quality (lower LTV → lower spread)
    3. Pool type (mortgage > public sector in some jurisdictions)
    """
    ref = discount_curve.reference_date

    # Spread benefit from dual recourse
    oc_benefit = min(cover_pool.overcollateralisation - 1.0, 0.30) * 50  # up to 15bp from OC
    ltv_benefit = max(0.80 - cover_pool.avg_ltv, 0) * 30  # lower LTV → lower spread
    pool_benefit = {"mortgage": 10, "public_sector": 15, "ship": 0, "aircraft": 0}.get(
        cover_pool.pool_type, 5)

    total_benefit = oc_benefit + ltv_benefit + pool_benefit
    # Covered bond spread = issuer spread × (1 - benefit_factor)
    # Typical: 30-60% of senior unsecured
    benefit_factor = min(total_benefit / max(issuer_spread_bp, 1), 0.70)
    covered_spread_bp = issuer_spread_bp * (1.0 - benefit_factor)
    covered_spread = covered_spread_bp / 10_000

    # Price with the covered bond spread
    cpn = coupon / freq * face
    pv = 0.0
    for i in range(1, int(maturity_years * freq) + 1):
        t = i / freq
        d = date.fromordinal(ref.toordinal() + int(t * 365))
        df = discount_curve.df(d) * math.exp(-covered_spread * t)
        pv += cpn * df
    d_mat = date.fromordinal(ref.toordinal() + int(maturity_years * 365))
    pv += face * discount_curve.df(d_mat) * math.exp(-covered_spread * maturity_years)

    return CoveredBondResult(
        price=pv,
        spread_bp=covered_spread_bp,
        issuer_spread_bp=issuer_spread_bp,
        cover_pool_benefit_bp=total_benefit,
        oc_cushion_pct=(cover_pool.overcollateralisation - 1.0) * 100,
    )
