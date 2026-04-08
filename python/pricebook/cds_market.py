"""Single-name CDS market making: curve building, pricing ladder, MTM.

Build survival curves from market par spreads, generate bid/ask ladders,
mark-to-market seasoned CDS, and compute roll P&L.

    from pricebook.cds_market import (
        build_cds_curve, pricing_ladder, mark_to_market, roll_pnl,
    )

    curve = build_cds_curve(ref, spreads, discount_curve)
    ladder = pricing_ladder(cds, discount_curve, curve)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.cds import CDS, bootstrap_credit_curve, risky_annuity
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---- Curve building ----

# Standard CDS tenors
STANDARD_TENORS = [1, 3, 5, 7, 10]
# Standard running coupons (post-Big Bang)
STANDARD_COUPONS = [0.01, 0.05]  # 100bp or 500bp


def build_cds_curve(
    reference_date: date,
    par_spreads: dict[int, float],
    discount_curve: DiscountCurve,
    recovery: float = 0.4,
) -> SurvivalCurve:
    """Build a survival curve from market par spreads.

    Args:
        reference_date: valuation date.
        par_spreads: tenor_years -> par_spread (e.g. {1: 0.005, 5: 0.01}).
        discount_curve: risk-free discount curve.
        recovery: recovery rate assumption.

    Returns:
        Bootstrapped SurvivalCurve.
    """
    cds_inputs = []
    for tenor in sorted(par_spreads.keys()):
        mat = date(reference_date.year + tenor, reference_date.month, reference_date.day)
        cds_inputs.append((mat, par_spreads[tenor]))

    return bootstrap_credit_curve(
        reference_date, cds_inputs, discount_curve, recovery=recovery,
    )


def reprice_spreads(
    reference_date: date,
    survival_curve: SurvivalCurve,
    discount_curve: DiscountCurve,
    tenors: list[int] | None = None,
    recovery: float = 0.4,
) -> dict[int, float]:
    """Reprice par spreads from a survival curve (round-trip check)."""
    if tenors is None:
        tenors = STANDARD_TENORS
    result = {}
    for tenor in tenors:
        mat = date(reference_date.year + tenor, reference_date.month, reference_date.day)
        cds = CDS(reference_date, mat, spread=0.01, notional=1.0, recovery=recovery)
        result[tenor] = cds.par_spread(discount_curve, survival_curve)
    return result


# ---- Upfront / running conversion ----

def spread_to_upfront(
    reference_date: date,
    maturity_years: int,
    market_spread: float,
    running_coupon: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    """Convert a par spread to upfront payment.

    upfront = (market_spread - running_coupon) * RPV01
    """
    mat = date(reference_date.year + maturity_years, reference_date.month, reference_date.day)
    rpv01 = risky_annuity(reference_date, mat, discount_curve, survival_curve)
    return (market_spread - running_coupon) * rpv01


def upfront_to_spread(
    reference_date: date,
    maturity_years: int,
    upfront: float,
    running_coupon: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    """Convert an upfront payment back to equivalent par spread.

    spread = running_coupon + upfront / RPV01
    """
    mat = date(reference_date.year + maturity_years, reference_date.month, reference_date.day)
    rpv01 = risky_annuity(reference_date, mat, discount_curve, survival_curve)
    if abs(rpv01) < 1e-12:
        return running_coupon
    return running_coupon + upfront / rpv01


# ---- Pricing ladder ----

@dataclass
class LadderRung:
    """One level of a pricing ladder."""
    spread: float
    pv: float
    upfront: float


def pricing_ladder(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    bumps_bps: list[float] | None = None,
) -> list[LadderRung]:
    """Generate a bid/ask pricing ladder around the current spread.

    Args:
        cds: reference CDS.
        discount_curve: discount curve.
        survival_curve: credit curve.
        bumps_bps: spread bumps in basis points (default: ±1,2,5,10,25).

    Returns:
        List of LadderRung, sorted by spread.
    """
    if bumps_bps is None:
        bumps_bps = [-25, -10, -5, -2, -1, 0, 1, 2, 5, 10, 25]

    base_spread = cds.spread
    ladder = []
    for bump in sorted(bumps_bps):
        spread = base_spread + bump / 10000.0
        if spread <= 0:
            continue
        trial = CDS(
            cds.start, cds.end, spread,
            notional=cds.notional, recovery=cds.recovery,
            frequency=cds.frequency, day_count=cds.day_count,
        )
        pv = trial.pv(discount_curve, survival_curve)
        upfront = pv / cds.notional
        ladder.append(LadderRung(spread=spread, pv=pv, upfront=upfront))

    return ladder


# ---- Mark-to-market ----

def mark_to_market(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> dict[str, float]:
    """Mark-to-market a seasoned CDS.

    Returns PV, par spread, upfront, and RPV01.
    """
    pv = cds.pv(discount_curve, survival_curve)
    par = cds.par_spread(discount_curve, survival_curve)
    upfront = cds.upfront(discount_curve, survival_curve)
    rpv01 = risky_annuity(
        cds.start, cds.end, discount_curve, survival_curve,
        frequency=cds.frequency, day_count=cds.day_count,
    )

    return {
        "pv": pv,
        "par_spread": par,
        "upfront": upfront,
        "rpv01": rpv01,
        "running_spread": cds.spread,
        "spread_to_par": par - cds.spread,
    }


# ---- Roll P&L ----

def roll_pnl(
    old_cds: CDS,
    new_end: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    new_spread: float | None = None,
) -> dict[str, float]:
    """P&L from rolling a CDS to a new on-the-run contract.

    Unwind old CDS at market, enter new CDS at par (or given spread).
    Roll P&L = unwind_pv of old + entry_pv of new.

    Args:
        old_cds: the existing CDS to roll.
        new_end: maturity of the new on-the-run contract.
        discount_curve: discount curve.
        survival_curve: credit curve.
        new_spread: spread for new contract (None = par spread).
    """
    # Unwind old: PV at current market
    old_pv = old_cds.pv(discount_curve, survival_curve)

    # New contract
    if new_spread is None:
        # Enter at par
        temp = CDS(old_cds.start, new_end, spread=0.01,
                    notional=old_cds.notional, recovery=old_cds.recovery)
        new_spread = temp.par_spread(discount_curve, survival_curve)

    new_cds = CDS(
        old_cds.start, new_end, new_spread,
        notional=old_cds.notional, recovery=old_cds.recovery,
    )
    new_pv = new_cds.pv(discount_curve, survival_curve)

    return {
        "old_pv": old_pv,
        "new_pv": new_pv,
        "roll_pnl": old_pv + new_pv,  # unwind old (receive PV) + enter new
        "old_spread": old_cds.spread,
        "new_spread": new_spread,
        "old_end": old_cds.end,
        "new_end": new_end,
    }
