"""CDS trading strategies: curve trades, basis, recovery lock.

Trade construction and analytics for credit desk strategies.

    from pricebook.cds_strategies import flatten, cds_bond_basis, recovery_lock_pv

    trade = flatten(ref, disc, sc, short_tenor=5, long_tenor=10)
    basis = cds_bond_basis(cds_spread=0.005, asw_spread=0.006)

References:
    O'Kane, D. (2008). Modelling Single-name and Multi-name Credit
    Derivatives. Wiley, Ch. 7 — CDS Strategies.
    Choudhry, M. (2006). The Credit Default Swap Basis. Bloomberg Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pricebook.cds import CDS
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import _register


# ---- Curve trades ----

@dataclass
class CurveTradePnL:
    """P&L decomposition for a CDS curve trade."""
    total_pv: float
    short_pv: float
    long_pv: float
    carry: float
    parallel_cs01: float
    short_cs01: float
    long_cs01: float

    def to_dict(self) -> dict:
        return {"total_pv": self.total_pv, "short_pv": self.short_pv,
                "long_pv": self.long_pv, "carry": self.carry,
                "parallel_cs01": self.parallel_cs01,
                "short_cs01": self.short_cs01, "long_cs01": self.long_cs01}


class CDSCurveTrade:
    """CDS curve trade: buy protection at one tenor, sell at another.

    Flattener: buy short protection, sell long protection.
    Steepener: sell short protection, buy long protection.

    DV01-neutral: notionals sized so parallel CS01 ≈ 0.

    Args:
        short_cds: the shorter-maturity CDS (buy protection).
        long_cds: the longer-maturity CDS (sell protection).
    """

    _SERIAL_TYPE = "cds_curve_trade"

    def __init__(
        self,
        short_cds: CDS,
        long_cds: CDS,
    ):
        self.short_cds = short_cds
        self.long_cds = long_cds

    def pnl(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> CurveTradePnL:
        """Compute P&L and risk decomposition."""
        short_pv = self.short_cds.pv(discount_curve, survival_curve)
        long_pv = self.long_cds.pv(discount_curve, survival_curve)
        total = short_pv - long_pv  # buy short protection, sell long

        short_cs01 = self.short_cds.cs01(discount_curve, survival_curve)
        long_cs01 = self.long_cds.cs01(discount_curve, survival_curve)
        parallel = short_cs01 - long_cs01

        carry = (self.short_cds.carry(discount_curve, survival_curve, 30)
                 - self.long_cds.carry(discount_curve, survival_curve, 30))

        return CurveTradePnL(
            total_pv=total, short_pv=short_pv, long_pv=long_pv,
            carry=carry, parallel_cs01=parallel,
            short_cs01=short_cs01, long_cs01=long_cs01,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "short_cds": self.short_cds.to_dict(),
            "long_cds": self.long_cds.to_dict(),
        }}

    @classmethod
    def from_dict(cls, d: dict) -> CDSCurveTrade:
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(short_cds=_fd(p["short_cds"]), long_cds=_fd(p["long_cds"]))


_register(CDSCurveTrade)


def flatten(
    reference_date: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    short_tenor: int = 5,
    long_tenor: int = 10,
    notional: float = 10_000_000.0,
    dv01_neutral: bool = True,
) -> CDSCurveTrade:
    """Construct a flattener: buy short protection, sell long protection.

    If dv01_neutral, sizes the long leg so parallel CS01 ≈ 0.
    """
    short_end = reference_date + timedelta(days=365 * short_tenor)
    long_end = reference_date + timedelta(days=365 * long_tenor)

    short_spread = -math.log(survival_curve.survival(short_end)) / max(
        year_fraction(reference_date, short_end, DayCountConvention.ACT_365_FIXED), 1e-10
    ) * 0.6
    long_spread = -math.log(survival_curve.survival(long_end)) / max(
        year_fraction(reference_date, long_end, DayCountConvention.ACT_365_FIXED), 1e-10
    ) * 0.6

    short_cds = CDS(reference_date, short_end, spread=short_spread, notional=notional)
    long_cds = CDS(reference_date, long_end, spread=long_spread, notional=notional)

    if dv01_neutral:
        short_cs01 = abs(short_cds.cs01(discount_curve, survival_curve))
        long_cs01 = abs(long_cds.cs01(discount_curve, survival_curve))
        if long_cs01 > 1e-10:
            ratio = short_cs01 / long_cs01
            long_cds = CDS(reference_date, long_end, spread=long_spread,
                           notional=notional * ratio)

    return CDSCurveTrade(short_cds, long_cds)


def steepen(
    reference_date: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    short_tenor: int = 5,
    long_tenor: int = 10,
    notional: float = 10_000_000.0,
    dv01_neutral: bool = True,
) -> CDSCurveTrade:
    """Construct a steepener: sell short protection, buy long protection."""
    trade = flatten(reference_date, discount_curve, survival_curve,
                    short_tenor, long_tenor, notional, dv01_neutral)
    # Swap directions by swapping short and long
    return CDSCurveTrade(trade.long_cds, trade.short_cds)


# ---- CDS-bond basis ----

@dataclass
class BasisTradeResult:
    """CDS-bond basis trade analytics."""
    basis_bp: float             # CDS spread - asset swap spread (in bp)
    is_negative: bool           # negative basis = potential arbitrage
    pv_if_no_default: float     # PV of the package assuming no default
    breakeven_funding: float    # funding rate at which basis trade breaks even

    def to_dict(self) -> dict:
        return {"basis_bp": self.basis_bp, "is_negative": self.is_negative,
                "pv_if_no_default": self.pv_if_no_default,
                "breakeven_funding": self.breakeven_funding}


def cds_bond_basis(cds_spread: float, asw_spread: float) -> float:
    """CDS-bond basis in basis points.

    basis = CDS_spread - ASW_spread

    Positive basis: CDS is wider (protection costs more than bond implies).
    Negative basis: CDS is tighter (potential arb: buy bond + buy CDS).
    """
    return (cds_spread - asw_spread) * 10_000


def basis_trade(
    cds_spread: float,
    asw_spread: float,
    notional: float = 10_000_000.0,
    funding_rate: float = 0.03,
    maturity_years: float = 5.0,
) -> BasisTradeResult:
    """Analyse a CDS-bond negative basis trade.

    Package: buy bond (earn ASW spread), buy CDS protection (pay CDS spread).
    Net carry = ASW spread - CDS spread - funding cost.
    """
    basis = cds_spread - asw_spread
    basis_bp = basis * 10_000
    is_negative = basis < 0

    # PV if no default: carry over maturity
    net_carry = (asw_spread - cds_spread) * notional * maturity_years
    pv_no_default = net_carry

    # Breakeven funding: at what funding rate does the trade become unprofitable?
    # Net = asw - cds - funding_spread
    breakeven = asw_spread - cds_spread

    return BasisTradeResult(
        basis_bp=basis_bp, is_negative=is_negative,
        pv_if_no_default=pv_no_default,
        breakeven_funding=breakeven,
    )


# ---- Recovery lock ----

def recovery_lock_pv(
    cds: CDS,
    lock_recovery: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """PV of a recovery lock at a fixed recovery rate.

    Isolates the default probability from recovery uncertainty.
    PV = (actual_recovery - lock_recovery) × expected_default_pv

    If lock_recovery > market recovery: seller profits.
    """
    from pricebook.cds import protection_leg_pv
    # Protection at market recovery
    prot_market = protection_leg_pv(
        cds.start, cds.end, discount_curve, survival_curve,
        cds.recovery, cds.notional,
    )
    # Protection at lock recovery
    prot_lock = protection_leg_pv(
        cds.start, cds.end, discount_curve, survival_curve,
        lock_recovery, cds.notional,
    )
    return prot_market - prot_lock


def digital_cds_spread(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """Implied default probability from CDS spread.

    digital_spread = par_spread / (1 - recovery)

    This is the spread of a digital CDS that pays 1 on default.
    """
    par = cds.par_spread(discount_curve, survival_curve)
    return par / (1 - cds.recovery) if cds.recovery < 1 else 0.0
