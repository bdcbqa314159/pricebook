"""CLN trading desk: risk metrics, carry, P&L attribution, book management.

Same depth as trs_desk.py / repo_desk.py — production desk infrastructure
for Credit-Linked Notes.

    from pricebook.cln_desk import (
        cln_risk_metrics, CLNRiskMetrics,
        cln_carry_decomposition, CLNCarryDecomposition,
        cln_daily_pnl, CLNDailyPnL,
        CLNBook, CLNBookEntry,
        cln_dashboard, CLNDashboard,
    )

Known out-of-scope:
- Real-time Greeks refresh: pricing library, not trading system.
- Corporate actions on reference entity: edge case, per-issuer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.cln import CreditLinkedNote, CLNResult
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class CLNRiskMetrics:
    """Complete risk metrics for a CLN position."""
    pv: float
    dv01: float                    # parallel rate shift +1bp (centred)
    cs01: float                    # credit spread (hazard) shift +1bp
    recovery_sensitivity: float    # recovery +1%
    jump_to_default_pnl: float     # immediate default: R×N - PV
    notional: float
    leverage: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "dv01": self.dv01, "cs01": self.cs01,
            "recovery_sensitivity": self.recovery_sensitivity,
            "jtd": self.jump_to_default_pnl,
            "notional": self.notional, "leverage": self.leverage,
        }


def cln_risk_metrics(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> CLNRiskMetrics:
    """Compute all risk metrics for a CLN via bump-and-reprice.

    DV01: centred difference on discount curve (O(h²)).
    CS01: +1bp hazard rate shift.
    Recovery: +1% recovery bump.
    JTD: loss on immediate default = R × N - PV.
    """
    base = cln.dirty_price(discount_curve, survival_curve)
    greeks = cln.greeks(discount_curve, survival_curve)

    # Centred DV01 (override one-sided from greeks)
    h = 0.0001
    pv_up = cln.dirty_price(discount_curve.bumped(h), survival_curve)
    pv_dn = cln.dirty_price(discount_curve.bumped(-h), survival_curve)
    dv01 = (pv_up - pv_dn) / 2

    # JTD: on immediate default, investor receives recovery × notional
    # but loses their investment (current PV)
    jtd = cln.recovery * cln.notional - base

    return CLNRiskMetrics(
        pv=base,
        dv01=dv01,
        cs01=greeks["cs01"],
        recovery_sensitivity=greeks["recovery_sensitivity"],
        jump_to_default_pnl=jtd,
        notional=cln.notional,
        leverage=cln.leverage,
    )
