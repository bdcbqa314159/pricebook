"""Risk participation: unfunded and sub-participation credit risk transfer.

A risk participation is a bilateral contract where the originating bank
retains the loan on its balance sheet but transfers credit risk to a
participant. Unlike a loan sale/assignment, the borrower has no
relationship with the participant.

Types:
    - Funded: participant posts cash collateral (like a deposit), earns spread
    - Unfunded: CDS-like — participant covers losses, pays nothing upfront
    - Sub-participation: intermediary re-sells portion to a third party

Capital relief: Under Basel IRB, the originator reduces K by
    participation_rate × (1 - regulatory_haircut).

    from pricebook.credit.risk_participation import (
        RiskParticipation, SubParticipation,
        risk_participation_capital_relief,
    )

References:
    LSTA (2022). Handbook of Loan Syndications and Trading, Ch. 16-17.
    Basel Committee (2023). CRE22 — Credit risk mitigation.
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from scipy.stats import norm as _norm

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.cds import premium_leg_pv, protection_leg_pv
from pricebook.core.schedule import Frequency
from pricebook.core.serialisable import serialisable as _serialisable


# ---------------------------------------------------------------------------
# Risk Participation
# ---------------------------------------------------------------------------

@dataclass
class RiskParticipationResult:
    """Pricing result for a risk participation."""
    pv: float                   # net PV to participant (fee - expected loss)
    fee_pv: float               # PV of participation fee (income)
    protection_pv: float        # PV of protection obligation (cost)
    participation_notional: float
    par_spread: float           # spread at which PV = 0
    capital_relief: float       # originator's capital saving

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "fee_pv": self.fee_pv,
            "protection_pv": self.protection_pv,
            "participation_notional": self.participation_notional,
            "par_spread": self.par_spread,
            "capital_relief": self.capital_relief,
        }


class RiskParticipation:
    """Bilateral unfunded risk participation.

    The originator retains the loan. The participant assumes credit risk
    on a portion (participation_rate) of the notional in exchange for a
    running fee (spread). On default, participant pays (1-R) × participation_notional.

    This is economically equivalent to selling CDS protection on the
    underlying credit, sized to participation_rate × loan_notional.

    Args:
        start: effective date of participation.
        end: maturity (typically matches loan maturity).
        loan_notional: total loan notional.
        participation_rate: fraction of notional at risk (0 to 1).
        spread: annual fee paid to participant (decimal, e.g. 0.02 = 200bp).
        recovery: expected recovery rate on default.
        upfront_fee: one-time fee to participant (decimal of participation notional).
        settlement: "cash" or "physical" (determines payoff on default).
    """

    def __init__(
        self,
        start: date,
        end: date,
        loan_notional: float,
        participation_rate: float,
        spread: float,
        recovery: float = 0.40,
        upfront_fee: float = 0.0,
        settlement: str = "cash",
    ):
        if not 0 < participation_rate <= 1:
            raise ValueError(f"participation_rate must be in (0, 1], got {participation_rate}")
        if loan_notional <= 0:
            raise ValueError(f"loan_notional must be positive, got {loan_notional}")
        if not 0 <= recovery <= 1:
            raise ValueError(f"recovery must be in [0, 1], got {recovery}")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        self.start = start
        self.end = end
        self.loan_notional = loan_notional
        self.participation_rate = participation_rate
        self.spread = spread
        self.recovery = recovery
        self.upfront_fee = upfront_fee
        self.settlement = settlement
        self.notional = loan_notional * participation_rate

    def price(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> RiskParticipationResult:
        """Price the risk participation from the participant's perspective.

        Participant receives: running fee (premium leg)
        Participant pays: (1-R) × notional on default (protection leg)

        PV > 0 means the fee overcompensates for expected loss.
        """
        # Fee leg: participant receives spread × notional × risky annuity
        fee_pv = premium_leg_pv(
            self.start, self.end, self.spread,
            discount_curve, survival_curve,
            notional=self.notional,
            frequency=Frequency.QUARTERLY,
        )

        # Upfront fee
        upfront_pv = self.upfront_fee * self.notional * discount_curve.df(self.start)

        # Protection leg: participant pays (1-R) × notional on default
        prot_pv = protection_leg_pv(
            self.start, self.end,
            discount_curve, survival_curve,
            recovery=self.recovery,
            notional=self.notional,
        )

        # Net PV to participant = fees received - protection cost
        pv = fee_pv + upfront_pv - prot_pv

        # Par spread: spread at which PV = 0 (like CDS par spread)
        rpv01 = premium_leg_pv(
            self.start, self.end, 1.0,
            discount_curve, survival_curve,
            notional=self.notional,
        )
        par_spread = prot_pv / rpv01 if abs(rpv01) > 1e-15 else 0.0

        # Capital relief for originator
        capital_relief = risk_participation_capital_relief(
            self.loan_notional, self.participation_rate,
        )

        return RiskParticipationResult(
            pv=pv,
            fee_pv=fee_pv + upfront_pv,
            protection_pv=prot_pv,
            participation_notional=self.notional,
            par_spread=par_spread,
            capital_relief=capital_relief,
        )

    def cs01(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        shift: float = 0.0001,
    ) -> float:
        """Credit spread sensitivity: PV change per 1bp hazard shift (centred)."""
        pv_up = self.price(discount_curve, survival_curve.bumped(shift)).pv
        pv_dn = self.price(discount_curve, survival_curve.bumped(-shift)).pv
        return (pv_up - pv_dn) / 2

    def jtd(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Jump-to-default: P&L if default happens immediately."""
        pv = self.price(discount_curve, survival_curve).pv
        # On default, participant pays (1-R) × notional, loses PV
        return -(1 - self.recovery) * self.notional - pv


# ---------------------------------------------------------------------------
# Sub-participation
# ---------------------------------------------------------------------------

class SubParticipation:
    """Sub-participation: intermediary re-sells portion of their participation.

    Party A participates in a loan (original participation).
    Party A then sub-participates a fraction to Party B.

    Party B's economics:
        - Receives: sub_spread × sub_notional × risky annuity
        - Pays: (1-R) × sub_notional on default

    Party A's residual:
        - Original spread on retained portion
        - Spread differential (original_spread - sub_spread) on sub'd portion

    Args:
        original: the underlying RiskParticipation.
        sub_rate: fraction of original participation to sub-participate (0 to 1).
        sub_spread: spread passed to sub-participant.
    """

    def __init__(
        self,
        original: RiskParticipation,
        sub_rate: float,
        sub_spread: float,
    ):
        if not 0 < sub_rate <= 1:
            raise ValueError(f"sub_rate must be in (0, 1], got {sub_rate}")
        self.original = original
        self.sub_rate = sub_rate
        self.sub_spread = sub_spread

        # Sub-participant's position
        self.sub_participation = RiskParticipation(
            start=original.start,
            end=original.end,
            loan_notional=original.notional,  # sub of the participation notional
            participation_rate=sub_rate,
            spread=sub_spread,
            recovery=original.recovery,
            settlement=original.settlement,
        )

        # Intermediary's retained notional
        self.retained_notional = original.notional * (1 - sub_rate)
        self.sub_notional = original.notional * sub_rate

    def price_sub(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> RiskParticipationResult:
        """Price from sub-participant's perspective."""
        return self.sub_participation.price(discount_curve, survival_curve)

    def intermediary_pnl(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> dict:
        """P&L for the intermediary (Party A).

        Intermediary earns:
        - Full spread on retained portion
        - Spread differential on sub'd portion (if original_spread > sub_spread)
        """
        # PV of retained portion at original spread
        retained = RiskParticipation(
            self.original.start, self.original.end,
            self.original.notional, 1 - self.sub_rate,
            self.original.spread, self.original.recovery,
        )
        retained_result = retained.price(discount_curve, survival_curve)

        # PV of spread differential on sub'd portion
        spread_diff = self.original.spread - self.sub_spread
        diff_pv = premium_leg_pv(
            self.original.start, self.original.end, spread_diff,
            discount_curve, survival_curve,
            notional=self.sub_notional,
        )

        return {
            "retained_pv": retained_result.pv,
            "spread_diff_pv": diff_pv,
            "total_pv": retained_result.pv + diff_pv,
            "retained_notional": self.retained_notional,
            "sub_notional": self.sub_notional,
            "spread_pickup_bp": spread_diff * 10_000,
        }


# ---------------------------------------------------------------------------
# Capital relief
# ---------------------------------------------------------------------------

def risk_participation_capital_relief(
    loan_notional: float,
    participation_rate: float,
    pd: float = 0.01,
    lgd: float = 0.45,
    maturity_years: float = 5.0,
    regulatory_haircut: float = 0.0,
) -> float:
    """Basel IRB capital relief from risk participation.

    The originator's capital requirement is reduced proportionally
    to the participation rate, subject to a regulatory haircut
    for maturity/quality mismatches.

    K = LGD × [Φ((Φ⁻¹(PD) + √ρ × Φ⁻¹(0.999)) / √(1-ρ)) - PD]
        × maturity_adjustment

    Capital relief = K × loan_notional × participation_rate × (1 - haircut)

    Args:
        loan_notional: total loan notional.
        participation_rate: fraction covered by participation.
        pd: probability of default (annual).
        lgd: loss given default.
        maturity_years: effective maturity.
        regulatory_haircut: haircut for maturity/quality mismatch (0 to 1).
    """
    # Asset correlation (Basel II formula)
    rho = 0.12 * (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)) + \
          0.24 * (1 - (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)))

    # Conditional PD at 99.9% confidence
    cpd = _norm.cdf(
        (_norm.ppf(pd) + math.sqrt(rho) * _norm.ppf(0.999)) / math.sqrt(1 - rho)
    )

    # Maturity adjustment
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    ma = (1 + (maturity_years - 2.5) * b) / (1 - 1.5 * b)

    # Capital requirement per unit
    k = lgd * (cpd - pd) * ma

    # Capital relief
    relief = k * loan_notional * participation_rate * (1 - regulatory_haircut)
    return relief

_serialisable("risk_participation", ['start', 'end', 'loan_notional', 'participation_rate', 'spread', 'recovery', 'upfront_fee', 'settlement'])(RiskParticipation)
