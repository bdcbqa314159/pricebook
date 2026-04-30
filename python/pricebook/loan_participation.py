"""Loan participation and partial funded structures.

L2: LoanParticipation — funded credit risk transfer on a loan.
L3: PartialFundedParticipation — split into funded + unfunded (CDS-like) legs.

    from pricebook.loan_participation import LoanParticipation, PartialFundedParticipation

    # Full participation
    part = LoanParticipation(loan, participation_rate=0.10, recovery=0.6)
    pv = part.pv(disc, proj, surv)

    # Partial funded
    pfp = PartialFundedParticipation(loan, funded_rate=0.60, unfunded_spread=0.02)
    total = pfp.total_pv(disc, proj, surv)

References:
    LSTA (2022). The Handbook of Loan Syndications and Trading, Ch. 15-16.
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives, Ch. 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.loan import TermLoan, RevolvingFacility
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import _register, _serialise_atom


# ---------------------------------------------------------------------------
# Layer 2: LoanParticipation
# ---------------------------------------------------------------------------

@dataclass
class ParticipationResult:
    """Loan participation pricing result."""
    pv: float
    funded_amount: float
    coupon_pv: float
    principal_pv: float
    recovery_pv: float
    expected_loss: float
    net_carry: float          # annual carry rate

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "funded_amount": self.funded_amount,
            "coupon_pv": self.coupon_pv, "principal_pv": self.principal_pv,
            "recovery_pv": self.recovery_pv, "expected_loss": self.expected_loss,
            "net_carry": self.net_carry,
        }


class LoanParticipation:
    """Funded loan participation — investor funds a fraction and takes credit risk.

    The participant funds participation_rate × notional and receives pro-rata
    interest payments. On default, the participant loses (1-R) × funded_amount.

    Args:
        underlying: the reference loan (TermLoan or RevolvingFacility).
        participation_rate: fraction of the loan purchased (0 to 1).
        funding_cost: participant's cost of funding (annual rate).
        recovery: recovery rate on default.
        trade_type: "participation" (contractual) or "assignment" (full transfer).
    """

    _SERIAL_TYPE = "loan_participation"

    def __init__(
        self,
        underlying: TermLoan | RevolvingFacility,
        participation_rate: float = 1.0,
        funding_cost: float = 0.03,
        recovery: float = 0.6,
        trade_type: str = "participation",
    ):
        if not 0.0 < participation_rate <= 1.0:
            raise ValueError(f"participation_rate must be in (0, 1], got {participation_rate}")
        if trade_type not in ("participation", "assignment"):
            raise ValueError(f"trade_type must be 'participation' or 'assignment'")
        self.underlying = underlying
        self.participation_rate = participation_rate
        self.funding_cost = funding_cost
        self.recovery = recovery
        self.trade_type = trade_type

    @property
    def notional(self) -> float:
        return getattr(self.underlying, "notional", getattr(self.underlying, "max_commitment", 0))

    @property
    def funded_amount(self) -> float:
        return self.notional * self.participation_rate

    def pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> ParticipationResult:
        """PV of the participation with credit risk.

        If survival_curve is None, assumes risk-free (no default).
        """
        proj = projection_curve or discount_curve
        flows = self.underlying.cashflows(proj)
        T = year_fraction(self.underlying.start, self.underlying.end, self.underlying.day_count)

        coupon_pv = 0.0
        principal_pv_total = 0.0
        recovery_pv = 0.0
        prev_date = self.underlying.start

        for d, interest, principal in flows:
            df = discount_curve.df(d)
            if survival_curve is not None:
                surv = survival_curve.survival(d)
                surv_prev = survival_curve.survival(prev_date)
                dpd = max(surv_prev - surv, 0.0)
            else:
                surv = 1.0
                dpd = 0.0

            # Pro-rata coupon, weighted by survival
            coupon_pv += self.participation_rate * interest * df * surv
            # Principal return, weighted by survival
            principal_pv_total += self.participation_rate * principal * df * surv
            # Recovery on default in this period
            recovery_pv += self.recovery * self.funded_amount * dpd * df
            prev_date = d

        # Expected loss
        if survival_curve is not None:
            surv_T = survival_curve.survival(self.underlying.end)
            expected_loss = (1 - self.recovery) * self.funded_amount * (1 - surv_T)
        else:
            expected_loss = 0.0

        total = coupon_pv + principal_pv_total + recovery_pv
        net_carry = (coupon_pv / self.funded_amount / max(T, 0.01) - self.funding_cost
                     if self.funded_amount > 0 else 0.0)

        return ParticipationResult(
            pv=total, funded_amount=self.funded_amount,
            coupon_pv=coupon_pv, principal_pv=principal_pv_total,
            recovery_pv=recovery_pv, expected_loss=expected_loss,
            net_carry=net_carry,
        )

    def assignment_premium(self, counterparty_spread: float = 0.001) -> float:
        """Extra value from assignment vs participation.

        Assignment = direct claim (lower counterparty risk).
        Premium ≈ counterparty_spread × funded_amount × WAL.
        """
        if self.trade_type == "assignment":
            return 0.0  # already assignment
        # Approximate WAL
        wal = year_fraction(self.underlying.start, self.underlying.end,
                            self.underlying.day_count) * 0.6  # rough
        return counterparty_spread * self.funded_amount * wal

    def pv_ctx(self, ctx) -> float:
        sc = None
        if ctx.credit_curves:
            sc = next(iter(ctx.credit_curves.values()))
        proj = None
        if ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()))
        return self.pv(ctx.discount_curve, proj, sc).pv

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "underlying": self.underlying.to_dict(),
            "participation_rate": self.participation_rate,
            "funding_cost": self.funding_cost,
            "recovery": self.recovery,
            "trade_type": self.trade_type,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> LoanParticipation:
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(
            underlying=_fd(p["underlying"]),
            participation_rate=p.get("participation_rate", 1.0),
            funding_cost=p.get("funding_cost", 0.03),
            recovery=p.get("recovery", 0.6),
            trade_type=p.get("trade_type", "participation"),
        )


_register(LoanParticipation)


# ---------------------------------------------------------------------------
# Layer 3: PartialFundedParticipation
# ---------------------------------------------------------------------------

class PartialFundedParticipation:
    """Split loan exposure into funded + unfunded (CDS-like) legs.

    Funded leg: investor puts up cash, earns loan coupons, takes credit risk.
    Unfunded leg: investor earns a premium for taking default protection.

    The unfunded leg is economically a CDS on the reference loan.

    Args:
        underlying: the reference loan.
        funded_rate: fraction that is cash-funded (e.g. 0.6 = 60% funded).
        unfunded_spread: premium earned on unfunded portion (annual).
        funding_cost: cost of funding the funded portion.
        recovery: recovery rate on default.
    """

    _SERIAL_TYPE = "partial_funded"

    def __init__(
        self,
        underlying: TermLoan | RevolvingFacility,
        funded_rate: float = 0.6,
        unfunded_spread: float = 0.02,
        funding_cost: float = 0.03,
        recovery: float = 0.6,
    ):
        if not 0.0 <= funded_rate <= 1.0:
            raise ValueError(f"funded_rate must be in [0, 1], got {funded_rate}")
        self.underlying = underlying
        self.funded_rate = funded_rate
        self.unfunded_spread = unfunded_spread
        self.funding_cost = funding_cost
        self.recovery = recovery

    @property
    def notional(self) -> float:
        return getattr(self.underlying, "notional", getattr(self.underlying, "max_commitment", 0))

    @property
    def unfunded_rate(self) -> float:
        return 1.0 - self.funded_rate

    @property
    def cash_outlay(self) -> float:
        """Actual cash invested."""
        return self.funded_rate * self.notional

    @property
    def leverage(self) -> float:
        """Total exposure / cash invested."""
        return self.notional / self.cash_outlay if self.cash_outlay > 0 else float("inf")

    def pv_funded(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """PV of the funded portion."""
        part = LoanParticipation(
            self.underlying, self.funded_rate, self.funding_cost,
            self.recovery, "participation",
        )
        return part.pv(discount_curve, projection_curve, survival_curve).pv

    def pv_unfunded(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """PV of the unfunded (CDS-like) portion.

        Premium leg: earn unfunded_spread on unfunded_notional.
        Protection leg: pay (1-R) × unfunded_notional on default.
        """
        unfunded_notional = self.unfunded_rate * self.notional
        T = year_fraction(self.underlying.start, self.underlying.end, self.underlying.day_count)

        if survival_curve is None:
            # No credit risk: pure premium income
            df_avg = discount_curve.df(self.underlying.end)
            return self.unfunded_spread * unfunded_notional * T * df_avg * 0.5

        # Premium leg (earn spread while alive)
        from pricebook.cds import risky_annuity
        ann = risky_annuity(self.underlying.start, self.underlying.end,
                            discount_curve, survival_curve)
        premium_pv = self.unfunded_spread * unfunded_notional * ann

        # Protection leg (pay on default)
        from pricebook.cds import protection_leg_pv
        prot_pv = protection_leg_pv(
            self.underlying.start, self.underlying.end,
            discount_curve, survival_curve,
            self.recovery, unfunded_notional,
        )

        return premium_pv - prot_pv  # net: premium received - protection cost

    def total_pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """Total PV = funded + unfunded."""
        return (self.pv_funded(discount_curve, projection_curve, survival_curve)
                + self.pv_unfunded(discount_curve, survival_curve))

    def pv_ctx(self, ctx) -> float:
        sc = None
        if ctx.credit_curves:
            sc = next(iter(ctx.credit_curves.values()))
        proj = None
        if ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()))
        return self.total_pv(ctx.discount_curve, proj, sc)

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "underlying": self.underlying.to_dict(),
            "funded_rate": self.funded_rate,
            "unfunded_spread": self.unfunded_spread,
            "funding_cost": self.funding_cost,
            "recovery": self.recovery,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> PartialFundedParticipation:
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(
            underlying=_fd(p["underlying"]),
            funded_rate=p.get("funded_rate", 0.6),
            unfunded_spread=p.get("unfunded_spread", 0.02),
            funding_cost=p.get("funding_cost", 0.03),
            recovery=p.get("recovery", 0.6),
        )


_register(PartialFundedParticipation)
