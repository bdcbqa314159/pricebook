"""Unitranche & direct lending: FOLO, call protection, DDTL, HTM yield.

    from pricebook.credit.unitranche import Unitranche, FOLO, DelayedDrawTermLoan

    loan = Unitranche(start, end, spread=0.055, notional=50_000_000,
                       folo=FOLO(30_000_000, 20_000_000, 0.04, 0.075))

References:
    Ares Management (2022). Direct Lending Market Overview.
    LSTA (2022). The Handbook of Loan Syndications and Trading.
    Golub & Crum (2018). The Art of Private Equity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.loan import TermLoan
from pricebook.core.schedule import Frequency
from pricebook.core.solvers import brentq


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class FOLO:
    """First-out / last-out split within a unitranche.

    First-out lender has priority on recovery (absolute priority within
    the unitranche). Last-out absorbs losses first.
    """
    first_out_notional: float
    last_out_notional: float
    first_out_spread: float   # e.g. 0.04 (SOFR + 400)
    last_out_spread: float    # e.g. 0.075 (SOFR + 750)
    first_out_recovery_cap: float = 1.0  # FO gets recovery first up to this fraction

    @property
    def total_notional(self) -> float:
        return self.first_out_notional + self.last_out_notional

    @property
    def first_out_pct(self) -> float:
        return self.first_out_notional / self.total_notional if self.total_notional > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "first_out_notional": self.first_out_notional,
            "last_out_notional": self.last_out_notional,
            "first_out_spread": self.first_out_spread,
            "last_out_spread": self.last_out_spread,
            "first_out_pct": self.first_out_pct,
            "total_notional": self.total_notional,
        }


@dataclass
class FOLORecoveryResult:
    """Recovery allocation under FOLO structure."""
    total_recovery: float
    first_out_recovery_pct: float
    last_out_recovery_pct: float
    first_out_loss: float
    last_out_loss: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CallProtectionSchedule:
    """Non-call / step-down / par call schedule.

    Typical: NC2 / 101 / par (no call for 2 years, then 101%, then par).
    """
    par_dates: list[date]   # dates at which premium steps down
    premiums: list[float]   # call premium at each step (e.g. [0.02, 0.01, 0.0])

    def call_price(self, call_date: date) -> float:
        """Call price (per 100) at a given date.

        premiums[i] applies between par_dates[i] and par_dates[i+1].
        Before par_dates[0]: non-callable (inf).
        After par_dates[-1]: premiums[-1].
        """
        if not self.par_dates:
            return 100.0 * (1 + self.premiums[0]) if self.premiums else 100.0

        if call_date < self.par_dates[0]:
            return float('inf')  # non-call period

        # Find which interval the call_date falls in
        for i in range(len(self.par_dates) - 1):
            if call_date < self.par_dates[i + 1]:
                prem_idx = min(i, len(self.premiums) - 1)
                return 100.0 * (1 + self.premiums[prem_idx])

        # After last par_date
        return 100.0 * (1 + self.premiums[-1]) if self.premiums else 100.0

    def is_callable(self, call_date: date) -> bool:
        """Whether the loan is callable at this date."""
        return len(self.par_dates) == 0 or call_date >= self.par_dates[0]

    def to_dict(self) -> dict:
        return {
            "par_dates": [d.isoformat() for d in self.par_dates],
            "premiums": self.premiums,
        }


@dataclass
class DirectLendingYield:
    """All-in yield decomposition for a direct lending position."""
    coupon_yield: float        # spread + estimated base rate
    oid_amortisation: float    # OID accreted per year over WAL
    upfront_fee_amort: float   # upfront fee amortised over WAL
    all_in_yield: float

    def to_dict(self) -> dict:
        return vars(self)


# ═══════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════

def folo_recovery_split(
    total_recovery_pct: float,
    folo: FOLO,
) -> FOLORecoveryResult:
    """Allocate recovery between first-out and last-out.

    First-out gets paid first (absolute priority within unitranche).
    """
    total_notional = folo.total_notional
    total_recovery = total_notional * total_recovery_pct

    # FO gets recovery first up to its notional
    fo_recovery = min(total_recovery, folo.first_out_notional * folo.first_out_recovery_cap)
    lo_recovery = max(total_recovery - fo_recovery, 0.0)

    fo_pct = fo_recovery / folo.first_out_notional if folo.first_out_notional > 0 else 0.0
    lo_pct = lo_recovery / folo.last_out_notional if folo.last_out_notional > 0 else 0.0

    return FOLORecoveryResult(
        total_recovery=total_recovery,
        first_out_recovery_pct=min(fo_pct, 1.0),
        last_out_recovery_pct=min(lo_pct, 1.0),
        first_out_loss=max(folo.first_out_notional - fo_recovery, 0.0),
        last_out_loss=max(folo.last_out_notional - lo_recovery, 0.0),
    )


def unitranche_blended_spread(
    fo_spread: float,
    lo_spread: float,
    fo_pct: float,
) -> float:
    """Weighted average spread: fo_pct × fo_spread + (1 - fo_pct) × lo_spread."""
    return fo_pct * fo_spread + (1 - fo_pct) * lo_spread


def direct_lending_economics(
    spread: float,
    base_rate: float,
    oid: float = 0.0,
    upfront_fee: float = 0.0,
    wal: float = 4.0,
) -> DirectLendingYield:
    """All-in yield decomposition for a direct lending position.

    Args:
        spread: credit spread (e.g. 0.055).
        base_rate: estimated floating base rate (e.g. 0.05).
        oid: original issue discount (e.g. 0.02 = 98 OID).
        upfront_fee: upfront arrangement fee (e.g. 0.01 = 1%).
        wal: weighted average life for amortising OID/fee.
    """
    coupon_yield = spread + base_rate
    oid_amort = oid / wal if wal > 0 else 0.0
    fee_amort = upfront_fee / wal if wal > 0 else 0.0

    return DirectLendingYield(
        coupon_yield=coupon_yield,
        oid_amortisation=oid_amort,
        upfront_fee_amort=fee_amort,
        all_in_yield=coupon_yield + oid_amort + fee_amort,
    )


def hold_to_maturity_yield(
    loan: TermLoan,
    market_price: float,
    projection_curve: DiscountCurve,
) -> float:
    """Yield-to-maturity assuming no default, no prepayment.

    Solves for the spread s such that PV(cashflows, discount=flat(s)) = market_price.

    Args:
        loan: TermLoan or subclass.
        market_price: current clean price per 100.
        projection_curve: for forward rate computation.
    """
    target_pv = market_price / 100.0 * loan.notional

    def objective(y: float) -> float:
        # Discount all cashflows at flat yield y
        total = 0.0
        for d, interest, principal in loan.cashflows(projection_curve):
            t = year_fraction(loan.start, d, DayCountConvention.ACT_365_FIXED)
            if t <= 0:
                t = 1 / 365
            df = 1.0 / (1 + y) ** t
            total += df * (interest + principal)
        return total - target_pv

    return brentq(objective, -0.10, 1.0, tol=1e-8)


# ═══════════════════════════════════════════════════════════════
# Unitranche
# ═══════════════════════════════════════════════════════════════

class Unitranche(TermLoan):
    """Unitranche term loan with optional FOLO structure.

    A single-tranche facility that blends senior and subordinated
    economics. The FOLO (first-out/last-out) structure governs
    the internal priority of payments and recovery allocation
    via an Agreement Among Lenders (AAL).

    Args:
        folo: optional FOLO split. If provided, blended_spread is derived.
        oid: original issue discount (e.g. 0.02 means price 98).
        call_protection: optional call protection schedule.
    """

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.055,
        notional: float = 10_000_000.0,
        amort_rate: float = 0.01,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        folo: FOLO | None = None,
        oid: float = 0.0,
        call_protection: CallProtectionSchedule | None = None,
    ):
        super().__init__(start, end, spread, notional, amort_rate, frequency, day_count)
        self.folo = folo
        self.oid = oid
        self.call_protection = call_protection

    def blended_spread(self) -> float:
        """Blended unitranche spread (from FOLO if present, else raw spread)."""
        if self.folo:
            return unitranche_blended_spread(
                self.folo.first_out_spread, self.folo.last_out_spread,
                self.folo.first_out_pct,
            )
        return self.spread

    def folo_recovery(self, total_recovery_pct: float) -> FOLORecoveryResult:
        """Allocate recovery between first-out and last-out."""
        if self.folo is None:
            raise ValueError("No FOLO structure defined")
        return folo_recovery_split(total_recovery_pct, self.folo)

    def proceeds(self) -> float:
        """Net proceeds after OID: notional × (1 - OID)."""
        return self.notional * (1 - self.oid)

    def to_dict(self) -> dict:
        d = {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "spread": self.spread, "notional": self.notional,
            "amort_rate": self.amort_rate, "oid": self.oid,
            "blended_spread": self.blended_spread(),
        }
        if self.folo:
            d["folo"] = self.folo.to_dict()
        if self.call_protection:
            d["call_protection"] = self.call_protection.to_dict()
        return d


# ═══════════════════════════════════════════════════════════════
# Delayed Draw Term Loan
# ═══════════════════════════════════════════════════════════════

class DelayedDrawTermLoan(TermLoan):
    """Committed but undrawn term loan with ticking fee.

    Before draw_date: pays ticking_fee on committed amount (no principal).
    After draw_date: behaves as normal TermLoan on drawn amount.

    Args:
        draw_date: expected draw date (None = draw at start).
        ticking_fee: annual fee on undrawn commitment.
        drawn_pct: fraction actually drawn on draw_date (1.0 = fully drawn).
    """

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.05,
        notional: float = 10_000_000.0,
        amort_rate: float = 0.01,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        draw_date: date | None = None,
        ticking_fee: float = 0.01,
        drawn_pct: float = 1.0,
    ):
        super().__init__(start, end, spread, notional, amort_rate, frequency, day_count)
        self.draw_date = draw_date or start
        self.ticking_fee = ticking_fee
        self.drawn_pct = drawn_pct

    def cashflows(
        self,
        projection_curve: DiscountCurve,
    ) -> list[tuple[date, float, float]]:
        """Generate cashflows: ticking fee before draw, normal coupon after."""
        drawn_amount = self.notional * self.drawn_pct
        outstanding = 0.0
        amort_amount = drawn_amount * self.amort_rate
        flows = []

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)

            if t_end <= self.draw_date:
                # Pre-draw: ticking fee on committed amount
                interest = self.notional * self.ticking_fee * yf
                principal = 0.0
            else:
                if outstanding == 0.0:
                    # Draw happens this period
                    outstanding = drawn_amount
                    # Partial period: ticking fee + drawn coupon
                    if t_start < self.draw_date:
                        pre_yf = year_fraction(t_start, self.draw_date, self.day_count)
                        post_yf = year_fraction(self.draw_date, t_end, self.day_count)
                    else:
                        pre_yf = 0.0
                        post_yf = yf

                    df1 = projection_curve.df(t_start)
                    df2 = projection_curve.df(t_end)
                    fwd = (df1 - df2) / (yf * df2) if yf > 0 else 0.0

                    tick = self.notional * self.ticking_fee * pre_yf
                    coupon = outstanding * (fwd + self.spread) * post_yf
                    interest = tick + coupon
                else:
                    df1 = projection_curve.df(t_start)
                    df2 = projection_curve.df(t_end)
                    fwd = (df1 - df2) / (yf * df2) if yf > 0 else 0.0
                    interest = outstanding * (fwd + self.spread) * yf

                if i == len(self.schedule) - 1:
                    principal = outstanding
                else:
                    principal = min(amort_amount, outstanding) if outstanding > 0 else 0.0

                outstanding -= principal
                outstanding = max(outstanding, 0.0)

            flows.append((t_end, interest, principal))

        return flows

    def commitment_value(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        market_spread: float | None = None,
    ) -> float:
        """Value of the commitment = PV(ticking fee) + PV(spread advantage).

        If market_spread is given, the spread advantage is the difference
        between the loan's spread and current market.
        """
        proj = projection_curve or discount_curve
        # PV of all cashflows at current curves
        base_pv = self.pv(discount_curve, proj)

        if market_spread is not None and market_spread != self.spread:
            # Compute PV with market spread (what it would cost to get same loan today)
            market_loan = DelayedDrawTermLoan(
                self.start, self.end,
                spread=market_spread, notional=self.notional,
                amort_rate=self.amort_rate, frequency=self.frequency,
                day_count=self.day_count, draw_date=self.draw_date,
                ticking_fee=self.ticking_fee, drawn_pct=self.drawn_pct,
            )
            market_pv = market_loan.pv(discount_curve, proj)
            return base_pv - market_pv  # positive if spread < market
        return base_pv

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "spread": self.spread, "notional": self.notional,
            "draw_date": self.draw_date.isoformat(),
            "ticking_fee": self.ticking_fee,
            "drawn_pct": self.drawn_pct,
        }
