"""
Loan instruments: term loans and leveraged loans.

Term loan: amortising floating-rate instrument with credit spread.
Leveraged loan: credit-risky floating-rate, marked via discount margin.

    loan = TermLoan(start, end, spread=0.03, notional=1_000_000,
                    amort_rate=0.01)
    pv = loan.pv(discount_curve, projection_curve)
"""

from __future__ import annotations

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, generate_schedule
from pricebook.solvers import brentq


class TermLoan:
    """Amortising floating-rate term loan.

    Principal amortises by a fixed percentage each period.
    Coupon = (forward_rate + spread) * outstanding_notional * year_frac.

    Args:
        start: drawdown date.
        end: final maturity.
        spread: credit spread over the floating index.
        notional: initial notional.
        amort_rate: fraction of initial notional repaid each period (0 = bullet).
        frequency: payment frequency.
        day_count: day count for accrual.
    """

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.03,
        notional: float = 1_000_000.0,
        amort_rate: float = 0.0,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        self.start = start
        self.end = end
        self.spread = spread
        self.notional = notional
        self.amort_rate = amort_rate
        self.frequency = frequency
        self.day_count = day_count
        self.schedule = generate_schedule(start, end, frequency)

    def cashflows(
        self,
        projection_curve: DiscountCurve,
    ) -> list[tuple[date, float, float]]:
        """Generate cashflows: (date, interest, principal_repayment).

        Args:
            projection_curve: for computing forward rates.
        """
        outstanding = self.notional
        amort_amount = self.notional * self.amort_rate
        flows = []

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df1 = projection_curve.df(t_start)
            df2 = projection_curve.df(t_end)
            fwd = (df1 - df2) / (yf * df2)

            interest = outstanding * (fwd + self.spread) * yf

            if i == len(self.schedule) - 1:
                # Final period: repay all remaining
                principal = outstanding
            else:
                principal = min(amort_amount, outstanding)

            flows.append((t_end, interest, principal))
            outstanding -= principal
            outstanding = max(outstanding, 0.0)

        return flows

    def pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """PV of all cashflows (interest + principal)."""
        proj = projection_curve if projection_curve is not None else discount_curve
        total = 0.0
        for d, interest, principal in self.cashflows(proj):
            df = discount_curve.df(d)
            total += df * (interest + principal)
        return total

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Price per 100 face."""
        return self.pv(discount_curve, projection_curve) / self.notional * 100.0

    def discount_margin(
        self,
        market_price: float,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Discount margin: spread adjustment to match market price.

        Solves for dm such that loan with spread+dm reprices to market_price.
        """
        def objective(dm: float) -> float:
            shifted = TermLoan(
                self.start, self.end,
                spread=self.spread + dm,
                notional=self.notional,
                amort_rate=self.amort_rate,
                frequency=self.frequency,
                day_count=self.day_count,
            )
            return shifted.dirty_price(discount_curve, projection_curve) - market_price

        return brentq(objective, -0.10, 0.10)

    def weighted_average_life(
        self,
        projection_curve: DiscountCurve,
    ) -> float:
        """WAL: weighted average time to principal repayment.

        WAL = sum(t_i * principal_i) / total_principal
        """
        flows = self.cashflows(projection_curve)
        ref = self.start
        num = 0.0
        den = 0.0
        for d, _, principal in flows:
            t = year_fraction(ref, d, self.day_count)
            num += t * principal
            den += principal
        return num / den if den > 0 else 0.0


class RevolvingFacility:
    """Revolving credit facility — drawn and undrawn components.

    The TRS total return includes both drawn interest and undrawn commitment fees.
    Notional changes as the facility is drawn/repaid.

        revolver = RevolvingFacility(
            start, end, max_commitment=100_000_000,
            drawn_amount=60_000_000, drawn_spread=0.03,
            undrawn_fee=0.005)
        flows = revolver.cashflows(projection_curve)
    """

    def __init__(
        self,
        start: date,
        end: date,
        max_commitment: float,
        drawn_amount: float,
        drawn_spread: float = 0.03,
        undrawn_fee: float = 0.005,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        self.start = start
        self.end = end
        self.max_commitment = max_commitment
        self.drawn_amount = drawn_amount
        self.drawn_spread = drawn_spread
        self.undrawn_fee = undrawn_fee
        self.frequency = frequency
        self.day_count = day_count
        self.notional = max_commitment  # for TRS compatibility

    def cashflows(
        self,
        projection_curve: DiscountCurve,
    ) -> list[tuple[date, float, float]]:
        """Generate cashflows: (date, total_income, principal_return).

        Total income = drawn_interest + undrawn_commitment_fee.
        Principal return = 0 (revolving — no scheduled amortisation).
        """
        dates = generate_schedule(self.start, self.end, self.frequency)
        flows = []

        for i in range(1, len(dates)):
            yf = year_fraction(dates[i - 1], dates[i], self.day_count)
            fwd = projection_curve.forward_rate(dates[i - 1], dates[i])

            # Drawn interest
            drawn_interest = self.drawn_amount * (fwd + self.drawn_spread) * yf

            # Undrawn commitment fee
            undrawn = self.max_commitment - self.drawn_amount
            commitment_fee = undrawn * self.undrawn_fee * yf

            total_income = drawn_interest + commitment_fee
            flows.append((dates[i], total_income, 0.0))

        return flows

    def pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        proj = projection_curve or discount_curve
        return sum(
            discount_curve.df(d) * (interest + principal)
            for d, interest, principal in self.cashflows(proj)
        )

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        return self.pv(discount_curve, projection_curve) / self.max_commitment * 100.0

    @property
    def utilization(self) -> float:
        """Drawn / committed ratio."""
        return self.drawn_amount / self.max_commitment if self.max_commitment > 0 else 0.0
