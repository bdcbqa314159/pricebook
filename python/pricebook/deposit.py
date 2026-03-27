"""Money market deposit instrument."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction


class Deposit:
    """
    A money market deposit.

    A deposit is a loan at a fixed rate for a fixed term. At maturity the
    borrower repays the notional plus accrued interest:

        cashflow = notional * (1 + rate * year_fraction)

    The implied discount factor from spot to maturity is:

        df = 1 / (1 + rate * year_fraction)
    """

    def __init__(
        self,
        start: date,
        end: date,
        rate: float,
        notional: float = 1.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")

        self.start = start
        self.end = end
        self.rate = rate
        self.notional = notional
        self.day_count = day_count

    @property
    def year_fraction(self) -> float:
        return year_fraction(self.start, self.end, self.day_count)

    @property
    def discount_factor(self) -> float:
        """Implied discount factor from start to end."""
        return 1.0 / (1.0 + self.rate * self.year_fraction)

    @property
    def cashflow(self) -> float:
        """Cashflow at maturity: notional * (1 + rate * yf)."""
        return self.notional * (1.0 + self.rate * self.year_fraction)

    def pv(self, df: float) -> float:
        """Present value of the deposit given an external discount factor."""
        return self.cashflow * df - self.notional
