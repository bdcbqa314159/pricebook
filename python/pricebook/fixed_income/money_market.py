"""Money market instruments beyond deposits and T-bills.

Classes:
    - CertificateOfDeposit — interest-bearing CD priced off a discount curve
    - CommercialPaper      — corporate discount instrument (add-on or discount basis)
    - BankersAcceptance    — bank-guaranteed discount instrument with acceptance fee
    - RepoRate             — implied repo / haircut-adjusted funding rate helpers

References:
    Fabozzi, F.J., *Money Market: An Overview*, CFA Institute, 2022.
    Stigum, M. & Crescenzi, A., *Stigum's Money Market*, McGraw-Hill, 2007, Ch. 5-9.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


# ---------------------------------------------------------------------------
# CertificateOfDeposit
# ---------------------------------------------------------------------------

class CertificateOfDeposit:
    """Interest-bearing certificate of deposit.

    A CD pays a single coupon at maturity equal to:

        cashflow = face_value × (1 + coupon_rate × year_fraction)

    The dirty price is the PV of that cashflow discounted at the curve rate.
    The clean price strips accrued interest earned since settlement.

    Args:
        settlement:   settlement date.
        maturity:     maturity date.
        face_value:   par amount (default 100).
        coupon_rate:  annual coupon rate (e.g. 0.05 for 5%).
        day_count:    day-count convention (default ACT/360).
    """

    def __init__(
        self,
        settlement: date,
        maturity: date,
        face_value: float = 100.0,
        coupon_rate: float = 0.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if settlement >= maturity:
            raise ValueError(f"settlement ({settlement}) must be before maturity ({maturity})")
        if face_value <= 0:
            raise ValueError(f"face_value must be positive, got {face_value}")
        if coupon_rate < 0:
            raise ValueError(f"coupon_rate must be non-negative, got {coupon_rate}")

        self.settlement = settlement
        self.maturity = maturity
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.day_count = day_count

    # ---- Constructors ----

    @classmethod
    def from_yield(
        cls,
        settlement: date,
        maturity: date,
        ytm: float,
        face_value: float = 100.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ) -> CertificateOfDeposit:
        """Build a CD where the coupon rate equals the yield (par CD)."""
        return cls(settlement, maturity, face_value, ytm, day_count)

    # ---- Basic measures ----

    @property
    def _yf_total(self) -> float:
        return year_fraction(self.settlement, self.maturity, self.day_count)

    @property
    def maturity_cashflow(self) -> float:
        """Total cashflow at maturity: face × (1 + rate × yf)."""
        return self.face_value * (1.0 + self.coupon_rate * self._yf_total)

    @property
    def accrued_interest(self) -> float:
        """Accrued interest from settlement to today (zero; settlement is pricing date)."""
        # For a freshly-settled CD accrued interest starts at zero.
        # This property exists so callers can compute it for a seasoned CD by
        # constructing with the original issue date as settlement and today as settlement.
        return 0.0

    @property
    def duration(self) -> float:
        """Macaulay duration: single cashflow at maturity (years, ACT/365)."""
        return (self.maturity - self.settlement).days / 365.0

    # ---- Curve-dependent pricing ----

    def dirty_price(self, curve: DiscountCurve) -> float:
        """Dirty (full) price: PV of maturity cashflow.

            dirty = face × (1 + coupon × yf_total) × df(maturity)
        """
        return self.maturity_cashflow * curve.df(self.maturity)

    def clean_price(self, curve: DiscountCurve) -> float:
        """Clean price: dirty price minus accrued interest."""
        return self.dirty_price(curve) - self.accrued_interest

    def yield_to_maturity(self, price: float) -> float:
        """Yield implied by a given dirty price.

        Inverts: price = face × (1 + rate × yf) / (1 + ytm × yf)

            ytm = (maturity_cashflow / price − 1) / yf
        """
        yf = self._yf_total
        if yf <= 0 or price <= 0:
            return 0.0
        return (self.maturity_cashflow / price - 1.0) / yf


# ---------------------------------------------------------------------------
# CommercialPaper
# ---------------------------------------------------------------------------

class CommercialPaper:
    """Corporate commercial paper (discount instrument).

    CP is quoted on a bank-discount basis identical to T-bills but is
    unsecured corporate paper and therefore trades at a spread to risk-free.

    Args:
        settlement: settlement date (typically T+0 or T+1).
        maturity:   maturity date (≤ 270 days from issuance for SEC exemption).
        face_value: par amount (default 100).
        day_count:  day-count convention (default ACT/360).
    """

    def __init__(
        self,
        settlement: date,
        maturity: date,
        face_value: float = 100.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if settlement >= maturity:
            raise ValueError(f"settlement ({settlement}) must be before maturity ({maturity})")
        if face_value <= 0:
            raise ValueError(f"face_value must be positive, got {face_value}")

        self.settlement = settlement
        self.maturity = maturity
        self.face_value = face_value
        self.day_count = day_count

    @property
    def days(self) -> int:
        """Actual days from settlement to maturity."""
        return (self.maturity - self.settlement).days

    def price_from_discount(self, discount_rate: float) -> float:
        """Price given a bank-discount rate.

            price = face × (1 − discount_rate × days / 360)
        """
        return self.face_value * (1.0 - discount_rate * self.days / 360.0)

    def price_from_yield(self, yield_rate: float) -> float:
        """Price given a money-market (add-on) yield.

            price = face / (1 + yield_rate × days / 360)
        """
        return self.face_value / (1.0 + yield_rate * self.days / 360.0)

    def discount_rate(self, price: float) -> float:
        """Implied bank-discount rate from a given price.

            d = (face − price) / face × (360 / days)
        """
        days = self.days
        if days <= 0 or price <= 0:
            return 0.0
        return (self.face_value - price) / self.face_value * (360.0 / days)

    def bond_equiv_yield(self, price: float) -> float:
        """Bond-equivalent yield (365-day basis, add-on to price paid).

            BEY = (face − price) / price × (365 / days)

        Note: The quadratic adjustment for > 182 days is not applied here
        because CP maturities are capped at 270 days in US markets and the
        single-period formula is the market convention (Fabozzi, 2022, Ch. 1).
        """
        days = self.days
        if days <= 0 or price <= 0:
            return 0.0
        return (self.face_value - price) / price * (365.0 / days)

    def credit_spread(self, price: float, risk_free_rate: float) -> float:
        """Credit spread over risk-free rate in discount-yield space.

            spread = discount_rate(price) − risk_free_rate
        """
        return self.discount_rate(price) - risk_free_rate


# ---------------------------------------------------------------------------
# BankersAcceptance
# ---------------------------------------------------------------------------

class BankersAcceptance:
    """Banker's acceptance (BA): bank-guaranteed time draft.

    A BA is economically identical to CP but carries an explicit bank
    guarantee, typically resulting in a tighter credit spread. The all-in
    cost to the borrower adds the acceptance fee on top of the discount rate.

    Args:
        settlement: settlement date.
        maturity:   maturity date.
        face_value: par amount (default 100).
        day_count:  day-count convention (default ACT/360).
    """

    def __init__(
        self,
        settlement: date,
        maturity: date,
        face_value: float = 100.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if settlement >= maturity:
            raise ValueError(f"settlement ({settlement}) must be before maturity ({maturity})")
        if face_value <= 0:
            raise ValueError(f"face_value must be positive, got {face_value}")

        self.settlement = settlement
        self.maturity = maturity
        self.face_value = face_value
        self.day_count = day_count

    @property
    def days(self) -> int:
        return (self.maturity - self.settlement).days

    def price_from_discount(self, discount_rate: float) -> float:
        """Price from bank-discount rate (same as CP)."""
        return self.face_value * (1.0 - discount_rate * self.days / 360.0)

    def price_from_yield(self, yield_rate: float) -> float:
        """Price from money-market add-on yield."""
        return self.face_value / (1.0 + yield_rate * self.days / 360.0)

    def discount_rate(self, price: float) -> float:
        """Implied bank-discount rate from price."""
        days = self.days
        if days <= 0 or price <= 0:
            return 0.0
        return (self.face_value - price) / self.face_value * (360.0 / days)

    def bond_equiv_yield(self, price: float) -> float:
        """Bond-equivalent yield (365-day basis)."""
        days = self.days
        if days <= 0 or price <= 0:
            return 0.0
        return (self.face_value - price) / price * (365.0 / days)

    def credit_spread(self, price: float, risk_free_rate: float) -> float:
        """Credit spread over risk-free discount rate."""
        return self.discount_rate(price) - risk_free_rate

    def all_in_cost(self, price: float, acceptance_fee: float) -> float:
        """All-in financing cost including the bank acceptance fee.

        The acceptance fee (expressed as an annual rate, e.g. 0.005 for 50 bps)
        is added to the discount rate to give the borrower's true cost:

            all_in = discount_rate(price) + acceptance_fee

        Args:
            price:          dirty price per 100 face.
            acceptance_fee: bank acceptance fee as an annual rate.

        Returns:
            All-in cost as an annual rate (discount basis).
        """
        return self.discount_rate(price) + acceptance_fee


# ---------------------------------------------------------------------------
# RepoRate helpers
# ---------------------------------------------------------------------------

class RepoRate:
    """Repurchase agreement rate helpers.

    These are stateless utility methods; instantiation is optional but
    provides a namespace consistent with the rest of the module.
    """

    @staticmethod
    def implied_repo(
        purchase_price: float,
        sale_price: float,
        days: int,
        day_count_basis: int = 360,
    ) -> float:
        """Implied repo rate from a cash-and-carry transaction.

        Computes the annualised financing rate embedded in a repo where
        collateral is purchased today and sold (or repoed out) at a known
        forward price:

            r_repo = (sale_price / purchase_price − 1) × (basis / days)

        Args:
            purchase_price:   initial purchase price (e.g. dirty price paid).
            sale_price:       forward sale / repurchase price including repo interest.
            days:             term of the repo in calendar days.
            day_count_basis:  360 (money-market, default) or 365 (gilt repo).

        Returns:
            Implied repo rate as a decimal (e.g. 0.053 for 5.3%).
        """
        if purchase_price <= 0 or days <= 0:
            return 0.0
        return (sale_price / purchase_price - 1.0) * (day_count_basis / days)

    @staticmethod
    def haircut_adjusted_rate(repo_rate: float, haircut: float) -> float:
        """Effective funding rate after accounting for repo haircut.

        A haircut h means the lender advances only (1 − h) of collateral
        value. The borrower therefore pays the repo rate on a smaller cash
        amount relative to the collateral, raising the effective cost:

            effective_rate = repo_rate / (1 − haircut)

        Args:
            repo_rate: quoted repo rate as a decimal.
            haircut:   haircut fraction (e.g. 0.02 for a 2% haircut).

        Returns:
            Effective borrowing cost as a decimal.

        Raises:
            ValueError: if haircut ≥ 1.
        """
        if haircut >= 1.0:
            raise ValueError(f"haircut must be < 1, got {haircut}")
        return repo_rate / (1.0 - haircut)
