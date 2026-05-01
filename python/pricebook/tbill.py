"""Treasury bill: discount instrument with US money market conventions.

T-bills are zero-coupon government securities quoted on a discount yield basis.
Three yield conventions coexist in the market:

* **Discount yield** (bank discount basis): d = (face − price) / face × (360 / days)
* **Bond equivalent yield** (BEY): annualised holding-period return on price paid
  - days ≤ 182: BEY = (face − price) / price × (365 / days)
  - days > 182: quadratic formula (semi-annual compounding convention)
* **Money market yield** (CD equivalent): MMY = (face − price) / price × (360 / days)

    from pricebook.tbill import TreasuryBill

    bill = TreasuryBill.from_discount_yield(settlement, maturity, 0.05)
    print(bill.price, bill.bey, bill.money_market_yield)

References:
    Stigum & Crescenzi, *Stigum's Money Market*, McGraw-Hill, 2007, Ch. 3.
    Fabozzi, *Fixed Income Analysis*, CFA Institute, 2022, Ch. 1.
    US Treasury, *Treasury Bills: Terms and Conditions*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


class TreasuryBill:
    """US Treasury bill (zero-coupon discount instrument).

    Args:
        settlement: settlement date (T+1 for T-bills).
        maturity: maturity date.
        price: clean price per 100 face.
        face_value: par amount (default 100).
    """

    def __init__(
        self,
        settlement: date,
        maturity: date,
        price: float,
        face_value: float = 100.0,
    ):
        if settlement >= maturity:
            raise ValueError(f"settlement ({settlement}) must be before maturity ({maturity})")
        if price <= 0 or price > face_value:
            raise ValueError(f"price must be in (0, {face_value}], got {price}")
        if face_value <= 0:
            raise ValueError(f"face_value must be positive, got {face_value}")

        self.settlement = settlement
        self.maturity = maturity
        self.price = price
        self.face_value = face_value

    @property
    def days_to_maturity(self) -> int:
        """Actual days from settlement to maturity."""
        return (self.maturity - self.settlement).days

    # ---- Constructors ----

    @classmethod
    def from_discount_yield(
        cls,
        settlement: date,
        maturity: date,
        discount_yield: float,
        face_value: float = 100.0,
    ) -> TreasuryBill:
        """Build from discount yield (bank discount basis).

        Price = face × (1 − d × days / 360)
        """
        days = (maturity - settlement).days
        price = face_value * (1.0 - discount_yield * days / 360.0)
        return cls(settlement, maturity, price, face_value)

    @classmethod
    def from_bond_equivalent_yield(
        cls,
        settlement: date,
        maturity: date,
        bey: float,
        face_value: float = 100.0,
    ) -> TreasuryBill:
        """Build from bond equivalent yield.

        For days ≤ 182: Price = face / (1 + bey × days / 365)
        For days > 182: solve quadratic (semi-annual convention).
        """
        days = (maturity - settlement).days
        if days <= 182:
            price = face_value / (1.0 + bey * days / 365.0)
        else:
            # Quadratic: P/F = (-bey × days/365 + sqrt((bey×days/365)² + ...)) / ...
            # From Stigum: 2×days/365 terms
            t = days / 365.0
            # P = F × 2 / (t × bey + sqrt(t² × bey² - (2t-1)(1 - F/P)))
            # Simpler: solve P × (1 + bey/2)^(2t) = F for semi-annual
            # Direct: P = F / (1 + bey × t) as approximation, then refine
            # Exact Stigum formula for days > 182:
            a = t / 2.0
            b = bey * a
            # P = F × (−b + sqrt(b² + 2b + 1)) = F / (1 + b)... no.
            # Correct formula: solve for P from BEY definition
            # BEY = [-2d/365 + 2√((d/365)² - (2d/365 - 1)(1 - F/P))] / (2d/365 - 1)
            # Let x = F/P, dt = days/365
            # BEY = [-dt + √(dt² - (2dt-1)(1-x))] × 2 / (2dt-1)
            # Rearrange: solve quadratic in x = F/P
            dt = days / 365.0
            # From BEY definition (Fabozzi): x = F/P satisfies
            # (1 + BEY/2)^(2×(days/365)) = F/P
            # So P = F / (1 + BEY/2)^(2×days/365)
            price = face_value / (1.0 + bey / 2.0) ** (2.0 * dt)
        return cls(settlement, maturity, price, face_value)

    # ---- Yield measures ----

    @property
    def discount_yield(self) -> float:
        """Discount yield (bank discount basis).

        d = (face − price) / face × (360 / days)
        """
        days = self.days_to_maturity
        if days <= 0:
            return 0.0
        return (self.face_value - self.price) / self.face_value * (360.0 / days)

    @property
    def bond_equivalent_yield(self) -> float:
        """Bond equivalent yield (BEY).

        days ≤ 182: BEY = (face − price) / price × (365 / days)
        days > 182: quadratic formula (Stigum / Fabozzi convention).
        """
        days = self.days_to_maturity
        if days <= 0:
            return 0.0

        if days <= 182:
            return (self.face_value - self.price) / self.price * (365.0 / days)
        else:
            # Stigum formula for T > 6 months:
            # BEY = 2 × [(F/P)^(365/(2×days)) − 1]
            dt = days / 365.0
            x = self.face_value / self.price
            return 2.0 * (x ** (1.0 / (2.0 * dt)) - 1.0)

    @property
    def bey(self) -> float:
        """Alias for bond_equivalent_yield."""
        return self.bond_equivalent_yield

    @property
    def money_market_yield(self) -> float:
        """Money market yield (CD equivalent yield).

        MMY = (face − price) / price × (360 / days)
        """
        days = self.days_to_maturity
        if days <= 0:
            return 0.0
        return (self.face_value - self.price) / self.price * (360.0 / days)

    # ---- Cross-conversions (static) ----

    @staticmethod
    def discount_to_bey(discount_yield: float, days: int) -> float:
        """Convert discount yield to BEY."""
        price = 100.0 * (1.0 - discount_yield * days / 360.0)
        if days <= 182:
            return (100.0 - price) / price * (365.0 / days)
        dt = days / 365.0
        return 2.0 * ((100.0 / price) ** (1.0 / (2.0 * dt)) - 1.0)

    @staticmethod
    def discount_to_mmyield(discount_yield: float, days: int) -> float:
        """Convert discount yield to money market yield."""
        price = 100.0 * (1.0 - discount_yield * days / 360.0)
        return (100.0 - price) / price * (360.0 / days)

    @staticmethod
    def bey_to_discount(bey: float, days: int) -> float:
        """Convert BEY to discount yield."""
        if days <= 182:
            price = 100.0 / (1.0 + bey * days / 365.0)
        else:
            dt = days / 365.0
            price = 100.0 / (1.0 + bey / 2.0) ** (2.0 * dt)
        return (100.0 - price) / 100.0 * (360.0 / days)

    # ---- PV and risk ----

    def pv(self, discount_curve: DiscountCurve) -> float:
        """Present value: face × df(maturity)."""
        return self.face_value * discount_curve.df(self.maturity)

    def dv01(self, discount_curve: DiscountCurve, shift: float = 0.0001) -> float:
        """DV01: PV change for 1bp parallel rate shift."""
        pv_base = self.pv(discount_curve)
        pv_bumped = self.pv(discount_curve.bumped(shift))
        return pv_bumped - pv_base

    @property
    def duration(self) -> float:
        """Macaulay duration = time to maturity (zero coupon)."""
        return self.days_to_maturity / 365.0

    @property
    def modified_duration(self) -> float:
        """Modified duration ≈ duration / (1 + y/2) where y = BEY."""
        bey = self.bond_equivalent_yield
        return self.duration / (1.0 + bey / 2.0)

    @staticmethod
    def implied_repo_rate(
        spot_price: float,
        forward_price: float,
        days: int,
    ) -> float:
        """Implied repo rate from T-bill cash-and-carry.

        r_repo = (forward / spot − 1) × (360 / days)
        """
        if spot_price <= 0 or days <= 0:
            return 0.0
        return (forward_price / spot_price - 1.0) * (360.0 / days)

    # ---- Serialisation ----

    def to_dict(self) -> dict:
        return {"type": "tbill", "params": {
            "settlement": self.settlement.isoformat(),
            "maturity": self.maturity.isoformat(),
            "price": self.price,
            "face_value": self.face_value,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> TreasuryBill:
        p = d["params"]
        return cls(
            settlement=date.fromisoformat(p["settlement"]),
            maturity=date.fromisoformat(p["maturity"]),
            price=p["price"],
            face_value=p.get("face_value", 100.0),
        )


from pricebook.serialisable import _register
TreasuryBill._SERIAL_TYPE = "tbill"
_register(TreasuryBill)
