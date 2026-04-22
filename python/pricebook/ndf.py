"""Non-Deliverable Forward (NDF).

An NDF is an FX forward that settles in cash (typically USD) rather
than physical delivery of the non-convertible currency.

At fixing: settlement = notional × (fixing_rate - contracted_rate) / fixing_rate
(paid in the settlement currency, typically USD).

Used for: CNY, KRW, INR, BRL, TWD, and other restricted currencies.

    from pricebook.ndf import NDF

    ndf = NDF("USD/CNY", date(2026,10,21), 7.25, 1_000_000)
    pv = ndf.pv(spot=7.20, base_curve=usd, quote_curve=cny)

References:
    ISDA, *1998 FX and Currency Option Definitions*.
    EMTA, *NDF Market Practice Guidelines*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


class NDF:
    """Non-Deliverable Forward.

    Args:
        pair: currency pair string (e.g. "USD/CNY").
        maturity: fixing/settlement date.
        contracted_rate: agreed NDF rate.
        notional: notional in base currency.
        settlement_currency: currency of cash settlement (default: base).
    """

    def __init__(
        self,
        pair: str,
        maturity: date,
        contracted_rate: float,
        notional: float = 1_000_000.0,
        settlement_currency: str = "base",
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if contracted_rate <= 0:
            raise ValueError(f"contracted_rate must be positive, got {contracted_rate}")

        self.pair = pair
        self.maturity = maturity
        self.contracted_rate = contracted_rate
        self.notional = notional
        self.settlement_currency = settlement_currency

    def forward_rate(
        self,
        spot: float,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """Implied NDF forward rate via CIP: F = S × df_base / df_quote."""
        return spot * base_curve.df(self.maturity) / quote_curve.df(self.maturity)

    def pv(
        self,
        spot: float,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """PV of the NDF (in settlement currency, discounted).

        PV = notional × (F - K) × df_settlement(T)
        For USD-settled NDFs, settlement df = base_curve.df(T).
        """
        fwd = self.forward_rate(spot, base_curve, quote_curve)
        df_settle = base_curve.df(self.maturity)
        return self.notional * (fwd - self.contracted_rate) * df_settle

    def settlement_amount(self, fixing_rate: float) -> float:
        """Cash settlement at fixing.

        For standard NDF (base currency settlement):
            settlement = notional × (fixing_rate - contracted_rate)

        Positive = buyer receives.
        """
        return self.notional * (fixing_rate - self.contracted_rate)

    def fx_delta(
        self,
        spot: float,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
        shift: float = 0.0001,
    ) -> float:
        """Spot delta: PV change per 1-pip spot move."""
        pv_base = self.pv(spot, base_curve, quote_curve)
        return self.pv(spot + shift, base_curve, quote_curve) - pv_base
