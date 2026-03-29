"""FX forward pricing via covered interest rate parity."""

from datetime import date

from pricebook.currency import Currency, CurrencyPair
from pricebook.discount_curve import DiscountCurve


class FXForward:
    """
    An FX forward contract.

    Covered interest rate parity (CIP):
        F = S * df_base / df_quote

    where:
        S = spot rate (units of quote per 1 unit of base)
        df_base = discount factor in the base currency
        df_quote = discount factor in the quote currency

    The forward buyer agrees to buy base currency at rate K at maturity.
    PV (to the buyer) = notional * (F - K) * df_quote(maturity)
    """

    def __init__(
        self,
        pair: CurrencyPair,
        maturity: date,
        strike: float,
        notional: float = 1_000_000.0,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")

        self.pair = pair
        self.maturity = maturity
        self.strike = strike
        self.notional = notional

    @staticmethod
    def forward_rate(
        spot: float,
        maturity: date,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """
        CIP forward rate: F = S * df_base / df_quote.

        base_curve: discount curve for the base currency.
        quote_curve: discount curve for the quote currency.
        """
        df_base = base_curve.df(maturity)
        df_quote = quote_curve.df(maturity)
        return spot * df_base / df_quote

    @staticmethod
    def forward_points(
        spot: float,
        maturity: date,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """Forward points: F - S, typically quoted in pips (multiply by 10000)."""
        fwd = FXForward.forward_rate(spot, maturity, base_curve, quote_curve)
        return fwd - spot

    def pv(
        self,
        spot: float,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """
        PV of the forward contract (in quote currency units).

        PV = notional * (F - K) * df_quote(maturity)
        """
        fwd = self.forward_rate(spot, self.maturity, base_curve, quote_curve)
        df_quote = quote_curve.df(self.maturity)
        return self.notional * (fwd - self.strike) * df_quote
