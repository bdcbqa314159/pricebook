"""FX swap: near leg + far leg."""

from datetime import date

from pricebook.currency import CurrencyPair
from pricebook.discount_curve import DiscountCurve
from pricebook.fx_forward import FXForward


class FXSwap:
    """
    An FX swap: simultaneous buy/sell of a currency pair at two different dates.

    Near leg: buy base at near_rate on near_date
    Far leg: sell base at far_rate on far_date

    PV = PV(near_leg) + PV(far_leg)

    Swap points = far_rate - near_rate (the market quote for an FX swap).
    """

    def __init__(
        self,
        pair: CurrencyPair,
        near_date: date,
        far_date: date,
        near_rate: float,
        far_rate: float,
        notional: float = 1_000_000.0,
    ):
        if near_date >= far_date:
            raise ValueError(f"near_date ({near_date}) must be before far_date ({far_date})")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if near_rate <= 0 or far_rate <= 0:
            raise ValueError("rates must be positive")

        self.pair = pair
        self.near_date = near_date
        self.far_date = far_date
        self.near_rate = near_rate
        self.far_rate = far_rate
        self.notional = notional

    @property
    def swap_points(self) -> float:
        """Far rate minus near rate."""
        return self.far_rate - self.near_rate

    def pv(
        self,
        spot: float,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """
        PV of the FX swap (in quote currency).

        Near leg: buy base at near_rate -> PV = notional * (F_near - near_rate) * df_quote(near)
        Far leg: sell base at far_rate -> PV = -notional * (F_far - far_rate) * df_quote(far)
        """
        fwd_near = FXForward.forward_rate(spot, self.near_date, base_curve, quote_curve)
        fwd_far = FXForward.forward_rate(spot, self.far_date, base_curve, quote_curve)
        df_near = quote_curve.df(self.near_date)
        df_far = quote_curve.df(self.far_date)

        pv_near = self.notional * (fwd_near - self.near_rate) * df_near
        pv_far = -self.notional * (fwd_far - self.far_rate) * df_far

        return pv_near + pv_far

    @staticmethod
    def fair_swap_points(
        spot: float,
        near_date: date,
        far_date: date,
        base_curve: DiscountCurve,
        quote_curve: DiscountCurve,
    ) -> float:
        """Fair swap points: difference between far and near forward rates."""
        fwd_near = FXForward.forward_rate(spot, near_date, base_curve, quote_curve)
        fwd_far = FXForward.forward_rate(spot, far_date, base_curve, quote_curve)
        return fwd_far - fwd_near
