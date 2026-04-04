"""Cross-currency swap: full notional exchange with floating legs in two currencies."""

from __future__ import annotations

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, generate_schedule


class CrossCurrencySwap:
    """Cross-currency basis swap with notional exchange.

    Structure (from domestic payer's perspective):
    - Initial exchange: pay domestic notional, receive foreign notional at spot
    - Periodic: pay domestic floating + spread, receive foreign floating
    - Final exchange: pay foreign notional, receive domestic notional

    The PV reflects that initial exchange already occurred.
    Future cashflows: spread coupons + remaining floating + final notional exchange.

    Args:
        start: effective date.
        end: maturity date.
        domestic_notional: notional in domestic currency.
        fx_rate: spot FX rate at inception (foreign per domestic).
        domestic_spread: basis spread on domestic leg.
        frequency: payment frequency for both legs.
        domestic_dc: day count for domestic leg.
        foreign_dc: day count for foreign leg.
        mtm_reset: if True, foreign notional resets to market FX each period.
    """

    def __init__(
        self,
        start: date,
        end: date,
        domestic_notional: float,
        fx_rate: float,
        domestic_spread: float = 0.0,
        frequency: Frequency = Frequency.QUARTERLY,
        domestic_dc: DayCountConvention = DayCountConvention.ACT_360,
        foreign_dc: DayCountConvention = DayCountConvention.ACT_360,
        mtm_reset: bool = False,
    ):
        self.start = start
        self.end = end
        self.domestic_notional = domestic_notional
        self.foreign_notional = domestic_notional * fx_rate
        self.fx_rate = fx_rate
        self.domestic_spread = domestic_spread
        self.frequency = frequency
        self.domestic_dc = domestic_dc
        self.foreign_dc = foreign_dc
        self.mtm_reset = mtm_reset

        self.payment_dates = generate_schedule(start, end, frequency)

    def pv(
        self,
        domestic_curve: DiscountCurve,
        foreign_curve: DiscountCurve,
        current_fx: float,
    ) -> float:
        """PV of the swap in domestic currency (from domestic payer's view).

        We receive the foreign leg and pay the domestic leg.

        Domestic leg PV (what we owe):
          - floating coupons (par → telescoping) + spread coupons + final notional
          - PV_dom = N_dom * [df(start) - df(end)] + spread_annuity + N_dom * df(end)
                   = N_dom * df(start) + spread_annuity

        Foreign leg PV (what we receive, converted to domestic):
          - floating coupons (par → telescoping) + final notional
          - PV_for = N_for * [df_for(start) - df_for(end)] + N_for * df_for(end)
                   = N_for * df_for(start)
          - in domestic: PV_for / current_fx

        Net PV = foreign_received - domestic_owed
        """
        N_dom = self.domestic_notional
        N_for = self.foreign_notional

        # Domestic leg: floating telescopes to df(start), plus spread annuity
        dom_floating = N_dom * domestic_curve.df(self.start)
        dom_spread = self._spread_annuity(domestic_curve)
        dom_total = dom_floating + dom_spread

        # Foreign leg: floating telescopes to df_for(start), in foreign currency
        if self.mtm_reset:
            for_total = self._foreign_mtm_pv(foreign_curve, current_fx)
        else:
            for_pv_foreign = N_for * foreign_curve.df(self.start)
            for_total = for_pv_foreign / current_fx

        return for_total - dom_total

    def _annuity(self, curve: DiscountCurve) -> float:
        """PV01 of the domestic leg coupon stream: sum(N * tau * df)."""
        N = self.domestic_notional
        total = 0.0
        for i, d in enumerate(self.payment_dates):
            prev = self.start if i == 0 else self.payment_dates[i - 1]
            tau = year_fraction(prev, d, self.domestic_dc)
            total += N * tau * curve.df(d)
        return total

    def _spread_annuity(self, curve: DiscountCurve) -> float:
        """PV of the basis spread coupon stream."""
        return self.domestic_spread * self._annuity(curve)

    def _foreign_mtm_pv(
        self,
        curve: DiscountCurve,
        current_fx: float,
    ) -> float:
        """Foreign leg with MTM reset, PV in domestic.

        With MTM reset, the foreign notional resets each period to
        N_dom * current_fx. This eliminates FX risk on notionals.
        Each period's notional exchange cancels, leaving only
        the floating coupon stream.
        """
        N_dom = self.domestic_notional
        # With MTM, foreign leg = N_dom * current_fx * df_for(start) / current_fx
        # = N_dom * df_for(start) (FX cancels!)
        return N_dom * curve.df(self.start)

    def par_spread(
        self,
        domestic_curve: DiscountCurve,
        foreign_curve: DiscountCurve,
        current_fx: float,
    ) -> float:
        """Par basis spread that makes PV = 0."""
        # PV = for_total - dom_floating - spread_annuity = 0
        # spread_annuity = for_total - dom_floating
        # spread * sum(N * tau * df) = for_total - dom_floating
        N = self.domestic_notional
        N_for = self.foreign_notional

        dom_floating = N * domestic_curve.df(self.start)

        if self.mtm_reset:
            for_total = self._foreign_mtm_pv(domestic_curve, current_fx)
        else:
            for_total = N_for * foreign_curve.df(self.start) / current_fx

        target = for_total - dom_floating
        annuity_pv01 = self._annuity(domestic_curve)

        if abs(annuity_pv01) < 1e-12:
            return 0.0

        return target / annuity_pv01

    def dv01_domestic(
        self,
        domestic_curve: DiscountCurve,
        foreign_curve: DiscountCurve,
        current_fx: float,
        shift: float = 0.0001,
    ) -> float:
        """Sensitivity to 1bp parallel shift in domestic rates."""
        pv_base = self.pv(domestic_curve, foreign_curve, current_fx)
        bumped = domestic_curve.bumped(shift)
        return self.pv(bumped, foreign_curve, current_fx) - pv_base

    def dv01_foreign(
        self,
        domestic_curve: DiscountCurve,
        foreign_curve: DiscountCurve,
        current_fx: float,
        shift: float = 0.0001,
    ) -> float:
        """Sensitivity to 1bp parallel shift in foreign rates."""
        pv_base = self.pv(domestic_curve, foreign_curve, current_fx)
        bumped = foreign_curve.bumped(shift)
        return self.pv(domestic_curve, bumped, current_fx) - pv_base

    def fx_delta(
        self,
        domestic_curve: DiscountCurve,
        foreign_curve: DiscountCurve,
        current_fx: float,
        shift: float = 0.01,
    ) -> float:
        """Sensitivity to FX rate move."""
        pv_base = self.pv(domestic_curve, foreign_curve, current_fx)
        return self.pv(domestic_curve, foreign_curve, current_fx + shift) - pv_base
