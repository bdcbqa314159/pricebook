"""Funding curve and collateralisation-aware pricing.

FundingCurve = OIS + funding spread (bank's unsecured borrowing cost).
CollateralisedPricer selects the correct discount curve based on CSA terms.

    from pricebook.funding_curve import FundingCurve, CollateralisedPricer

    funding = FundingCurve.flat_spread(ois, 0.005)
    pricer = CollateralisedPricer({"USD": ois}, funding)
    result = pricer.price(trade, ctx, csa=my_csa)

References:
    Piterbarg, V. (2010). Funding beyond discounting.
    Lou, W. (2015). Liability-side pricing of derivatives.
    Henrard, M. (2014). Interest Rate Modelling in the Multi-Curve
    Framework, Ch. 5 — collateral discounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.csa import (
    CSA, CollateralType,
    csa_discount_curve,
    non_cash_collateral_discount_rate,
    NonCashCollateralAsset,
)
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.rfr import SpreadCurve


class FundingCurve:
    """Bank-specific funding curve = OIS + funding spread.

    The funding spread reflects the bank's unsecured borrowing cost
    above the risk-free rate. For uncollateralised trades, this is
    the appropriate discount rate (Piterbarg 2010).

    Args:
        ois_curve: the risk-free OIS discount curve.
        funding_spread_curve: term structure of the funding spread.
    """

    def __init__(
        self,
        ois_curve: DiscountCurve,
        funding_spread_curve: SpreadCurve,
    ):
        self._ois = ois_curve
        self._spread = funding_spread_curve

    @property
    def reference_date(self) -> date:
        return self._ois.reference_date

    @property
    def ois_curve(self) -> DiscountCurve:
        return self._ois

    @property
    def funding_spread_curve(self) -> SpreadCurve:
        return self._spread

    def df(self, d: date) -> float:
        """Discount factor at date d using funding rate = OIS zero + spread.

        Piterbarg (2010) Eq 3: df_funding(T) = df_OIS(T) × exp(-∫₀ᵀ s(u) du)

        For piecewise constant spread (linear interp on pillars), the integral
        is computed exactly by summing over spread segments.
        """
        t = year_fraction(self.reference_date, d, DayCountConvention.ACT_365_FIXED)
        if t <= 0:
            return 1.0
        # Integrate spread over [0, T] using piecewise segments
        integral = self._integrate_spread(d)
        return self._ois.df(d) * math.exp(-integral)

    def _integrate_spread(self, d: date) -> float:
        """Integrate spread(u) du from reference_date to d.

        Uses trapezoidal rule on spread pillar grid for piecewise linear spread.
        """
        from pricebook.day_count import DayCountConvention, year_fraction
        ref = self.reference_date
        t_end = year_fraction(ref, d, DayCountConvention.ACT_365_FIXED)
        if t_end <= 0:
            return 0.0

        # Sample at spread pillars + endpoint
        pillar_times = []
        for pd in self._spread.dates:
            pt = year_fraction(ref, pd, DayCountConvention.ACT_365_FIXED)
            if 0 < pt < t_end:
                pillar_times.append(pt)
        times = sorted(set([0.0] + pillar_times + [t_end]))

        integral = 0.0
        for i in range(len(times) - 1):
            t1, t2 = times[i], times[i + 1]
            mid_t = (t1 + t2) / 2
            from datetime import timedelta
            mid_d = ref + timedelta(days=int(mid_t * 365))
            s_mid = self._spread.spread(mid_d)
            integral += s_mid * (t2 - t1)
        return integral

    def funding_rate(self, d: date) -> float:
        """Continuously compounded funding rate at date d.

        r_funding(T) = r_OIS(T) + s(T)
        """
        return self._ois.zero_rate(d) + self._spread.spread(d)

    def forward_funding_rate(self, d1: date, d2: date) -> float:
        """Simply-compounded forward funding rate between d1 and d2."""
        df1 = self.df(d1)
        df2 = self.df(d2)
        tau = year_fraction(d1, d2, DayCountConvention.ACT_365_FIXED)
        if tau < 1e-10:
            return self.funding_rate(d1)
        return (df1 / df2 - 1.0) / tau

    def as_discount_curve(self) -> DiscountCurve:
        """Convert to a DiscountCurve for use in standard pricing functions.

        Samples the funding curve at the OIS pillar dates plus additional
        points to capture the spread term structure.
        """
        ref = self.reference_date
        # Sample at standard tenors
        tenors_days = [30, 91, 182, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300]
        dates = [ref + timedelta(days=d) for d in tenors_days]
        dfs = [self.df(d) for d in dates]
        return DiscountCurve(ref, dates, dfs)

    def bumped(self, shift: float) -> FundingCurve:
        """Return a FundingCurve with parallel-shifted OIS curve."""
        return FundingCurve(self._ois.bumped(shift), self._spread)

    @classmethod
    def flat_spread(cls, ois_curve: DiscountCurve, spread: float) -> FundingCurve:
        """Convenience: constant funding spread over OIS."""
        sc = SpreadCurve(
            ois_curve.reference_date,
            [ois_curve.reference_date + timedelta(days=1)],
            [spread],
        )
        return cls(ois_curve, sc)


@dataclass
class CollateralisedResult:
    """Result of collateralisation-aware pricing."""
    pv: float
    discount_curve_type: str   # "ois", "funding", "xccy", "non_cash"
    csa_type: str              # "none", "cash_same_ccy", "cash_foreign", "non_cash"
    funding_adjustment: float  # PV(uncollateralised) - PV(collateralised)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "pv": self.pv,
            "discount_curve_type": self.discount_curve_type,
            "csa_type": self.csa_type,
            "funding_adjustment": self.funding_adjustment,
        }


class CollateralisedPricer:
    """Selects discount curve based on CSA terms and prices accordingly.

    Decision tree (Piterbarg 2010, Henrard 2014 §5):
    1. No CSA → discount on FundingCurve (unsecured)
    2. Cash CSA, same currency → discount on OIS
    3. Cash CSA, foreign currency → discount on foreign OIS + xccy basis
    4. Non-cash CSA → Lou 2017 CTD-adjusted rate

    Args:
        ois_curves: {currency: OIS DiscountCurve}.
        funding_curve: bank's unsecured funding curve.
        xccy_basis_curves: {currency_pair: xccy-adjusted DiscountCurve}.
    """

    def __init__(
        self,
        ois_curves: dict[str, DiscountCurve],
        funding_curve: FundingCurve,
        xccy_basis_curves: dict[str, DiscountCurve] | None = None,
    ):
        self._ois = ois_curves
        self._funding = funding_curve
        self._xccy = xccy_basis_curves or {}

    def discount_curve_for(
        self,
        trade_currency: str,
        csa: CSA | None = None,
        collateral_assets: list[NonCashCollateralAsset] | None = None,
    ) -> tuple[DiscountCurve, str]:
        """Select the correct discount curve for the trade.

        Returns:
            (curve, csa_type) tuple.
        """
        if csa is None:
            # Uncollateralised: discount on funding curve
            return self._funding.as_discount_curve(), "none"

        # Check if non-cash collateral
        has_non_cash = any(
            ct != CollateralType.CASH for ct in csa.eligible_collateral
        )

        if has_non_cash and collateral_assets:
            # Lou 2017: cheapest-to-deliver
            result = non_cash_collateral_discount_rate(
                collateral_assets,
                funding_rate=self._funding.funding_rate(
                    self._funding.reference_date + timedelta(days=1825)),
                cash_rate=self._ois.get(trade_currency,
                    self._funding.ois_curve).zero_rate(
                    self._funding.reference_date + timedelta(days=1825)),
            )
            # Build adjusted curve from effective rate
            ref = self._funding.reference_date
            base_ois = self._ois.get(trade_currency, self._funding.ois_curve)
            adj_spread = result.effective_rate - base_ois.zero_rate(
                ref + timedelta(days=1825))
            return FundingCurve.flat_spread(base_ois, adj_spread).as_discount_curve(), "non_cash"

        # Cash collateral
        coll_ccy = csa.currency

        if coll_ccy == trade_currency:
            # Same currency: discount on OIS
            if trade_currency in self._ois:
                return self._ois[trade_currency], "cash_same_ccy"
            return self._funding.ois_curve, "cash_same_ccy"

        # Foreign currency: look for xccy basis curve
        pair_key = f"{trade_currency}_{coll_ccy}"
        alt_key = f"{coll_ccy}_{trade_currency}"
        if pair_key in self._xccy:
            return self._xccy[pair_key], "cash_foreign"
        if alt_key in self._xccy:
            return self._xccy[alt_key], "cash_foreign"

        # Fallback: use collateral currency OIS
        if coll_ccy in self._ois:
            return self._ois[coll_ccy], "cash_foreign"

        # Last resort: OIS in trade currency
        return self._ois.get(trade_currency, self._funding.ois_curve), "cash_same_ccy"

    def price(
        self,
        trade,
        ctx: PricingContext,
        csa: CSA | None = None,
        collateral_assets: list[NonCashCollateralAsset] | None = None,
    ) -> CollateralisedResult:
        """Price a trade with CSA-aware discounting.

        Computes PV under the CSA-appropriate curve, and the funding
        adjustment vs the uncollateralised case.
        """
        trade_ccy = getattr(ctx, "currency", "USD")
        curve, csa_type = self.discount_curve_for(trade_ccy, csa, collateral_assets)

        # Price with the selected curve
        adjusted_ctx = ctx.replace(discount_curve=curve)
        if hasattr(trade, "pv_ctx"):
            pv = trade.pv_ctx(adjusted_ctx)
        elif hasattr(trade, "pv"):
            pv = trade.pv(adjusted_ctx)
        else:
            raise ValueError(f"Trade {type(trade).__name__} has no pv or pv_ctx method")

        # Compute funding adjustment: PV(funding) - PV(CSA)
        funding_curve_dc = self._funding.as_discount_curve()
        funding_ctx = ctx.replace(discount_curve=funding_curve_dc)
        if hasattr(trade, "pv_ctx"):
            pv_funding = trade.pv_ctx(funding_ctx)
        else:
            pv_funding = trade.pv(funding_ctx)

        funding_adj = pv_funding - pv

        return CollateralisedResult(
            pv=pv,
            discount_curve_type=csa_type if csa_type != "none" else "funding",
            csa_type=csa_type,
            funding_adjustment=funding_adj,
        )
