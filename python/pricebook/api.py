"""Unified API — price anything in 5 lines.

The entry point for pricebook. Wraps 300+ modules into a clean interface.

    import pricebook.api as pb

    # Build curves
    curve = pb.build_curve("USD", ref, deposits=[(d1, 0.04)], swaps=[(d2, 0.039)])

    # Price products — all return PV (float)
    pb.irs(ref, "5Y", 0.04, curve)
    pb.bond(ref, "10Y", 0.04, curve)
    pb.cds(ref, "5Y", 0.01, curve, surv)
    pb.swaption(ref, "1Y", "5Y", 0.04, vol=0.30, curve=curve)
    pb.cap(ref, "5Y", 0.04, vol=0.30, curve=curve)

    # Rates and analytics — return float
    pb.par_rate(ref, "5Y", curve)
    pb.fx_forward_rate("EUR/USD", 1.10, "6M", eur_curve, usd_curve)
    pb.swap_dv01(ref, "5Y", 0.04, curve)
    pb.bond_duration(ref, "10Y", 0.04, 100.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.bootstrap import bootstrap
from pricebook.capfloor import CapFloor
from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.fra import FRA
from pricebook.fx_forward import FXForward
from pricebook.greeks import Greeks
from pricebook.ois import bootstrap_ois
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.swaption import Swaption, SwaptionType
from pricebook.vol_surface import FlatVol
from pricebook.black76 import OptionType


# ---- Tenor parsing ----

def _parse_tenor(ref: date, tenor: str) -> date:
    """Parse '1Y', '6M', '3M', '10Y' etc. into a date."""
    tenor = tenor.upper().strip()
    if tenor.endswith("Y"):
        return ref + relativedelta(years=int(tenor[:-1]))
    elif tenor.endswith("M"):
        return ref + relativedelta(months=int(tenor[:-1]))
    elif tenor.endswith("W"):
        return ref + relativedelta(weeks=int(tenor[:-1]))
    elif tenor.endswith("D"):
        return ref + relativedelta(days=int(tenor[:-1]))
    raise ValueError(f"Cannot parse tenor '{tenor}'. Use '5Y', '6M', '2W', '30D'.")


# ---- Convention engine ----

@dataclass(frozen=True)
class CcyConventions:
    """Market conventions for a currency."""
    calendar: str
    fixed_dc: DayCountConvention
    float_dc: DayCountConvention
    fixed_freq: Frequency
    float_freq: Frequency
    settlement_days: int


_CCY_CONVENTIONS = {
    "USD": CcyConventions("US", DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 2),
    "EUR": CcyConventions("TARGET", DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
                           Frequency.ANNUAL, Frequency.QUARTERLY, 2),
    "GBP": CcyConventions("UK", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 0),
    "JPY": CcyConventions("JP", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_360,
                           Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL, 2),
    "CHF": CcyConventions("CH", DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
                           Frequency.ANNUAL, Frequency.QUARTERLY, 2),
    "CAD": CcyConventions("CA", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 1),
    "AUD": CcyConventions("AU", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 2),
}

_DEFAULT = _CCY_CONVENTIONS["USD"]


def conventions(ccy: str) -> CcyConventions:
    """Get market conventions for a currency."""
    return _CCY_CONVENTIONS.get(ccy.upper(), _DEFAULT)


# ---- Curve building ----

def build_curve(
    ccy: str,
    reference_date: date,
    deposits: list[tuple[date, float]] | None = None,
    swaps: list[tuple[date, float]] | None = None,
    ois_rates: list[tuple[date, float]] | None = None,
) -> DiscountCurve:
    """Build a discount curve from market quotes.

        curve = pb.build_curve("USD", ref,
                               deposits=[(d1, 0.043)],
                               swaps=[(d5, 0.039)])
    """
    conv = conventions(ccy)
    if ois_rates:
        return bootstrap_ois(reference_date, ois_rates,
                             day_count=conv.float_dc, fixed_frequency=conv.fixed_freq)
    return bootstrap(reference_date, deposits or [], swaps or [],
                     fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
                     fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq)


def build_credit_curve(
    reference_date: date,
    spreads: dict[str, float],
    discount_curve: DiscountCurve,
    recovery: float = 0.4,
) -> SurvivalCurve:
    """Build a credit (survival) curve from CDS par spreads.

        surv = pb.build_credit_curve(ref, {"1Y": 0.005, "5Y": 0.01}, curve)
    """
    cds_inputs = sorted(
        [(_parse_tenor(reference_date, t), s) for t, s in spreads.items()],
        key=lambda x: x[0],
    )
    return bootstrap_credit_curve(reference_date, cds_inputs, discount_curve, recovery=recovery)


# ---- Product pricing (all return PV as float) ----

def irs(
    start: date, tenor: str, fixed_rate: float, curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    ccy: str = "USD", notional: float = 1_000_000.0, direction: str = "payer",
) -> float:
    """PV of a vanilla interest rate swap.

        pv = pb.irs(ref, "5Y", 0.04, curve)
    """
    conv = conventions(ccy)
    d = SwapDirection.PAYER if direction == "payer" else SwapDirection.RECEIVER
    return InterestRateSwap(
        start, _parse_tenor(start, tenor), fixed_rate, direction=d, notional=notional,
        fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq,
        fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
    ).pv(curve, projection_curve)


def par_rate(
    start: date, tenor: str, curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None, ccy: str = "USD",
) -> float:
    """Par swap rate for a given tenor.

        rate = pb.par_rate(ref, "5Y", curve)
    """
    conv = conventions(ccy)
    return InterestRateSwap(
        start, _parse_tenor(start, tenor), 0.0, notional=1.0,
        fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq,
        fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
    ).par_rate(curve, projection_curve)


def swap_dv01(
    start: date, tenor: str, fixed_rate: float, curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None, ccy: str = "USD",
    notional: float = 1_000_000.0,
) -> float:
    """Parallel DV01 of an interest rate swap.

        dv01 = pb.swap_dv01(ref, "5Y", 0.04, curve)
    """
    conv = conventions(ccy)
    return InterestRateSwap(
        start, _parse_tenor(start, tenor), fixed_rate, notional=notional,
        fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq,
        fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
    ).dv01(curve, projection_curve)


def fra(
    start: date, tenor: str, strike: float, curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    notional: float = 1_000_000.0,
) -> float:
    """PV of a forward rate agreement.

        pv = pb.fra(ref, "3M", 0.04, curve)
    """
    end = _parse_tenor(start, tenor)
    return FRA(start, end, strike, notional).pv(curve, projection_curve)


def bond(
    start: date, maturity: date | str, coupon_rate: float, curve: DiscountCurve,
    face_value: float = 100.0,
) -> float:
    """Dirty price of a fixed-rate bond (per 100 face).

        px = pb.bond(issue, "10Y", 0.04, curve)
    """
    if isinstance(maturity, str):
        maturity = _parse_tenor(start, maturity)
    return FixedRateBond(start, maturity, coupon_rate, face_value=face_value).dirty_price(curve)


def bond_duration(
    start: date, maturity: date | str, coupon_rate: float, market_price: float,
    settlement: date | None = None,
) -> float:
    """Modified duration of a fixed-rate bond.

        dur = pb.bond_duration(issue, "10Y", 0.04, 100.0)
    """
    if isinstance(maturity, str):
        maturity = _parse_tenor(start, maturity)
    b = FixedRateBond(start, maturity, coupon_rate)
    ytm = b.yield_to_maturity(market_price, settlement)
    return b.modified_duration(ytm, settlement)


def cds(
    start: date, tenor: str, spread: float,
    discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    recovery: float = 0.4, notional: float = 1_000_000.0,
) -> float:
    """PV of a CDS (protection buyer).

        pv = pb.cds(ref, "5Y", 0.01, curve, surv)
    """
    return CDS(start, _parse_tenor(start, tenor), spread,
               notional=notional, recovery=recovery).pv(discount_curve, survival_curve)


def fx_forward_rate(
    pair: str, spot: float, tenor: str,
    base_curve: DiscountCurve, quote_curve: DiscountCurve,
    reference_date: date | None = None,
) -> float:
    """FX forward rate via CIP.

        fwd = pb.fx_forward_rate("EUR/USD", 1.10, "6M", eur_curve, usd_curve)
    """
    ref = reference_date or base_curve.reference_date
    return FXForward.forward_rate(spot, _parse_tenor(ref, tenor), base_curve, quote_curve)


def swaption(
    reference_date: date, expiry_tenor: str, swap_tenor: str,
    strike: float, vol: float, curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    swaption_type: str = "payer", notional: float = 1_000_000.0,
    return_greeks: bool = False,
) -> float | Greeks:
    """Price a European swaption. Set return_greeks=True for Greeks.

        pv = pb.swaption(ref, "1Y", "5Y", 0.04, 0.30, curve)
        g  = pb.swaption(ref, "1Y", "5Y", 0.04, 0.30, curve, return_greeks=True)
    """
    expiry = _parse_tenor(reference_date, expiry_tenor)
    swap_end = _parse_tenor(expiry, swap_tenor)
    st = SwaptionType.PAYER if swaption_type == "payer" else SwaptionType.RECEIVER
    swpn = Swaption(expiry, swap_end, strike, swaption_type=st, notional=notional)
    if return_greeks:
        return swpn.greeks(curve, FlatVol(vol), projection_curve)
    return swpn.pv(curve, FlatVol(vol), projection_curve)


def cap(
    start: date, tenor: str, strike: float, vol: float, curve: DiscountCurve,
    notional: float = 1_000_000.0,
) -> float:
    """PV of an interest rate cap.

        pv = pb.cap(ref, "5Y", 0.04, 0.30, curve)
    """
    return CapFloor(start, _parse_tenor(start, tenor), strike,
                    OptionType.CALL, notional).pv(curve, FlatVol(vol))


def floor(
    start: date, tenor: str, strike: float, vol: float, curve: DiscountCurve,
    notional: float = 1_000_000.0,
) -> float:
    """PV of an interest rate floor.

        pv = pb.floor(ref, "5Y", 0.03, 0.30, curve)
    """
    return CapFloor(start, _parse_tenor(start, tenor), strike,
                    OptionType.PUT, notional).pv(curve, FlatVol(vol))
