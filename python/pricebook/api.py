"""Unified API — price anything in 5 lines.

The entry point for pricebook. Wraps 300+ modules into a clean interface.

    import pricebook.api as pb

    # Build curves
    curves = pb.curves("USD", ref, deposits=[(d1, 0.04)], swaps=[(d2, 0.039)])

    # Price products
    swap_pv = pb.irs(ref, "5Y", 0.04, curves=curves)
    bond_px = pb.bond(issue, maturity, 0.04, curves=curves)
    cds_pv  = pb.cds(ref, "5Y", 0.01, curves=curves, recovery=0.4)
    fwd     = pb.fx_forward("EUR/USD", 1.10, "6M", curves=curves)

    # Greeks
    g = pb.swaption_greeks(ref, "1Y", "5Y", 0.04, vol=0.30, curves=curves)
"""

from __future__ import annotations

import math
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Any

from pricebook.bond import FixedRateBond
from pricebook.bootstrap import bootstrap
from pricebook.calendar import USSettlementCalendar, Calendar
from pricebook.capfloor import CapFloor
from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.commodity import CommodityForwardCurve, CommoditySwap
from pricebook.currency import Currency, CurrencyPair
from pricebook.curve_builder import build_curves
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.equity_forward import EquityForward
from pricebook.fra import FRA
from pricebook.frn import FloatingRateNote
from pricebook.fx_forward import FXForward
from pricebook.greeks import Greeks
from pricebook.ois import OISSwap, bootstrap_ois
from pricebook.rate_index import get_rate_index, RateIndex
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.swaption import Swaption, SwaptionType
from pricebook.vol_surface import FlatVol


# ---- Tenor parsing ----

def _parse_tenor(ref: date, tenor: str) -> date:
    """Parse '1Y', '6M', '3M', '10Y' etc. into a date."""
    tenor = tenor.upper().strip()
    if tenor.endswith("Y"):
        years = int(tenor[:-1])
        return ref + relativedelta(years=years)
    elif tenor.endswith("M"):
        months = int(tenor[:-1])
        return ref + relativedelta(months=months)
    elif tenor.endswith("W"):
        weeks = int(tenor[:-1])
        return ref + relativedelta(weeks=weeks)
    elif tenor.endswith("D"):
        days = int(tenor[:-1])
        return ref + relativedelta(days=days)
    raise ValueError(f"Cannot parse tenor '{tenor}'. Use '5Y', '6M', '2W', '30D'.")


# ---- Convention engine ----

_CCY_CONVENTIONS = {
    "USD": {"calendar": "US", "fixed_dc": DayCountConvention.THIRTY_360,
            "float_dc": DayCountConvention.ACT_360, "fixed_freq": Frequency.SEMI_ANNUAL,
            "float_freq": Frequency.QUARTERLY, "settlement": 2},
    "EUR": {"calendar": "TARGET", "fixed_dc": DayCountConvention.THIRTY_360,
            "float_dc": DayCountConvention.ACT_360, "fixed_freq": Frequency.ANNUAL,
            "float_freq": Frequency.QUARTERLY, "settlement": 2},
    "GBP": {"calendar": "UK", "fixed_dc": DayCountConvention.ACT_365_FIXED,
            "float_dc": DayCountConvention.ACT_365_FIXED, "fixed_freq": Frequency.SEMI_ANNUAL,
            "float_freq": Frequency.QUARTERLY, "settlement": 0},
    "JPY": {"calendar": "JP", "fixed_dc": DayCountConvention.ACT_365_FIXED,
            "float_dc": DayCountConvention.ACT_360, "fixed_freq": Frequency.SEMI_ANNUAL,
            "float_freq": Frequency.SEMI_ANNUAL, "settlement": 2},
}


def conventions(ccy: str) -> dict:
    """Get market conventions for a currency."""
    return _CCY_CONVENTIONS.get(ccy.upper(), _CCY_CONVENTIONS["USD"])


# ---- Curve building ----

def curves(
    ccy: str,
    reference_date: date,
    deposits: list[tuple[date, float]] | None = None,
    swaps: list[tuple[date, float]] | None = None,
    ois_rates: list[tuple[date, float]] | None = None,
) -> DiscountCurve:
    """Build a discount curve from market quotes.

    Uses bootstrap for deposits+swaps, or OIS bootstrap for OIS rates.

        curve = pb.curves("USD", date(2026,4,21),
                          deposits=[(d1, 0.043)],
                          swaps=[(d5, 0.039), (d10, 0.040)])
    """
    conv = conventions(ccy)

    if ois_rates:
        return bootstrap_ois(reference_date, ois_rates,
                             day_count=conv["float_dc"],
                             fixed_frequency=conv["fixed_freq"])

    deps = deposits or []
    swps = swaps or []
    return bootstrap(reference_date, deps, swps,
                     fixed_day_count=conv["fixed_dc"],
                     float_day_count=conv["float_dc"],
                     fixed_frequency=conv["fixed_freq"],
                     float_frequency=conv["float_freq"])


def credit_curve(
    reference_date: date,
    spreads: dict[str, float],
    discount_curve: DiscountCurve,
    recovery: float = 0.4,
) -> SurvivalCurve:
    """Build a credit (survival) curve from CDS par spreads.

        surv = pb.credit_curve(ref, {"1Y": 0.005, "5Y": 0.01}, curve)
    """
    cds_inputs = []
    for tenor, spread in sorted(spreads.items(), key=lambda x: _parse_tenor(reference_date, x[0])):
        mat = _parse_tenor(reference_date, tenor)
        cds_inputs.append((mat, spread))
    return bootstrap_credit_curve(reference_date, cds_inputs, discount_curve, recovery=recovery)


# ---- Product pricing ----

def irs(
    start: date,
    tenor: str,
    fixed_rate: float,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    ccy: str = "USD",
    notional: float = 1_000_000.0,
    direction: str = "payer",
) -> float:
    """Price a vanilla interest rate swap.

        pv = pb.irs(ref, "5Y", 0.04, curve=curve)
    """
    conv = conventions(ccy)
    end = _parse_tenor(start, tenor)
    d = SwapDirection.PAYER if direction == "payer" else SwapDirection.RECEIVER
    swap = InterestRateSwap(start, end, fixed_rate, direction=d, notional=notional,
                            fixed_frequency=conv["fixed_freq"],
                            float_frequency=conv["float_freq"],
                            fixed_day_count=conv["fixed_dc"],
                            float_day_count=conv["float_dc"])
    return swap.pv(curve, projection_curve)


def par_rate(
    start: date,
    tenor: str,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    ccy: str = "USD",
) -> float:
    """Par swap rate for a given tenor.

        rate = pb.par_rate(ref, "5Y", curve)
    """
    conv = conventions(ccy)
    end = _parse_tenor(start, tenor)
    swap = InterestRateSwap(start, end, 0.0, notional=1.0,
                            fixed_frequency=conv["fixed_freq"],
                            float_frequency=conv["float_freq"],
                            fixed_day_count=conv["fixed_dc"],
                            float_day_count=conv["float_dc"])
    return swap.par_rate(curve, projection_curve)


def bond(
    issue_date: date,
    maturity: date | str,
    coupon_rate: float,
    curve: DiscountCurve,
    face_value: float = 100.0,
) -> float:
    """Price a fixed-rate bond (dirty price per 100).

        px = pb.bond(issue, date(2036,1,15), 0.04, curve)
    """
    if isinstance(maturity, str):
        maturity = _parse_tenor(issue_date, maturity)
    b = FixedRateBond(issue_date, maturity, coupon_rate, face_value=face_value)
    return b.dirty_price(curve)


def cds(
    start: date,
    tenor: str,
    spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve | None = None,
    recovery: float = 0.4,
    notional: float = 1_000_000.0,
) -> float:
    """Price a CDS (protection buyer PV).

        pv = pb.cds(ref, "5Y", 0.01, curve, surv)
    """
    end = _parse_tenor(start, tenor)
    c = CDS(start, end, spread, notional=notional, recovery=recovery)
    if survival_curve is None:
        # At par: PV = 0
        return 0.0
    return c.pv(discount_curve, survival_curve)


def fx_forward(
    pair: str,
    spot: float,
    tenor: str,
    base_curve: DiscountCurve,
    quote_curve: DiscountCurve,
    reference_date: date | None = None,
) -> float:
    """FX forward rate via CIP.

        fwd = pb.fx_forward("EUR/USD", 1.10, "6M", eur_curve, usd_curve)
    """
    if reference_date is None:
        reference_date = base_curve.reference_date
    maturity = _parse_tenor(reference_date, tenor)
    return FXForward.forward_rate(spot, maturity, base_curve, quote_curve)


def swaption(
    reference_date: date,
    expiry_tenor: str,
    swap_tenor: str,
    strike: float,
    vol: float,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    swaption_type: str = "payer",
    notional: float = 1_000_000.0,
) -> float:
    """Price a European swaption.

        pv = pb.swaption(ref, "1Y", "5Y", 0.04, 0.30, curve)
    """
    expiry = _parse_tenor(reference_date, expiry_tenor)
    swap_end = _parse_tenor(expiry, swap_tenor)
    st = SwaptionType.PAYER if swaption_type == "payer" else SwaptionType.RECEIVER
    swpn = Swaption(expiry, swap_end, strike, swaption_type=st, notional=notional)
    return swpn.pv(curve, FlatVol(vol), projection_curve)


def swaption_greeks(
    reference_date: date,
    expiry_tenor: str,
    swap_tenor: str,
    strike: float,
    vol: float,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    swaption_type: str = "payer",
    notional: float = 1_000_000.0,
) -> Greeks:
    """Swaption Greeks (delta, gamma, vega).

        g = pb.swaption_greeks(ref, "1Y", "5Y", 0.04, 0.30, curve)
    """
    expiry = _parse_tenor(reference_date, expiry_tenor)
    swap_end = _parse_tenor(expiry, swap_tenor)
    st = SwaptionType.PAYER if swaption_type == "payer" else SwaptionType.RECEIVER
    swpn = Swaption(expiry, swap_end, strike, swaption_type=st, notional=notional)
    return swpn.greeks(curve, FlatVol(vol), projection_curve)


def cap(
    start: date,
    tenor: str,
    strike: float,
    vol: float,
    curve: DiscountCurve,
    notional: float = 1_000_000.0,
) -> float:
    """Price an interest rate cap.

        pv = pb.cap(ref, "5Y", 0.04, 0.30, curve)
    """
    from pricebook.black76 import OptionType
    end = _parse_tenor(start, tenor)
    c = CapFloor(start, end, strike, OptionType.CALL, notional)
    return c.pv(curve, FlatVol(vol))


def floor(
    start: date,
    tenor: str,
    strike: float,
    vol: float,
    curve: DiscountCurve,
    notional: float = 1_000_000.0,
) -> float:
    """Price an interest rate floor.

        pv = pb.floor(ref, "5Y", 0.03, 0.30, curve)
    """
    from pricebook.black76 import OptionType
    end = _parse_tenor(start, tenor)
    c = CapFloor(start, end, strike, OptionType.PUT, notional)
    return c.pv(curve, FlatVol(vol))
