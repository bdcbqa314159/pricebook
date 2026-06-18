"""Canadian fixed income derivatives.

CORRA swaps, CGB bonds, Canadian IRS, provincial bonds,
Real Return Bonds (RRBs), and breakeven inflation.

    from pricebook.fixed_income.canadian import (
        CORRASwap, CGBBond, CanadianIRS, ProvincialBond, RRBBond,
        build_corra_curve, breakeven_inflation_ca,
    )

Conventions:
- Day count: ACT/365 for all CAD instruments
- CORRA: Canadian Overnight Repo Rate Average (BOC)
- CGB: Canadian Government Bond, semi-annual, ACT/365F, T+1
- Provincial: same conventions as CGB, spread over federal curve
- RRB: Real Return Bond (CPI_CA-linked, 3-month lag, deflation floor)

References:
    BOC (2024). Bank of Canada — CORRA.
    CSA (2024). Canadian Securities Administrators.
    IIROC (2024). Investment Industry Regulatory Organization of Canada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def synthetic_corra_strip(reference_date: date, corra: float = 0.0425,
                           n: int = 10, slope_bp: float = 5.0) -> list[dict]:
    """Synthetic CORRA swap strip. Canada: ~4.25% overnight rate."""
    cal = get_calendar("CAD")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = corra + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_corra_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_365_FIXED, InterpolationMethod.LOG_LINEAR)


@dataclass
class CORRASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class CORRASwap:
    """Canadian CORRA overnight swap. ACT/365."""
    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 10_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> CORRASwapResult:
        tau = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        df_s, df_e = curve.df(self.start), curve.df(self.end)
        float_pv = df_s - df_e
        fixed_pv = self.fixed_rate * tau * df_e
        pv = self.direction * self.notional * (fixed_pv - float_pv)
        par = float_pv / (tau * df_e) if tau * df_e > 0 else 0
        pv_up = self.direction * self.notional * ((self.fixed_rate + 0.0001) * tau * df_e - float_pv)
        return CORRASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "corra_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


@dataclass
class RRBResult:
    real_price: float; nominal_price: float; cpi_ratio: float; real_yield: float
    def to_dict(self) -> dict: return dict(vars(self))


class RRBBond:
    """Canadian Real Return Bond (RRB). CPI_CA-linked, 3-month lag.

    Like US TIPS: principal indexed to CPI. Semi-annual real coupon.
    Deflation floor: principal protected at par.
    """
    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100.0):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_cpi: float) -> RRBResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        cpi_ratio = max(current_cpi / self.base_cpi, 1.0)  # deflation floor

        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i],
                  DayCountConvention.ACT_365_FIXED) * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return RRBResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "rrb", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


# ═══════════════════════════════════════════════════════════════
# CGB (Canadian Government Bond)
# ═══════════════════════════════════════════════════════════════


class CGBBond:
    """Canadian Government Bond. Semi-annual coupon, ACT/365F, T+1.

    Standard benchmark tenors: 2Y, 5Y, 10Y, 30Y.
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        dc = DayCountConvention.ACT_365_FIXED
        pv = sum(self.face * self.coupon * year_fraction(schedule[i-1], schedule[i], dc)
                 * curve.df(schedule[i]) for i in range(1, len(schedule)))
        return pv + self.face * curve.df(self.maturity)

    def yield_to_maturity(self, market_price: float, ref: date) -> float:
        """Solve for yield that reprices the bond."""
        from scipy.optimize import brentq
        from pricebook.core.interpolation import InterpolationMethod

        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        if T <= 0:
            return 0.0

        def _price_at_yield(y):
            dates = [self.maturity]
            dfs = [math.exp(-y * T)]
            trial = DiscountCurve(ref, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
            return self.dirty_price(trial) - market_price

        try:
            return brentq(_price_at_yield, -0.05, 0.30)
        except ValueError:
            return 0.0

    def to_dict(self) -> dict:
        return {"type": "cgb", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


def synthetic_cgb_strip(reference_date: date) -> list[dict]:
    """Synthetic CGB benchmark quotes (realistic Nov 2024)."""
    from dateutil.relativedelta import relativedelta
    return [
        {"tenor": "2Y", "maturity": reference_date + relativedelta(years=2),
         "coupon": 0.0375, "price": 99.2},
        {"tenor": "5Y", "maturity": reference_date + relativedelta(years=5),
         "coupon": 0.035, "price": 97.5},
        {"tenor": "10Y", "maturity": reference_date + relativedelta(years=10),
         "coupon": 0.0325, "price": 94.0},
        {"tenor": "30Y", "maturity": reference_date + relativedelta(years=30),
         "coupon": 0.03, "price": 82.0},
    ]


# ═══════════════════════════════════════════════════════════════
# Canadian IRS (Interest Rate Swap)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CanadianIRSResult:
    pv: float; fixed_rate: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class CanadianIRS:
    """Canadian interest rate swap. Fixed semi-annual ACT/365F vs CORRA compound.

    The standard CAD IRS pays fixed semi-annually and receives
    compounded CORRA (overnight) on the floating leg.

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: annual fixed rate.
        notional: notional amount in CAD.
        direction: +1 = pay fixed, -1 = receive fixed.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 10_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> CanadianIRSResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        dc = DayCountConvention.ACT_365_FIXED
        schedule = generate_schedule(self.start, self.end, Frequency.SEMI_ANNUAL)

        # Fixed leg: Σ fixed_rate × τ × df(t_i)
        fixed_pv = sum(self.fixed_rate * year_fraction(schedule[i-1], schedule[i], dc)
                       * curve.df(schedule[i]) for i in range(1, len(schedule)))

        # Floating (CORRA compound): telescoping = df(start) - df(end)
        float_pv = curve.df(self.start) - curve.df(self.end)

        pv = self.direction * self.notional * (fixed_pv - float_pv)

        # Par rate
        annuity = sum(year_fraction(schedule[i-1], schedule[i], dc)
                      * curve.df(schedule[i]) for i in range(1, len(schedule)))
        par = float_pv / annuity if annuity > 0 else 0

        # DV01
        pv_up = self.direction * self.notional * (
            (self.fixed_rate + 0.0001) * annuity - float_pv)
        dv01 = abs(pv_up - pv)

        return CanadianIRSResult(pv, self.fixed_rate, par, dv01, self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "cad_irs", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# Provincial Bonds
# ═══════════════════════════════════════════════════════════════

# Provincial spread over federal CGB (typical Nov 2024, bp)
PROVINCIAL_SPREADS = {
    "ON": 35,   # Ontario
    "QC": 40,   # Quebec
    "BC": 25,   # British Columbia
    "AB": 30,   # Alberta
    "MB": 45,   # Manitoba
    "SK": 40,   # Saskatchewan
}


class ProvincialBond:
    """Canadian provincial bond. Semi-annual ACT/365F, spread over federal.

    Provincial bonds are obligations of provincial governments,
    not backed by the federal government. They trade at a spread
    over CGB reflecting provincial credit risk.

    Args:
        province: 2-letter code (ON, QC, BC, AB, etc.).
        issue_date, maturity: bond dates.
        coupon: annual coupon rate.
        face: face value.
    """

    def __init__(self, province: str, issue_date: date, maturity: date,
                 coupon: float, face: float = 100):
        self.province = province.upper()
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face
        self.spread_bp = PROVINCIAL_SPREADS.get(self.province, 40)

    def dirty_price(self, federal_curve: DiscountCurve) -> float:
        """Price using federal CGB curve + provincial spread."""
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        dc = DayCountConvention.ACT_365_FIXED
        spread = self.spread_bp / 10_000

        pv = 0.0
        for i in range(1, len(schedule)):
            tau = year_fraction(schedule[i-1], schedule[i], dc)
            t = year_fraction(self.issue_date, schedule[i], dc)
            df = federal_curve.df(schedule[i]) * math.exp(-spread * t)
            pv += self.face * self.coupon * tau * df

        t_mat = year_fraction(self.issue_date, self.maturity, dc)
        pv += self.face * federal_curve.df(self.maturity) * math.exp(-spread * t_mat)
        return pv

    def spread_duration(self, federal_curve: DiscountCurve) -> float:
        """dP/d(spread) per 1bp."""
        base = self.dirty_price(federal_curve)
        bump = self.spread_bp + 1
        saved = self.spread_bp
        self.spread_bp = bump
        bumped = self.dirty_price(federal_curve)
        self.spread_bp = saved
        return base - bumped  # positive: price falls when spread widens

    def to_dict(self) -> dict:
        return {"type": "provincial", "province": self.province,
                "maturity": self.maturity.isoformat(), "coupon": self.coupon,
                "spread_bp": self.spread_bp}


# ═══════════════════════════════════════════════════════════════
# Breakeven Inflation
# ═══════════════════════════════════════════════════════════════

def breakeven_inflation_ca(
    corra_curve: DiscountCurve,
    rrb_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """Canadian breakeven inflation from CORRA nominal vs RRB real curves.

    BEI ≈ nominal_rate - real_rate.

    Args:
        corra_curve: nominal CORRA discount curve.
        rrb_curve: real (RRB) discount curve.
        maturities_years: horizons to compute BEI.

    Returns:
        List of dicts with years, nominal_rate, real_rate, bei.
    """
    from dateutil.relativedelta import relativedelta

    if maturities_years is None:
        maturities_years = [2, 5, 10, 20, 30]

    ref = reference_date or corra_curve.reference_date
    results = []

    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom = corra_curve.df(mat)
        df_real = rrb_curve.df(mat)

        if df_nom > 0 and df_real > 0 and T > 0:
            nom_rate = -math.log(df_nom) / T
            real_rate = -math.log(df_real) / T
            bei = nom_rate - real_rate
        else:
            nom_rate = real_rate = bei = 0.0

        results.append({"years": T, "nominal_rate": nom_rate,
                         "real_rate": real_rate, "bei": bei})

    return results
