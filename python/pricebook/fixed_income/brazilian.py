"""Brazilian fixed income derivatives.

DI futures, DI swaps (Pré × CDI), LFT (CDI-linked floating sovereign),
cupom cambial, and FRA de cupom. All using BUS/252 day count convention.

    from pricebook.fixed_income.brazilian import (
        DIFuture, DISwap, LFTBond, CupomCambial, FRACupom,
        build_cdi_curve_from_di, synthetic_di_strip,
    )

Conventions:
- Day count: BUS/252 (business days in São Paulo calendar / 252)
- Compounding: discrete annual (1+r)^(bd/252)
- Face value: R$ 1,000 for bonds, R$ 100,000 for DI futures
- CDI: overnight rate, compounded daily

References:
    B3 (2024). DI Futures Contract Specifications.
    Tesouro Nacional (2024). Títulos Públicos — Metodologia de Cálculo.
    Securato (2008). Cálculo Financeiro das Tesourarias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _bus_days(start: date, end: date) -> int:
    """Count business days between two dates using São Paulo calendar."""
    cal = get_calendar("BRL")
    count = 0
    d = start + timedelta(days=1)
    while d <= end:
        if cal.is_business_day(d):
            count += 1
        d += timedelta(days=1)
    return count


def _di_discount_factor(rate: float, bus_days: int) -> float:
    """Discount factor from DI rate: df = 1 / (1 + rate)^(bd/252)."""
    if bus_days <= 0:
        return 1.0
    return 1.0 / (1 + rate) ** (bus_days / 252.0)


def _di_rate_from_df(df: float, bus_days: int) -> float:
    """Implied DI rate from discount factor: rate = df^(-252/bd) - 1."""
    if bus_days <= 0 or df <= 0:
        return 0.0
    return df ** (-252.0 / bus_days) - 1


# ═══════════════════════════════════════════════════════════════
# Synthetic market data
# ═══════════════════════════════════════════════════════════════

def synthetic_di_strip(
    reference_date: date,
    selic: float = 0.1050,
    n_contracts: int = 15,
    slope_bp_per_year: float = 30.0,
) -> list[dict]:
    """Generate realistic DI futures strip.

    Produces contracts maturing on first business day of each
    month/quarter, with rates reflecting typical Brazilian
    upward-sloping term structure.

    Args:
        reference_date: pricing date.
        selic: current Selic target rate.
        n_contracts: number of contracts to generate.
        slope_bp_per_year: term premium per year in basis points.

    Returns:
        List of {"maturity": date, "rate": float, "bus_days": int, "pu": float}.
    """
    contracts = []
    cal = get_calendar("BRL")

    for i in range(1, n_contracts + 1):
        # Maturity: first business day of month i from now
        mat_month = reference_date.month + i * 2  # every 2 months for liquid contracts
        mat_year = reference_date.year + (mat_month - 1) // 12
        mat_month = ((mat_month - 1) % 12) + 1
        mat = date(mat_year, mat_month, 1)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)

        bd = _bus_days(reference_date, mat)
        # Rate: Selic + term premium
        years_to_mat = bd / 252.0
        rate = selic + slope_bp_per_year / 10_000 * years_to_mat
        pu = 100_000 * _di_discount_factor(rate, bd)

        contracts.append({
            "maturity": mat,
            "rate": rate,
            "bus_days": bd,
            "pu": pu,
            "years": years_to_mat,
        })

    return contracts


# ═══════════════════════════════════════════════════════════════
# CDI Discount Curve
# ═══════════════════════════════════════════════════════════════

def build_cdi_curve_from_di(
    reference_date: date,
    di_strip: list[dict],
    interpolation=None,
) -> DiscountCurve:
    """Build CDI discount curve from DI futures strip.

    Args:
        reference_date: curve date.
        di_strip: list of {"maturity": date, "rate": float} dicts.
        interpolation: interpolation method (default log-linear).

    Returns:
        DiscountCurve with CDI discount factors.
    """
    from pricebook.core.interpolation import InterpolationMethod
    interp = interpolation or InterpolationMethod.LOG_LINEAR

    pillar_dates = []
    pillar_dfs = []

    for contract in sorted(di_strip, key=lambda c: c["maturity"]):
        mat = contract["maturity"]
        rate = contract["rate"]
        bd = contract.get("bus_days", _bus_days(reference_date, mat))
        df = _di_discount_factor(rate, bd)

        pillar_dates.append(mat)
        pillar_dfs.append(df)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.BUS_252,
        interpolation=interp,
    )


# ═══════════════════════════════════════════════════════════════
# DI Futures
# ═══════════════════════════════════════════════════════════════

@dataclass
class DIFutureResult:
    """DI futures pricing result."""
    pu: float               # unit price (PU)
    rate: float              # implied DI rate
    bus_days: int
    dv01: float              # BRL change per 1bp
    convexity: float
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


class DIFuture:
    """Brazilian DI futures contract (B3).

    PU = 100,000 / (1 + DI_rate)^(bus_days / 252)

    The contract settles daily against CDI. At maturity PU → 100,000.
    Quoted as an annualised rate.

    Args:
        maturity: contract maturity date (first bday of month).
        rate: quoted DI rate.
        notional: contract size (default R$ 100,000 per contract).
    """

    def __init__(self, maturity: date, rate: float, notional: float = 100_000.0):
        self.maturity = maturity
        self.rate = rate
        self.notional = notional

    def pu(self, reference_date: date) -> float:
        """Unit price (PU) at reference date."""
        bd = _bus_days(reference_date, self.maturity)
        return self.notional * _di_discount_factor(self.rate, bd)

    def price(self, reference_date: date) -> DIFutureResult:
        """Full pricing result."""
        bd = _bus_days(reference_date, self.maturity)
        pu = self.notional * _di_discount_factor(self.rate, bd)

        # DV01: dPU/drate per 1bp
        rate_up = self.rate + 0.0001
        rate_dn = self.rate - 0.0001
        pu_up = self.notional * _di_discount_factor(rate_up, bd)
        pu_dn = self.notional * _di_discount_factor(rate_dn, bd)
        dv01 = (pu_dn - pu_up) / 2  # positive: rate up → PU down

        # Convexity
        convexity = (pu_up + pu_dn - 2 * pu) / (0.0001 ** 2)

        return DIFutureResult(pu, self.rate, bd, dv01, convexity, self.notional)

    def implied_rate(self, pu: float, reference_date: date) -> float:
        """Back out DI rate from PU."""
        bd = _bus_days(reference_date, self.maturity)
        df = pu / self.notional
        return _di_rate_from_df(df, bd)

    def pv_ctx(self, ctx) -> float:
        """PV using PricingContext."""
        return self.pu(ctx.valuation_date) - self.notional

    def to_dict(self) -> dict:
        return {"type": "di_future", "maturity": self.maturity.isoformat(),
                "rate": self.rate, "notional": self.notional}


# ═══════════════════════════════════════════════════════════════
# DI Swap (Pré × CDI)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DISwapResult:
    """DI swap pricing result."""
    pv: float
    fixed_factor: float
    float_factor: float
    par_rate: float
    dv01: float
    bus_days: int
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


class DISwap:
    """Brazilian DI swap: fixed rate (Pré) vs CDI compounded.

    Fixed leg: (1 + pré_rate)^(bus_days/252) - 1
    Float leg: Π(1 + CDI_daily) - 1 (accrued CDI)

    PV = notional × [(1+pré)^(bd/252) - projected_float] × df

    At inception, par rate = DI rate for the maturity.

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: annual fixed rate (pré).
        notional: swap notional in BRL.
        direction: +1 = pay fixed / receive CDI (most common).
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_rate: float,
        notional: float = 10_000_000.0,
        direction: int = 1,
    ):
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.direction = direction

    def price(self, cdi_curve: DiscountCurve) -> DISwapResult:
        """Price the DI swap using CDI discount curve.

        In the CDI curve world:
        - Float leg = par (CDI compounds to exactly the DF)
        - Fixed leg = (1+pré)^(bd/252)
        - PV = direction × notional × [fixed_factor - 1/df(T)] × df(T)
        Wait, more precisely:
        - Float factor at T = 1/df(T) (CDI accumulation = inverse of DF)
        - Fixed factor at T = (1+pré)^(bd/252)
        - PV = direction × notional × (fixed_factor - float_factor) × df(T)
        But df(T) is from CDI curve, so float_factor × df(T) = 1.
        So PV = direction × notional × [fixed_factor × df(T) - 1]
        """
        bd = _bus_days(self.start, self.end)
        if bd <= 0:
            return DISwapResult(0, 1, 1, self.fixed_rate, 0, 0, self.notional)

        fixed_factor = (1 + self.fixed_rate) ** (bd / 252.0)
        df_T = cdi_curve.df(self.end)
        float_factor = 1.0 / df_T if df_T > 0 else 1.0

        # PV from fixed payer's perspective
        pv = self.direction * self.notional * (fixed_factor * df_T - 1.0)

        # Par rate: rate that makes PV = 0
        # (1+par)^(bd/252) × df(T) = 1 → par = df(T)^(-252/bd) - 1
        par_rate = _di_rate_from_df(df_T, bd)

        # DV01
        bd_plus = bd  # same bus days
        rate_up = self.fixed_rate + 0.0001
        pv_up = self.direction * self.notional * ((1 + rate_up) ** (bd / 252.0) * df_T - 1.0)
        dv01 = abs(pv_up - pv)

        return DISwapResult(pv, fixed_factor, float_factor, par_rate, dv01, bd, self.notional)

    def pv_ctx(self, ctx) -> float:
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        return self.price(curve).pv

    def to_dict(self) -> dict:
        return {"type": "di_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate,
                "notional": self.notional, "direction": self.direction}


# ═══════════════════════════════════════════════════════════════
# LFT (Letra Financeira do Tesouro) — CDI-linked floating
# ═══════════════════════════════════════════════════════════════

@dataclass
class LFTResult:
    """LFT pricing result."""
    dirty_price: float       # per R$ 1,000 face
    accrued_vna: float       # VNA (Valor Nominal Atualizado)
    spread_duration: float
    spread_dv01: float

    def to_dict(self) -> dict:
        return vars(self)


class LFTBond:
    """LFT (Letra Financeira do Tesouro) — CDI-linked sovereign bond.

    The face value (VNA) accrues daily at the CDI rate:
        VNA(t) = 1000 × Π(1 + CDI_i)^(1/252)

    Traded at a spread (ágio/deságio) over CDI:
        Price = VNA(t) / (1 + spread)^(remaining_bd / 252)

    When spread = 0, price = VNA (par). LFTs typically trade at
    tiny spreads (±5bp), making them quasi-cash instruments.

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        spread: annual spread over CDI (e.g. 0.001 = 10bp ágio).
        face_value: nominal face (R$ 1,000 standard).
    """

    def __init__(
        self,
        issue_date: date,
        maturity: date,
        spread: float = 0.0,
        face_value: float = 1000.0,
    ):
        self.issue_date = issue_date
        self.maturity = maturity
        self.spread = spread
        self.face_value = face_value

    def vna(self, reference_date: date, cdi_curve: DiscountCurve) -> float:
        """Valor Nominal Atualizado (updated face value) at reference date.

        VNA = face × (1/df_CDI(issue, ref)) = face × CDI_accrual
        """
        if reference_date <= self.issue_date:
            return self.face_value
        df_issue_to_ref = cdi_curve.df(reference_date) / cdi_curve.df(self.issue_date) \
            if cdi_curve.df(self.issue_date) > 0 else 1.0
        # VNA = face / df = face × accumulation factor
        return self.face_value / max(df_issue_to_ref, 1e-10)

    def price(self, reference_date: date, cdi_curve: DiscountCurve) -> LFTResult:
        """Price the LFT.

        Price = VNA(ref) / (1 + spread)^(remaining_bd / 252)
        """
        vna = self.vna(reference_date, cdi_curve)
        remaining_bd = _bus_days(reference_date, self.maturity)

        if remaining_bd <= 0:
            return LFTResult(vna, vna, 0.0, 0.0)

        # Price = VNA × df_spread
        df_spread = _di_discount_factor(self.spread, remaining_bd) if self.spread != 0 else 1.0
        dirty = vna * df_spread

        # Spread duration and DV01
        years = remaining_bd / 252.0
        spread_duration = years / (1 + self.spread) if self.spread > -1 else years
        spread_dv01 = dirty * spread_duration * 0.0001

        return LFTResult(dirty, vna, spread_duration, spread_dv01)

    def pv_ctx(self, ctx) -> float:
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        return self.price(ctx.valuation_date, curve).dirty_price

    def to_dict(self) -> dict:
        return {"type": "lft_bond", "issue_date": self.issue_date.isoformat(),
                "maturity": self.maturity.isoformat(), "spread": self.spread,
                "face_value": self.face_value}


# ═══════════════════════════════════════════════════════════════
# Cupom Cambial
# ═══════════════════════════════════════════════════════════════

def cupom_cambial(
    fx_spot: float,
    fx_forward: float,
    di_rate: float,
    bus_days: int,
) -> float:
    """Compute cupom cambial (USD rate implied from FX forward).

    cupom = [(F/S) × (1 + DI)^(bd/252)]^(252/bd) - 1

    This is the USD interest rate implied by covered interest parity
    in the BRL market. The cupom cambial is always lower than the DI
    rate (reflecting the interest rate differential BRL vs USD).

    Args:
        fx_spot: USDBRL spot rate.
        fx_forward: USDBRL forward rate.
        di_rate: DI (CDI) rate for the same maturity.
        bus_days: business days to maturity.
    """
    if bus_days <= 0 or fx_spot <= 0:
        return 0.0

    # Forward factor in BRL: (1+DI)^(bd/252)
    brl_factor = (1 + di_rate) ** (bus_days / 252.0)

    # USD factor: F/S × BRL_factor (but F > S means USD rate < BRL rate)
    # Actually: CIP: F/S = BRL_factor / USD_factor
    # → USD_factor = BRL_factor × S / F
    usd_factor = brl_factor * fx_spot / fx_forward

    # Annualise: cupom = USD_factor^(252/bd) - 1
    cupom = usd_factor ** (252.0 / bus_days) - 1

    return cupom


def cupom_cambial_curve(
    reference_date: date,
    fx_spot: float,
    fx_forwards: list[dict],
    di_strip: list[dict],
) -> list[dict]:
    """Build cupom cambial term structure from FX forwards + DI strip.

    Args:
        fx_forwards: [{"maturity": date, "forward": float}, ...]
        di_strip: [{"maturity": date, "rate": float}, ...]

    Returns:
        [{"maturity": date, "cupom": float, "bus_days": int}, ...]
    """
    di_dict = {c["maturity"]: c["rate"] for c in di_strip}
    results = []

    for fwd in sorted(fx_forwards, key=lambda x: x["maturity"]):
        mat = fwd["maturity"]
        fx_fwd = fwd["forward"]
        bd = _bus_days(reference_date, mat)

        # Find closest DI rate
        di_rate = di_dict.get(mat)
        if di_rate is None:
            # Interpolate from nearest
            closest = min(di_dict.keys(), key=lambda d: abs((d - mat).days))
            di_rate = di_dict[closest]

        cc = cupom_cambial(fx_spot, fx_fwd, di_rate, bd)
        results.append({
            "maturity": mat,
            "cupom": cc,
            "bus_days": bd,
            "fx_forward": fx_fwd,
            "di_rate": di_rate,
        })

    return results


def breakeven_inflation_br(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """BRL BEI from CDI nominal vs IPCA real curves."""
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [1, 2, 3, 5, 10, 20]
    ref = reference_date or nominal_curve.reference_date
    results = []
    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom, df_real = nominal_curve.df(mat), real_curve.df(mat)
        if df_nom > 0 and df_real > 0 and T > 0:
            nom, real = -math.log(df_nom) / T, -math.log(df_real) / T
            bei = nom - real
        else:
            nom = real = bei = 0.0
        results.append({"years": T, "nominal_rate": nom, "real_rate": real, "bei": bei})
    return results
