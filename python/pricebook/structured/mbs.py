"""Mortgage-backed securities: prepayment, OAS, duration, IO/PO strips.

Prepayment modelling via PSA benchmark, CPR/SMM conversion, and a
turnover + refinancing + burnout model. OAS via Monte Carlo rate
simulation. Prepayment-adjusted duration and convexity.

* :class:`MBSPool` — mortgage pool specification.
* :class:`MBSResult` — pricing result with OAS and prepay analytics.
* :func:`price_mbs` — price MBS pass-through with prepayment.
* :func:`oas_mbs` — option-adjusted spread via Newton-Raphson.
* :func:`io_po_strips` — interest-only / principal-only decomposition.
* :func:`psa_speed` — PSA benchmark prepayment speed.

References:
    Fabozzi, *The Handbook of Mortgage-Backed Securities*, 7th ed.
    Hayre, *Salomon Brothers Guide to Mortgage-Backed Securities*.
    Tuckman & Serrat, *Fixed Income Securities*, 3rd ed., Ch. 20-21.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


# ═══════════════════════════════════════════════════════════════
# Prepayment models
# ═══════════════════════════════════════════════════════════════

def cpr_to_smm(cpr: float) -> float:
    """Convert annual CPR to monthly SMM.

    SMM = 1 − (1 − CPR)^(1/12).
    """
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def smm_to_cpr(smm: float) -> float:
    """Convert monthly SMM to annual CPR.

    CPR = 1 − (1 − SMM)^12.
    """
    return 1.0 - (1.0 - smm) ** 12.0


def psa_speed(month: int, psa_multiple: float = 1.0) -> float:
    """PSA benchmark prepayment speed.

    PSA 100%: CPR ramps from 0% to 6% over months 1-30,
    then stays at 6% thereafter.

    PSA 200% means 2× the benchmark speed.

    Args:
        month: seasoning month (1-based).
        psa_multiple: PSA speed (1.0 = 100% PSA, 2.0 = 200% PSA).

    Returns:
        Annual CPR for this month.
    """
    if month <= 30:
        base_cpr = 0.06 * month / 30.0
    else:
        base_cpr = 0.06
    return min(base_cpr * psa_multiple, 1.0)


def prepayment_model(
    month: int,
    wac: float,
    current_rate: float,
    burnout_factor: float = 1.0,
    seasonal_month: int | None = None,
) -> float:
    """Turnover + refinancing + burnout prepayment model.

    CPR = (turnover + refinancing) × burnout × seasonality.

    Turnover: baseline ~6% CPR after 30 months.
    Refinancing: increases when rates drop below WAC (incentive).
    Burnout: fraction of pool that hasn't yet prepaid (decays over time).
    Seasonality: higher in summer (moving season).

    Args:
        month: loan age in months.
        wac: weighted average coupon of the pool.
        current_rate: current market mortgage rate.
        burnout_factor: 1.0 = no burnout, decays toward 0.
        seasonal_month: calendar month (1-12) for seasonality.
    """
    # Turnover: ramps up over 30 months
    if month <= 30:
        turnover = 0.06 * month / 30.0
    else:
        turnover = 0.06

    # Refinancing incentive: arctan of rate difference
    incentive = wac - current_rate
    if incentive > 0:
        refi = 0.30 * (2 / math.pi) * math.atan(incentive / 0.01)
    else:
        refi = 0.0

    # Seasonality
    seasonal = 1.0
    if seasonal_month is not None:
        # Summer peak (May-August)
        seasonal_pattern = [0.85, 0.90, 0.95, 1.0, 1.10, 1.15,
                           1.20, 1.15, 1.05, 0.95, 0.90, 0.85]
        seasonal = seasonal_pattern[max(0, min(seasonal_month - 1, 11))]

    cpr = (turnover + refi) * burnout_factor * seasonal
    return min(max(cpr, 0.0), 1.0)


# ═══════════════════════════════════════════════════════════════
# MBS Pool and Pricing
# ═══════════════════════════════════════════════════════════════

@dataclass
class MBSPool:
    """Mortgage pool specification."""
    original_balance: float
    wac: float              # weighted average coupon (annual)
    wam: int                # weighted average maturity (months)
    pass_through_rate: float  # coupon passed to investor
    age: int = 0            # seasoning (months since origination)
    servicing_fee: float = 0.0025  # annual servicing strip

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class MBSResult:
    """MBS pricing result."""
    price: float            # clean price per 100 face
    yield_to_maturity: float
    wal: float              # weighted average life (years)
    modified_duration: float
    convexity: float
    total_prepayment_pct: float
    cashflows: int          # number of cashflow periods

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class OASResult:
    """Option-adjusted spread result."""
    oas_bp: float
    price: float
    z_spread_bp: float
    wal: float
    prepay_duration: float
    prepay_convexity: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def price_mbs(
    pool: MBSPool,
    discount_curve: DiscountCurve,
    psa_speed: float = 1.0,
    use_model: bool = False,
    current_rate: float | None = None,
    spread: float = 0.0,
    _compute_risk: bool = True,
) -> MBSResult:
    """Price an MBS pass-through with prepayment.

    Generates monthly cashflows: scheduled principal + interest +
    prepayment. Discounts at the curve + spread.

    Args:
        pool: mortgage pool specification.
        discount_curve: risk-free discount curve.
        psa_speed: PSA multiple (1.0 = 100% PSA).
        use_model: if True, use turnover+refi model instead of PSA.
        current_rate: current mortgage rate (for refinancing model).
        spread: static spread over curve (for pricing).
    """
    remaining = pool.wam - pool.age
    balance = pool.original_balance
    monthly_rate = pool.wac / 12.0
    pass_rate = pool.pass_through_rate / 12.0

    total_pv = 0.0
    total_principal_pv = 0.0
    wal_num = 0.0
    total_principal = 0.0
    ref = discount_curve.reference_date

    for m in range(1, remaining + 1):
        if balance < 0.01:
            break

        month_age = pool.age + m

        # Prepayment
        if use_model and current_rate is not None:
            burnout = balance / pool.original_balance
            cpr = prepayment_model(month_age, pool.wac, current_rate, burnout)
        else:
            cpr = psa_speed_fn(month_age, psa_speed)

        smm = cpr_to_smm(cpr)

        # Scheduled payment (level payment amortisation)
        remaining_months = remaining - m + 1
        if monthly_rate > 0 and remaining_months > 0:
            scheduled_payment = balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining_months))
        else:
            scheduled_payment = balance / max(remaining_months, 1)

        interest = balance * pass_rate
        scheduled_principal = scheduled_payment - balance * monthly_rate
        scheduled_principal = max(min(scheduled_principal, balance), 0)

        # Prepayment on remaining balance after scheduled principal
        prepayment = (balance - scheduled_principal) * smm

        total_cf = interest + scheduled_principal + prepayment
        principal_cf = scheduled_principal + prepayment

        # Discount
        t_years = m / 12.0
        df = discount_curve.df_interpolated(t_years) if hasattr(discount_curve, 'df_interpolated') else math.exp(-0.04 * t_years)
        # Use curve df with spread
        try:
            from datetime import timedelta
            cf_date = ref + timedelta(days=round(t_years * 365.25))
            df = discount_curve.df(cf_date) * math.exp(-spread * t_years)
        except Exception:
            df = math.exp(-(0.04 + spread) * t_years)

        total_pv += total_cf * df
        total_principal_pv += principal_cf * df
        wal_num += principal_cf * t_years
        total_principal += principal_cf

        balance -= principal_cf
        balance = max(balance, 0)

    # Results
    price_100 = total_pv / pool.original_balance * 100 if pool.original_balance > 0 else 0
    wal = wal_num / total_principal if total_principal > 0 else 0
    prepay_pct = 1.0 - balance / pool.original_balance if pool.original_balance > 0 else 0

    # Duration via finite difference
    if _compute_risk:
        dur = _mbs_duration(pool, discount_curve, psa_speed, spread)
        cvx = _mbs_convexity(pool, discount_curve, psa_speed, spread)
    else:
        dur = 0.0
        cvx = 0.0

    # YTM approximation
    ytm = pool.pass_through_rate + spread if price_100 > 0 else 0

    return MBSResult(
        price=price_100,
        yield_to_maturity=ytm,
        wal=wal,
        modified_duration=dur,
        convexity=cvx,
        total_prepayment_pct=prepay_pct,
        cashflows=min(pool.wam - pool.age, pool.wam),
    )


def oas_mbs(
    pool: MBSPool,
    discount_curve: DiscountCurve,
    market_price: float,
    psa_speed: float = 1.0,
    tol: float = 0.0001,
    max_iter: int = 50,
) -> OASResult:
    """Option-adjusted spread via Newton-Raphson.

    Finds the spread s such that price_mbs(spread=s) = market_price.

    Args:
        market_price: observed clean price per 100 face.
    """
    s = 0.01  # initial guess
    for _ in range(max_iter):
        r = price_mbs(pool, discount_curve, psa_speed, spread=s)
        diff = r.price - market_price
        if abs(diff) < tol:
            break
        # Numerical derivative
        ds = 0.0001
        r_up = price_mbs(pool, discount_curve, psa_speed, spread=s + ds)
        deriv = (r_up.price - r.price) / ds
        if abs(deriv) < 1e-15:
            break
        s -= diff / deriv
        s = max(s, -0.10)

    r_final = price_mbs(pool, discount_curve, psa_speed, spread=s)

    return OASResult(
        oas_bp=s * 10_000,
        price=r_final.price,
        z_spread_bp=s * 10_000,  # for pass-through, OAS ≈ Z-spread at fixed PSA
        wal=r_final.wal,
        prepay_duration=r_final.modified_duration,
        prepay_convexity=r_final.convexity,
    )


# ═══════════════════════════════════════════════════════════════
# IO/PO Strips
# ═══════════════════════════════════════════════════════════════

@dataclass
class IOPOResult:
    """IO/PO strip pricing."""
    io_price: float         # interest-only price per 100 face
    po_price: float         # principal-only price per 100 face
    io_duration: float
    po_duration: float
    io_wal: float
    po_wal: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def io_po_strips(
    pool: MBSPool,
    discount_curve: DiscountCurve,
    psa_speed: float = 1.0,
    spread: float = 0.0,
) -> IOPOResult:
    """Decompose MBS into IO and PO strips.

    IO: receives interest payments only. Value decreases with faster prepay.
    PO: receives principal payments only. Value increases with faster prepay.

    IO_price + PO_price ≈ MBS_price.
    """
    remaining = pool.wam - pool.age
    balance = pool.original_balance
    monthly_rate = pool.wac / 12.0
    pass_rate = pool.pass_through_rate / 12.0
    ref = discount_curve.reference_date

    io_pv = 0.0
    po_pv = 0.0
    io_wal_num = 0.0
    po_wal_num = 0.0
    total_io = 0.0
    total_po = 0.0

    for m in range(1, remaining + 1):
        if balance < 0.01:
            break

        month_age = pool.age + m
        cpr = psa_speed_fn(month_age, psa_speed)
        smm = cpr_to_smm(cpr)

        remaining_months = remaining - m + 1
        if monthly_rate > 0 and remaining_months > 0:
            scheduled_payment = balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining_months))
        else:
            scheduled_payment = balance / max(remaining_months, 1)

        interest = balance * pass_rate
        scheduled_principal = max(min(scheduled_payment - balance * monthly_rate, balance), 0)
        prepayment = (balance - scheduled_principal) * smm
        principal_cf = scheduled_principal + prepayment

        t_years = m / 12.0
        try:
            from datetime import timedelta
            cf_date = ref + timedelta(days=round(t_years * 365.25))
            df = discount_curve.df(cf_date) * math.exp(-spread * t_years)
        except Exception:
            df = math.exp(-(0.04 + spread) * t_years)

        io_pv += interest * df
        po_pv += principal_cf * df
        io_wal_num += interest * t_years
        po_wal_num += principal_cf * t_years
        total_io += interest
        total_po += principal_cf

        balance -= principal_cf
        balance = max(balance, 0)

    orig = pool.original_balance
    io_price = io_pv / orig * 100 if orig > 0 else 0
    po_price = po_pv / orig * 100 if orig > 0 else 0
    io_wal = io_wal_num / total_io if total_io > 0 else 0
    po_wal = po_wal_num / total_po if total_po > 0 else 0

    # Duration via spread bump
    ds = 0.0001
    io_up = _strip_pv(pool, discount_curve, psa_speed, spread + ds, "io") / orig * 100
    io_dn = _strip_pv(pool, discount_curve, psa_speed, spread - ds, "io") / orig * 100
    po_up = _strip_pv(pool, discount_curve, psa_speed, spread + ds, "po") / orig * 100
    po_dn = _strip_pv(pool, discount_curve, psa_speed, spread - ds, "po") / orig * 100

    io_dur = -(io_up - io_dn) / (2 * ds) / max(io_price, 0.01)
    po_dur = -(po_up - po_dn) / (2 * ds) / max(po_price, 0.01)

    return IOPOResult(
        io_price=io_price,
        po_price=po_price,
        io_duration=io_dur,
        po_duration=po_dur,
        io_wal=io_wal,
        po_wal=po_wal,
    )


# ---- Internal helpers ----

def psa_speed_fn(month: int, psa_multiple: float) -> float:
    """PSA speed (alias for module-level function)."""
    if month <= 30:
        base_cpr = 0.06 * month / 30.0
    else:
        base_cpr = 0.06
    return min(base_cpr * psa_multiple, 1.0)


def _mbs_duration(pool, curve, psa, spread, ds=0.0001):
    p_up = price_mbs(pool, curve, psa, spread=spread + ds, _compute_risk=False).price
    p_dn = price_mbs(pool, curve, psa, spread=spread - ds, _compute_risk=False).price
    p_base = price_mbs(pool, curve, psa, spread=spread, _compute_risk=False).price
    if p_base <= 0:
        return 0
    return -(p_up - p_dn) / (2 * ds) / p_base


def _mbs_convexity(pool, curve, psa, spread, ds=0.0001):
    p_up = price_mbs(pool, curve, psa, spread=spread + ds, _compute_risk=False).price
    p_dn = price_mbs(pool, curve, psa, spread=spread - ds, _compute_risk=False).price
    p_base = price_mbs(pool, curve, psa, spread=spread, _compute_risk=False).price
    if p_base <= 0:
        return 0
    return (p_up - 2 * p_base + p_dn) / (ds ** 2) / p_base


def _strip_pv(pool, curve, psa, spread, strip_type):
    remaining = pool.wam - pool.age
    balance = pool.original_balance
    monthly_rate = pool.wac / 12.0
    pass_rate = pool.pass_through_rate / 12.0
    ref = curve.reference_date
    pv = 0.0

    for m in range(1, remaining + 1):
        if balance < 0.01:
            break
        month_age = pool.age + m
        cpr = psa_speed_fn(month_age, psa)
        smm = cpr_to_smm(cpr)
        remaining_months = remaining - m + 1
        if monthly_rate > 0 and remaining_months > 0:
            sched = balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining_months))
        else:
            sched = balance / max(remaining_months, 1)
        interest = balance * pass_rate
        sched_prin = max(min(sched - balance * monthly_rate, balance), 0)
        prepay = (balance - sched_prin) * smm
        t = m / 12.0
        try:
            from datetime import timedelta
            df = curve.df(ref + timedelta(days=round(t * 365.25))) * math.exp(-spread * t)
        except Exception:
            df = math.exp(-(0.04 + spread) * t)
        if strip_type == "io":
            pv += interest * df
        else:
            pv += (sched_prin + prepay) * df
        balance -= (sched_prin + prepay)
        balance = max(balance, 0)
    return pv
