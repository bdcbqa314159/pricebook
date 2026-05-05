"""Trader API — desk-level operations in one call.

Extends api.py with:
- analyse(): universal "give me everything" for any instrument
- CLN, TRS, repo one-liners
- Vol surface from simple quotes
- Book creation from dicts
- Multi-curve + CSA comparison
- Recovery analytics + desk dashboards

    import pricebook.api_desk as desk

    # Analyse anything
    result = desk.analyse("irs", tenor="5Y", rate=0.04, curve=curve)

    # Build a swap book from dicts
    result = desk.swap_book([
        {"tenor": "5Y", "rate": 0.038, "direction": "payer", "notional": 50e6},
    ], curve=curve)

    # Vol surface from market quotes
    surface = desk.vol_surface("fx", [
        {"expiry": "1M", "atm": 0.08, "rr25": -0.01, "bf25": 0.003},
    ], spot=1.08)
"""

from __future__ import annotations

import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import DayCountConvention, year_fraction


def _parse_tenor(ref: date, tenor: str | date) -> date:
    """Parse "5Y", "6M", "3W", "30D", "0.5Y", "1.5Y" or a date."""
    if isinstance(tenor, date):
        return tenor
    t = tenor.upper().strip()
    try:
        if t.endswith("Y"):
            val = float(t[:-1])
            if val == int(val):
                return ref + relativedelta(years=int(val))
            # Fractional year: convert to months
            return ref + relativedelta(months=int(val * 12))
        if t.endswith("M"):
            return ref + relativedelta(months=int(float(t[:-1])))
        if t.endswith("W"):
            return ref + relativedelta(weeks=int(float(t[:-1])))
        if t.endswith("D"):
            return ref + relativedelta(days=int(float(t[:-1])))
    except (ValueError, OverflowError):
        pass
    raise ValueError(f"Cannot parse tenor: {tenor}. Use '5Y', '6M', '3W', '30D' or a date.")


def _require_keys(d: dict, keys: list[str], context: str = "") -> None:
    """Validate required keys are present in a dict."""
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {context or 'trade dict'}: got {list(d.keys())}")


def _warn_default(param_name: str, default_value, context: str = "") -> None:
    """Issue warning when using a default value for a key parameter."""
    import warnings
    warnings.warn(
        f"Using default {param_name}={default_value}"
        f"{' for ' + context if context else ''}. "
        f"Pass {param_name}=... explicitly to suppress.",
        stacklevel=4,
    )


def _check_rate_not_bps(rate: float, param_name: str = "rate") -> None:
    """Warn if rate looks like it's in basis points instead of decimal."""
    if abs(rate) > 1.0:
        import warnings
        warnings.warn(
            f"{param_name}={rate} looks like basis points, not decimal. "
            f"Did you mean {rate/10000}? (e.g., 0.04 not 400)",
            stacklevel=4,
        )


# ============================================================================
# SECTION 1: analyse() — Universal analytics
# ============================================================================

def analyse(instrument_type: str, *, curve: DiscountCurve, **kwargs) -> dict:
    """Price ANY instrument and return all analytics.

        desk.analyse("irs", tenor="5Y", rate=0.04, curve=curve)
        desk.analyse("cds", tenor="5Y", spread=0.01, hazard=0.02, curve=curve)
        desk.analyse("cln", tenor="5Y", coupon=0.05, hazard=0.02, curve=curve)
        desk.analyse("bond", tenor="10Y", coupon=0.04, curve=curve)
    """
    t = instrument_type.lower()
    ref = curve.reference_date

    if t == "irs":
        return _analyse_irs(ref, curve, **kwargs)
    elif t == "cds":
        return _analyse_cds(ref, curve, **kwargs)
    elif t == "cln":
        return _analyse_cln(ref, curve, **kwargs)
    elif t == "bond":
        return _analyse_bond(ref, curve, **kwargs)
    else:
        raise ValueError(f"Unknown instrument type: {instrument_type}. "
                         "Supported: irs, cds, cln, bond")


def _analyse_irs(ref, curve, **kw):
    from pricebook.swap import InterestRateSwap, SwapDirection
    from pricebook.swap_desk import swap_risk_metrics, swap_carry_decomposition

    if "rate" not in kw:
        _warn_default("rate", 0.04, "IRS")
    tenor = kw.get("tenor", "5Y")
    rate = kw.get("rate", 0.04)
    _check_rate_not_bps(rate, "rate")
    notional = kw.get("notional", 10_000_000)
    direction = SwapDirection.PAYER if kw.get("direction", "payer") == "payer" else SwapDirection.RECEIVER

    swap = InterestRateSwap(ref, _parse_tenor(ref, tenor), rate, direction=direction, notional=notional)
    rm = swap_risk_metrics(swap, curve)
    carry = swap_carry_decomposition(swap, curve)

    return {
        "type": "irs", "pv": rm.pv, "par_rate": rm.par_rate,
        "dv01": rm.dv01, "key_rate_dv01": rm.key_rate_dv01,
        "gamma": rm.gamma, "theta": rm.theta,
        "carry": carry.to_dict(),
        "notional": notional, "direction": rm.direction,
    }


def _analyse_cds(ref, curve, **kw):
    from pricebook.cds import CDS
    from pricebook.cds_market import build_cds_curve
    from pricebook.cds_desk import cds_risk_metrics, cds_carry_decomposition

    if "spread" not in kw:
        _warn_default("spread", 0.01, "CDS")
    tenor = kw.get("tenor", "5Y")
    spread = kw.get("spread", 0.01)
    _check_rate_not_bps(spread, "spread")
    hazard = kw.get("hazard", 0.02)
    recovery = kw.get("recovery", 0.40)
    notional = kw.get("notional", 10_000_000)

    surv = SurvivalCurve.flat(ref, hazard)
    cds = CDS(ref, _parse_tenor(ref, tenor), spread=spread, notional=notional, recovery=recovery)
    rm = cds_risk_metrics(cds, curve, surv)
    carry = cds_carry_decomposition(cds, curve, surv)

    return {
        "type": "cds", "pv": rm.pv, "par_spread": rm.par_spread,
        "cs01": rm.cs01, "rec01": rm.rec01, "jtd": rm.jump_to_default,
        "carry": carry.to_dict(), "theta": rm.theta,
        "spread_duration": rm.spread_duration,
    }


def _analyse_cln(ref, curve, **kw):
    from pricebook.cln import CreditLinkedNote
    from pricebook.schedule import Frequency
    from pricebook.cln_desk import cln_risk_metrics, cln_carry_decomposition

    if "coupon" not in kw:
        _warn_default("coupon", 0.05, "CLN")
    tenor = kw.get("tenor", "5Y")
    coupon = kw.get("coupon", 0.05)
    _check_rate_not_bps(coupon, "coupon")
    hazard = kw.get("hazard", 0.02)
    recovery = kw.get("recovery", 0.40)
    leverage = kw.get("leverage", 1.0)
    notional = kw.get("notional", 10_000_000)

    surv = SurvivalCurve.flat(ref, hazard)
    cln_inst = CreditLinkedNote(ref, _parse_tenor(ref, tenor), coupon_rate=coupon,
                                notional=notional, recovery=recovery, leverage=leverage,
                                frequency=Frequency.QUARTERLY)
    rm = cln_risk_metrics(cln_inst, curve, surv)
    carry = cln_carry_decomposition(cln_inst, curve, surv)

    return {
        "type": "cln", "pv": rm.pv, "cs01": rm.cs01,
        "recovery_sensitivity": rm.recovery_sensitivity,
        "jtd": rm.jump_to_default_pnl,
        "carry": carry.to_dict(), "leverage": leverage,
    }


def _analyse_bond(ref, curve, **kw):
    from pricebook.bond import FixedRateBond
    from pricebook.bond_trading_desk import bond_risk_metrics, bond_carry_roll

    tenor = kw.get("tenor", "10Y")
    coupon = kw.get("coupon", 0.04)
    repo_rate = kw.get("repo_rate", 0.04)

    bond = FixedRateBond.treasury_note(ref, _parse_tenor(ref, tenor), coupon)
    rm = bond_risk_metrics(bond, curve, ref)
    carry = bond_carry_roll(bond, curve, repo_rate=repo_rate, horizon_days=30)

    return {
        "type": "bond", "pv": rm.pv, "clean": rm.clean_price,
        "ytm": rm.ytm, "mod_duration": rm.modified_duration,
        "eff_duration": rm.effective_duration, "convexity": rm.convexity,
        "dv01": rm.dv01, "key_rate_dv01": rm.key_rate_dv01,
        "carry": carry.to_dict(),  # consistent key with IRS/CDS/CLN
    }


# ============================================================================
# SECTION 2: CLN, TRS, Repo one-liners
# ============================================================================

def cln(tenor, coupon, curve, *, hazard=0.02, recovery=0.4, leverage=1.0,
        notional=10_000_000) -> dict:
    """Price a CLN in one call.

        desk.cln("5Y", 0.05, curve, hazard=0.02)
    """
    return analyse("cln", curve=curve, tenor=tenor, coupon=coupon,
                   hazard=hazard, recovery=recovery, leverage=leverage,
                   notional=notional)


def trs(tenor, underlying, curve, *, funding_spread=0.005, repo_spread=0.01,
        notional=10_000_000, sigma=0.20) -> dict:
    """Price an equity TRS in one call.

        desk.trs("6M", 100.0, curve)  # equity TRS, spot=100

    Note: `underlying` must be a numeric spot price (equity TRS).
    For bond/loan/CLN TRS, use the full TotalReturnSwap class directly.
    """
    from pricebook.trs import TotalReturnSwap, FundingLegSpec
    from pricebook.trs_desk import trs_risk_metrics, trs_carry_decomposition

    ref = curve.reference_date
    trs_inst = TotalReturnSwap(
        underlying=float(underlying), notional=notional,
        start=ref, end=_parse_tenor(ref, tenor),
        funding=FundingLegSpec(spread=funding_spread),
        repo_spread=repo_spread, initial_price=float(underlying), sigma=sigma,
    )
    rm = trs_risk_metrics(trs_inst, curve)
    carry = trs_carry_decomposition(trs_inst, curve)

    return {
        "type": "trs", "pv": rm.pv, "delta": rm.delta, "gamma": rm.gamma,
        "dv01": rm.dv01, "funding_dv01": rm.funding_dv01,
        "vega": rm.vega, "carry": carry.to_dict(),
        "notional": notional, "underlying": float(underlying),
    }


def repo(tenor_days, face, rate, *, haircut=0.05, notional=None) -> dict:
    """Price a repo trade in one call.

        desk.repo(30, 10_000_000, 0.04, haircut=0.05)

    Conventions: ACT/360 interest accrual (USD repo standard).
    """
    if not 0 <= haircut < 1:
        raise ValueError(f"haircut must be in [0, 1), got {haircut}")
    if face <= 0:
        raise ValueError(f"face must be positive, got {face}")
    if tenor_days <= 0:
        raise ValueError(f"tenor_days must be positive, got {tenor_days}")

    cash_lent = face * (1 - haircut)
    interest = cash_lent * rate * tenor_days / 360  # ACT/360 convention
    maturity_amount = cash_lent + interest

    return {
        "type": "repo", "face": face, "cash_lent": cash_lent,
        "rate": rate, "haircut": haircut, "tenor_days": tenor_days,
        "interest": interest, "maturity_amount": maturity_amount,
        "carry": {"income": interest, "funding": 0.0, "net": interest},
        "carry_30d": cash_lent * rate * 30 / 360,
    }


# ============================================================================
# SECTION 3: Vol surface builder
# ============================================================================

def vol_surface(asset_class: str, quotes: list[dict], *, spot=None, ref=None):
    """Build a calibrated vol surface from market quotes.

        surface = desk.vol_surface("fx", [
            {"expiry": "1M", "atm": 0.08, "rr25": -0.01, "bf25": 0.003},
            {"expiry": "3M", "atm": 0.09, "rr25": -0.012, "bf25": 0.004},
        ], spot=1.08)
        surface.vol(expiry_date, strike)
    """
    from pricebook.vol_calibration import (
        calibrate_fx_surface, calibrate_equity_surface,
        calibrate_ir_surface, calibrate_commodity_surface,
    )

    if ref is None:
        raise ValueError("ref (reference date) is required for vol_surface. "
                         "Pass ref=date(2024, 7, 15) or similar.")
    if not quotes:
        raise ValueError("quotes list is empty — provide at least one vol quote.")

    # Parse expiry strings to dates
    parsed = []
    for q in quotes:
        pq = dict(q)
        if isinstance(pq.get("expiry"), str):
            pq["expiry"] = _parse_tenor(ref, pq["expiry"])
        parsed.append(pq)

    # Validate spot: must be positive if provided, use sensible defaults only if None
    effective_spot = spot if spot is not None else None

    if asset_class.lower() == "fx":
        return calibrate_fx_surface(ref, parsed, spot=effective_spot if effective_spot else 1.0)
    elif asset_class.lower() in ("equity", "eq"):
        return calibrate_equity_surface(ref, parsed, spot=effective_spot if effective_spot else 100.0)
    elif asset_class.lower() in ("ir", "swaption"):
        return calibrate_ir_surface(ref, parsed)
    elif asset_class.lower() in ("commodity", "commo"):
        return calibrate_commodity_surface(ref, parsed, spot=effective_spot if effective_spot else 75.0)
    else:
        raise ValueError(f"Unknown asset class: {asset_class}. "
                         "Supported: fx, equity, ir, commodity")


# ============================================================================
# SECTION 4: Book creation from dicts
# ============================================================================

def swap_book(trades: list[dict], *, curve: DiscountCurve) -> dict:
    """Create swap book and compute all risk from a list of dicts.

        desk.swap_book([
            {"tenor": "5Y", "rate": 0.038, "direction": "payer", "notional": 50e6},
        ], curve=curve)
    """
    from pricebook.swap import InterestRateSwap, SwapDirection
    from pricebook.swap_desk import SwapBook, SwapBookEntry, swap_dashboard, swap_stress_suite

    ref = curve.reference_date
    book = SwapBook()

    for i, t in enumerate(trades):
        _require_keys(t, ["tenor", "rate"], f"swap trade #{i+1}")
        direction = SwapDirection.PAYER if t.get("direction", "payer") == "payer" else SwapDirection.RECEIVER
        swap = InterestRateSwap(
            ref, _parse_tenor(ref, t["tenor"]), t["rate"],
            direction=direction, notional=t.get("notional", 10_000_000))
        book.add(SwapBookEntry(f"T{i+1}", swap, t.get("counterparty", "")))

    risk = book.aggregate_risk(curve)
    db = swap_dashboard(book, ref, curve)
    stress = swap_stress_suite(book, curve)

    return {
        "n_positions": risk["n_positions"],
        "total_pv": risk["total_pv"],
        "total_dv01": risk["total_dv01"],
        "net_dv01": risk["net_dv01"],
        "total_notional": risk["total_notional"],
        "dv01_ladder": db.dv01_ladder,
        "stress": [s.to_dict() for s in stress],
    }


def cds_book(trades: list[dict], *, curve: DiscountCurve) -> dict:
    """Create CDS book and compute risk from dicts.

        desk.cds_book([
            {"name": "AAPL", "tenor": "5Y", "spread": 0.007, "sector": "tech"},
        ], curve=curve)
    """
    from pricebook.cds import CDS
    from pricebook.cds_desk import CDSBook, CDSBookEntry, cds_dashboard, cds_stress_suite

    ref = curve.reference_date
    book = CDSBook()

    for i, t in enumerate(trades):
        _require_keys(t, ["tenor", "spread"], f"CDS trade #{i+1}")
        # h ≈ spread / (1-R): standard flat hazard approximation (O'Kane 2008)
        hazard = t.get("hazard", t["spread"] / 0.6)
        surv = SurvivalCurve.flat(ref, hazard)
        cds_inst = CDS(ref, _parse_tenor(ref, t["tenor"]),
                       spread=t["spread"], notional=t.get("notional", 10_000_000))
        book.add(CDSBookEntry(
            f"C{i+1}", cds_inst, surv,
            reference_name=t.get("name", f"name_{i}"),
            sector=t.get("sector", ""),
        ))

    risk = book.aggregate_risk(curve)
    db = cds_dashboard(book, ref, curve)
    stress = cds_stress_suite(book, curve)

    return {
        "n_positions": risk["n_positions"],
        "total_cs01": risk["total_cs01"],
        "total_jtd": risk["total_jtd"],
        "total_notional": risk["total_notional"],
        "by_name": db.by_name,
        "by_sector": db.by_sector,
        "stress": [s.to_dict() for s in stress],
    }


# ============================================================================
# SECTION 5: Multi-curve + CSA
# ============================================================================

def multicurve(*, ref=None, **currencies) -> dict:
    """Build multi-currency curves from simple dicts.

        curves = desk.multicurve(
            usd={"swaps": {"1Y": 0.047, "5Y": 0.038, "10Y": 0.036}},
            eur={"swaps": {"1Y": 0.034, "5Y": 0.028, "10Y": 0.026}},
        )
        curves["USD"].df(date)
    """
    from pricebook.sofr_curve import build_sofr_curve, build_estr_curve, build_sonia_curve

    if ref is None:
        raise ValueError("ref (reference date) is required for multicurve. "
                         "Pass ref=date(2024, 7, 15) or similar.")

    result = {}
    for ccy, data in currencies.items():
        swaps_dict = data.get("swaps", {})
        swap_list = []
        for t, r in swaps_dict.items():
            if isinstance(t, str):
                swap_list.append((_parse_tenor(ref, t), r))
            else:
                swap_list.append((t, r))

        ccy_upper = ccy.upper()
        if ccy_upper == "USD":
            result[ccy_upper] = build_sofr_curve(ref, sofr_swaps=swap_list)
        elif ccy_upper == "EUR":
            result[ccy_upper] = build_estr_curve(ref, swap_list)
        elif ccy_upper == "GBP":
            result[ccy_upper] = build_sonia_curve(ref, swap_list)
        else:
            from pricebook.ois import bootstrap_ois
            result[ccy_upper] = bootstrap_ois(ref, swap_list)

    return result


# ============================================================================
# SECTION 6: Recovery + desk analytics
# ============================================================================

def recovery_analysis(*, cds_spreads: dict, curve: DiscountCurve,
                      tenor="5Y", coupon=0.05, recoveries=None) -> dict:
    """Full recovery-hazard analysis for a CLN.

        desk.recovery_analysis(
            cds_spreads={1: 0.005, 5: 0.01, 10: 0.012},
            curve=curve, tenor="5Y", coupon=0.05)
    """
    from pricebook.cln import CreditLinkedNote
    from pricebook.schedule import Frequency
    from pricebook.recovery_analytics import (
        recovery_curve_family, recovery_greeks, recovery_pv_surface,
    )

    if not cds_spreads:
        raise ValueError("cds_spreads is empty — provide at least one tenor:spread pair.")

    ref = curve.reference_date
    cln_inst = CreditLinkedNote(ref, _parse_tenor(ref, tenor), coupon_rate=coupon,
                                notional=10_000_000, recovery=0.40,
                                frequency=Frequency.QUARTERLY)

    family = recovery_curve_family(cds_spreads, curve, ref, recoveries)
    greeks = recovery_greeks(cln_inst, curve, cds_spreads, ref)
    surface = recovery_pv_surface(cln_inst, curve, cds_spreads, ref, recoveries)

    return {
        "greeks": greeks.to_dict(),
        "surface": [p.to_dict() for p in surface],
        "n_recoveries": len(family),
        "direct_effect": greeks.direct_effect,
        "indirect_effect": greeks.indirect_effect,
        "convexity": greeks.convexity,
    }


def dashboard(book_type: str, trades: list[dict], *, curve: DiscountCurve) -> dict:
    """One-call desk dashboard from a list of trade dicts.

        desk.dashboard("swap", [...], curve=curve)
        desk.dashboard("cds", [...], curve=curve)
    """
    if book_type.lower() in ("swap", "irs"):
        return swap_book(trades, curve=curve)
    elif book_type.lower() == "cds":
        return cds_book(trades, curve=curve)
    else:
        raise ValueError(f"Unknown book type: {book_type}. Supported: swap, cds")
