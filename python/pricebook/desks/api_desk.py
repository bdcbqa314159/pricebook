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

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


def _parse_tenor(ref: date, tenor: str | date) -> date:
    """Parse "5Y", "6M", "3W", "30D", "0.5Y", "1.5Y" or a date.

    Raises ValueError for non-positive tenors (maturity must be after reference).
    """
    if isinstance(tenor, date):
        if tenor <= ref:
            raise ValueError(f"Tenor date {tenor} must be after reference date {ref}.")
        return tenor
    t = tenor.upper().strip()
    try:
        if t.endswith("Y"):
            raw = t[:-1]
            if raw.lstrip("-").isdigit():
                years = int(raw)
                if years <= 0:
                    raise ValueError(f"Tenor must be positive, got '{tenor}'.")
                return ref + relativedelta(years=years)
            val = float(raw)
            if val <= 0:
                raise ValueError(f"Tenor must be positive, got '{tenor}'.")
            months = round(val * 12)
            if months <= 0:
                raise ValueError(f"Tenor must be positive, got '{tenor}'.")
            return ref + relativedelta(months=months)
        if t.endswith("M"):
            months = int(float(t[:-1]))
            if months <= 0:
                raise ValueError(f"Tenor must be positive, got '{tenor}'.")
            return ref + relativedelta(months=months)
        if t.endswith("W"):
            weeks = int(float(t[:-1]))
            if weeks <= 0:
                raise ValueError(f"Tenor must be positive, got '{tenor}'.")
            return ref + relativedelta(weeks=weeks)
        if t.endswith("D"):
            days = int(float(t[:-1]))
            if days <= 0:
                raise ValueError(f"Tenor must be positive, got '{tenor}'.")
            return ref + relativedelta(days=days)
    except ValueError:
        raise
    except OverflowError:
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
        stacklevel=3,
    )


def _check_rate_not_bps(rate: float, param_name: str = "rate") -> None:
    """Warn if rate looks like it's in basis points instead of decimal."""
    if abs(rate) > 1.0:
        import warnings
        warnings.warn(
            f"{param_name}={rate} looks like basis points, not decimal. "
            f"Did you mean {rate/10000}? (e.g., 0.04 not 400)",
            stacklevel=3,
        )


def _apply_notional_profile(notional, profile, ref, end, frequency,
                            final_notional=None):
    """Generate a notional schedule from a profile name.

    Returns the notional unchanged (scalar or list) if profile is None.
    """
    from pricebook.core.schedule import generate_schedule
    if profile is None:
        return notional
    if isinstance(notional, list):
        raise ValueError("Cannot specify both notional as list and notional_profile.")
    schedule = generate_schedule(ref, end, frequency)
    n = len(schedule) - 1
    if profile == "amortising":
        return [max(float(notional) * (1.0 - i / n), float(notional) / n)
                for i in range(n)]
    elif profile == "accreting":
        final = final_notional if final_notional is not None else notional * 2
        return [float(notional) + (final - notional) * i / max(n - 1, 1)
                for i in range(n)]
    else:
        raise ValueError(f"Unknown notional_profile: {profile}. "
                         "Use 'amortising' or 'accreting'.")


# ============================================================================
# SECTION 1: analyse() — Universal analytics
# ============================================================================

def analyse(instrument_type: str, *, curve: DiscountCurve, **kwargs) -> dict:
    """Price ANY instrument and return all analytics.

        desk.analyse("irs", tenor="5Y", rate=0.04, curve=curve)
        desk.analyse("cds", tenor="5Y", spread=0.01, hazard=0.02, curve=curve)
        desk.analyse("cln", tenor="5Y", coupon=0.05, hazard=0.02, curve=curve)
        desk.analyse("bond", tenor="10Y", coupon=0.04, curve=curve)

        # Amortising swap: pass notional as list or use notional_profile
        desk.analyse("irs", notional=[50e6, 40e6, 30e6, 20e6, 10e6], ...)
        desk.analyse("irs", notional=50e6, notional_profile="amortising", ...)
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
    from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
    from pricebook.desks.swap_desk import swap_risk_metrics, swap_carry_decomposition
    from pricebook.core.schedule import Frequency

    if "rate" not in kw:
        _warn_default("rate", 0.04, "IRS")
    if "notional" not in kw:
        _warn_default("notional", "10_000_000", "IRS")
    tenor = kw.get("tenor", "5Y")
    rate = kw.get("rate", 0.04)
    _check_rate_not_bps(rate, "rate")
    notional = kw.get("notional", 10_000_000)
    end = _parse_tenor(ref, tenor)
    notional = _apply_notional_profile(
        notional, kw.get("notional_profile"), ref, end,
        Frequency.SEMI_ANNUAL, final_notional=kw.get("final_notional"))

    dir_str = kw.get("direction", "payer").lower().strip()
    if dir_str in ("payer", "pay", "p"):
        direction = SwapDirection.PAYER
    elif dir_str in ("receiver", "recv", "receive", "r"):
        direction = SwapDirection.RECEIVER
    else:
        raise ValueError(f"Invalid direction '{kw.get('direction')}'. "
                         "Use 'payer'/'pay'/'p' or 'receiver'/'recv'/'r'.")

    swap = InterestRateSwap(ref, end, rate, direction=direction, notional=notional)
    rm = swap_risk_metrics(swap, curve)
    carry = swap_carry_decomposition(swap, curve)

    result = {
        "type": "irs", "pv": rm.pv, "par_rate": rm.par_rate,
        "dv01": rm.dv01, "key_rate_dv01": rm.key_rate_dv01,
        "gamma": rm.gamma, "theta": rm.theta,
        "carry": carry.to_dict(),
        "notional": swap.notional, "direction": rm.direction,
    }

    # Enrich with schedule info when notional varies
    if isinstance(notional, list):
        result["notional_schedule"] = swap.notional_schedule
        result["average_notional"] = swap.average_notional

    return result


def _analyse_cds(ref, curve, **kw):
    from pricebook.credit.cds import CDS
    from pricebook.desks.cds_desk import cds_risk_metrics, cds_carry_decomposition
    from pricebook.core.schedule import Frequency

    if "spread" not in kw:
        _warn_default("spread", 0.01, "CDS")
    if "notional" not in kw:
        _warn_default("notional", "10_000_000", "CDS")
    tenor = kw.get("tenor", "5Y")
    spread = kw.get("spread", 0.01)
    _check_rate_not_bps(spread, "spread")
    hazard = kw.get("hazard", 0.02)
    if hazard <= 0:
        raise ValueError(f"hazard must be positive, got {hazard}")
    recovery = kw.get("recovery", 0.40)
    if not 0 <= recovery < 1:
        raise ValueError(f"recovery must be in [0, 1), got {recovery}")
    notional = kw.get("notional", 10_000_000)
    end = _parse_tenor(ref, tenor)
    notional = _apply_notional_profile(
        notional, kw.get("notional_profile"), ref, end,
        Frequency.QUARTERLY, final_notional=kw.get("final_notional"))

    surv = SurvivalCurve.flat(ref, hazard)
    cds = CDS(ref, end, spread=spread, notional=notional, recovery=recovery)
    rm = cds_risk_metrics(cds, curve, surv)
    carry = cds_carry_decomposition(cds, curve, surv)

    result = {
        "type": "cds", "pv": rm.pv, "par_spread": rm.par_spread,
        "cs01": rm.cs01, "rec01": rm.rec01, "jtd": rm.jump_to_default,
        "carry": carry.to_dict(), "theta": rm.theta,
        "spread_duration": rm.spread_duration,
        "notional": cds.notional,
    }
    if isinstance(notional, list):
        result["notional_schedule"] = cds.notional_schedule
        result["average_notional"] = cds.average_notional
    return result


def _analyse_cln(ref, curve, **kw):
    from pricebook.credit.cln import CreditLinkedNote
    from pricebook.core.schedule import Frequency
    from pricebook.desks.cln_desk import cln_risk_metrics, cln_carry_decomposition

    if "coupon" not in kw:
        _warn_default("coupon", 0.05, "CLN")
    if "notional" not in kw:
        _warn_default("notional", "10_000_000", "CLN")
    tenor = kw.get("tenor", "5Y")
    coupon = kw.get("coupon", 0.05)
    _check_rate_not_bps(coupon, "coupon")
    hazard = kw.get("hazard", 0.02)
    if hazard <= 0:
        raise ValueError(f"hazard must be positive, got {hazard}")
    recovery = kw.get("recovery", 0.40)
    if not 0 <= recovery < 1:
        raise ValueError(f"recovery must be in [0, 1), got {recovery}")
    leverage = kw.get("leverage", 1.0)
    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")
    notional = kw.get("notional", 10_000_000)
    end = _parse_tenor(ref, tenor)
    notional = _apply_notional_profile(
        notional, kw.get("notional_profile"), ref, end,
        Frequency.QUARTERLY, final_notional=kw.get("final_notional"))

    surv = SurvivalCurve.flat(ref, hazard)
    cln_inst = CreditLinkedNote(ref, end, coupon_rate=coupon,
                                notional=notional, recovery=recovery, leverage=leverage,
                                frequency=Frequency.QUARTERLY)
    rm = cln_risk_metrics(cln_inst, curve, surv)
    carry = cln_carry_decomposition(cln_inst, curve, surv)

    result = {
        "type": "cln", "pv": rm.pv, "cs01": rm.cs01,
        "recovery_sensitivity": rm.recovery_sensitivity,
        "jtd": rm.jump_to_default_pnl,
        "carry": carry.to_dict(), "leverage": leverage,
        "notional": cln_inst.notional,
    }
    if isinstance(notional, list):
        result["notional_schedule"] = cln_inst.notional_schedule
        result["average_notional"] = cln_inst.average_notional
    return result


def _analyse_bond(ref, curve, **kw):
    from pricebook.fixed_income.bond import FixedRateBond
    from pricebook.desks.bond_trading_desk import bond_risk_metrics, bond_carry_roll

    if "coupon" not in kw:
        _warn_default("coupon", 0.04, "bond")
    if "repo_rate" not in kw:
        _warn_default("repo_rate", 0.04, "bond")
    tenor = kw.get("tenor", "10Y")
    coupon = kw.get("coupon", 0.04)
    _check_rate_not_bps(coupon, "coupon")
    repo_rate = kw.get("repo_rate", 0.04)
    _check_rate_not_bps(repo_rate, "repo_rate")

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
    if not 0 <= recovery < 1:
        raise ValueError(f"recovery must be in [0, 1), got {recovery}")
    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")
    return analyse("cln", curve=curve, tenor=tenor, coupon=coupon,
                   hazard=hazard, recovery=recovery, leverage=leverage,
                   notional=notional)


def trs(tenor, underlying, curve, *, funding_spread=0.005, repo_spread=0.01,
        notional=None, sigma=None) -> dict:
    """Price an equity TRS in one call.

        desk.trs("6M", 100.0, curve)  # equity TRS, spot=100
        desk.trs("6M", 100.0, curve, sigma=0.25)  # with explicit vol

    Note: `underlying` must be a numeric spot price (equity TRS).
    For bond/loan/CLN TRS, use the full TotalReturnSwap class directly.
    """
    if float(underlying) <= 0:
        raise ValueError(f"underlying (spot price) must be positive, got {underlying}")
    _check_rate_not_bps(funding_spread, "funding_spread")
    _check_rate_not_bps(repo_spread, "repo_spread")
    if notional is None:
        _warn_default("notional", "10_000_000", "TRS")
        notional = 10_000_000
    if sigma is None:
        _warn_default("sigma", 0.20, "TRS vega")
        sigma = 0.20

    from pricebook.equity.trs import TotalReturnSwap, FundingLegSpec
    from pricebook.desks.trs_desk import trs_risk_metrics, trs_carry_decomposition

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


def repo(tenor_days, face, rate, *, haircut=0.05) -> dict:
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
    _check_rate_not_bps(rate, "rate")

    cash_lent = face * (1 - haircut)
    interest = cash_lent * rate * tenor_days / 360  # ACT/360 convention
    maturity_amount = cash_lent + interest

    return {
        "type": "repo", "face": face, "cash_lent": cash_lent,
        "rate": rate, "haircut": haircut, "tenor_days": tenor_days,
        "interest": interest, "maturity_amount": maturity_amount,
        "carry": {"interest": interest, "funding": 0.0, "net": interest},
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
    from pricebook.options.vol_calibration import (
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
    for i, q in enumerate(quotes):
        if "expiry" not in q:
            raise ValueError(f"Missing 'expiry' key in vol quote #{i+1}: got {list(q.keys())}")
        pq = dict(q)
        if isinstance(pq["expiry"], str):
            pq["expiry"] = _parse_tenor(ref, pq["expiry"])
        parsed.append(pq)

    # Validate spot
    if spot is not None and spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")

    ac = asset_class.lower()
    if ac == "fx":
        return calibrate_fx_surface(ref, parsed, spot=spot if spot is not None else 1.0)
    elif ac in ("equity", "eq"):
        return calibrate_equity_surface(ref, parsed, spot=spot if spot is not None else 100.0)
    elif ac in ("ir", "swaption"):
        return calibrate_ir_surface(ref, parsed)
    elif ac in ("commodity", "commo"):
        return calibrate_commodity_surface(ref, parsed, spot=spot if spot is not None else 75.0)
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
    if not trades:
        raise ValueError("trades list is empty — provide at least one swap trade.")

    from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
    from pricebook.desks.swap_desk import SwapBook, SwapBookEntry, swap_dashboard, swap_stress_suite

    ref = curve.reference_date
    book = SwapBook()

    for i, t in enumerate(trades):
        _require_keys(t, ["tenor", "rate"], f"swap trade #{i+1}")
        _check_rate_not_bps(t["rate"], f"rate in swap trade #{i+1}")
        if "notional" not in t:
            _warn_default("notional", "10_000_000", f"swap trade #{i+1}")
        dir_str = t.get("direction", "payer").lower().strip()
        if dir_str in ("payer", "pay", "p"):
            direction = SwapDirection.PAYER
        elif dir_str in ("receiver", "recv", "receive", "r"):
            direction = SwapDirection.RECEIVER
        else:
            raise ValueError(f"Invalid direction '{t.get('direction')}' in swap trade #{i+1}. "
                             "Use 'payer'/'pay'/'p' or 'receiver'/'recv'/'r'.")
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
    if not trades:
        raise ValueError("trades list is empty — provide at least one CDS trade.")

    from pricebook.credit.cds import CDS
    from pricebook.desks.cds_desk import CDSBook, CDSBookEntry, cds_dashboard, cds_stress_suite

    ref = curve.reference_date
    book = CDSBook()

    for i, t in enumerate(trades):
        _require_keys(t, ["tenor", "spread"], f"CDS trade #{i+1}")
        _check_rate_not_bps(t["spread"], f"spread in CDS trade #{i+1}")
        if "notional" not in t:
            _warn_default("notional", "10_000_000", f"CDS trade #{i+1}")
        # h ≈ spread / (1-R): standard flat hazard approximation (O'Kane 2008)
        recovery = t.get("recovery", 0.40)
        if not 0 <= recovery < 1:
            raise ValueError(f"recovery must be in [0, 1) in CDS trade #{i+1}, got {recovery}")
        hazard = t.get("hazard", t["spread"] / (1 - recovery))
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
    from pricebook.fixed_income.sofr_curve import build_sofr_curve, build_estr_curve, build_sonia_curve

    if ref is None:
        raise ValueError("ref (reference date) is required for multicurve. "
                         "Pass ref=date(2024, 7, 15) or similar.")
    if not currencies:
        raise ValueError("No currencies provided — pass at least one (e.g., usd={...}).")

    result = {}
    for ccy, data in currencies.items():
        swaps_dict = data.get("swaps", {})
        if not swaps_dict:
            raise ValueError(f"No swap rates provided for {ccy.upper()}. "
                             f"Pass swaps={{...}} with at least one tenor:rate pair.")
        swap_list = []
        for t, r in swaps_dict.items():
            if isinstance(t, date):
                swap_list.append((t, r))
            elif isinstance(t, str):
                swap_list.append((_parse_tenor(ref, t), r))
            elif isinstance(t, (int, float)):
                # Numeric tenor interpreted as years (e.g., 5 → "5Y")
                if t <= 0:
                    raise ValueError(f"Tenor must be positive, got {t} for {ccy.upper()}")
                swap_list.append((_parse_tenor(ref, f"{t}Y"), r))
            else:
                raise ValueError(f"Unsupported tenor key type: {type(t)} for {t}")

        ccy_upper = ccy.upper()
        if ccy_upper == "USD":
            result[ccy_upper] = build_sofr_curve(ref, sofr_swaps=swap_list)
        elif ccy_upper == "EUR":
            result[ccy_upper] = build_estr_curve(ref, swap_list)
        elif ccy_upper == "GBP":
            result[ccy_upper] = build_sonia_curve(ref, swap_list)
        else:
            from pricebook.fixed_income.ois import bootstrap_ois
            result[ccy_upper] = bootstrap_ois(ref, swap_list)

    return result


# ============================================================================
# SECTION 6: Recovery + desk analytics
# ============================================================================

def recovery_analysis(*, cds_spreads: dict, curve: DiscountCurve,
                      tenor="5Y", coupon=0.05, recovery=0.40,
                      recoveries=None) -> dict:
    """Full recovery-hazard analysis for a CLN.

        desk.recovery_analysis(
            cds_spreads={1: 0.005, 5: 0.01, 10: 0.012},
            curve=curve, tenor="5Y", coupon=0.05)

    Args:
        recovery: Base recovery assumption for the CLN (default 0.40).
        recoveries: List of recovery values to scan (default [0.2, 0.3, ..., 0.8]).
    """
    from pricebook.credit.cln import CreditLinkedNote
    from pricebook.core.schedule import Frequency
    from pricebook.credit.recovery_analytics import (
        recovery_curve_family, recovery_greeks, recovery_pv_surface,
    )

    if not cds_spreads:
        raise ValueError("cds_spreads is empty — provide at least one tenor:spread pair.")
    _check_rate_not_bps(coupon, "coupon")
    if not 0 <= recovery < 1:
        raise ValueError(f"recovery must be in [0, 1), got {recovery}")

    ref = curve.reference_date
    cln_inst = CreditLinkedNote(ref, _parse_tenor(ref, tenor), coupon_rate=coupon,
                                notional=10_000_000, recovery=recovery,
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
