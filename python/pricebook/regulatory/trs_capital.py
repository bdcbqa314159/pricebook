"""Regulatory capital for Total Return Swaps.

SA-CCR exposure, SIMM sensitivities, KVA, and leverage ratio treatment
for TRS positions across equity, bond, loan, and CLN underlyings.

    from pricebook.regulatory.trs_capital import (
        trs_sa_ccr_add_on, trs_simm_sensitivities, trs_kva, trs_leverage_exposure,
    )

References:
    Basel III: CRE52 (SA-CCR), MAR50 (CVA), CRE10 (leverage).
    ISDA SIMM Methodology v2.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.regulatory.counterparty import (
    calculate_maturity_factor,
    calculate_supervisory_duration,
    SA_CCR_SUPERVISORY_FACTORS,
)


# ---- SA-CCR classification ----

_TRS_ASSET_CLASS_MAP = {
    "equity": "EQ_SINGLE",
    "bond": "CR_BBB",        # default; can override with rating
    "loan": "CR_BBB",
    "cln": "CR_BBB",
    "unknown": "EQ_SINGLE",
}

_RATING_TO_CR = {
    "AAA": "CR_AAA_AA", "AA": "CR_AAA_AA",
    "A": "CR_A",
    "BBB": "CR_BBB",
    "BB": "CR_BB",
    "B": "CR_B",
    "CCC": "CR_CCC",
}


def _trs_asset_class(underlying_type: str, rating: str = "BBB") -> str:
    """Map TRS underlying type to SA-CCR asset class."""
    if underlying_type in ("bond", "loan", "cln"):
        return _RATING_TO_CR.get(rating, "CR_BBB")
    return _TRS_ASSET_CLASS_MAP.get(underlying_type, "EQ_SINGLE")


@dataclass
class TRSSACCRResult:
    """SA-CCR result for a TRS position."""
    ead: float
    replacement_cost: float
    pfe: float
    add_on: float
    asset_class: str
    supervisory_factor: float
    maturity_factor: float
    adjusted_notional: float


def trs_sa_ccr_add_on(
    trs,
    curve: DiscountCurve,
    rating: str = "BBB",
    is_margined: bool = False,
    mpor_days: int = 10,
    alpha: float = 1.4,
) -> TRSSACCRResult:
    """Compute SA-CCR EAD for a TRS position.

    EAD = alpha × (RC + PFE)
    RC  = max(V, 0)  (replacement cost, unmargined)
    PFE = multiplier × AddOn

    AddOn = SF × delta × d × MF × adjusted_notional

    Args:
        trs: TotalReturnSwap instance.
        curve: discount curve for MTM.
        rating: credit rating of reference entity (for bond/loan/CLN).
        is_margined: whether the position is margined.
        alpha: SA-CCR alpha multiplier (1.4 standard).
    """
    # Current MTM
    result = trs.price(curve)
    mtm = result.value

    # Replacement cost
    rc = max(mtm, 0.0)

    # Asset class and supervisory factor
    asset_class = _trs_asset_class(trs._underlying_type, rating)
    sf_data = SA_CCR_SUPERVISORY_FACTORS.get(asset_class, {"SF": 32.0})
    sf = sf_data["SF"] / 100.0  # convert from pct

    # Maturity
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    mf = calculate_maturity_factor(T, is_margined, mpor_days)

    # Adjusted notional
    if trs._underlying_type == "equity":
        # For equity: adjusted_notional = notional × spot / initial
        spot = float(trs.underlying) if isinstance(trs.underlying, (int, float)) else 100.0
        initial = trs.initial_price or spot
        adj_notional = trs.notional * spot / initial
    else:
        # For credit: adjusted_notional = notional × supervisory_duration
        sd = calculate_supervisory_duration(T)
        adj_notional = trs.notional * sd

    # Delta: +1 for long (TR receiver), assuming long position
    delta = 1.0

    # Add-on
    add_on = sf * delta * mf * adj_notional

    # PFE with multiplier
    # Simplified: multiplier = 1 when no excess collateral
    pfe = add_on

    ead = alpha * (rc + pfe)

    return TRSSACCRResult(
        ead=ead, replacement_cost=rc, pfe=pfe, add_on=add_on,
        asset_class=asset_class, supervisory_factor=sf,
        maturity_factor=mf, adjusted_notional=adj_notional,
    )


# ---- SIMM tenor mapping ----

# GIRR tenors (years) for bucketing
_GIRR_TENOR_MAP = [
    (2/52, "2W"), (1/12, "1M"), (3/12, "3M"), (6/12, "6M"),
    (1, "1Y"), (2, "2Y"), (3, "3Y"), (5, "5Y"),
    (10, "10Y"), (15, "15Y"), (20, "20Y"), (30, "30Y"),
]


def _map_time_to_girr_tenor(t: float) -> str:
    """Map a year fraction to the nearest GIRR tenor bucket."""
    best = "1Y"
    best_dist = float("inf")
    for tenor_t, tenor_name in _GIRR_TENOR_MAP:
        dist = abs(t - tenor_t)
        if dist < best_dist:
            best_dist = dist
            best = tenor_name
    return best


# ---- SIMM sensitivities ----

@dataclass
class TRSSIMMSensitivities:
    """SIMM sensitivity extraction from a TRS."""
    delta_sensitivities: list[dict]
    vega_sensitivities: list[dict]
    risk_class: str
    total_delta: float
    total_vega: float


def trs_simm_sensitivities(
    trs,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> TRSSIMMSensitivities:
    """Extract SIMM delta and vega sensitivities from a TRS.

    Maps TRS greeks to SIMM risk classes:
    - Equity TRS → EQ (delta, vega)
    - Bond/Loan TRS → GIRR (rate delta) + CSR (credit spread delta)
    - CLN TRS → CSR (credit spread delta)
    """
    greeks = trs.greeks(curve, projection_curve)

    if trs._underlying_type == "equity":
        risk_class = "EQ"
        # For equity TRS, delta ≈ notional / S0 (number of shares)
        # When at inception, bump-and-reprice gives 0 because S0 = spot.
        # The SIMM delta is the position's market value sensitivity.
        spot = float(trs.underlying) if isinstance(trs.underlying, (int, float)) else 100.0
        equity_delta = trs.notional / spot if spot > 0 else trs.notional
        delta_sens = [{
            "risk_class": "EQ",
            "bucket": "single_name",
            "tenor": "spot",
            "delta": equity_delta,
        }]
        vega_sens = [{
            "risk_class": "EQ",
            "bucket": "single_name",
            "tenor": "spot",
            "vega": trs.sigma * trs.notional * 0.01,  # 1% vol bump
        }]
    elif trs._underlying_type in ("bond", "loan"):
        risk_class = "GIRR"
        base_val = trs.price(curve, projection_curve).value

        # Per-pillar rate sensitivity → multi-tenor GIRR bucketing
        pillar_times = [t for t in curve.pillar_times if t > 0]
        n_pillars = len(pillar_times)
        delta_sens = []
        for i in range(n_pillars):
            bumped_i = curve.bumped_at(i, 0.0001)
            dv01_i = trs.price(bumped_i, projection_curve).value - base_val
            tenor = _map_time_to_girr_tenor(pillar_times[i])
            delta_sens.append({
                "risk_class": "GIRR",
                "bucket": "USD",
                "tenor": tenor,
                "delta": dv01_i,
            })

        # CSR: credit spread sensitivity via spread bump on funding leg
        old_spread = trs.funding.spread
        trs.funding = type(trs.funding)(
            spread=old_spread + 0.0001,
            **{k: v for k, v in trs.funding.__dict__.items() if k != "spread"},
        )
        cs_delta = trs.price(curve, projection_curve).value - base_val
        trs.funding = type(trs.funding)(
            spread=old_spread,
            **{k: v for k, v in trs.funding.__dict__.items() if k != "spread"},
        )

        delta_sens.append({
            "risk_class": "CSR",
            "bucket": "IG_corporate",
            "tenor": "5Y",
            "delta": cs_delta,
        })
        vega_sens = []
    elif trs._underlying_type == "cln":
        risk_class = "CSR"
        # CLN: credit spread sensitivity via survival curve bump
        base_val = trs.price(curve, projection_curve).value
        bumped = curve.bumped(0.0001)
        cs_delta = trs.price(bumped, projection_curve).value - base_val
        delta_sens = [{
            "risk_class": "CSR",
            "bucket": "IG_corporate",
            "tenor": "5Y",
            "delta": cs_delta,
        }]
        vega_sens = []
    else:
        risk_class = "CSR"
        delta_sens = [{
            "risk_class": "CSR",
            "bucket": "IG_corporate",
            "tenor": "5Y",
            "delta": trs.notional * 0.0001,
        }]
        vega_sens = []

    total_delta = sum(abs(s["delta"]) for s in delta_sens)
    total_vega = sum(abs(s.get("vega", 0)) for s in vega_sens)

    return TRSSIMMSensitivities(
        delta_sensitivities=delta_sens,
        vega_sensitivities=vega_sens,
        risk_class=risk_class,
        total_delta=total_delta,
        total_vega=total_vega,
    )


# ---- KVA for TRS ----

@dataclass
class TRSKVAResult:
    """KVA result for a TRS position."""
    kva: float
    ead_profile: list[float]
    capital_profile: list[float]
    time_grid: list[float]
    hurdle_rate: float


def trs_kva(
    trs,
    curve: DiscountCurve,
    hurdle_rate: float = 0.10,
    rating: str = "BBB",
    rwa_multiplier: float = 1.0,
    n_steps: int = 4,
) -> TRSKVAResult:
    """Compute KVA for a TRS using SA-CCR EAD profile.

    KVA = Σ K(t_i) × hurdle_rate × dt_i × df(t_i)

    Capital K(t_i) = 8% × RW × EAD(t_i), where RW is based on rating.

    Args:
        hurdle_rate: cost of capital (e.g. 10%).
        rwa_multiplier: risk weight multiplier (default 1.0 = 100% RW).
        n_steps: number of time steps for capital profile.
    """
    from pricebook.xva import kva

    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    dt = T / n_steps

    # Build EAD and capital profiles over time
    time_grid = [i * dt for i in range(n_steps + 1)]
    ead_profile = []
    capital_profile = []

    base_ead = trs_sa_ccr_add_on(trs, curve, rating=rating)

    for t in time_grid:
        # EAD decays as maturity shortens (simplified: linear decay)
        remaining = max(T - t, 0.0)
        scale = remaining / T if T > 0 else 0.0
        ead_t = base_ead.ead * scale
        ead_profile.append(ead_t)

        # Capital = 8% × RW × EAD
        capital_t = 0.08 * rwa_multiplier * ead_t
        capital_profile.append(capital_t)

    # KVA via numerical integration
    kva_value = kva(
        np.array(capital_profile),
        time_grid,
        curve,
        hurdle_rate,
    )

    return TRSKVAResult(
        kva=kva_value,
        ead_profile=ead_profile,
        capital_profile=capital_profile,
        time_grid=time_grid,
        hurdle_rate=hurdle_rate,
    )


# ---- Leverage ratio ----

@dataclass
class TRSLeverageResult:
    """Leverage ratio exposure for a TRS."""
    exposure: float
    mtm_component: float
    add_on_component: float
    is_off_balance_sheet: bool


def trs_leverage_exposure(
    trs,
    curve: DiscountCurve,
    rating: str = "BBB",
) -> TRSLeverageResult:
    """Compute leverage ratio exposure for a TRS.

    TRS is off-balance-sheet: exposure = max(0, MTM) + SA-CCR add-on.
    For financing TRS (repo replacement), notional may be used instead.
    """
    result = trs.price(curve)
    mtm = max(result.value, 0.0)

    sa_ccr = trs_sa_ccr_add_on(trs, curve, rating=rating)

    return TRSLeverageResult(
        exposure=mtm + sa_ccr.add_on,
        mtm_component=mtm,
        add_on_component=sa_ccr.add_on,
        is_off_balance_sheet=True,
    )
