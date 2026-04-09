"""FRTB Standardised Approach: SbM (delta/vega/curvature), DRC, RRAO.

Full sensitivities-based method across all 5 risk classes (GIRR, CSR, EQ, FX, COM),
Default Risk Charge with obligor netting, and Residual Risk Add-On.

    from pricebook.regulatory.market_risk_sa import (
        calculate_delta_capital, calculate_drc_charge, calculate_frtb_sa,
    )
"""

from __future__ import annotations

import math


# ---- Risk Weights (MAR21) ----

GIRR_RISK_WEIGHTS: dict[float, float] = {
    0.25: 1.7, 0.5: 1.7, 1: 1.6, 2: 1.3, 3: 1.2,
    5: 1.1, 10: 1.1, 15: 1.1, 20: 1.1, 30: 1.1,
}
GIRR_INFLATION_RW = 1.6
GIRR_CROSS_CURRENCY_RW = 1.6

CSR_RISK_WEIGHTS: dict[str, tuple[float, ...]] = {
    "sovereign_IG": (0.5, 0.5, 0.5, 0.5),
    "sovereign_HY": (3.0, 3.0, 3.0, 3.0),
    "corporate_IG": (1.0, 1.0, 1.0, 1.0),
    "corporate_HY": (5.0, 5.0, 5.0, 5.0),
    "financial_IG": (1.5, 1.5, 1.5, 1.5),
    "financial_HY": (7.5, 7.5, 7.5, 7.5),
    "covered_bond": (1.0, 1.0, 1.0, 1.0),
}

EQ_RISK_WEIGHTS: dict[str, tuple[float, float]] = {
    "large_cap_developed": (20, 0.55),
    "large_cap_emerging": (30, 0.60),
    "small_cap_developed": (30, 0.60),
    "small_cap_emerging": (50, 0.75),
    "volatility_index": (70, 0.10),
    "other": (50, 0.75),
}

FX_RISK_WEIGHT = 15.0
FX_LIQUID_PAIRS_RW = 11.25

COM_RISK_WEIGHTS: dict[str, float] = {
    "energy_solid": 30, "energy_liquid": 25, "energy_electricity": 60,
    "freight": 80, "metals_precious": 20, "metals_non_precious": 30,
    "agriculture_grains": 25, "agriculture_softs": 30, "other": 50,
}

CORRELATIONS: dict[str, float] = {
    "GIRR_same_curve": 0.99, "GIRR_different_curve": 0.50,
    "CSR_same_bucket": 0.75, "CSR_different_bucket": 0.25,
    "EQ_same_bucket": 0.20, "EQ_different_bucket": 0.15,
    "FX": 0.60,
    "COM_same_bucket": 0.55, "COM_different_bucket": 0.20,
}


# ---- Aggregation ----

def aggregate_within_bucket(sensitivities: list[float], correlation: float = 0.50) -> float:
    """K_b = sqrt(Σs_i² + ρ × ((Σs_i)² - Σs_i²))."""
    sum_s = sum(sensitivities)
    sum_sq = sum(s ** 2 for s in sensitivities)
    return math.sqrt(max(sum_sq + correlation * (sum_s ** 2 - sum_sq), 0))


def aggregate_across_buckets(
    bucket_capitals: dict[str, float],
    bucket_net_sens: dict[str, float],
    inter_corr: float = 0.25,
) -> float:
    """K = sqrt(ΣK_b² + 2 × Σγ_{bc} S_b S_c)."""
    buckets = list(bucket_capitals.keys())
    sum_k_sq = sum(k ** 2 for k in bucket_capitals.values())
    cross = 0.0
    for i, b1 in enumerate(buckets):
        for b2 in buckets[i + 1:]:
            cross += inter_corr * bucket_net_sens.get(b1, 0) * bucket_net_sens.get(b2, 0)
    return math.sqrt(max(sum_k_sq + 2 * cross, 0))


# ---- Delta capital ----

def calculate_delta_capital(positions: list[dict], risk_class: str = "GIRR") -> dict:
    """Delta capital charge for a risk class.

    positions: list of {bucket, sensitivity, risk_weight}.
    """
    if risk_class == "GIRR":
        intra, inter = CORRELATIONS["GIRR_same_curve"], CORRELATIONS["GIRR_different_curve"]
    elif risk_class == "CSR":
        intra, inter = CORRELATIONS["CSR_same_bucket"], CORRELATIONS["CSR_different_bucket"]
    elif risk_class == "EQ":
        intra, inter = CORRELATIONS["EQ_same_bucket"], CORRELATIONS["EQ_different_bucket"]
    elif risk_class == "FX":
        intra, inter = CORRELATIONS["FX"], CORRELATIONS["FX"]
    else:
        intra, inter = CORRELATIONS["COM_same_bucket"], CORRELATIONS["COM_different_bucket"]

    buckets: dict[str, list[float]] = {}
    for pos in positions:
        b = pos.get("bucket", "default")
        s = pos.get("sensitivity", 0) * pos.get("risk_weight", 1.0) / 100
        buckets.setdefault(b, []).append(s)

    bucket_caps = {b: aggregate_within_bucket(ws, intra) for b, ws in buckets.items()}
    bucket_net = {b: sum(ws) for b, ws in buckets.items()}
    k = aggregate_across_buckets(bucket_caps, bucket_net, inter)

    return {
        "risk_class": risk_class, "component": "delta", "capital": k,
        "bucket_capitals": bucket_caps, "bucket_net_sensitivities": bucket_net,
    }


# ---- Vega capital ----

def calculate_vega_capital(positions: list[dict], risk_class: str = "EQ") -> dict:
    """Vega capital charge for a risk class."""
    if risk_class == "EQ":
        intra, inter = 0.60, 0.20
    else:
        intra, inter = 0.50, 0.25

    buckets: dict[str, list[float]] = {}
    for pos in positions:
        b = pos.get("bucket", "default")
        v = pos.get("vega", 0) * pos.get("vega_risk_weight", 50) / 100
        buckets.setdefault(b, []).append(v)

    bucket_caps = {b: aggregate_within_bucket(ws, intra) for b, ws in buckets.items()}
    bucket_net = {b: sum(ws) for b, ws in buckets.items()}
    k = aggregate_across_buckets(bucket_caps, bucket_net, inter)

    return {"risk_class": risk_class, "component": "vega", "capital": k, "bucket_capitals": bucket_caps}


# ---- Curvature capital ----

def calculate_curvature_capital(positions: list[dict], risk_class: str = "EQ") -> dict:
    """Curvature capital charge."""
    buckets: dict[str, dict[str, float]] = {}
    for pos in positions:
        b = pos.get("bucket", "default")
        if b not in buckets:
            buckets[b] = {"cvr_up": 0, "cvr_down": 0}
        buckets[b]["cvr_up"] += pos.get("cvr_up", 0)
        buckets[b]["cvr_down"] += pos.get("cvr_down", 0)

    bucket_caps = {b: max(c["cvr_up"], c["cvr_down"], 0) for b, c in buckets.items()}
    k = sum(bucket_caps.values())

    return {"risk_class": risk_class, "component": "curvature", "capital": k, "bucket_capitals": bucket_caps}


# ---- SbM total ----

def calculate_sbm_capital(
    delta_positions: list[dict],
    vega_positions: list[dict] | None = None,
    curvature_positions: list[dict] | None = None,
    risk_class: str = "EQ",
) -> dict:
    """Total SbM capital = delta + vega + curvature."""
    d = calculate_delta_capital(delta_positions, risk_class)
    v = calculate_vega_capital(vega_positions or [], risk_class) if vega_positions else {"capital": 0}
    c = calculate_curvature_capital(curvature_positions or [], risk_class) if curvature_positions else {"capital": 0}
    total = d["capital"] + v.get("capital", 0) + c.get("capital", 0)
    return {
        "risk_class": risk_class, "approach": "SbM",
        "delta_capital": d["capital"], "vega_capital": v.get("capital", 0),
        "curvature_capital": c.get("capital", 0), "total_capital": total,
        "delta_detail": d, "vega_detail": v, "curvature_detail": c,
    }


# ---- DRC (MAR22) ----

DRC_RISK_WEIGHTS: dict[str, float] = {
    "AAA": 0.5, "AA": 0.5, "A": 1.0, "BBB": 2.0, "BB": 5.0,
    "B": 10.0, "CCC": 15.0, "D": 30.0, "unrated": 15.0,
}

DRC_LGD: dict[str, float] = {
    "senior": 0.75, "subordinated": 1.0, "covered_bond": 0.625, "equity": 1.0,
}


def calculate_drc_charge(positions: list[dict]) -> dict:
    """Default Risk Charge with obligor netting.

    positions: list of {obligor, notional, rating, seniority, sector, is_long}.
    """
    obligors: dict[str, dict] = {}
    for pos in positions:
        ob = pos.get("obligor", "default")
        if ob not in obligors:
            obligors[ob] = {"positions": [], "sector": pos.get("sector", "other"), "rating": pos.get("rating", "BBB")}
        obligors[ob]["positions"].append(pos)

    total_drc = 0.0
    by_obligor: dict[str, dict] = {}
    for ob, data in obligors.items():
        jtd_long = jtd_short = 0.0
        for p in data["positions"]:
            lgd = DRC_LGD.get(p.get("seniority", "senior"), 0.75)
            jtd = p["notional"] * lgd
            if p.get("is_long", True):
                jtd_long += jtd
            else:
                jtd_short += jtd

        net = jtd_long - jtd_short
        rw = DRC_RISK_WEIGHTS.get(data["rating"], DRC_RISK_WEIGHTS["unrated"]) / 100
        drc = max(net, 0) * rw
        by_obligor[ob] = {"net_jtd": net, "risk_weight": rw, "drc": drc}
        total_drc += drc

    return {"approach": "DRC", "total_drc": total_drc, "rwa": total_drc * 12.5, "obligors": by_obligor}


# ---- RRAO (MAR23) ----

RRAO_EXOTIC_RATE = 0.01
RRAO_OTHER_RATE = 0.001


def calculate_rrao(positions: list[dict]) -> dict:
    """Residual Risk Add-On."""
    exotic = sum(p.get("notional", 0) for p in positions if p.get("is_exotic"))
    other = sum(p.get("notional", 0) for p in positions if p.get("has_other_residual_risk"))
    ec = exotic * RRAO_EXOTIC_RATE
    oc = other * RRAO_OTHER_RATE
    total = ec + oc
    return {
        "approach": "RRAO", "exotic_notional": exotic, "exotic_charge": ec,
        "other_notional": other, "other_charge": oc,
        "total_rrao": total, "rwa": total * 12.5,
    }


# ---- Total FRTB SA ----

def calculate_frtb_sa(
    delta_positions: dict[str, list[dict]],
    vega_positions: dict[str, list[dict]] | None = None,
    curvature_positions: dict[str, list[dict]] | None = None,
    drc_positions: list[dict] | None = None,
    rrao_positions: list[dict] | None = None,
) -> dict:
    """Full FRTB-SA: K = SbM + DRC + RRAO.

    delta_positions: {risk_class: [positions]}.
    """
    vega_positions = vega_positions or {}
    curvature_positions = curvature_positions or {}

    sbm_results = {}
    total_sbm = 0.0
    for rc, pos in delta_positions.items():
        r = calculate_sbm_capital(pos, vega_positions.get(rc), curvature_positions.get(rc), rc)
        sbm_results[rc] = r
        total_sbm += r["total_capital"]

    drc = calculate_drc_charge(drc_positions) if drc_positions else {"total_drc": 0, "rwa": 0}
    rrao = calculate_rrao(rrao_positions) if rrao_positions else {"total_rrao": 0, "rwa": 0}

    total = total_sbm + drc["total_drc"] + rrao["total_rrao"]
    return {
        "approach": "FRTB-SA",
        "sbm_capital": total_sbm, "sbm_by_risk_class": sbm_results,
        "drc_capital": drc["total_drc"], "drc_detail": drc,
        "rrao_capital": rrao["total_rrao"], "rrao_detail": rrao,
        "total_capital": total, "total_rwa": total * 12.5,
    }
