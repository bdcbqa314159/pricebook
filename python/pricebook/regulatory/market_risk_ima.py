"""FRTB Internal Models Approach (MAR30-33).

ES with liquidity horizons, Stressed ES, NMRF charge, internal DRC,
backtesting, P&L attribution test, and total IMA capital.

Capital formula:
    IMA Capital = IMCC + DRC_IMA
    IMCC = max(ES_t-1, m_c × ES_avg) + max(SES_t-1, m_c × SES_avg) + NMRF
    m_c = 1.5 × (1 + plus_factor)

    from pricebook.regulatory.market_risk_ima import (
        ESRiskFactor, DRCPosition, FRTBIMAConfig,
        calculate_liquidity_adjusted_es, calculate_stressed_es,
        calculate_nmrf_charge, calculate_ima_drc, calculate_imcc,
        calculate_frtb_ima_capital, evaluate_backtesting, evaluate_pla,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from pricebook.regulatory.market_risk_sa import calculate_frtb_sa
from pricebook.regulatory.ratings import RATING_TO_PD


# ---- Dataclasses ----

@dataclass
class ESRiskFactor:
    """A single risk factor for Expected Shortfall calculation."""
    risk_class: str
    sub_category: str
    es_10day: float
    is_modellable: bool = True
    stressed_es_10day: float | None = None


@dataclass
class DRCPosition:
    """A single position for the internal DRC model."""
    position_id: str
    obligor: str
    notional: float
    market_value: float
    pd: float
    lgd: float = 0.45
    seniority: str = "senior_unsecured"
    sector: str = "corporate"
    systematic_factor: float = 0.20
    is_long: bool = True


@dataclass
class DeskPLA:
    """P&L Attribution Test results for a single desk."""
    desk_id: str
    spearman_correlation: float
    kl_divergence: float


@dataclass
class FRTBIMAConfig:
    """Configuration for FRTB-IMA."""
    multiplication_factor: float = 1.5
    plus_factor: float = 0.0
    es_confidence: float = 0.975
    drc_confidence: float = 0.999
    drc_horizon_years: float = 1.0
    drc_num_simulations: int = 50_000
    backtesting_exceptions: int = 0


# ---- Liquidity Horizons (MAR31.13) ----

LIQUIDITY_HORIZONS: dict[tuple[str, str], int] = {
    ("IR", "major"): 10,
    ("IR", "other"): 20,
    ("CR", "IG_sovereign"): 20,
    ("EQ", "large_cap"): 40,
    ("FX", "major"): 40,
    ("CR", "IG_corporate"): 40,
    ("EQ", "small_cap"): 60,
    ("FX", "other"): 60,
    ("COM", "energy"): 60,
    ("COM", "precious_metals"): 60,
    ("CR", "HY"): 60,
    ("EQ", "other"): 120,
    ("COM", "other"): 120,
    ("CR", "other"): 120,
}

LH_STEPS = [10, 20, 40, 60, 120]


def get_liquidity_horizon(risk_class: str, sub_category: str) -> int:
    """Map (risk_class, sub_category) → liquidity horizon in days."""
    return LIQUIDITY_HORIZONS.get((risk_class, sub_category), 120)


# ---- Liquidity-adjusted Expected Shortfall ----

def calculate_liquidity_adjusted_es(risk_factors: list[ESRiskFactor]) -> dict:
    """Liquidity-adjusted ES via cascading sum (MAR31.12).

    ES = sqrt(Σ_j [ES_j(10) × sqrt((LH_j - LH_{j-1}) / 10)]²)
    """
    modellable = [rf for rf in risk_factors if rf.is_modellable]
    if not modellable:
        return {"es_total": 0.0, "es_by_bucket": {}}

    factor_lh: dict[int, list[ESRiskFactor]] = {}
    for rf in modellable:
        lh = get_liquidity_horizon(rf.risk_class, rf.sub_category)
        factor_lh.setdefault(lh, []).append(rf)

    es_by_bucket: dict[int, dict] = {}
    variance_sum = 0.0
    prev_lh = 0
    for lh in LH_STEPS:
        factors = factor_lh.get(lh, [])
        es_10 = sum(rf.es_10day for rf in factors)
        if es_10 > 0:
            scale = math.sqrt((lh - prev_lh) / 10.0)
            contrib = (es_10 * scale) ** 2
            variance_sum += contrib
            es_by_bucket[lh] = {"es_10day_sum": es_10, "scale_factor": scale, "contribution": contrib}
        prev_lh = lh

    es_total = math.sqrt(variance_sum) if variance_sum > 0 else 0.0
    return {"es_total": es_total, "es_by_bucket": es_by_bucket, "num_factors": len(modellable)}


def calculate_stressed_es(
    risk_factors: list[ESRiskFactor],
    es_full_current: float,
    es_reduced_current: float,
) -> dict:
    """Stressed ES (MAR31.16-18) via ratio scaling.

    SES = ES_reduced_stressed × (ES_full_current / ES_reduced_current)
    """
    stressed = [
        ESRiskFactor(
            risk_class=rf.risk_class,
            sub_category=rf.sub_category,
            es_10day=rf.stressed_es_10day if rf.stressed_es_10day is not None else rf.es_10day,
            is_modellable=rf.is_modellable,
        )
        for rf in risk_factors
        if rf.is_modellable and rf.stressed_es_10day is not None
    ]
    if not stressed or es_reduced_current <= 0:
        return {"ses_total": 0.0, "ratio": 0.0, "es_reduced_stressed": 0.0}

    es_reduced_stressed = calculate_liquidity_adjusted_es(stressed)["es_total"]
    ratio = es_full_current / es_reduced_current if es_reduced_current > 0 else 1.0
    ses_total = es_reduced_stressed * ratio

    return {
        "ses_total": ses_total, "ratio": ratio,
        "es_reduced_stressed": es_reduced_stressed,
        "es_full_current": es_full_current, "es_reduced_current": es_reduced_current,
    }


def calculate_nmrf_charge(risk_factors: list[ESRiskFactor]) -> dict:
    """NMRF add-on (MAR31.24-31): zero-diversification sum."""
    nmrf = [rf for rf in risk_factors if not rf.is_modellable]
    if not nmrf:
        return {"nmrf_total": 0.0, "factors": []}

    details = []
    total = 0.0
    for rf in nmrf:
        charge = rf.stressed_es_10day if rf.stressed_es_10day is not None else rf.es_10day
        lh = get_liquidity_horizon(rf.risk_class, rf.sub_category)
        scaled = charge * math.sqrt(lh / 10.0)
        total += scaled
        details.append({
            "risk_class": rf.risk_class, "sub_category": rf.sub_category,
            "charge_10day": charge, "liquidity_horizon": lh, "scaled_charge": scaled,
        })

    return {"nmrf_total": total, "num_factors": len(nmrf), "factors": details}


# ---- Internal DRC (MAR32) ----

def simulate_drc_portfolio(
    positions: list[DRCPosition],
    num_simulations: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo DRC via two-factor Gaussian copula.

    Z_i = ρ_i X + sqrt(1 - ρ_i²) ε_i
    Default if Z_i < Φ⁻¹(PD_i)
    """
    rng = np.random.default_rng(seed)

    obligor_pos: dict[str, list[DRCPosition]] = {}
    for pos in positions:
        obligor_pos.setdefault(pos.obligor, []).append(pos)

    obligor_info = []
    for ob, pos_list in obligor_pos.items():
        threshold = float(norm.ppf(max(min(pos_list[0].pd, 0.999999), 1e-10)))
        rho = pos_list[0].systematic_factor
        net_loss_on_default = 0.0
        for p in pos_list:
            sign = 1.0 if p.is_long else -1.0
            net_loss_on_default += sign * p.notional * p.lgd
        obligor_info.append((threshold, rho, max(net_loss_on_default, 0.0)))

    n_obligors = len(obligor_info)
    if n_obligors == 0:
        return np.zeros(num_simulations)

    # Vectorised: shape (n_sims, n_obligors)
    systematic = rng.standard_normal(num_simulations)
    idio = rng.standard_normal((num_simulations, n_obligors))

    losses = np.zeros(num_simulations)
    for j, (thresh, rho, loss) in enumerate(obligor_info):
        z = rho * systematic + math.sqrt(1.0 - rho * rho) * idio[:, j]
        defaults = z < thresh
        losses += defaults * loss

    return losses


def calculate_ima_drc(
    positions: list[DRCPosition],
    config: FRTBIMAConfig | None = None,
) -> dict:
    """Internal DRC charge: 99.9th percentile of MC loss distribution."""
    if config is None:
        config = FRTBIMAConfig()
    if not positions:
        return {"drc_charge": 0.0, "mean_loss": 0.0, "num_simulations": 0}

    losses = simulate_drc_portfolio(positions, config.drc_num_simulations)
    n = len(losses)
    sorted_losses = np.sort(losses)
    idx_999 = min(int(n * config.drc_confidence), n - 1)
    drc_charge = float(sorted_losses[idx_999])

    return {
        "drc_charge": drc_charge,
        "mean_loss": float(losses.mean()),
        "percentile_95": float(sorted_losses[int(n * 0.95)]),
        "percentile_99": float(sorted_losses[int(n * 0.99)]),
        "percentile_999": drc_charge,
        "max_loss": float(sorted_losses[-1]),
        "num_simulations": n,
        "num_positions": len(positions),
        "num_obligors": len({p.obligor for p in positions}),
    }


# ---- Backtesting (MAR33) ----

PLUS_FACTOR_TABLE = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00,
    5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85,
}


def evaluate_backtesting(num_exceptions: int, num_observations: int = 250) -> dict:
    """Traffic light zone + plus factor (MAR33)."""
    if num_exceptions <= 4:
        zone = "green"
    elif num_exceptions <= 9:
        zone = "yellow"
    else:
        zone = "red"
    plus_factor = PLUS_FACTOR_TABLE.get(num_exceptions, 1.0)
    rate = num_exceptions / num_observations if num_observations > 0 else 0.0
    return {
        "zone": zone, "plus_factor": plus_factor,
        "num_exceptions": num_exceptions, "num_observations": num_observations,
        "exception_rate_pct": rate * 100,
    }


# ---- P&L Attribution Test (PLA) ----

PLA_THRESHOLDS = {
    "spearman": {"green": 0.80, "amber": 0.70},
    "kl_divergence": {"green": 0.09, "amber": 0.12},
}


def evaluate_pla(desks: list[DeskPLA]) -> dict:
    """PLA test per desk: Spearman correlation + KL divergence."""
    results = []
    summary = {"green": 0, "amber": 0, "red": 0}
    for desk in desks:
        if desk.spearman_correlation >= PLA_THRESHOLDS["spearman"]["green"]:
            sp = "green"
        elif desk.spearman_correlation >= PLA_THRESHOLDS["spearman"]["amber"]:
            sp = "amber"
        else:
            sp = "red"

        if desk.kl_divergence <= PLA_THRESHOLDS["kl_divergence"]["green"]:
            kl = "green"
        elif desk.kl_divergence <= PLA_THRESHOLDS["kl_divergence"]["amber"]:
            kl = "amber"
        else:
            kl = "red"

        order = {"green": 0, "amber": 1, "red": 2}
        overall = max([sp, kl], key=lambda z: order[z])
        results.append({
            "desk_id": desk.desk_id,
            "spearman_correlation": desk.spearman_correlation,
            "spearman_zone": sp,
            "kl_divergence": desk.kl_divergence,
            "kl_zone": kl,
            "overall_zone": overall,
            "ima_eligible": overall != "red",
        })
        summary[overall] += 1

    return {
        "desks": results, "summary": summary, "total_desks": len(desks),
        "ima_eligible_desks": summary["green"] + summary["amber"],
        "sa_fallback_desks": summary["red"],
    }


# ---- IMCC ----

def calculate_imcc(
    risk_factors: list[ESRiskFactor],
    es_current: float,
    ses: float,
    es_avg_60: float | None = None,
    ses_avg_60: float | None = None,
    config: FRTBIMAConfig | None = None,
) -> dict:
    """IMCC = max(ES, m_c × ES_avg) + max(SES, m_c × SES_avg) + NMRF."""
    if config is None:
        config = FRTBIMAConfig()
    if es_avg_60 is None:
        es_avg_60 = es_current
    if ses_avg_60 is None:
        ses_avg_60 = ses

    m_c = config.multiplication_factor * (1.0 + config.plus_factor)
    es_comp = max(es_current, m_c * es_avg_60)
    ses_comp = max(ses, m_c * ses_avg_60)

    nmrf_result = calculate_nmrf_charge(risk_factors)
    nmrf = nmrf_result["nmrf_total"]
    imcc = es_comp + ses_comp + nmrf

    return {
        "imcc": imcc, "es_component": es_comp, "ses_component": ses_comp, "nmrf": nmrf,
        "multiplication_factor_mc": m_c,
        "es_current": es_current, "ses_current": ses,
        "es_avg_60": es_avg_60, "ses_avg_60": ses_avg_60,
        "nmrf_detail": nmrf_result,
    }


# ---- Total IMA capital ----

def calculate_frtb_ima_capital(
    risk_factors: list[ESRiskFactor],
    drc_positions: list[DRCPosition],
    config: FRTBIMAConfig | None = None,
    es_avg_60: float | None = None,
    ses_avg_60: float | None = None,
    desks: list[DeskPLA] | None = None,
) -> dict:
    """Full FRTB-IMA: IMCC + DRC."""
    if config is None:
        config = FRTBIMAConfig()

    es_result = calculate_liquidity_adjusted_es(risk_factors)
    es_current = es_result["es_total"]

    reduced = [rf for rf in risk_factors if rf.stressed_es_10day is not None]
    if reduced:
        reduced_current = calculate_liquidity_adjusted_es(reduced)["es_total"]
    else:
        reduced_current = es_current

    ses_result = calculate_stressed_es(risk_factors, es_current, reduced_current)
    ses = ses_result["ses_total"]

    imcc_result = calculate_imcc(
        risk_factors, es_current, ses,
        es_avg_60=es_avg_60, ses_avg_60=ses_avg_60, config=config,
    )

    drc_result = calculate_ima_drc(drc_positions, config)
    bt_result = evaluate_backtesting(config.backtesting_exceptions)
    pla_result = evaluate_pla(desks) if desks else None

    total_capital = imcc_result["imcc"] + drc_result["drc_charge"]

    return {
        "approach": "FRTB-IMA",
        "total_capital": total_capital,
        "total_rwa": total_capital * 12.5,
        "imcc": imcc_result["imcc"], "imcc_detail": imcc_result,
        "es": es_result, "ses": ses_result,
        "drc_charge": drc_result["drc_charge"], "drc_detail": drc_result,
        "backtesting": bt_result, "pla": pla_result,
    }


# ---- Convenience ----

def quick_frtb_ima(
    es_10day_total: float,
    stressed_es_10day_total: float,
    drc_positions: list[dict] | None = None,
    plus_factor: float = 0.0,
) -> dict:
    """Quick FRTB-IMA from minimal inputs."""
    config = FRTBIMAConfig(plus_factor=plus_factor)
    rf = ESRiskFactor(
        risk_class="IR", sub_category="major",
        es_10day=es_10day_total, stressed_es_10day=stressed_es_10day_total,
    )
    drc_pos = []
    if drc_positions:
        for i, p in enumerate(drc_positions):
            drc_pos.append(DRCPosition(
                position_id=p.get("position_id", f"pos_{i}"),
                obligor=p.get("obligor", f"obligor_{i}"),
                notional=p.get("notional", 0),
                market_value=p.get("market_value", p.get("notional", 0)),
                pd=p.get("pd", RATING_TO_PD.get(p.get("rating", "BBB"), 0.004)),
                lgd=p.get("lgd", 0.45),
                is_long=p.get("is_long", True),
            ))
    return calculate_frtb_ima_capital([rf], drc_pos, config)


def compare_ima_vs_sa(
    risk_factors: list[ESRiskFactor],
    drc_positions_ima: list[DRCPosition],
    delta_positions_sa: dict,
    drc_positions_sa: list[dict] | None = None,
    config: FRTBIMAConfig | None = None,
) -> dict:
    """Side-by-side IMA vs SA capital comparison."""
    ima = calculate_frtb_ima_capital(risk_factors, drc_positions_ima, config)
    sa = calculate_frtb_sa(delta_positions=delta_positions_sa, drc_positions=drc_positions_sa or [])
    return {
        "ima_capital": ima["total_capital"], "sa_capital": sa["total_capital"],
        "ima_rwa": ima["total_rwa"], "sa_rwa": sa["total_rwa"],
        "savings_pct": (1 - ima["total_capital"] / sa["total_capital"]) * 100 if sa["total_capital"] > 0 else 0,
        "ima_detail": ima, "sa_detail": sa,
    }
