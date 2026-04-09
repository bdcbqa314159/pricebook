"""Securitisation RWA: SEC-SA, SEC-IRBA, ERBA, and trade-specific RWA.

Basel III/IV securitisation framework (CRE40):
- SEC-SA: Standardised approach using K_SA of underlying pool
- SEC-IRBA: Internal Ratings-Based using K_IRB of underlying pool
- ERBA: External Ratings-Based using CQS-mapped risk weights (CRR2)

Trade-specific RWA helpers for CDS, repo, TRS, loans.

    from pricebook.regulatory.securitization import (
        calculate_sec_sa_rwa, calculate_sec_irba_rwa, calculate_erba_rwa,
        calculate_cds_rwa, calculate_repo_rwa, calculate_trs_rwa, calculate_loan_rwa,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.regulatory.credit_rwa import (
    calculate_capital_requirement, calculate_irb_rwa, calculate_sa_rwa,
    SAExposure,
)


# ---- SEC-SA (CRE40) ----

def calculate_sec_sa_p(n: int, lgd: float = 0.50) -> float:
    """SEC-SA supervisory parameter p.

    p = max(0.3, 0.5 × (1 - LGD))                  for n >= 25
    p = max(0.3, 0.5 × (1 - LGD) + 0.5/n × LGD)    for n < 25
    """
    if n >= 25:
        return max(0.3, 0.5 * (1 - lgd))
    return max(0.3, 0.5 * (1 - lgd) + 0.5 / n * lgd)


def calculate_sec_sa_rw(
    ksa: float,
    attachment: float,
    detachment: float,
    n: int = 25,
    lgd: float = 0.50,
    w: float = 0.0,
    is_sts: bool = False,
) -> float:
    """SEC-SA risk weight via supervisory formula approach.

    Args:
        ksa: K_SA of underlying pool (decimal).
        attachment: tranche attachment point.
        detachment: tranche detachment point.
        n: effective number of exposures in pool.
        lgd: average LGD.
        w: ratio of delinquent exposures.
        is_sts: simple, transparent, standardised → lower floor.
    """
    ksa_adj = ksa * (1 - w)
    p = calculate_sec_sa_p(n, lgd)

    if ksa_adj <= 1e-12:
        return 1250  # max risk weight

    a = -1 / (p * ksa_adj)
    u = detachment - ksa_adj
    l = max(attachment - ksa_adj, 0)

    if detachment <= ksa_adj:
        # Tranche entirely within first-loss → 1250% risk weight (full deduction)
        k_ssfa = 1.0
    elif attachment >= ksa_adj:
        # Tranche entirely above K_SA
        k_ssfa = ksa_adj * (math.exp(a * u) - math.exp(a * l)) / (a * (detachment - attachment))
    else:
        # Tranche spans K_SA
        k_ssfa = ksa_adj - attachment + ksa_adj * (math.exp(a * u) - 1) / (a * (detachment - attachment))

    risk_weight = k_ssfa * 12.5 * 100
    floor = 10 if is_sts else 15
    return min(max(risk_weight, floor), 1250)


def calculate_sec_sa_rwa(
    ead: float,
    attachment: float,
    detachment: float,
    ksa: float = 0.08,
    n: int = 25,
    lgd: float = 0.50,
    w: float = 0.0,
    is_sts: bool = False,
    is_resecuritization: bool = False,
) -> dict:
    """SEC-SA RWA for one tranche."""
    rw = calculate_sec_sa_rw(ksa, attachment, detachment, n, lgd, w, is_sts)
    if is_resecuritization:
        rw = max(rw * 1.5, 100)

    rwa = ead * rw / 100.0
    return {
        "approach": "SEC-SA",
        "ead": ead, "attachment": attachment, "detachment": detachment,
        "thickness": detachment - attachment, "ksa": ksa,
        "n": n, "lgd": lgd, "w": w,
        "is_sts": is_sts, "is_resecuritization": is_resecuritization,
        "risk_weight_pct": rw, "rwa": rwa,
    }


# ---- SEC-IRBA (CRE40) ----

def calculate_sec_irba_kirb(exposures: list[dict]) -> float:
    """K_IRB of underlying pool (weighted average capital requirement).

    K_IRB = Σ K_i × EAD_i / Σ EAD_i
    """
    if not exposures:
        return 0.08

    total_k_ead = 0.0
    total_ead = 0.0
    for exp in exposures:
        ead = exp.get("ead", 0)
        k = calculate_capital_requirement(
            pd=exp.get("pd", 0.01),
            lgd=exp.get("lgd", 0.45),
            maturity=exp.get("maturity", 2.5),
            asset_class=exp.get("asset_class", "corporate"),
        )
        total_k_ead += k * ead
        total_ead += ead

    return total_k_ead / total_ead if total_ead > 0 else 0.08


def calculate_sec_irba_rwa(
    ead: float,
    attachment: float,
    detachment: float,
    kirb: float | None = None,
    underlying_exposures: list[dict] | None = None,
    n: int = 25,
    lgd: float = 0.45,
    w: float = 0.0,
    is_sts: bool = False,
) -> dict:
    """SEC-IRBA RWA for one tranche.

    Uses the same SSFA formula as SEC-SA but with K_IRB instead of K_SA.
    """
    if kirb is None:
        kirb = calculate_sec_irba_kirb(underlying_exposures or [])

    rw = calculate_sec_sa_rw(kirb, attachment, detachment, n, lgd, w, is_sts)
    rwa = ead * rw / 100.0
    return {
        "approach": "SEC-IRBA",
        "ead": ead, "attachment": attachment, "detachment": detachment,
        "kirb": kirb, "n": n, "lgd": lgd, "w": w,
        "is_sts": is_sts,
        "risk_weight_pct": rw, "rwa": rwa,
    }


# ---- ERBA (CRR2 Articles 263-264) ----

# RW base table: CQS → (senior, non-senior) for 1Y maturity
ERBA_RW_BASE = {
    1:  (0.15, 0.15),
    2:  (0.15, 0.25),
    3:  (0.25, 0.40),
    4:  (0.30, 0.50),
    5:  (0.40, 0.65),
    6:  (0.45, 0.85),
    7:  (0.55, 1.05),
    8:  (0.75, 1.35),
    9:  (0.95, 1.70),
    10: (1.20, 2.25),
    11: (1.55, 2.80),
    12: (1.95, 3.40),
    13: (2.50, 4.15),
    14: (4.00, 5.00),
    15: (5.00, 6.25),
    16: (6.25, 7.50),
    17: (7.50, 8.25),
}

# Coefficients (a, b, c) per CQS
ERBA_COEFFICIENTS = {
    1:  (0.01, 0.20, 0.40), 2:  (0.01, 0.20, 0.40),
    3:  (0.03, 0.22, 0.40),
    4:  (0.05, 0.25, 0.35), 5:  (0.05, 0.25, 0.35), 6:  (0.05, 0.25, 0.35),
    7:  (0.09, 0.30, 0.30), 8:  (0.09, 0.30, 0.30), 9:  (0.09, 0.30, 0.30),
    10: (0.09, 0.30, 0.30), 11: (0.09, 0.30, 0.30),
    12: (0.10, 0.35, 0.25), 13: (0.10, 0.35, 0.25), 14: (0.10, 0.35, 0.25),
    15: (0.10, 0.35, 0.25), 16: (0.10, 0.35, 0.25), 17: (0.10, 0.35, 0.25),
}

# PD → CQS mapping (upper bounds)
PD_TO_CQS_THRESHOLDS = [
    (0.0001, 1), (0.0005, 2), (0.0010, 3), (0.0020, 4), (0.0030, 5),
    (0.0050, 6), (0.0080, 7), (0.0130, 8), (0.0200, 9), (0.0350, 10),
    (0.0550, 11), (0.0800, 12), (0.1500, 13), (0.2500, 14), (0.3500, 15),
    (0.5000, 16), (1.0000, 17),
]


def pd_to_cqs(pd: float) -> int:
    """Map PD to Credit Quality Step 1-17."""
    for threshold, cqs in PD_TO_CQS_THRESHOLDS:
        if pd <= threshold:
            return cqs
    return 17


def calculate_erba_rwa(
    ead: float,
    cqs: int,
    seniority: str = "senior",
    maturity: float = 5.0,
    thickness: float | None = None,
    is_sts: bool = False,
) -> dict:
    """ERBA RWA per CRR2 Articles 263-264.

    Senior:     RW = RW_base × (a + b × M_T)
    Non-Senior: RW = RW_base × (a + b × M_T) × T^(-c)

    Args:
        ead: tranche EAD.
        cqs: Credit Quality Step (1-17).
        seniority: "senior" or "non_senior".
        maturity: tranche maturity (floored at 1, capped at 5).
        thickness: tranche thickness (required for non-senior).
        is_sts: STS securitisation → lower floor.
    """
    if cqs not in ERBA_RW_BASE:
        raise ValueError(f"Invalid CQS: {cqs}. Must be 1-17.")

    rw_senior, rw_non_senior = ERBA_RW_BASE[cqs]
    a, b, c = ERBA_COEFFICIENTS[cqs]

    M_T = max(1.0, min(maturity, 5.0))
    rw_base = rw_senior if seniority == "senior" else rw_non_senior

    rw = rw_base * (a + b * M_T)

    if seniority != "senior":
        if thickness is None:
            raise ValueError("Thickness required for non-senior tranches")
        T = max(thickness, 0.001)
        rw = rw * T ** (-c)

    rw_pct = rw * 100
    floor = 10 if is_sts else 15
    rw_pct = min(max(rw_pct, floor), 1250)

    return {
        "approach": "ERBA",
        "ead": ead, "cqs": cqs, "seniority": seniority,
        "maturity": M_T, "thickness": thickness,
        "is_sts": is_sts,
        "risk_weight_pct": rw_pct,
        "rwa": ead * rw_pct / 100.0,
    }


# ---- Trade-specific RWA ----

def calculate_cds_rwa(
    notional: float,
    reference_pd: float,
    counterparty_pd: float,
    maturity: float = 5.0,
    is_protection_buyer: bool = True,
    recovery: float = 0.40,
    approach: str = "irb",
) -> dict:
    """Total RWA for a CDS position.

    Components:
    - Counterparty Credit Risk on the dealer (SA-CCR-style EAD)
    - Reference entity credit risk (only for protection seller)
    """
    # Counterparty EAD (simplified SA-CCR add-on)
    addon_factor = 0.05  # 5% supervisory factor for credit
    sf_maturity = math.sqrt(min(maturity, 1.0))
    counterparty_ead = abs(notional) * addon_factor * sf_maturity

    # CCR RWA on counterparty
    if approach == "irb":
        ccr_result = calculate_irb_rwa(
            counterparty_ead, counterparty_pd, lgd=0.45, maturity=maturity,
        )
    else:
        ccr_result = calculate_sa_rwa(
            SAExposure(ead=counterparty_ead, asset_class="corporate", rating="BBB"),
        )
    ccr_rwa = ccr_result["rwa"]

    # Reference entity risk (protection seller only)
    ref_rwa = 0.0
    if not is_protection_buyer:
        if approach == "irb":
            ref_result = calculate_irb_rwa(notional, reference_pd, lgd=1 - recovery, maturity=maturity)
        else:
            ref_result = calculate_sa_rwa(
                SAExposure(ead=notional, asset_class="corporate", rating="BBB"),
            )
        ref_rwa = ref_result["rwa"]

    total_rwa = ccr_rwa + ref_rwa
    return {
        "approach": f"CDS-{approach.upper()}",
        "notional": notional,
        "ccr_ead": counterparty_ead, "ccr_rwa": ccr_rwa,
        "reference_rwa": ref_rwa,
        "total_rwa": total_rwa,
        "capital_requirement": total_rwa * 0.08,
    }


def calculate_repo_rwa(
    cash_lent: float,
    collateral_value: float,
    counterparty_pd: float,
    counterparty_lgd: float = 0.45,
    haircut: float = 0.04,
    maturity: float = 0.25,
    approach: str = "irb",
) -> dict:
    """Repo RWA (reverse repo from cash lender perspective).

    EAD = max(0, cash_lent - collateral × (1 - haircut))
    """
    adjusted_collateral = collateral_value * (1 - haircut)
    ead = max(0, cash_lent - adjusted_collateral)

    if ead == 0:
        return {"approach": f"Repo-{approach.upper()}", "ead": 0, "rwa": 0, "capital_requirement": 0}

    if approach == "irb":
        result = calculate_irb_rwa(ead, counterparty_pd, lgd=counterparty_lgd, maturity=maturity)
    else:
        result = calculate_sa_rwa(SAExposure(ead=ead, asset_class="bank", rating="BBB"))

    return {
        "approach": f"Repo-{approach.upper()}",
        "cash_lent": cash_lent, "collateral_value": collateral_value,
        "haircut": haircut, "ead": ead,
        "rwa": result["rwa"],
        "capital_requirement": result["rwa"] * 0.08,
    }


def calculate_trs_rwa(
    notional: float,
    reference_pd: float,
    counterparty_pd: float,
    maturity: float = 1.0,
    is_total_return_payer: bool = False,
    approach: str = "irb",
) -> dict:
    """Total Return Swap RWA.

    The TRS receiver synthetically owns the reference asset.
    """
    # CCR component (similar to swap)
    addon_factor = 0.10  # equity TRS supervisory factor
    sf_maturity = math.sqrt(min(maturity, 1.0))
    ccr_ead = abs(notional) * addon_factor * sf_maturity

    if approach == "irb":
        ccr_result = calculate_irb_rwa(ccr_ead, counterparty_pd, lgd=0.45, maturity=maturity)
    else:
        ccr_result = calculate_sa_rwa(SAExposure(ead=ccr_ead, asset_class="bank", rating="BBB"))
    ccr_rwa = ccr_result["rwa"]

    # Reference asset risk (TRS receiver gets full economic exposure)
    ref_rwa = 0.0
    if not is_total_return_payer:
        if approach == "irb":
            ref_result = calculate_irb_rwa(notional, reference_pd, lgd=0.45, maturity=maturity)
        else:
            ref_result = calculate_sa_rwa(SAExposure(ead=notional, asset_class="corporate", rating="BBB"))
        ref_rwa = ref_result["rwa"]

    total = ccr_rwa + ref_rwa
    return {
        "approach": f"TRS-{approach.upper()}",
        "notional": notional, "ccr_rwa": ccr_rwa, "reference_rwa": ref_rwa,
        "total_rwa": total, "capital_requirement": total * 0.08,
    }


def calculate_loan_rwa(
    ead: float,
    pd: float,
    lgd: float = 0.45,
    maturity: float = 5.0,
    asset_class: str = "corporate",
    approach: str = "irb",
) -> dict:
    """Loan RWA (corporate or retail)."""
    if approach == "irb":
        return calculate_irb_rwa(ead, pd, lgd, maturity, asset_class)
    return calculate_sa_rwa(SAExposure(ead=ead, asset_class=asset_class, rating="BBB"))
