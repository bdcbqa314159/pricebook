"""Smart contract and DeFi composability risk scoring.

* :func:`contract_risk_score` — risk score for a smart contract.
* :func:`composability_risk` — risk from nested DeFi dependencies.
* :func:`oracle_risk` — oracle failure and manipulation risk.

References:
    Werner et al., *SoK: Decentralized Finance*, 2022.
    Perez & Livshits, *Smart Contract Vulnerabilities*, 2021.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuditStatus(Enum):
    UNAUDITED = "unaudited"
    SINGLE_AUDIT = "single_audit"
    MULTI_AUDIT = "multi_audit"
    FORMALLY_VERIFIED = "formally_verified"


@dataclass
class ContractRiskResult:
    """Smart contract risk score."""
    score: float                # 0–100 (higher = riskier)
    components: dict[str, float]
    risk_level: str
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
        }


def contract_risk_score(
    audit_status: AuditStatus = AuditStatus.UNAUDITED,
    age_days: int = 0,
    tvl_usd: float = 0.0,
    is_upgradeable: bool = True,
    multisig_signers: int = 0,
    multisig_threshold: int = 0,
    has_timelock: bool = False,
    bug_bounty_usd: float = 0.0,
    n_forks: int = 0,
) -> ContractRiskResult:
    """Score smart contract risk.

    Lower score = safer. Factors:
    - Audit: formally verified > multi-audit > single > unaudited.
    - Age: older = more battle-tested.
    - TVL: higher TVL = more eyes, but also bigger target.
    - Upgradeability: proxy contracts can be rugged.
    - Multisig: higher threshold = harder to rug.
    - Timelock: gives users time to exit on malicious upgrade.
    - Bug bounty: incentivises white-hat disclosure.
    """
    # Audit score (0–25)
    audit_score = {
        AuditStatus.FORMALLY_VERIFIED: 0,
        AuditStatus.MULTI_AUDIT: 5,
        AuditStatus.SINGLE_AUDIT: 12,
        AuditStatus.UNAUDITED: 25,
    }.get(audit_status, 25)

    # Age score (0–15)
    if age_days > 730:
        age_score = 0
    elif age_days > 365:
        age_score = 5
    elif age_days > 90:
        age_score = 10
    else:
        age_score = 15

    # Upgradeability (0–20)
    if not is_upgradeable:
        upgrade_score = 0
    elif has_timelock and multisig_threshold >= 3:
        upgrade_score = 5
    elif multisig_threshold >= 2:
        upgrade_score = 10
    else:
        upgrade_score = 20

    # TVL concentration (0–15)
    if tvl_usd > 1_000_000_000:
        tvl_score = 5  # big target but well-monitored
    elif tvl_usd > 100_000_000:
        tvl_score = 3
    elif tvl_usd > 10_000_000:
        tvl_score = 8
    else:
        tvl_score = 15  # low TVL = less battle-tested

    # Bug bounty (0–10)
    bounty_score = 10 if bug_bounty_usd == 0 else max(0, 10 - bug_bounty_usd / 100_000)

    # Governance (0–15)
    if multisig_signers == 0:
        gov_score = 15  # single key = maximum rug risk
    elif multisig_threshold / multisig_signers >= 0.6:
        gov_score = 3
    else:
        gov_score = 8

    total = audit_score + age_score + upgrade_score + tvl_score + bounty_score + gov_score
    total = min(total, 100)

    level = "low" if total < 25 else "medium" if total < 50 else "high" if total < 75 else "critical"

    recs = []
    if audit_status == AuditStatus.UNAUDITED:
        recs.append("Get smart contract audited")
    if is_upgradeable and not has_timelock:
        recs.append("Add timelock to upgrades")
    if multisig_signers < 3:
        recs.append("Increase multisig signers")
    if bug_bounty_usd == 0:
        recs.append("Launch bug bounty program")

    return ContractRiskResult(total, {
        "audit": audit_score, "age": age_score, "upgrade": upgrade_score,
        "tvl": tvl_score, "bounty": bounty_score, "governance": gov_score,
    }, level, recs)


@dataclass
class ComposabilityRiskResult:
    """DeFi composability (dependency chain) risk."""
    total_score: float
    chain_depth: int
    weakest_link: str
    weakest_score: float
    protocols: list[str]

    def to_dict(self) -> dict:
        return vars(self)


def composability_risk(
    protocol_scores: dict[str, float],
    dependency_chain: list[str],
) -> ComposabilityRiskResult:
    """Risk from nested DeFi protocol dependencies.

    Total risk compounds: failure of any link breaks the chain.
    R_total ≈ 1 − Π(1 − R_i) for independent failures.

    Args:
        protocol_scores: {protocol_name: risk_score (0–100)}.
        dependency_chain: ordered list of protocols in the dependency chain.
    """
    if not dependency_chain:
        return ComposabilityRiskResult(0, 0, "", 0, [])

    # Compound risk
    survival = 1.0
    weakest = ("", 0.0)
    for proto in dependency_chain:
        score = protocol_scores.get(proto, 50)  # default medium
        survival *= (1 - score / 100)
        if score > weakest[1]:
            weakest = (proto, score)

    total = (1 - survival) * 100

    return ComposabilityRiskResult(
        total_score=total,
        chain_depth=len(dependency_chain),
        weakest_link=weakest[0],
        weakest_score=weakest[1],
        protocols=dependency_chain,
    )


@dataclass
class OracleRiskResult:
    """Oracle risk assessment."""
    score: float
    n_sources: int
    update_frequency: str
    manipulation_cost_usd: float
    risk_level: str

    def to_dict(self) -> dict:
        return vars(self)


def oracle_risk(
    n_price_sources: int = 1,
    update_frequency_seconds: int = 3600,
    min_manipulation_cost_usd: float = 100_000.0,
    uses_twap: bool = False,
    twap_window_seconds: int = 1800,
) -> OracleRiskResult:
    """Oracle failure and manipulation risk.

    Safer oracles:
    - Multiple independent sources (Chainlink = 21 nodes).
    - Frequent updates (every block vs every hour).
    - TWAP (harder to manipulate than spot).
    - High cost to manipulate (deep liquidity).

    Args:
        n_price_sources: number of independent oracle feeds.
        update_frequency_seconds: how often price updates.
        min_manipulation_cost_usd: estimated cost to manipulate price.
        uses_twap: time-weighted average price (harder to flash-loan attack).
        twap_window_seconds: TWAP averaging window.
    """
    # Source diversity (0–30)
    if n_price_sources >= 10:
        source_score = 0
    elif n_price_sources >= 3:
        source_score = 10
    elif n_price_sources >= 2:
        source_score = 20
    else:
        source_score = 30

    # Update frequency (0–25)
    if update_frequency_seconds <= 15:
        freq_score = 0
    elif update_frequency_seconds <= 300:
        freq_score = 10
    elif update_frequency_seconds <= 3600:
        freq_score = 20
    else:
        freq_score = 25

    # Manipulation cost (0–25)
    if min_manipulation_cost_usd >= 10_000_000:
        manip_score = 0
    elif min_manipulation_cost_usd >= 1_000_000:
        manip_score = 10
    elif min_manipulation_cost_usd >= 100_000:
        manip_score = 18
    else:
        manip_score = 25

    # TWAP (0–20)
    twap_score = 0 if uses_twap and twap_window_seconds >= 600 else 10 if uses_twap else 20

    total = source_score + freq_score + manip_score + twap_score
    total = min(total, 100)

    if update_frequency_seconds < 60:
        freq_label = "per-block"
    elif update_frequency_seconds < 3600:
        freq_label = f"{update_frequency_seconds}s"
    else:
        freq_label = f"{update_frequency_seconds // 3600}h"

    level = "low" if total < 25 else "medium" if total < 50 else "high" if total < 75 else "critical"

    return OracleRiskResult(total, n_price_sources, freq_label,
                             min_manipulation_cost_usd, level)
