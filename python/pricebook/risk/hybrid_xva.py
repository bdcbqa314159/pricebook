"""XVA for hybrids: CVA on multi-asset, wrong-way risk, FVA.

* :func:`hybrid_cva` — CVA on multi-asset exotic.
* :func:`wrong_way_risk_adjustment` — equity down → credit spread up.
* :func:`hybrid_fva` — FVA for long-dated hybrid products.

References:
    Gregory, *Counterparty Credit Risk and CVA*, Wiley, 2012.
    Brigo, Morini & Pallavicini, *Counterparty Credit Risk, Collateral and
    Funding*, Wiley, 2013.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class HybridCVAResult:
    cva: float
    expected_positive_exposure: float
    expected_negative_exposure: float
    peak_exposure: float
    wrong_way_adjustment: float


    def to_dict(self) -> dict:
        return vars(self)
def hybrid_cva(
    exposure_paths: np.ndarray,     # (n_paths, n_steps+1)
    default_probs: np.ndarray,      # (n_steps+1,) cumulative PD
    recovery: float = 0.4,
    wrong_way_factor: float = 0.0,
    discount_factors: np.ndarray | None = None,  # (n_steps+1,) P(0,t_i)
) -> HybridCVAResult:
    """CVA on multi-asset exotic from exposure profiles.

    Textbook form:

        CVA = (1 − R) × Σ_i D(0, t_i) × EPE(t_i) × ΔPD(i)

    Fix T4-RISK21: pre-fix omitted the discount-factor term D(0, t)
    entirely, computing the *future-valued* CVA sum instead of the
    present-valued one.  For a 10y trade with EPE in current notional
    and rates ~5%, the pre-fix overstated CVA by ~30%-40%.
    ``discount_factors`` is now an explicit optional input;
    backwards compatibility: when omitted, the function assumes
    the caller has pre-discounted the exposure paths (i.e. each
    ``exposure_paths[p, i]`` is already PV at t=0).  Either pass
    discount_factors explicitly OR pre-discount the paths upstream.

    wrong_way_factor: scales CVA up when default is more likely
    (positive factor → exposure increases near default events).
    """
    n_paths, n_steps_p1 = exposure_paths.shape
    n_steps = n_steps_p1 - 1

    # Marginal PD
    marginal_pd = np.diff(default_probs, prepend=0)

    epe = np.maximum(exposure_paths, 0).mean(axis=0)
    ene = np.minimum(exposure_paths, 0).mean(axis=0)
    peak = float(epe.max())

    # Discount the EPE if discount factors supplied.
    if discount_factors is not None:
        df = np.asarray(discount_factors, dtype=float)
        if df.shape != (n_steps_p1,):
            raise ValueError(
                f"discount_factors must have shape ({n_steps_p1},); got {df.shape}"
            )
        epe_discounted = epe * df
    else:
        epe_discounted = epe

    # CVA
    lgd = 1 - recovery
    cva = lgd * float(np.sum(epe_discounted * marginal_pd))

    # Wrong-way adjustment: scale CVA by (1 + factor)
    wwr = cva * wrong_way_factor
    cva_total = cva + wwr

    return HybridCVAResult(float(cva_total), float(epe.mean()), float(ene.mean()),
                             peak, float(wwr))


@dataclass
class WrongWayRiskResult:
    base_cva: float
    adjusted_cva: float
    adjustment_pct: float
    correlation_equity_credit: float


    def to_dict(self) -> dict:
        return vars(self)
def wrong_way_risk_adjustment(
    equity_paths: np.ndarray,       # (n_paths, n_steps+1)
    exposure_paths: np.ndarray,
    base_default_probs: np.ndarray,
    recovery: float = 0.4,
    equity_credit_sensitivity: float = -0.5,
) -> WrongWayRiskResult:
    """Wrong-way risk: equity down → credit spread up → higher default prob.

    Adjust default probability conditional on equity level:
    PD_adjusted(t) = PD_base(t) × exp(−β × equity_return).
    """
    n_paths, n_steps_p1 = equity_paths.shape
    eq_returns = equity_paths[:, -1] / equity_paths[:, 0] - 1

    # Base CVA
    marginal_pd = np.diff(base_default_probs, prepend=0)
    epe = np.maximum(exposure_paths, 0).mean(axis=0)
    lgd = 1 - recovery
    base_cva = lgd * float(np.sum(epe * marginal_pd))

    # Adjusted: paths where equity is down have higher PD
    # Scale exposure by conditional factor
    eq_factor = np.exp(-equity_credit_sensitivity * eq_returns)
    weighted_epe = np.maximum(exposure_paths, 0) * eq_factor[:, np.newaxis]
    adj_epe = weighted_epe.mean(axis=0)
    adj_cva = lgd * float(np.sum(adj_epe * marginal_pd))

    adj_pct = (adj_cva - base_cva) / max(abs(base_cva), 1e-10) * 100

    return WrongWayRiskResult(float(base_cva), float(adj_cva), float(adj_pct),
                                equity_credit_sensitivity)


@dataclass
class HybridFVAResult:
    fva: float
    funding_spread_bps: float
    expected_funding: float


    def to_dict(self) -> dict:
        return vars(self)
def hybrid_fva(
    exposure_paths: np.ndarray,
    funding_spread_bps: float,
    dt: float,
    discount_factors: np.ndarray | None = None,
) -> HybridFVAResult:
    """FVA for long-dated hybrid: funding cost of uncollateralised exposure.

    Textbook form:

        FVA = funding_spread × Σ_i D(0, t_i) × E[V(t_i)] × dt

    Fix T4-RISK22: pre-fix omitted the discount factor, computing
    *future-valued* FVA.  Same shape as the CVA fix above.  Pass
    ``discount_factors`` explicitly or pre-discount the exposure paths.
    """
    n_paths, n_steps = exposure_paths.shape
    spread = funding_spread_bps / 10000
    expected_exposure = exposure_paths.mean(axis=0)
    if discount_factors is not None:
        df = np.asarray(discount_factors, dtype=float)
        if df.shape != expected_exposure.shape:
            raise ValueError(
                f"discount_factors shape {df.shape} != exposure time-step "
                f"shape {expected_exposure.shape}"
            )
        expected_exposure_pv = expected_exposure * df
    else:
        expected_exposure_pv = expected_exposure
    fva = spread * float(np.sum(expected_exposure_pv * dt))
    expected_funding = float(np.sum(np.abs(expected_exposure_pv) * dt)) * spread

    return HybridFVAResult(float(fva), funding_spread_bps, float(expected_funding))
