"""Consolidated counterparty XVA report — the L6 valuation adjustments object (A6.2).

A real XVA number is per-counterparty across a *netting set* — a book of trades, so an
L6 concept. `xva_report` simulates the netting set's exposure ONCE (offsetting trades net
on shared paths) and returns every adjustment from that single pass, instead of the six
separate L5 calls each re-running the simulation:

    CVA, DVA, BCVA  — counterparty/own default on the netted EPE/ENE (protection leg)
    FVA             — funding of the net position
    MVA             — funding of initial margin (dynamic IM = the high-quantile PFE)
    KVA             — cost of capital on the netting-set SA-CCR EAD runoff

The economy (discount + both parties' survival curves) is carried by `model.market` (A5),
so the integrators read it directly. A single-trade netting set reproduces the standalone
L5 values exactly (same draws); a hedged set nets the exposure down.

Provenance:
  quarry: python/pricebook/risk/ (xva) + desks (counterparty aggregation)
  source: Gregory, The xVA Challenge (netting-set XVA); redesign/02_spine.md A6.2
  oracle: single-trade report == standalone CVA/DVA/BCVA/FVA/KVA/MVA; mirror hedge nets to 0
  slice:  xva-report
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.risk.exposure import netting_set_exposure
from pricebook_ng.risk.saccr import netting_set_ead
from pricebook_ng.risk.xva import (
    CreditParty,
    ExposureProfile,
    cva,
    dva,
    fva,
    kva,
    mva,
)

_CAPITAL_RATIO = 0.08  # regulatory capital = 8% of RWA


@dataclass(frozen=True)
class XvaReportConfig:
    """The counterparty-report parameters bundled so `xva_report` stays under the arg ceiling:
    the two defaultable parties (A5-keyed survival + recovery), the funding spread, the cost of
    capital, the counterparty risk weight, and the PFE confidence used as the dynamic-IM proxy."""

    counterparty: CreditParty
    self_party: CreditParty
    funding_spread: float
    cost_of_capital: float
    risk_weight: float
    pfe_quantile: float


@dataclass(frozen=True)
class XvaReport:
    """All valuation adjustments for a netting set + the profiles they were built from."""

    cva: float
    dva: float
    bcva: float
    fva: float
    kva: float
    mva: float
    epe: ExposureProfile
    ene: ExposureProfile
    pfe: ExposureProfile
    ead: ExposureProfile


def xva_report(
    swaps: list[VanillaSwap], model: HullWhite, numerics: NumericalConfig, config: XvaReportConfig
) -> XvaReport:
    """Every XVA for the netting set `swaps` from one exposure simulation (A6.2)."""
    market = model.market
    cpty, myself = config.counterparty, config.self_party

    exposure, pfe = netting_set_exposure(swaps, model, numerics, config.pfe_quantile)
    cva_ = cva(exposure.epe, market, cpty.key, cpty.recovery)
    dva_ = dva(exposure.ene, market, myself.key, myself.recovery)
    fva_ = fva(exposure, market, myself.key, config.funding_spread)
    mva_ = mva(pfe, market, myself.key, config.funding_spread)  # dynamic IM = the PFE

    # capital: the netting-set SA-CCR EAD runoff (as-of each grid date, ATM) -> 8% * RWA
    grid = exposure.epe.grid
    ead = ExposureProfile(
        grid, tuple(netting_set_ead([(s, 0.0) for s in swaps], t_j) for t_j in grid)
    )
    capital = ExposureProfile(grid, tuple(_CAPITAL_RATIO * config.risk_weight * e for e in ead.ee))
    kva_ = kva(capital, market, myself.key, config.cost_of_capital)

    return XvaReport(
        cva=cva_, dva=dva_, bcva=cva_ - dva_, fva=fva_, kva=kva_, mva=mva_,
        epe=exposure.epe, ene=exposure.ene, pfe=pfe, ead=ead,
    )
