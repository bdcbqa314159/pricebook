"""BlackModel — the lognormal-vol model for optionlets (L3).

The SECOND `CalibratedModel` (after `DiscountingModel`) — the first real rule-of-two test of the
capability model. Like `DiscountingModel` it is BUILT, not solved: it carries the `MarketSnapshot`
(A1) and reads its vol from `market.surfaces` (vol calibration/stripping is a later slice). It adds
the `BlackVol` capability; `black_vol` delegates to the surface under the `Q3′a` semantic contract
(lognormal, T-forward measure — see `models/protocols.py`).

Provenance:
  quarry: python/pricebook/models/black76.py (the model side)
  source: redesign/22 Q1 (2nd CalibratedModel) · Q3′a (semantic contract); §B4 (Black-76 model)
  oracle: caplet reprices to Black-76 off model.black_vol; DiscountingModel still prices linear
  slice:  black-caplet (C2 slice 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation import RateIndex, Tenor
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import SurfaceKey, SwaptionSurfaceKey, flat_surface
from pricebook_ng.models.black import vol_time_measure


@dataclass(frozen=True)
class BlackModel:
    """Carries `market` (A1). Satisfies `CalibratedModel` (`.market`), `BlackVol` (`.black_vol`,
    reads `SurfaceKey(index)`), AND `SwaptionVol` (`.swaption_vol`, reads `SwaptionSurfaceKey`) —
    opt-in capabilities on one model (doc 22 Q1). Adding `swaption_vol` did not change the meaning
    of `market` or `black_vol`; it extended the model additively."""

    market: MarketSnapshot

    def black_vol(self, index: RateIndex, expiry: date, strike: float) -> float:
        tm = vol_time_measure(self.market.valuation_date)  # the canonical vol clock (§3d, #4)
        return flat_surface(self.market.surfaces[SurfaceKey(index)]).at(expiry, strike, tm)

    def swaption_vol(self, index: RateIndex, expiry: date, swap_tenor: Tenor, strike: float) -> float:
        tm = vol_time_measure(self.market.valuation_date)
        return flat_surface(self.market.surfaces[SwaptionSurfaceKey(index, swap_tenor)]).at(expiry, strike, tm)
