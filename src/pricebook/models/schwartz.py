"""The Schwartz (1997) one-factor mean-reverting commodity model (L3). `CLARK-2014` ch 2.10.1.1.

Mean reversion in **log-spot**, which is what gives a commodity a *term structure* of volatility:

    dS = kappa (alpha_S - ln S) S dt + sigma S dW                            (2.147a)
    dX = kappa (alpha_X - X) dt + sigma dW,   X = ln S                       (2.147b)

with `alpha_X = alpha_S - sigma^2/(2 kappa)` by Ito, so the level the *log* reverts to is not the
log of the level the *spot* reverts to. Clark notes (2.147b) is the Vasicek SDE, which is why the
simulator this model reuses is the one tranche 57 built for the LGM.

**This is where `CommodityBlackModel`'s own prophecy comes true.** That docstring said a commodity's
volatility was "on a path to diverge from an equity's within this very arc"; here it does - the
Samuelson effect (near-dated futures more volatile than far-dated) is a term structure a flat Black
vol cannot express, which is exactly why the two models stayed separate types at tranche 60 rather
than being merged to save six lines.

**What this model carries, and the identifiability point behind it.** Clark's risk-neutral level is
`alpha_hat_X = alpha_X - lambda/kappa` with `lambda` the market price of risk, so unwinding the Ito
step:

    alpha_hat_X = alpha_S - sigma^2/(2 kappa) - lambda/kappa

Three inputs - the physical long-run level, the market price of risk, and the volatility - and they
reach every observable **only** through that one composite. No set of futures prices can separate
`alpha_S` from `lambda`, which is tranche 11's identifiability lesson arriving at design time rather
than at a calibration that fails to converge. So `level` **is** `alpha_hat_X`, the risk-neutral
long-run log level, and the decomposition is written here rather than folded in silently: what the
type carries is what is observable.

**`kappa > 0` is structural, not a convenience.** Every closed form divides by `kappa`, so zero is
not a slow model but an undefined one. The `kappa -> 0` limit is a real and useful anchor and it is
approached **analytically** (the variance tends to `sigma^2 T`, the futures price to
`S0 exp(sigma^2 T / 2)`); it is never evaluated at zero.

**Its own parameter type rather than `HullWhiteParams` plus a level.** The algebra is shared - both
are Ornstein-Uhlenbeck - but a *rate's* mean reversion and a *log-commodity-price's* are different
ideas, and the charter's DRY rule is about knowledge, not characters. Reusing the rates type would
make a commodity model carry a type named for an interest-rate model: shared word, not shared shape.
Tranche 57's LGM reuse was the opposite case and is the contrast that makes the line clear - the
LGM's numbers literally *are* Hull-White's two, reparameterised. These are two models that happen to
share an SDE. *(The ruling the tranche-64 spec left to Code; raised at the handoff.)*

Provenance:
  stock:  none - the archived predecessor had no mean-reverting commodity model
  source: CLARK-2014 (docs/references.md) ch 2.10.1.1 - eqs (2.147a/b) the SDE and the Ito
          conversion, (2.150a/b) the conditional moments, (2.154)/(2.156) the futures price with
          the market-price-of-risk adjustment, (2.155) its stationary limit; transcribed in both
          renderings and reconciled
  oracle: tests/L4/test_schwartz.py - the non-positive mean reversion, negative volatility and
          non-finite level refusals; the type asserted distinct from `HullWhiteParams`; the
          parameters are exercised by every price and every moment pin in that file
  crosscheck: none - a frozen three-field parameter bundle, no numerics of its own
  slice:  tranche 64 - the Schwartz one-factor model, the commodity arc capstone
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class SchwartzParams:
    """The one factor: mean reversion `kappa`, risk-neutral long-run log level, volatility `sigma`.

    A bundle rather than three loose floats on the model, for the reason `HullWhiteParams` gives:
    the grouping is the model's own structure, and three same-typed floats in a signature invite
    passing them in the wrong order. Concrete and frozen, no polymorphism (§6).

    `level` is `alpha_hat_X` - the **risk-neutral** long-run log level, not the physical one. See
    the module docstring for the decomposition and why only the composite is carried.
    """

    kappa: float
    level: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError(f"mean reversion must be finite and positive, got {self.kappa}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(
                f"volatility must be finite and non-negative, got {self.sigma}"
            )
        if not math.isfinite(self.level):
            raise ValueError(f"the long-run log level must be finite, got {self.level}")


@dataclass(frozen=True)
class SchwartzModel:
    """The Schwartz parameters, carrying the snapshot they belong to (§3)."""

    market: MarketSnapshot
    params: SchwartzParams
