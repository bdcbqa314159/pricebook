"""The CEV (constant elasticity of variance) forward-rate model (L3).

`dF_i(t) = sigma_i(t) F_i(t)^beta dW^{T_i}(t)` with zero an **absorbing** state — MCINERNEY-ZASTAWNIAK
eq (8.5). The exponent interpolates between the two tranche-71 models: `beta = 1` is lognormal
(Black) and `beta = 0` is normal (Bachelier, up to the barrier), and a `beta` between them fits a
skew neither can.

**`beta`, and the one unit consequence that is not cosmetic.** Two held sources name the exponent
differently: MCINERNEY-ZASTAWNIAK writes `beta` (eq 8.5), BROCKHAUS-2016 writes `rho` (eq 6.8).
`beta` is kept — it is the rates-standard letter, it is the arc's own source's letter, and `rho`
would collide with correlation throughout this tree. What both letters force is that **`vol` carries
units of rate^(1-beta)**, so a CEV volatility is *not comparable across exponents*: at `beta = 1` it
is a dimensionless 22%, at `beta = 0` an absolute ~85bp. Two `CevModel`s with different exponents and
the same `vol` are not two calibrations of one volatility, and nothing may be shared between them.

**Why `beta = 1` is refused rather than special-cased.** The SDE admits `0 <= beta <= 1`, but Cox's
closed form — the only way this model is priced here — is stated for `0 < beta < 1` **strictly** and
diverges at the endpoint: `1/(1-beta)` is the degrees of freedom, and the formula returns `nan` by
`beta = 0.99999` (measured). A model that cannot be priced is refused at construction, where the
input is simply outside the domain, rather than returning a `PricingFailure` from every engine call
or — worse — a `nan` that looks like a price. `beta = 0` is kept because it *is* reachable: there
`1/(1-beta)` is 1 and every term is finite.

**Nothing here generalises the tranche-71 models, deliberately.** §8.4 derives the displaced-diffusion
process *as an approximation to this one* (a Taylor expansion of `F^beta`), so the two are related by
an approximation rather than by a shared shape — a base class would suggest an interchangeability the
mathematics denies. Three distinct frozen types, no hierarchy (§6).

Provenance:
  stock:  none — the migration tree carried no CEV model
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §8.3 eq (8.5) — the SDE, the `0 <= beta <= 1`
          range, the absorbing boundary condition and "market practice is to put beta = 0.5";
          BROCKHAUS-2016 §6.1 eq (6.8) for the same process in its `rho` notation
  oracle: tests/L3/test_cev.py — the accepted range, the refusal at the closed form's undefined
          endpoint, the vol refusal, frozen-ness, and the absence of any hierarchy with the
          tranche-71 smile models
  crosscheck: quantlib has no CEV cap engine; scipy.stats.ncx2 backs the pricing primitive instead
          (see foundation/noncentral_chisquare.py) — deferred to the optional QuantLib tier
  slice:  tranche 72 — the CEV smile model
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class CevModel:
    """A CEV forward rate: `vol` is `sigma` in rate^(1-beta), `beta` the exponent in `[0, 1)`."""

    market: MarketSnapshot
    vol: float
    beta: float

    def __post_init__(self) -> None:
        if self.vol <= 0.0:
            raise ValueError(f"a CEV volatility must be positive, got {self.vol}")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(
                f"the CEV exponent must satisfy 0 <= beta < 1, got {self.beta}; Cox's closed form "
                f"is stated for beta strictly below 1 and diverges at the endpoint, where the "
                f"degrees of freedom 1/(1-beta) are unbounded — use the Black model for beta = 1"
            )
