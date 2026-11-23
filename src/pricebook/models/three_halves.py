"""The 3/2 stochastic-volatility model (L3). KWOK-ZHENG §2.75d, §4.2.2.

    dv = κ·v·(θ − v) dt + ε·v^{3/2} dW

**Non-affine, and the mean-reversion speed scales with the variance level** (`κv(θ−v)`, not Heston's
`κ(θ−v)`). Two consequences the rest of the tranche turns on: the deterministic limit is a
**logistic** ODE rather than an exponential one, and the reciprocal `Y = 1/v` follows a **CIR**
process (§4.2.2 eq 4.35), which is what makes the fair variance computable without a hypergeometric.

**`epsilon`, not `sigma`.** The source writes `σ_v` (eq 2.75d) and `σ` (the paper), and both letters
are taken in this tree: `sigma_v` is Heston's vol-of-vol on `HestonParams` and `sigma` is the Black
vol everywhere. A third meaning for either would be the T71 `β`-collision again, so the field is
named after its role in this model. §4.2.2's generalisation writes the same dynamics as
`V[p(t) − qV]`, i.e. `p = κθ` and `q = κ`; the paper's Proposition 3 uses `q = −κ` for the
coefficient of `V²`. Both mappings are asserted in the oracle rather than trusted.

**Sibling of `HestonParams` and `CevModel`, not a subclass** (§6): a different SDE with a different
mean-reversion shape. They share the words "stochastic volatility" and not the dynamics.

Provenance:
  stock:  none — the migration tree carried no 3/2 model
  source: KWOK-ZHENG (docs/references.md) eq (2.75d) — the 3/2 variance dynamics in exactly this
          parameterisation; §4.2.2 eq (4.34)/(4.35) — the generalised form and the reciprocal-CIR
          map; ZHAO-CUI-FLORESCU-2018 Proposition 3 for the parameter conventions cross-checked
          against it
  oracle: tests/L4/test_three_halves.py — the parameter map between the three conventions asserted
          pointwise, the reciprocal-CIR law, the refusals, and that the inverse moment always exists
          for a positive mean reversion
  crosscheck: none — QuantLib implements no non-affine 3/2 model
  slice:  tranche 77 — the 3/2-model variance swap
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class ThreeHalvesParams:
    """`v0, kappa, theta, epsilon` for `dv = κv(θ−v)dt + εv^{3/2}dW`.

    **`theta` is the reversion LEVEL, not the long-run variance** — the distinction is specific to
    this model and is why it is named here. For Heston the two coincide; for the 3/2 the `v^{3/2}`
    vol-of-vol pulls the stationary mean **below** `θ`, to `θ/(1 + ε²/(2κ))` — 26% below at `ε = 1.2`.
    Oracle 5 in `tests/L4/test_three_halves.py` derives and asserts it (T77, ruling 0411).
    """

    v0: float
    kappa: float
    theta: float
    epsilon: float

    def __post_init__(self) -> None:
        for name, value in (
            ("initial variance", self.v0),
            ("mean reversion", self.kappa),
            ("reversion level", self.theta),
            ("vol-of-vol", self.epsilon),
        ):
            if not value > 0.0:
                raise ValueError(f"the 3/2 model's {name} must be positive, got {value}")


@dataclass(frozen=True)
class ThreeHalvesModel:
    """The 3/2 parameters and the `MarketSnapshot` they were calibrated to (§3)."""

    market: MarketSnapshot
    params: ThreeHalvesParams
