"""The time-changed Lévy model (L3). KWOK-ZHENG §2.4, §4.4.

    ln(S_t/S_0) = (r−q)t + L_{Z_t} − κ_L(1)·Z_t          (eq 4.81)

A **Lévy process `L`** run on a **stochastic clock `Z`**, the two independent (Prop 4.3's
hypothesis). Two parts, two parameter objects, because they are two different objects and §6's
rule-of-two has not fired to merge them:

- **`LevyParams`** — the base process: a diffusion part `σ` plus a `Y = 0` CGMY (**Variance Gamma**)
  jump part `(C, G, M)`. The whole of pricing reaches it through the **cumulant exponent**
  `κ_L(u) = ln E[e^{uL₁}]`, which is elementary here; no special function is on the production path.
  Table 4.3's other rows (Kou, NIG) are elementary too, so the choice is tractability of the
  *simulation*, not of the formula — VG increments are an exact difference of two gamma draws.
- **`ActivityRateParams`** — the clock, as an **absolutely continuous** `Z_t = ∫₀ᵗ v_s ds` with a CIR
  activity rate. Not a subordinator: §4.4.4's convergence assumes `E[(ΔZ)²] ~ O(N⁻²)`, which an
  integrated rate gives and a jump clock does not (a gamma subordinator gives `O(N⁻¹)` and the
  discrete strike would then **not** converge to the continuous one at all).

**`v` is stationary with mean 1, which is forced, not chosen.** Eq (4.81) normalises the clock by
`E[Z_t] = t`, so the activity rate must have unit mean at every `t` — hence the long-run level and
the initial level are both 1 and neither is a free parameter. That is why `ActivityRateParams`
carries only `(κ, σ_v)`: a `theta` or `v0` field would be a fifth wheel the normalisation pins.

**Sibling of `HestonParams` / `ThreeHalvesParams` / `CevModel`, not a subclass** (§6): a jump process
on a stochastic clock is not a diffusion with a stochastic variance, and they share only the words.

Provenance:
  stock:  none — the migration tree carried no Lévy model
  source: KWOK-ZHENG (docs/references.md) §2.4 — time-changed Lévy processes; eq (4.81) — the log
          price under the time change and the `E[Z_t] = t` normalisation; Table 4.3 — the concrete
          Lévy processes, whose CGMY row is transcribed at `Y = 0`
  oracle: tests/L4/test_time_changed_levy.py — the cumulant exponent transcribed pointwise, the
          multiplier against Table 4.3's own row, the refusals, and `M > 1` shown to be exactly the
          condition for a finite `κ_L(1)` rather than asserted by fiat
  crosscheck: none — QuantLib has no time-changed-Lévy variance-swap engine
  slice:  tranche 78 — variance swaps under time-changed Lévy processes
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class LevyParams:
    """`σ` diffusion volatility plus a Variance-Gamma jump part `(C, G, M)`.

    `C` is the jump arrival intensity, `G` and `M` the exponential decay rates of the left and
    right tails: a **larger** `G`/`M` means **smaller** jumps, so the pure-diffusion limit is
    `C → 0` (or `G, M → ∞`). `G ≠ M` is a skewed jump measure, which is the empirically relevant
    case and the one Table 4.3 cites for the multiplier exceeding 2.
    """

    sigma: float
    c: float
    g: float
    m: float

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError(f"a diffusion volatility may not be negative, got {self.sigma}")
        if self.c < 0.0:
            raise ValueError(f"a jump arrival intensity may not be negative, got {self.c}")
        if not self.g > 0.0:
            raise ValueError(f"the left-tail decay G must be positive, got {self.g}")
        if not self.m > 1.0:
            # kappa_L(1) carries ln(M/(M-1)), which diverges at M = 1. Table 4.3 footnote b.
            raise ValueError(f"the right-tail decay M must exceed 1, got {self.m}")


@dataclass(frozen=True)
class ActivityRateParams:
    """The CIR activity rate `dv = κ(1 − v)dt + σ_v·√v·dW`, whose integral is the clock.

    The long-run level is **1 by the `E[Z_t] = t` normalisation** (see the module docstring), not a
    field. `σ_v = 0` is the deterministic clock `Z_t = t` — the reduction that switches the time
    change off, and the one under which the discrete adjustment keeps only its squared-mean term.
    """

    kappa: float
    vol_of_vol: float

    def __post_init__(self) -> None:
        if not self.kappa > 0.0:
            raise ValueError(f"the clock's mean reversion must be positive, got {self.kappa}")
        if self.vol_of_vol < 0.0:
            raise ValueError(
                f"the clock's vol-of-vol may not be negative, got {self.vol_of_vol}"
            )


@dataclass(frozen=True)
class TimeChangedLevyModel:
    """The Lévy and clock parameters with the `MarketSnapshot` they were calibrated to (§3)."""

    market: MarketSnapshot
    levy: LevyParams
    clock: ActivityRateParams
