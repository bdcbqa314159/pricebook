"""The two volatility-smile forward-rate models: Normal/Bachelier and displaced-diffusion (L3).

Chapter 8 of MCINERNEY-ZASTAWNIAK extends the Black caplet to a smile three ways. These are the two
that are a **reparameterisation of Black** — a normal forward instead of a lognormal one, and a
lognormal forward that has been *shifted*. Neither needs a new numerical primitive, which is why
they land together and CEV (§8.3) does not: that one needs a non-central chi-squared.

**Two types, no base class.** They share the word "volatility" and not its units: Bachelier's
`normal_vol` is a RATE per root-year (quoted in basis points — 85bp is a plausible value), while
the displaced-diffusion's is dimensionless like Black's (0.22). §6's rule of two triggers on a
shared shape, and a quantity measured in different units is not a shared shape — it is exactly the
real-leg / year-on-year-leg case the charter names. A base class here would invite passing one
where the other is meant, and nothing would complain.

**On the naming of the shift.** The source calls it `alpha` (eq 8.8) and reserves `beta` for the
CEV exponent that `alpha` is derived from: `alpha = (1-beta) F(0) / beta`. The field is
`displacement`, after what it IS, because both letters are already taken — `alpha` is this arc's
most overloaded symbol (Vasicek's mean reversion, G2++'s first factor) and `beta` becomes the CEV
exponent the moment tranche 72 lands. Tranche 66's `MertonParams.drift` is the precedent: a source's
letter collision is not a reason to propagate it into the tree.

**Eq (8.8) is deliberately absent.** It is the CEV link — the `beta` that *generates* a
displacement — and nothing here has a CEV process to be generated from. It arrives with tranche 72
or not at all (§6: nothing speculative). What IS here is eq (8.9), the at-the-money matching rule,
because the chapter's own numerical example and its skew plot both use it, so a product pulls it.

Provenance:
  stock:  none — the migration tree carried no normal or shifted-lognormal forward model
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §8.2 (the normal process, eq 8.3), §8.4 (the
          displaced-diffusion process, eq 8.7 and eq 8.8), §8.4's Example 8.1 (eq 8.9, the
          at-the-money matching rule, with its printed `sigma_D = 0.1`)
  oracle: tests/L3/test_smile.py — eq 8.9 against the chapter's own printed value, the zero-shift
          identity, the refusals, and the two types being genuinely distinct
  crosscheck: quantlib BachelierCapFloorEngine / shifted BlackCapFloorEngine — deferred to the
          optional QuantLib tier
  slice:  tranche 71 — the smile models
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class BachelierModel:
    """A NORMAL forward rate: `dF = sigma_N dW` under the forward measure (eq 8.3).

    `normal_vol` is `sigma_N`, an absolute rate volatility per root-year. The model carries the
    `MarketSnapshot` it was calibrated to (§3), so the engine cannot be handed a mismatched market.

    The forward can go negative in finite time with non-zero probability — the source says so
    plainly, and it is the model's defining trade-off rather than a defect to guard against. No
    refusal here enforces positivity, because the model does not claim it."""

    market: MarketSnapshot
    normal_vol: float

    def __post_init__(self) -> None:
        if self.normal_vol <= 0.0:
            raise ValueError(f"a normal volatility must be positive, got {self.normal_vol}")


@dataclass(frozen=True)
class DisplacedDiffusionModel:
    """A SHIFTED lognormal forward: `d(F + alpha) = sigma_D (F + alpha) dW` (eq 8.7).

    `F + displacement` is lognormal, so the model is Black's applied to a shifted forward and a
    shifted strike — which is exactly how the engine prices it. A positive displacement lets the
    forward fall to `-displacement` before the shifted quantity hits zero, and the skew it produces
    is the reason the chapter introduces it.

    `vol` is `sigma_D`, dimensionless, and it is the volatility of the SHIFTED forward — not of `F`.
    Eq (8.9) is the bridge when a Black vol on `F` is what you have."""

    market: MarketSnapshot
    vol: float
    displacement: float

    def __post_init__(self) -> None:
        if self.vol <= 0.0:
            raise ValueError(f"a displaced-diffusion volatility must be positive, got {self.vol}")


def atm_matched_displaced_vol(vol: float, forward: float, displacement: float) -> float:
    """Eq (8.9): `sigma_D = sigma * F / (F + alpha)` — the vol that matches Black at the money.

    A displaced forward carries more absolute variance at the same relative vol, so quoting the same
    `sigma` in both models would make them disagree even at the money. This scales it back. The
    chapter's Example 8.1 prints the result: `sigma = 0.2`, `F = 5%`, `alpha = 5%` gives exactly 0.1.

    The match is close but not exact — the source says "very close", and its own numbers are 0.003789
    against 0.003793. It is an at-the-money calibration convention, not an identity, and the
    oracles treat it as one.

    **Grouped as `sigma * (F / (F + alpha))` rather than `sigma * F / (F + alpha)`**, which is the
    same quantity in exact arithmetic and not the same in floating point: multiplying first leaves
    `atm_matched_displaced_vol(0.2, 0.05, 0.0)` at `0.20000000000000004`, one ULP off the `sigma` it
    must return when there is no shift to correct for. The ratio is also the honest reading of eq
    (8.9) — a dimensionless scaling factor applied to a vol — so the grouping that makes the
    zero-shift identity exact is also the one that matches the formula's meaning."""
    shifted = forward + displacement
    if shifted <= 0.0:
        raise ValueError(
            f"the shifted forward must be positive to match at the money, got {shifted} "
            f"from a forward of {forward} and a displacement of {displacement}"
        )
    return vol * (forward / shifted)
