"""The Hull-White one-factor core (L3). BERHAN-2012 §2.4; standard Hull-White.

`HullWhiteParams` (mean reversion and volatility) and `β(t,T) = (1 − e^{−κ(T−t)})/κ` — eqs 2.4.3
and 2.4.16 — the two pieces every Hull-White use in the library shares.

**Extracted from `models/jarrow_yildirim.py` at its second consumer**, exactly as Black was in
tranche 9. They lived inside JY while JY was the only user; the nominal-rates work is the second, so
§3d and the rule of two say extract now — one definition, so a Hull-White convention cannot diverge
between the inflation and the rates uses. The nominal Hull-White model *is* JY's nominal factor
standing alone.

**The move is verbatim and bit-identical.** The class body and `hw_beta` were lifted unchanged
rather than retyped: `β`'s `κ → 0` limit and the sign of the exponent are the kind of thing a
re-transcription gets subtly wrong, and tranche 4's oracles are the guard. 15 reference values
(`β` across three `κ` and three intervals, plus `C` and `V²` at three configurations) were captured
as `repr()` before the move and compared after.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py — the parameter bundle and `β`
          were purified into this tree in tranche 4; this module is their relocation, not a
          re-import
  source: BERHAN-2012 (docs/references.md) §2.4 eqs 2.4.3 / 2.4.16 — the Hull-White specification
          of the JY framework. The bond option and caplet come from MCINERNEY-ZASTAWNIAK §3.5
          (eqs 3.16-3.21), because BERHAN performs its §5.3.1 fit with a cap formula it declines
          to publish ("beyond the scope of this thesis") — so the forward model is McInerney's and
          only the calibration TARGET is BERHAN's.
  oracle: tests/L3/test_jarrow_yildirim.py (tranche 4, unchanged and re-run green) — `β`'s `κ → 0`
          limit is `T − t`, which catches every misplaced-`κ` variant; the relocation additionally
          proved bit-identical across 15 captured values
  crosscheck: quantlib HullWhite / HullWhiteModel — deferred to the optional QuantLib tier
  slice:  tranche 10(a) — the Hull-White core extracted to its own module, plus the bond option
          and caplet
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.inflation import CapFloorKind
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price


@dataclass(frozen=True)
class HullWhiteParams:
    """One Hull-White factor: mean reversion `kappa` and volatility `sigma`.

    A bundle rather than four loose floats on the model, because JY has *two* such factors and
    naming them `kappa_n, sigma_n, kappa_r, sigma_r` invites passing the nominal pair where the
    real belongs. The grouping is the model's own structure -- the thesis derives it as two
    factors plus the index."""

    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError(f"mean reversion must be positive, got {self.kappa}")
        if self.sigma < 0.0:
            raise ValueError(f"volatility cannot be negative, got {self.sigma}")


def hw_beta(kappa: float, t: float, maturity: float) -> float:
    """`beta(t,T) = (1 - e^{-kappa(T-t)})/kappa` -- eqs 2.4.3 / 2.4.16.

    Defined once and shared by both factors, so the nominal and real betas cannot drift. As
    `kappa -> 0` it tends to `T - t`, which is the limit that catches a misplaced `kappa`: every
    wrong-denominator variant still looks plausible at a realistic `kappa`."""
    return (1.0 - math.exp(-kappa * (maturity - t))) / kappa


@dataclass(frozen=True)
class CapletPeriod:
    """One caplet period as the discount curve sees it: `B(0,Tᵢ₋₁)`, `B(0,Tᵢ)`, and the accrual.

    A bundle because `hw_caplet` otherwise takes six arguments, one over the limit, and the rule's
    answer is always to group the cohesive ones rather than raise it. These three are cohesive in a
    real sense: they are exactly what the *curve* says about the period, as opposed to what the
    contract (`strike`, `kind`) or the model (`variance`) say."""

    df_start: float
    df_pay: float
    accrual: float

    def __post_init__(self) -> None:
        if self.df_start <= 0.0 or self.df_pay <= 0.0:
            raise ValueError(
                f"discount factors must be positive, got {self.df_start} and {self.df_pay}"
            )
        if self.accrual <= 0.0:
            raise ValueError(f"a caplet period needs a positive accrual, got {self.accrual}")


def bond_option_variance(
    params: HullWhiteParams, option_expiry: float, bond_maturity: float
) -> float:
    """`ν(0,S)`, the variance of `ln B(S,T)` — eq 3.20, the constant-σ case of 3.19.

        ν(0,S) = σ²/(2κ³) · (1 − e^{−κ(T−S)})² · (1 − e^{−2κS})

    **Written through `hw_beta` rather than as the printed closed form**, because the two are the
    same object and the shared one cannot drift: eq 3.19 gives `ν = D(S,T)²·∫₀ˢ σ²e^{−2κ(S−u)}du`
    with `D(S,T) = (1 − e^{−κ(T−S)})/κ`, which *is* `β`. That identity was verified against the
    printed 3.20 rather than assumed, and the oracle sweeps both forms against each other.

    Zero exactly when `σ = 0` (it is an overall factor) and when `T = S` (the bond matures at the
    option's expiry, so there is nothing left to be uncertain about) — the two reductions the
    oracles anchor on."""
    beta = hw_beta(params.kappa, option_expiry, bond_maturity)
    return params.sigma**2 * beta**2 * (
        1.0 - math.exp(-2.0 * params.kappa * option_expiry)
    ) / (2.0 * params.kappa)


def zero_bond_option(
    df_expiry: float, df_maturity: float, strike: float, variance: float, kind: CapFloorKind
) -> float:
    """An option on a zero-coupon bond — eqs 3.16 (`BC`, a call) and 3.17 (`BP`, a put).

    **Composed from `black_price`, not transcribed again**, because a bond option *is* Black on the
    forward bond price `B(0,T)/B(0,S)` with total variance `ν`, discounted by `B(0,S)`:

        B(0,S)·ω·[F·Φ(ω d₊) − K·Φ(ω d₋)]  =  B(0,T)Φ(d₊) − K·B(0,S)Φ(d₋)   for ω = +1

    which is 3.16 exactly, and 3.17 for `ω = −1`. Writing the formula a second time would duplicate
    the Black convention across two modules, which is what §3d exists to prevent; the oracle
    transcribes 3.16–3.18 longhand instead, so there is still independent code on each side.

    **The source drops `d₋`'s radical in both renderings**, printing `d₋ = d₊ − ν(0,S)` while `ν` is
    defined as a variance. Read literally it prices a call *below zero*, which settles it — and note
    that put-call parity does **not**: `BC − BP = B(0,T) − K·B(0,S)` holds identically for any
    `d₊`/`d₋`, so on this question parity is a check that cannot fail."""
    forward = df_maturity / df_expiry
    return df_expiry * black_price(forward, strike, variance, kind)


def hw_caplet(
    period: CapletPeriod, strike: float, variance: float, kind: CapFloorKind
) -> float:
    """A caplet or floorlet on the spot rate over the period — eq 3.21.

        Cplᵢ(0) = (1 + τᵢK) · BP(0; Tᵢ₋₁, Tᵢ, (1 + τᵢK)⁻¹)
        Flrᵢ(0) = (1 + τᵢK) · BC(0; Tᵢ₋₁, Tᵢ, (1 + τᵢK)⁻¹)

    **A caplet is a PUT on the bond and a floorlet is a CALL** — the direction is inverted because
    a bond price falls as its rate rises. That inversion is the thing an implementation silently
    reverses while every magnitude check stays plausible, so the oracle transcribes it from §3.21
    rather than inferring it."""
    scale = 1.0 + period.accrual * strike
    bond_strike = 1.0 / scale
    bond_kind = CapFloorKind.FLOOR if kind is CapFloorKind.CAP else CapFloorKind.CAP
    return scale * zero_bond_option(
        period.df_start, period.df_pay, bond_strike, variance, bond_kind
    )


@dataclass(frozen=True)
class HullWhiteModel:
    """The one-factor Hull-White model, carrying the snapshot it was calibrated to.

    Mirrors `JarrowYildirim`: the engine reaches market data only through `model.market`, never as a
    second argument, so a model/market mismatch cannot be expressed (§3)."""

    market: MarketSnapshot
    params: HullWhiteParams
