"""Market Model II — modelling the forward inflation indices (L3). BERHAN-2012 ch 4.

The **third** model in the library, and the third through tranche 12(a)'s `IndexRatioModel` seam
with no change to either pricer.

    Di(t) = σ_{I,i−1}[ τᵢσ_{n,i}Fn/(1+τᵢFn)·ρ_{I,n} + σ_{I,i−1} − σ_{I,i}ρ_{I,i} ](Tᵢ₋₁ − t)
    m     = I(t,Tᵢ)/I(t,Tᵢ₋₁)·e^{Di(t)} = [(1+τFn)/(1+τFr)]·e^{Di(t)}                     (4.1.6)
    V²    = [σ_{I,i−1}² + σ_{I,i}² − 2ρ_{I,i}σ_{I,i−1}σ_{I,i}](Tᵢ₋₁ − t) + σ_{I,i}²(Tᵢ − Tᵢ₋₁)

`I(t,Tᵢ)/I(t,Tᵢ₋₁)` is `(1+τFn)/(1+τFr)` by eq 3.1.5, so it is read straight off the curves — the
same quantity JY reaches with `(Pn/Pn)(Pr/Pr)`, which is why all three models meet at the same
zero-vol limit.

## MM-II's identity: no real-rate volatility, anywhere

The source's own words: *"contrary to Market Model I, the correction term does not depend on the
volatility of real rates."* There is no `σr` field on `MMIIParams` — not merely unused but absent,
because a carried-but-unread real vol would be both the §6 violation and a lie about what this model
is. The real *curve* still matters, through the forward levels in `1 + τFr`; it is the real
*volatility* that cannot reach this model.

## A second exact anchor, which no other model here has

*"The formula above is exact only when the correlations between `I(·,Tᵢ₋₁)` and `Fn(·,Tᵢ₋₁,Tᵢ)` are
assumed to be zero so that the nominal forward rate is zeroed out from `Di`."* At `ρ_{I,n} = 0` the
freezing approximation is **not used at all** and `Di` is a closed form with no forward rate in it —
an exact anchor that does not require zero volatility. Away from it the frozen drift is approximate
and, the source warns, *"may be rough for longer maturities"*; the oracle asserts that the
sensitivity **grows with maturity** rather than bounding it by an arbitrary smallness threshold.

## Two hazards in the source, handled rather than guessed

**1. Eq 4.1.3's denominator disagrees with `Di`'s, in BOTH renderings.** 4.1.3 prints
`τᵢσ_{n,i}Fn/(1 + Fn)`; `Di(t)` prints `.../(1 + τᵢFn)`. An internal inconsistency, not a rendering
artefact — so §4's derivation-step rule fires and finds a *discrepancy* rather than a confirmation.
`1 + τᵢFn` is taken as correct: it is the standard LMM coefficient, it is what Market Model I's
freezing relation uses consistently, `1 + Fn` adds a pure number to a rate, and `Di` is the formula
used downstream in 4.1.6 and 4.2.4. The two readings differ by more than 0.1% here, so the choice
is pinned by an oracle rather than left to the reader.

**2. `Vi(t)` is defined twice, for two different quantities.** Eq 4.2.2 gives
`sqrt([…][Tᵢ₋₁−t])`; a line before 4.2.4 redefines the same symbol as
`sqrt(Vᵢ²(t) + σ_{I,i}²(Tᵢ−Tᵢ₋₁))`. The caplet needs the augmented one, and the reason is the
measure rather than the notation: 4.2.1 is the distribution of the **forward**-index ratio at
`Tᵢ₋₁`, while the payoff is on the **spot** ratio at `Tᵢ`, so the diffusion from `Tᵢ₋₁` to `Tᵢ` must
be added. Dropping it would silently reuse MM-I's `σ_I²·span` shape and lose the whole lead term —
the missing-term trap §4 now names, so it has its own guard.

## Scope

One index vol, so `σ_{I,i−1} = σ_{I,i}` and `Vi²` collapses to `2σ_I²(1−ρ_{I,I})(Tᵢ₋₁−t)`. A
per-period index-vol term structure is what §5.4's calibration fits, so it arrives with that
consumer (T14) rather than speculatively here.

Provenance:
  stock:  — the quarry had no forward-CPI market model; built fresh from the source
  source: BERHAN-2012 (docs/references.md) ch 4 — eq 4.1.2/4.1.3 (the dynamics and the frozen
          drift), `Di(t)` and eq 4.1.6 (the year-on-year value), eqs 4.2.1–4.2.4 (the cap/floor and
          `Vi`), and §4.2's closing paragraph for the no-real-vol property and the exactness
          condition. Mercurio (2005) and Belgrade–Benhamou–Koehler (2004) are the originals and are
          `referenced-by` only — not cited, per §7.
  oracle: tests/L3/test_market_model_ii.py — pointwise transcriptions of 4.1.6 and 4.2.2 swept over
          periods (§4: properties never replace this); the **four-path** zero-vol reduction (MM-II,
          MM-I, JY and the curve ratio); the **ρ_{I,n}=0 exact anchor** and its converse (σ_n inert
          there); the no-real-vol identity proved structurally *with* the companion that MM-I's `m`
          does move with σ_r; `Vi²` required to differ from `σ_I²·span` and to reduce to it exactly
          at ρ_{I,I}=1; three models required to DISAGREE away from the limits; `cap − floor = the
          YoY leg` carrying a third model through the seam; and the freezing sensitivity asserted to
          GROW with maturity as the source warns
  crosscheck: none — QuantLib has no inflation forward-CPI market model
  slice:  tranche 13 — Market Model II
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Currency
from pricebook.market.curve_set import CurveRole
from pricebook.market.snapshot import MarketSnapshot

#: The key a FLAT term structure uses — a sentinel date no schedule can produce, so a flat set and
#: a real one cannot be confused for each other.
_FLAT = date(1, 1, 1)


@dataclass(frozen=True)
class MMIICorrelations:
    """MM-II's two correlations: index-to-nominal-forward, and index-to-index across periods.

    **A third correlation type, and deliberately so.** JY carries nominal-real / real-index /
    nominal-index; MM-I carries nominal-real / real-index; MM-II carries index-nominal /
    index-index. Three models, three different shapes — and §6's rule-of-two fires on a shared
    *shape*, not a shared word. Sharing one type would force two of the three to carry fields they
    never read."""

    index_nominal: float
    index_index: float

    def __post_init__(self) -> None:
        for name, rho in (
            ("index_nominal", self.index_nominal),
            ("index_index", self.index_index),
        ):
            if not -1.0 <= rho <= 1.0:
                raise ValueError(f"{name} correlation must lie in [-1, 1], got {rho}")


@dataclass(frozen=True)
class MMIIParams:
    """MM-II's parameters: a **term structure** of index vols, the nominal vol, the correlations.

    **No real-rate volatility, and its absence is the point** — see the module docstring.

    **`index_vols` is keyed by the forward index's own maturity, which is what makes it a term
    structure rather than a bag of numbers.** `σ_{I,i}` is the volatility of `I(·,Tᵢ)`, so a period
    `[Tᵢ₋₁, Tᵢ]` reads *two adjacent entries* — `index_vols[Tᵢ₋₁]` and `index_vols[Tᵢ]` — which is
    exactly why `Di` and `Vᵢ²` carry both. Tranche 13 carried a single scalar and said the term
    structure would arrive with §5.4's calibration; it has, because §5.4.2's free set per expiry is
    `{σ_{I,i−1}, σ_{I,i}, ρ_{I,i}, ρ_{I,n,i}}` and a scalar would silently collapse two of its four
    parameters into one — making any identifiability finding an artefact of our simplification
    rather than a property of the model.

    Use `MMIIParams.flat(...)` for the constant case; it is **bit-identical** to the tranche 13
    scalar form, verified across the six values that tranche measured.

    `nominal_vol` stays a scalar. §5.4.1 indexes it by maturity too, but it is **read** from the
    Black ATM cap quote rather than fitted, so one value per model instance is the natural
    granularity for a single-expiry fit; a per-maturity ATM cap curve arrives with a consumer that
    fits across maturities."""

    index_vols: Mapping[date, float]
    nominal_vol: float
    correlations: MMIICorrelations

    @classmethod
    def flat(
        cls, index_vol: float, nominal_vol: float, correlations: MMIICorrelations,
        dates: Sequence[date] = (),
    ) -> MMIIParams:
        """A constant index-vol term structure — the tranche 13 shape, now one case of the general.

        `dates` may be omitted: a flat structure answers every lookup with the same number, which
        `_index_vol` implements by falling back to the single value when a date is absent."""
        return cls({d: index_vol for d in dates} or {_FLAT: index_vol}, nominal_vol, correlations)

    def __post_init__(self) -> None:
        if not self.index_vols:
            raise ValueError("MM-II needs at least one index volatility")
        for key, vol in self.index_vols.items():
            if vol < 0.0:
                raise ValueError(f"index_vol cannot be negative, got {vol} at {key}")
        if self.nominal_vol < 0.0:
            raise ValueError(f"nominal_vol cannot be negative, got {self.nominal_vol}")

    def index_vol_at(self, maturity: date) -> float:
        """`σ_{I,i}` for the forward index maturing at `maturity`.

        A flat structure (built by `flat()` without dates) answers every maturity with its single
        value; a real term structure refuses an unknown date rather than interpolating, for the
        reason `ImpliedVolSurface.vol` refuses one — an interpolation scheme baked into a model
        parameter is an unexercised choice."""
        if _FLAT in self.index_vols:
            return self.index_vols[_FLAT]
        try:
            return self.index_vols[maturity]
        except KeyError:
            raise KeyError(
                f"no MM-II index volatility for the forward index maturing {maturity}; "
                f"the term structure holds {len(self.index_vols)} points and does not interpolate"
            ) from None


@dataclass(frozen=True)
class MarketModelII:
    """MM-II, carrying the snapshot it was calibrated to — and satisfying `IndexRatioModel`."""

    market: MarketSnapshot
    params: MMIIParams

    def expected_index_ratio(self, period_start: date, period_pay: date) -> float:
        """`m` — eq 4.1.6's forward index ratio times `e^{Di}`."""
        nominal, real, measure = self._curves()
        forward_index_ratio = (
            nominal.df(period_start) / nominal.df(period_pay)
        ) * (real.df(period_pay) / real.df(period_start))
        return forward_index_ratio * math.exp(self._drift(period_start, period_pay))

    def index_ratio_variance(self, period_start: date, period_pay: date) -> float:
        """`V² = Vᵢ²(t) + σ_{I,i}²(Tᵢ − Tᵢ₋₁)` — the augmented form, per hazard 2 above.

        The lead term is `[σ_{I,i−1}² + σ_{I,i}² − 2ρ_{I,I}σ_{I,i−1}σ_{I,i}](Tᵢ₋₁ − t)`, which with
        one index vol is `2σ_I²(1 − ρ_{I,I})(Tᵢ₋₁ − t)` — zero exactly at `ρ_{I,I} = 1`, where the
        variance reduces to MM-I's `σ_I²·span` and the two models' cap/floors coincide."""
        _, _, measure = self._curves()
        lead = measure.year_fraction(period_start)
        span = measure.year_fraction(period_pay) - lead
        sigma_pay = self.params.index_vol_at(period_pay)
        rho = self.params.correlations.index_index
        # At `lead = 0` the whole lead term vanishes, so `sigma_{I,i-1}` is multiplied by zero and
        # must NOT be looked up -- the first period's `T_0` is the curve anchor and has no forward
        # index maturing on it. That is also why the 1-Year floor identifies `sigma_{I,1}` alone
        # (BERHAN 5.4.2), which the calibration's bootstrap relies on.
        if lead == 0.0:
            return sigma_pay * sigma_pay * span
        sigma_start = self.params.index_vol_at(period_start)
        lead_variance = (
            sigma_start * sigma_start
            + sigma_pay * sigma_pay
            - 2.0 * rho * sigma_start * sigma_pay
        ) * lead
        return lead_variance + sigma_pay * sigma_pay * span

    def _drift(self, period_start: date, period_pay: date) -> float:
        """`Di(t)` — the correction term, with **no real-rate volatility in it**.

        The `τᵢFn/(1+τᵢFn)` coefficient is the denominator of `Di` as printed, not eq 4.1.3's
        `1 + Fn`; hazard 1 in the module docstring says why, and an oracle pins the choice."""
        nominal, _, measure = self._curves()
        lead = measure.year_fraction(period_start)
        tau_fn = nominal.df(period_start) / nominal.df(period_pay) - 1.0

        p = self.params
        # At `lead = 0` the whole drift vanishes (it carries `T_{i-1} - t` as a factor), so no
        # volatility is looked up -- the first period's start is the curve anchor.
        if lead == 0.0:
            return 0.0
        sigma_start = p.index_vol_at(period_start)   # sigma_{I,i-1}
        sigma_pay = p.index_vol_at(period_pay)       # sigma_{I,i}
        bracket = (
            p.nominal_vol * tau_fn / (1.0 + tau_fn) * p.correlations.index_nominal
            + sigma_start
            - sigma_pay * p.correlations.index_index
        )
        return sigma_start * bracket * lead

    def _curves(self):
        """The nominal and real curves and the shared time measure, in one lookup."""
        currency = self._currency()
        nominal = self.market.curves.discount(currency)
        return nominal, self.market.curves.real(currency), nominal.time_measure

    def _currency(self) -> Currency:
        """The one currency this model's snapshot discounts in — refusing when ambiguous, as the
        other two models do and for the same reason."""
        currencies = {
            key.currency for key in self.market.curves.curves if key.role is CurveRole.DISCOUNT
        }
        if len(currencies) != 1:
            raise KeyError(
                f"Market Model II needs exactly one discount curve to know its currency, "
                f"found {len(currencies)}"
            )
        return currencies.pop()
