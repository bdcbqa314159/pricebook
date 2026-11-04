"""Market Model I — an LMM for nominal and real forward rates (L3). BERHAN-2012 ch 3.

The second model in the library, and the reason tranche 12(a) built the `IndexRatioModel` seam. It
prices the *same* products as Jarrow–Yildirim through the *same* pricers, by answering the same two
questions differently.

## The formula

Under `Q^{Tᵢ}_n`, Mercurio's result (3.1.7–3.1.8) is that

    (Xᵢ, Yᵢ) = (ln Fn(Tᵢ₋₁,·)/Fn(t,·), ln Fr(Tᵢ₋₁,·)/Fr(t,·))

is bivariate normal with

    µx = −½σn²(Tᵢ₋₁−t),                     σx = σn√(Tᵢ₋₁−t)
    µy = [−½σr² − ρ_Ir·σI·σr](Tᵢ₋₁−t),      σy = σr√(Tᵢ₋₁−t)

and the `Tᵢ`-forward expectation of the index ratio is (3.1.11)

    m = E_z[ (1 + τFn·e^{ρ_nr σx z − ½σx²ρ_nr²}) / (1 + τFr·e^{µy + σy z}) ]

with `1 + τF = P(t,Tᵢ₋₁)/P(t,Tᵢ)` reading the forwards straight off the curves. No closed form, so
it is a Gauss-Hermite expectation (`foundation/quadrature.py`).

**The transcription is corroborated by a second path inside the source, which is stronger than two
renderings.** `pdftotext -layout` mangles 3.1.11 into broken columns, so the dual-rendering rule
could not settle it — but page 25 gives the integral in its **pre-substitution** form, over `y`
against the conditional density, before `z = (y − µy)/σy`. The two agree to 2.2e-16, and a misprint
would have to appear identically in both forms to survive that. Both are oracled.

## Where the approximation is — and it is NOT in 3.1.11

The spec expected the freezing procedure to make the leg value approximate. It does not: given its
per-period inputs, 3.1.11 is an **exact** expectation. Mercurio's freezing governs the *recursion
between periods* —

    σ_I,i−1 ≈ σ_I,i + σr·τFr(0)/(1+τFr(0)) − σn·τFn(0)/(1+τFn(0))

— where the exact relation carries the **stochastic** `F(t)` rather than its time-0 value. So the
approximation is a statement about the index-vol *term structure*, implemented as `index_vol_step`
and bounded by how far the frozen coefficient can move under a stated forward shock: at a 200bp
move it shifts `σ_I` by under 1e-2 in absolute terms. **That is a sensitivity bound, not a
distributional one** — a probability-weighted bound needs an LMM simulation of the forwards, which
this tranche does not build and does not pretend to.

## Scope

One `σn`, `σr`, `σI` and two correlations, constant across periods — which is what the freezing
exists to justify ("we can still assume that the volatilities σ_I,i are all constant"). A per-period
vol term structure arrives with a consumer that needs one, not before. `ρ_nI` is absent because
neither 3.1.11 nor §3.2 reads it, and a carried-unread field is what §6 forbids — hence
`MMICorrelations` with two fields rather than reusing JY's three.

Provenance:
  stock:  — the quarry had no market model for inflation; built fresh from the source
  source: BERHAN-2012 (docs/references.md) ch 3 — eqs 3.1.5–3.1.8 (the dynamics and the bivariate
          moments), the page-25 pre-substitution integral and eq 3.1.11 (the forward ratio), §3.2
          (the cap/floor, Black with `σ_I²(Tᵢ−Tᵢ₋₁)`), and the freezing relation above.
          MCINERNEY-ZASTAWNIAK ch 5 is held as LMM background (Bernardo ruled no dedicated LMM text
          is to be acquired — the LMM is not this project's topic).
  oracle: tests/L3/test_market_model_i.py — the two source forms agree (the corroboration above);
          **the three-path zero-vol reduction** (MM-I's integral, JY's `e^C → 1`, and the curves
          directly, all exactly equal) with its non-vacuity companion; MM-I and JY required to
          DISAGREE away from the limit; positivity across a parameter sweep; the cap/floor variance
          on the PERIOD not the expiry; the freezing bound measured; and `cap − floor = the YoY leg`
          now carrying a second model through tranche 12(a)'s seam
  crosscheck: quantlib — none for this model (QuantLib has no inflation LMM); the Gauss-Hermite rule
          is cross-checked against scipy.integrate.quad in the oracle, which is a different
          integrator on the other side of the check
  slice:  tranche 12(b) — Market Model I
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Currency
from pricebook.foundation.quadrature import standard_normal_expectation
from pricebook.market.curve_set import CurveRole
from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class MMICorrelations:
    """MM-I's two correlations: nominal-real, and real-index.

    **Two, not three, and deliberately not `JYCorrelations`.** Neither eq 3.1.11 nor §3.2 reads a
    nominal-index correlation, so carrying one would be the unread field §6 forbids — and sharing
    JY's type would force exactly that. Two models needing different correlation sets is two types;
    the rule of two fires on a shared *shape*, and these shapes differ."""

    nominal_real: float
    real_index: float

    def __post_init__(self) -> None:
        for name, rho in (("nominal_real", self.nominal_real), ("real_index", self.real_index)):
            if not -1.0 <= rho <= 1.0:
                raise ValueError(f"{name} correlation must lie in [-1, 1], got {rho}")


@dataclass(frozen=True)
class MMIParams:
    """MM-I's parameters: the three volatilities and the correlations.

    Constant across periods, which is what the freezing procedure exists to justify — the source's
    own words are that with it "we can still assume that the volatilities σ_I,i are all constant". A
    per-period term structure arrives with a consumer that needs one."""

    nominal_vol: float
    real_vol: float
    index_vol: float
    correlations: MMICorrelations

    def __post_init__(self) -> None:
        for name, vol in (
            ("nominal_vol", self.nominal_vol),
            ("real_vol", self.real_vol),
            ("index_vol", self.index_vol),
        ):
            if vol < 0.0:
                raise ValueError(f"{name} cannot be negative, got {vol}")


def index_vol_step(
    index_vol: float, nominal_vol: float, real_vol: float, tau_fn: float, tau_fr: float
) -> float:
    """`σ_I,i−1` from `σ_I,i` — the freezing relation of BERHAN ch 3, p.25.

        σ_I,i−1 ≈ σ_I,i + σr·τFr/(1+τFr) − σn·τFn/(1+τFn)

    **This is the whole of MM-I's approximation.** The exact relation carries the *stochastic*
    `F(t)`; freezing evaluates the coefficients at time 0. Each adjustment term carries a rate vol
    as a factor, so with `σn = σr = 0` the term structure is flat — the exact reduction the oracle
    anchors on. The error is bounded by how far `τF/(1+τF)` can move, which is measured in the
    oracle rather than asserted.

    **It REFUSES a non-positive result, and that is not defensive padding — it fires at realistic
    parameters.** Walking the recursion backwards from `σ_I = 0.012` with `σn = 25%`, `σr = 20%` on
    this library's own sloped test curves:

        i=4  +0.012000
        i=3  +0.005605
        i=2  −0.000542   ← not a volatility

    Two steps. The adjustment is negative whenever `σn·τFn/(1+τFn) > σr·τFr/(1+τFr)`, which is the
    ordinary case (nominal vol and nominal forwards both exceed real), so the relation **cannot be
    iterated far** and the source's "we can still assume that the volatilities σ_I,i are all
    constant" does not survive a realistic parameter set. Returning a negative volatility would let
    it reach a `√` and surface as a `nan` three layers away; raising names it where it happens."""
    stepped = (
        index_vol
        + real_vol * tau_fr / (1.0 + tau_fr)
        - nominal_vol * tau_fn / (1.0 + tau_fn)
    )
    if stepped <= 0.0:
        raise ValueError(
            f"the freezing relation gives a non-positive index volatility ({stepped}) from "
            f"sigma_I={index_vol}: the recursion has been iterated past where it is valid at "
            f"these rate volatilities"
        )
    return stepped


@dataclass(frozen=True)
class MarketModelI:
    """MM-I, carrying the snapshot it was calibrated to — and satisfying `IndexRatioModel`.

    Two fields, like every model here. The engine reaches market data only through `market`, so a
    model/market mismatch cannot be expressed (§3)."""

    market: MarketSnapshot
    params: MMIParams

    def expected_index_ratio(self, period_start: date, period_pay: date) -> float:
        """`m` — eq 3.1.11, as a Gauss-Hermite expectation.

        `1 + τF = P(t,Tᵢ₋₁)/P(t,Tᵢ)` reads each forward straight off its curve, so no forward rate
        is stored and none can drift from the curve it came from."""
        currency = self._currency()
        nominal = self.market.curves.discount(currency)
        real = self.market.curves.real(currency)
        measure = nominal.time_measure

        tau_fn = nominal.df(period_start) / nominal.df(period_pay) - 1.0
        tau_fr = real.df(period_start) / real.df(period_pay) - 1.0

        p = self.params
        lead = measure.year_fraction(period_start)        # T_{i-1} − t, with t = 0
        sigma_x = p.nominal_vol * math.sqrt(lead)
        sigma_y = p.real_vol * math.sqrt(lead)
        rho_nr = p.correlations.nominal_real
        mu_y = (
            -0.5 * p.real_vol * p.real_vol
            - p.correlations.real_index * p.index_vol * p.real_vol
        ) * lead

        def integrand(z: float) -> float:
            numerator = 1.0 + tau_fn * math.exp(
                rho_nr * sigma_x * z - 0.5 * sigma_x * sigma_x * rho_nr * rho_nr
            )
            return numerator / (1.0 + tau_fr * math.exp(mu_y + sigma_y * z))

        return standard_normal_expectation(integrand)

    def index_ratio_variance(self, period_start: date, period_pay: date) -> float:
        """`V² = σ_I²·(Tᵢ − Tᵢ₋₁)` — §3.2's inner expectation is Black with this variance.

        **The period, not the expiry.** The index ratio accumulates variance over `[Tᵢ₋₁, Tᵢ]`,
        which is the same distinction tranche 9 measured at up to 50% and tranche 10(b) found
        reversed for a rate caplet."""
        measure = self.market.curves.discount(self._currency()).time_measure
        span = measure.year_fraction(period_pay) - measure.year_fraction(period_start)
        return self.params.index_vol * self.params.index_vol * span

    def _currency(self) -> Currency:
        """The one currency this model's snapshot discounts in — refusing when ambiguous, as
        `JarrowYildirim` does and for the same reason."""
        currencies = {
            key.currency for key in self.market.curves.curves if key.role is CurveRole.DISCOUNT
        }
        if len(currencies) != 1:
            raise KeyError(
                f"Market Model I needs exactly one discount curve to know its currency, "
                f"found {len(currencies)}"
            )
        return currencies.pop()
