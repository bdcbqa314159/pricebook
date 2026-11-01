"""Strip quoted floors into floorlets and imply the volatility surface (L3). BERHAN-2012 §5.2.

A market quotes **floors**, each the sum of its floorlets. The surface is recovered by
differencing the strip and inverting Black node by node:

    floorletᵢ      = floorᵢ − floorᵢ₋₁                        with floor(T₀) = 0
    black_valueᵢ   = floorletᵢ / (N · ψᵢ · Pn(0,Tᵢ))
    mᵢ             = [Pn(0,Tᵢ₋₁)/Pn(0,Tᵢ)] · [Pr(0,Tᵢ)/Pr(0,Tᵢ₋₁)]
    σ_I(i)         = black_implied_vol(black_valueᵢ, mᵢ, 1 + κ, Tᵢ − Tᵢ₋₁, FLOOR)

**The model is the thesis's "simplified Jarrow–Yildirim", and §5.2 proves it is JY at
`σn = σr = 0`** (Prop 2.3.1) — which is why the forward `mᵢ` carries no convexity term and the
variance is `σ_I²·(Tᵢ − Tᵢ₋₁)`. Both facts are already oracled: `yoy_convexity` vanishes at
`σr = 0` by construction, and tranche 5's
`test_with_both_rate_vols_zero_the_variance_IS_index_vol_squared_times_the_period` is the second.
So this builder is not a new model — it is the inverse of a model the library already has.

**The variance horizon is the PERIOD LENGTH, not the time to expiry**, and that distinction is
load-bearing: the floorlet on `[2y, 3y]` expires at 3y and accumulates one year of variance. Passing
the expiry would under-state `σ_I` by `√(Tᵢ/(Tᵢ−Tᵢ₋₁))` — 1.7× at the 3y node — with every oracle
green, because the forward and the inverse would agree on the wrong horizon. It is why
`black_implied_vol`'s parameter was renamed from `expiry` to `variance_years` when this module
became its second consumer.

**L3 though it returns an L1 value.** It composes the L3 Black inverse, so it cannot sit at L1;
`acyclic` accepts the L3→L1 import (downward) and does not constrain the return type, which was
confirmed by probe before this was built. The surface it produces is market data and so is an L1
type — §2's "produces market data → L1" holds for a bootstrap using only L0+L1, while a market-data
object built *through a model analytic* is built at the analytic's layer.

**What this is NOT.** It builds the *market's* vol view. It is not a claim that JY reproduces that
view — §5.3.2 is explicit that JY fits ATM and not the smile, so the skew this surface shows is
exactly what the model cannot fit. Saying so is the honest scope, not a gap.

Provenance:
  stock:  _stock/quarry/pricebook/options/inflation_smile.py, inflation_vol.py — mined for the
          stripping SHAPE only; the Black inverse is read from standard Black-76 (models/black.py),
          not from the stock, which may carry a different parameterisation
  source: BERHAN-2012 (docs/references.md) §5.2 (the floorlet bootstrap and the implied surface of
          Figures 5.3/5.4), §2.6 eq 2.6.5 (the floorlet being inverted), Prop 2.3.1 (the
          simplified-JY equivalence that zeroes the convexity)
  oracle: tests/L3/test_implied_vol_surface.py — a known SLOPED generating vol is recovered at every
          node (the non-telescoping anchor: the test prices the floorlets itself), with the fixture's
          per-node distinctness ASSERTED and a reversed strip shown to give different numbers, so
          the check discriminates a mis-indexed difference rather than only a dropped one; flat and
          sloped reduction + companion; two strikes strip independently; a decreasing strip refused
  crosscheck: quantlib YoYInflationCapFloorStripper / implied-vol stripping — deferred to the
          optional QuantLib tier
  slice:  tranche 9(b) — the floor strip and the implied-vol surface
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from itertools import groupby

from pricebook.foundation.inflation import CapFloorKind
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.vol_surface import FloorQuote, ImpliedVolSurface
from pricebook.models.black import black_implied_vol


def _validate_strip(quotes: Sequence[FloorQuote]) -> None:
    """Nesting and ordering are **preconditions of differencing**, not coincidences.

    The same shape as tranche 8's bootstrap: each quote must extend the previous one, or a
    difference would subtract two floors that do not share their earlier periods."""
    previous: tuple[date, ...] = ()
    last_maturity: date | None = None
    for quote in quotes:
        dates = quote.schedule.dates
        maturity = dates[-1]
        if last_maturity is not None and maturity <= last_maturity:
            raise ValueError(
                f"floor quotes at one strike must be in strictly ascending maturity order, "
                f"got {maturity} after {last_maturity}"
            )
        if previous and dates[: len(previous)] != previous:
            raise ValueError(
                f"each floor quote's schedule must extend the previous one; the quote maturing "
                f"{maturity} does not share the earlier payment dates"
            )
        previous = dates
        last_maturity = maturity


def _strike_column(
    quotes: Sequence[FloorQuote], market: MarketSnapshot
) -> dict[tuple[date, float], float]:
    """One strike's floor strip → its implied-vol nodes."""
    _validate_strip(quotes)
    currency = quotes[0].notional.currency
    nominal = market.curves.discount(currency)
    real = market.curves.real(currency)
    measure = nominal.time_measure

    nodes: dict[tuple[date, float], float] = {}
    running = 0.0
    for quote in quotes:
        strike = 1.0 + quote.strike
        start, pay, psi = list(quote.schedule.periods())[-1]
        floorlet = quote.price.amount - running
        running = quote.price.amount
        if floorlet < 0.0:
            raise ValueError(
                f"the floor quotes imply a negative floorlet ({floorlet}) on the period ending "
                f"{pay}: a longer floor cannot be worth less than a shorter one at the same strike"
            )
        # Simplified JY: no convexity term, so the forward ratio is purely a curve relationship.
        forward = (nominal.df(start) / nominal.df(pay)) * (real.df(pay) / real.df(start))
        discounted_unit = quote.notional.amount * psi * nominal.df(pay)
        variance_years = measure.year_fraction(pay) - measure.year_fraction(start)
        nodes[(pay, quote.strike)] = black_implied_vol(
            floorlet / discounted_unit, forward, strike, variance_years, CapFloorKind.FLOOR
        )
    return nodes


def implied_vol_surface(
    market: MarketSnapshot, quotes: Sequence[FloorQuote]
) -> ImpliedVolSurface:
    """The implied-vol surface of §5.2, from a set of nested floor quotes.

    Quotes are grouped **by strike and stripped independently**, because the decomposition runs
    along maturity: differencing across strike would mix two unrelated option series while still
    producing a plausible node count."""
    if not quotes:
        raise ValueError("an implied-vol surface needs at least one floor quote")

    by_strike = sorted(quotes, key=lambda q: (q.strike, q.schedule.dates[-1]))
    nodes: dict[tuple[date, float], float] = {}
    for _, column in groupby(by_strike, key=lambda q: q.strike):
        nodes.update(_strike_column(list(column), market))
    return ImpliedVolSurface(anchor=market.valuation_date, nodes=nodes)
