"""Fitting `(κn, σn)` to ATM nominal cap volatilities (L4). BERHAN-2012 §5.3.1.

BERHAN's scheme, verbatim: *"for each maturity Tᵢ, we observe the ATM Cap (Black) volatility quote
σ_ATMᵢ and the accompanying ATM strike level K_ATMᵢ. We can then fit the nominal volatility
parameters κn, σn by performing a least squares optimization over*

    Cap_HullWhite(t, Tᵢ, ψᵢ, K_ATMᵢ, σn, κn) − Cap_Black(t, Tᵢ, ψᵢ, K_ATMᵢ, σ_ATMᵢ)"

**This closes the two-of-six calibration gap.** `σn` has been a hand-supplied input since tranche 6
(`JYFixedParams.sigma_nominal`), because BERHAN sources it here and then declines to publish the
forward model — "the implementation of nominal volatility structure calibration is a subject in
itself and is beyond the scope of this thesis". MCINERNEY-ZASTAWNIAK §3.5 supplied the model in
tranche 10(a); its eq 3.26 states the same scheme in vol space rather than price space.

**Price space, not vol space, and that is BERHAN's choice not a convenience.** §5.3.1 least-squares
the *price* difference. McInerney 3.26 instead solves `Cplᵢ(0) = Cpl^Black(0;σᵢ)` for a model-implied
caplet vol and matches vols. The two are equivalent at the optimum and differ in weighting — vol
space implicitly weights by `1/vega`, which flattens the long end. BERHAN's own result ("not too bad
in the long end") is reported against the price-space fit, so price space is what reproduces it.

**L4, because it inverts the engine.** It reprices through `price_nominal_cap` to solve for the
parameters, so by §2 it is engine-inverting and lives here, beside `jy_calibration` — not with the
models. It calls the pricer directly rather than the registry's `dispatch`, since dispatching on a
type it already holds is indirection with no consumer.

**The Black caplet's variance horizon is the EXPIRY — the opposite of the inflation case.** A rate
caplet's underlying `L(Tᵢ₋₁,Tᵢ)` is observed at `Tᵢ₋₁`, so its variance is `σ²·Tᵢ₋₁`. An inflation
floorlet on `I(Tᵢ)/I(Tᵢ₋₁)` accumulates over the period, so its variance is `σ_I²·(Tᵢ − Tᵢ₋₁)`.
Tranche 9 measured that confusion at up to 50% in the other direction; the two instruments genuinely
differ, which is exactly why `black_price`'s parameter is `variance_years` and not `expiry`.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/ — mined for the scheme's SHAPE only; the Black cap
          and the implied-vol solve are built on this tree's own `models/black.py`
  source: BERHAN-2012 (docs/references.md) §5.3.1 (the least-squares scheme and the ATM quote
          convention); MCINERNEY-ZASTAWNIAK §2.9 (the implied Black cap volatility), eq 3.26
  oracle: tests/L4/test_hw_cap_calibration.py — B1 recovers a known `(κn,σn)` from quotes generated
          through tranche 10(a); B2 is its identifiability companion (each parameter moves the
          strip, four starts land on one point, the sensitivity Jacobian is well conditioned, and a
          ONE-maturity strip is asserted to be unidentifiable so the multi-start check is known to
          have power); B3 the residual vanishes at the truth and descends from a wrong start; B4
          failure is a value
  crosscheck: quantlib HullWhite calibration / CapHelper — deferred to the optional QuantLib tier
  slice:  tranche 10(b) — the (κn, σn) fit to ATM nominal caps
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pricebook.engine.nominal_cap import price_nominal_cap
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.solvers import bracketed_root, least_squares_fit
from pricebook.market.curve import CurveHandle
from pricebook.market.curve_set import CurveRole
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price
from pricebook.models.hull_white import HullWhiteModel, HullWhiteParams
from pricebook.products.nominal_cap import NominalCap

_FLOOR = 1e-10   # a mean reversion and a volatility must be strictly positive to be fitted
_UNIT = 1.0      # the fit is notional-invariant; unit notional keeps residuals in readable units
_BP = 1.0e4      # residual scaling: price-per-unit-notional -> basis points (see `residuals`)


@dataclass(frozen=True)
class AtmCapVolQuote:
    """One ATM cap quote: the cap's `schedule`, its ATM `strike`, and the quoted flat Black vol.

    Three fields and no notional, because a Black volatility is notional-independent — carrying one
    would be a field the fit never reads."""

    schedule: CouponSchedule
    strike: float
    black_vol: float


@dataclass(frozen=True)
class HullWhiteCalibration:
    """A successful fit. `params` is a complete `HullWhiteParams`, so the output of calibration is
    directly a model's input — no unpacking step in which a caller can pair a fitted `σn` with the
    wrong `κn`."""

    params: HullWhiteParams
    rms_residual: float
    n_quotes: int
    iterations: int


def black_cap_price(quote: AtmCapVolQuote, curve: CurveHandle) -> float:
    """The market's Black price for a quote, per unit notional — the fit's right-hand side.

    Local to this module rather than promoted to `models/black.py`, because it has exactly one
    consumer: the rule of two does not fire, and keeping it here leaves `black.py` free of any
    market dependency.

    Takes the `CurveHandle` **protocol**, not the concrete `DiscountCurve` — it needs only `df` and
    the curve's own `time_measure`, which is exactly what the protocol promises. The first cut
    annotated the concrete class and pyright caught it: `CurveSet.discount` hands back a
    `CurveHandle`, and over-constraining to the implementation is what the protocol exists to
    avoid."""
    measure = curve.time_measure
    total = 0.0
    for start, pay, accrual in quote.schedule.periods():
        expiry = measure.year_fraction(start)
        forward = (curve.df(start) / curve.df(pay) - 1.0) / accrual
        total += accrual * curve.df(pay) * black_price(
            forward, quote.strike, quote.black_vol**2 * expiry, CapFloorKind.CAP
        )
    return total


def black_cap_implied_vol(
    price: float, schedule: CouponSchedule, strike: float, curve: CurveHandle
) -> float:
    """The flat Black volatility reproducing `price` — a monotone 1-D inverse on the whole cap.

    A *flat* vol across the strip, which is what a market cap quote is: one number for the cap, not
    a caplet-by-caplet surface. Monotone in the vol, so the bracket is structural — zero below, and
    doubled upward until the price is exceeded."""
    def excess(vol: float) -> float:
        return black_cap_price(AtmCapVolQuote(schedule, strike, vol), curve) - price

    if excess(0.0) >= 0.0:
        # At zero vol the cap is already worth at least the target, so no positive vol reproduces
        # it. Refuse rather than return the degenerate bracket's endpoint (the L3/L4 constructor
        # convention of tranche 8: `PricingFailure` is the engine contract, not this boundary).
        if abs(excess(0.0)) < 1e-15:
            return 0.0
        raise ValueError(
            f"a cap price of {price} is at or below its intrinsic; no Black volatility reproduces it"
        )
    hi = 0.5
    while excess(hi) < 0.0:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError(
                f"no Black volatility below {hi} reproduces a cap price of {price}"
            )
    return bracketed_root(excess, 0.0, hi)


def _sole_discount_currency(market: MarketSnapshot) -> Currency | None:
    """The currency of the snapshot's one discount curve, or `None` if that is ambiguous.

    **Inferred rather than passed, but it REFUSES when ambiguous** — which is the difference
    between inference and guessing. An `AtmCapVolQuote` carries no currency (a Black vol is
    currency-agnostic) and the quotes are all one market by construction, so threading a fourth
    argument through would be ceremony in the single-curve case this fit is for. Two discount
    curves means the caller has a choice to make, and making it for them would be exactly the kind
    of plausible wrong answer `CurveSet.discount` refuses to give."""
    currencies = {
        key.currency for key in market.curves.curves if key.role is CurveRole.DISCOUNT
    }
    return currencies.pop() if len(currencies) == 1 else None


def _hull_white_prices(
    quotes: Sequence[AtmCapVolQuote], market: MarketSnapshot, currency, x: Sequence[float]
) -> list[float] | PricingFailure:
    """Price every quoted cap under `(κn, σn) = x`, or say why it cannot be done."""
    model = HullWhiteModel(market, HullWhiteParams(x[0], x[1]))
    prices: list[float] = []
    for quote in quotes:
        cap = NominalCap(quote.schedule, quote.strike, Money(_UNIT, currency), CapFloorKind.CAP)
        out = price_nominal_cap(cap, model)
        if not isinstance(out, PricingResult):
            return PricingFailure(f"a quoted cap cannot be priced: {out.reason}")
        prices.append(out.pv.amount)
    return prices


def calibrate_hull_white_to_atm_caps(
    quotes: Sequence[AtmCapVolQuote], market: MarketSnapshot, start: tuple[float, float]
) -> HullWhiteCalibration | PricingFailure:
    """Fit `(κn, σn)` to `quotes` by least squares on the price residuals — §5.3.1's scheme.

    Overdetermined: two parameters against a maturity strip, which is what separates them — `κn`
    bends the implied cap-vol term structure while `σn` lifts its level. A single maturity is one
    equation in two unknowns and is *not* identifiable, which the oracle asserts rather than
    assumes."""
    if not quotes:
        return PricingFailure("cannot calibrate without at least one ATM cap quote")
    if any(v <= 0.0 for v in start):
        return PricingFailure(
            f"the starting mean reversion and volatility must be positive, got {start}"
        )

    currency = _sole_discount_currency(market)
    if currency is None:
        return PricingFailure(
            "the snapshot must carry exactly one discount curve for this fit to know which "
            "currency it is calibrating; found none, or more than one"
        )

    # Price once at the start point: this validates the market before entering the solver, which is
    # what keeps a missing-curve KeyError from escaping mid-iteration as something other than a
    # value (tranche 6's precedent).
    opening = _hull_white_prices(quotes, market, currency, start)
    if isinstance(opening, PricingFailure):
        return opening

    curve = market.curves.discount(currency)
    targets = [black_cap_price(q, curve) for q in quotes]

    def residuals(x: Sequence[float]) -> list[float]:
        prices = _hull_white_prices(quotes, market, currency, x)
        if isinstance(prices, PricingFailure):
            return [math.inf] * len(targets)
        # **Scaled to basis points of notional, and that is not cosmetic.** At unit notional the
        # price residuals are O(1e-3), so scipy's RELATIVE tolerances (ftol/xtol ~1e-8) are met
        # while the parameters are still ~0.1% off: the first cut recovered kappa = 0.08007 against
        # a truth of 0.08 and stopped. Basis points are also how caps are quoted, so the solver now
        # works in the units the market states the problem in. `rms_residual` is still reported in
        # price units, so the number a caller reads does not depend on the solver's scaling.
        return [(p - t) * _BP for p, t in zip(prices, targets)]

    outcome = least_squares_fit(
        residuals, start, bounds=((_FLOOR, _FLOOR), (math.inf, math.inf))
    )
    if not outcome.converged:
        return PricingFailure(
            f"the (kappa_n, sigma_n) fit did not converge from {start} on {len(quotes)} quotes"
        )

    # `cost` is half the sum of squared residuals, and those residuals are in basis points — so
    # undo both to report an RMS in price units per unit notional.
    rms = math.sqrt(2.0 * outcome.cost / len(quotes)) / _BP
    return HullWhiteCalibration(
        params=HullWhiteParams(outcome.x[0], outcome.x[1]),
        rms_residual=rms,
        n_quotes=len(quotes),
        iterations=outcome.iterations,
    )
