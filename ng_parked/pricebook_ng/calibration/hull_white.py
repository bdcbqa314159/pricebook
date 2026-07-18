"""Hull-White vol calibration — the unified calibration front's first model fit (L3).

`market -> calibrate -> model -> price` (Amendment A1): calibration turns a
`MarketSnapshot` + market vol quotes into a `CalibratedModel`. It lives at L3 and
depends only on L0/L1 + the model's own analytic building blocks — never the L4
engine (the spine: L3 calibration depends on L0 + L1). So the fit reprices with a
*model* closed form, exactly as the curve bootstraps reprice with the curve's own
`df`, not an engine.

The textbook HW vol instrument is a caplet — a European option on a zero-coupon
bond (Brigo & Mercurio s.3.3), which `HullWhite.zero_bond_option` prices in closed
form. Mean reversion `a` is taken as given (standard practice: `a` from history or
co-terminal structure, `sigma` to the vol quote); a joint `(a, sigma)` fit waits
for a second quote that makes the pair identifiable (rule of two).

This module establishes the front's shape — `calibrate_*(snapshot, quotes, …) ->
CalibratedModel` over a per-family solver. The curve bootstraps
(`bootstrap_discount_curve`, `bootstrap_survival_curve`) are the sibling rate/credit
solvers that migrate under this front as they gain a second consumer.

Provenance:
  quarry: python/pricebook/calibration/ (scattered per-family fits -> unified L3 front)
  source: Brigo & Mercurio, Interest Rate Models s.3.3 (HW cap/caplet calibration)
  oracle: round-trip sigma recovery + calibrated model reprices the quote < 1e-10
  slice:  calibration-front (HW to a ZCB-option / caplet)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.solvers import bisect_root, nelder_mead
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite

_SIGMA_MAX = 1.0  # rates HW vol is ~1e-3..5e-2; 1.0 brackets any real quote


@dataclass(frozen=True)
class ZCBOptionQuote:
    """A caplet-style HW vol instrument: the market price of a European option (at
    `expiry`) on the zero-coupon bond P(expiry, bond_maturity), struck at `strike`.
    `is_call` selects call (ZBC) vs put (ZBP)."""

    expiry: date
    bond_maturity: date
    strike: float
    is_call: bool
    price: float


def calibrate_hull_white(snapshot: MarketSnapshot, quote: ZCBOptionQuote, a: float) -> HullWhite:
    """Fit HW `sigma` (with `a` fixed) so the model reprices `quote`, returning the
    calibrated model bound to `snapshot`. The ZCB-option price is monotone increasing
    in `sigma` (more vol -> more time value), so one bracketed root pins it down.

    Raises `ValueError` (via the solver) if the quote is unreachable — a price below
    the deterministic intrinsic (no positive vol can match it)."""

    def mispricing(sigma: float) -> float:
        model = HullWhite(a=a, sigma=sigma, market=snapshot)
        priced = model.zero_bond_option(
            quote.expiry, quote.bond_maturity, quote.strike, quote.is_call
        )
        return priced - quote.price

    sigma = bisect_root(mispricing, 1e-10, _SIGMA_MAX)
    return HullWhite(a=a, sigma=sigma, market=snapshot)


def calibrate_hull_white_cap(
    snapshot: MarketSnapshot, quotes: list[ZCBOptionQuote]
) -> HullWhite:
    """Joint `(a, sigma)` fit to a cap strip by least squares. A cap is a strip of
    caplets (each a ZCB option); one caplet cannot separate mean reversion from vol,
    so a strip spanning expiries is needed (rule of two: 2+ quotes). Minimises the
    sum of squared repricing errors with Nelder-Mead — derivative-free, no engine.

    `sigma` is fitted through its magnitude (a sign has no meaning: the price depends
    on sigma only via sigma^2 and |sigma|), so the returned model carries `sigma >= 0`."""
    if len(quotes) < 2:
        raise ValueError("a joint (a, sigma) fit needs at least two cap-strip quotes")

    def sse(params: list[float]) -> float:
        a, sigma = params[0], abs(params[1])
        model = HullWhite(a=a, sigma=sigma, market=snapshot)
        return sum(
            (
                model.zero_bond_option(q.expiry, q.bond_maturity, q.strike, q.is_call)
                - q.price
            )
            ** 2
            for q in quotes
        )

    a, sigma = nelder_mead(sse, [0.03, 0.01])
    return HullWhite(a=a, sigma=abs(sigma), market=snapshot)
