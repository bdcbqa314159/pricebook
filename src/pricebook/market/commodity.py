"""Commodity spot quotes and the cost-of-carry forward (L1). `CLARK-2014` ch 2.1.2, 2.2.3, 2.2.6.

**The forward is the whole commodity content of a vanilla.** Once `F(T)` carries the cost of carry
and the convenience yield, the option price is Black-76's, and Black has been verified since
tranche 5 - so this module builds the carry and `engine/commodity_vanilla.py` reuses the rest. The
same shape `market/equity.py` + `engine/equity_vanilla.py` set at tranche 30.

**The convenience yield, and why it enters the drift but not the discounting.** Clark derives it by
explicit analogy with FX (§2.2.3): the tradeable asset is the spot times the return on holding the
commodity, `S_t B^f_t` with `dB^f = r^f B^f dt`, so `B^f` is a "foreign commodity bond" - more
precisely a continuously rolled long spot position. The resulting PDE (2.19) is

    dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r^d - r^f) S dV/dS - r^d V = 0

and the asymmetry is the point: `r^f` appears in the **convection** term and not in the **forcing**
term. A convenience yield changes what the spot drifts to; it does not change what a payoff is
discounted at. That is exactly why this module produces a forward and leaves discounting alone.

    F(0,T) = S_0 e^{(r^d - r^f)T}        (2.55)
    F(t,T) = S_t e^{(r^d - r^f)(T-t)}    (2.56)

transcribed from both renderings and reconciled (§4), and cross-checked against the **earlier
derivation step** the source prints on the same page - `F = S_0 E^d[exp(sigma W_T + (r^d - r^f -
1/2 sigma^2)T)]`, before the martingale is applied - which the oracle evaluates by Gauss-Hermite
quadrature rather than transcribing twice.

**Written against the curve, not a scalar `r`.** `1/df(T)` *is* `e^{rT}` on a flat curve and the
honest generalisation on a sloped one, so a term structure costs nothing and a single-rate signature
would have had to invent one. The convenience yield stays a scalar because a deterministic `delta`
is the spine; a seasonal or stochastic one is `to_go_deeper`.

**Not merged with `dividend_forward`, deliberately.** `S e^{-delta T}/df(T)` is arithmetically what
`market/equity.py` computes for a stock whose only convention is a continuous yield. They stay
separate types: an `EquityUnderlying` carries a cash-dividend schedule, proportional dividends and a
repo rate that a commodity does not have, so sharing it would hang carried-but-unread fields on every
commodity (§6) - and the charter's DRY rule is about knowledge, not characters: two blocks standing
for *different ideas* must not be merged, because it couples things that will diverge. These are
about to, at the arc's mean-reverting model.

Provenance:
  stock:  none - the archived predecessor's commodity work was book/P&L machinery, never a forward
  source: CLARK-2014 (docs/references.md) ch 2.1.2 (forwards, contango/backwardation), 2.2.3 (the
          convenience-yield Black-Scholes PDE 2.19) and 2.2.6 (eqs 2.55/2.56, with the
          pre-substitution expectation on the same page); transcribed in both renderings and
          reconciled
  oracle: tests/L4/test_commodity_vanilla.py - eq (2.56) transcribed POINTWISE over a (delta, T)
          grid; the source's own earlier derivation step reproduced by Gauss-Hermite quadrature (a
          different numerical path, not a second transcription); the delta=0 cost-of-carry reduction
          with a MANDATORY non-zero-delta companion measured material (T39 - the reduction is exact
          because it kills the very parameter under test); the sloped-curve companion proving the
          constructor reads the curve; contango and backwardation both reachable
  crosscheck: quantlib CommodityIndex / forward curve with a convenience yield - deferred to the
          optional QuantLib tier
  slice:  tranche 60 - the commodity forward and option; the commodity arc opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.market.curve import CurveHandle


@dataclass(frozen=True)
class CommodityAsset:
    """What the market quotes on - the key a snapshot's commodity data is held under."""

    symbol: str

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("a commodity asset needs a symbol")


@dataclass(frozen=True)
class CommodityUnderlying:
    """The quote side for one commodity: its spot and its convenience yield.

    **`convenience_yield` is carried and constitutive (§6), not speculative.** Two underlyings
    identical but for `delta` are different instruments: they imply different forwards at every
    maturity, which is the entire content of contango versus backwardation (Clark p.19). The test
    for a carried field is constitutive-of-identity rather than does-the-pricer-read-it, and this
    one passes on both counts - it is read by `commodity_forward` and seen by equality.
    """

    asset: CommodityAsset
    spot: float
    convenience_yield: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError(f"the commodity spot must be finite and positive, got {self.spot}")
        if not math.isfinite(self.convenience_yield):
            raise ValueError(
                f"the convenience yield must be finite, got {self.convenience_yield}"
            )


def commodity_forward(
    underlying: CommodityUnderlying, expiry: date, discount: CurveHandle
) -> float:
    """`F(T) = S e^{(r - delta)T}` - eq (2.56) at `t = 0`.

    Written as `S * e^{-delta T} / df(T)` because `1/df(T)` is the curve's own `e^{rT}`: the same
    object on a flat curve and the honest generalisation on a sloped one. The two factors are kept
    separate rather than collapsed into one exponent, since they come from different places - the
    discount curve and the carry parameter - and a reader should see which is which.

    A **negative** convenience yield is legal and means net storage cost, which steepens contango;
    only a non-finite one is refused, by the type.
    """
    years = discount.time_measure.year_fraction(expiry)
    return underlying.spot * math.exp(-underlying.convenience_yield * years) / discount.df(expiry)
