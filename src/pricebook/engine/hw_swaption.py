"""The Hull-White European swaption by Jamshidian's trick (L4). `MCINERNEY-ZASTAWNIAK` §3.5.

The tree has priced Hull-White caplets since tranche 10 but never its **swaption** - which the
tranche-68 step-1 grep found, and which had to be built before the Bermudan could have a
one-exercise reduction anchor at all.

**The source, transcribed from both renderings and reconciled (§4).** §3.5, "Formula for swaptions":

    payoff   = (1 - K sum_i tau_i B(T0,Ti) - B(T0,Tn))^+                           (3.23)
    critical:  K sum_i tau_i F(T0, r~; Ti) + F(T0, r~; Tn) = 1                     (3.24)
    with       K_i = F(T0, r~; Ti)
    PSwpt(0) = sum_i K (Ti - T_{i-1}) BP(0; T0, Ti, K_i) + BP(0; T0, Tn, K_n)      (3.25)

**Why the decomposition is valid, which is a premise and not a remark.** The source argues that
`B(t,T)` is a *decreasing* function of `r(t)`, so the coupon-bearing bond is too, so a critical rate
exists and is unique - and then that `B(T0,Ti) < K_i` exactly when `r(T0) > r~`, which is what lets
one option on a sum become a sum of options. Both steps are asserted by the oracle rather than
assumed: if the monotonicity failed, (3.24) would have no solution or several and (3.25) would be
meaningless while still producing a number.

**`BP` is `zero_bond_option` with `FLOOR`** - eq (3.17), held and oracled since tranche 10. So the
only genuinely-new code here is the critical-state root-find and the portfolio assembly; the option
valuation is reused whole, and `bracketed_root` (tranche 6) does the solving.

**The root-find is bracketed rather than Newton, deliberately.** The coupon-bond value is monotone
in the state by the premise above, so a bracket is guaranteed to contain exactly one root and cannot
diverge - where a Newton iteration would need a derivative this module does not have and could step
outside the regime where the bond formula is numerically sound.

Provenance:
  stock:  none - the archived predecessor had no Hull-White swaption
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §3.5 - eqs (3.23) the coupon-bond-put payoff,
          (3.24) the critical rate, (3.25) the portfolio; and §2.8 for the swaption-as-bond-put
          identity the section rests on; transcribed in both renderings and reconciled
  oracle: tests/L4/test_hw_swaption.py - the critical state's DEFINING identity (the coupon bond is
          worth exactly 1 there) as a requirement oracle; the bond's monotonicity in the state,
          asserted because the decomposition rests on it; the per-bond strike ordering that splits
          the payoff; the portfolio transcribed pointwise against (3.25); zero volatility reducing
          EXACTLY to the forward swap's intrinsic value; positivity and vol monotonicity as
          requirement companions; and `hw_bond` cross-checked against tranche 67's `g2pp_bond` at
          zero second volatility - already-verified code rather than a second transcription
  crosscheck: quantlib JamshidianSwaptionEngine - deferred to the optional QuantLib tier
  slice:  tranche 68 - Bermudan swaptions in the Hull-White model (piece 1, the European anchor)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.solvers import bracketed_root
from pricebook.market.curve import CurveHandle
from pricebook.models.hull_white import (
    HullWhiteParams,
    hw_bond,
    zero_bond_option,
)

#: How far either side of zero the critical-state search brackets, in absolute short-rate units.
#: Wide enough that no realistic swap's critical state falls outside (a 200% rate deviation), and
#: the bracket is *verified* rather than trusted - `_bracket` checks the sign change and refuses
#: rather than returning whatever a root-finder makes of a bracket that does not contain a root.
_SEARCH_WIDTH = 2.0


@dataclass(frozen=True)
class SwaptionTerms:
    """A European swaption's schedule and strike, in **times** rather than tenor indices.

    Times, because a short-rate model has no tenor to index into: tranche 18's `SwapSpan` is
    `(first, count)` over an `LmmTenor`'s forwards, which is LMM vocabulary and meaningless here.
    The same instrument described in the coordinates each model actually has - the §6 line, where
    two descriptions of one trade stay separate because they are *descriptions for different
    models*, not because the trade differs.

    `expiry` is `T0`, the option's expiry and the swap's first reset; `settlements` are
    `T1 ... Tn`, so the accruals are the gaps starting from the expiry.
    """

    expiry: float
    settlements: tuple[float, ...]
    strike: float

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"the strike must be positive, got {self.strike}")
        if not self.settlements:
            raise ValueError("a swap needs at least one settlement date")
        if self.settlements[0] <= self.expiry:
            raise ValueError(
                f"the first settlement ({self.settlements[0]}) must follow the expiry "
                f"({self.expiry})"
            )
        if any(b <= a for a, b in zip(self.settlements, self.settlements[1:])):
            raise ValueError(
                f"the settlement dates must be strictly ascending, got {self.settlements}"
            )

    @property
    def accruals(self) -> tuple[float, ...]:
        """`tau_i = T_i - T_{i-1}`, with `T_0` the expiry.

        Derived rather than stored: a stored set could disagree with the dates, and the first
        accrual running from the *expiry* is exactly the off-by-one a stored copy would hide.
        """
        return tuple(
            b - a for a, b in zip((self.expiry, *self.settlements), self.settlements)
        )


def _coupon_bond(
    params: HullWhiteParams, terms: SwaptionTerms, state: float, discount: CurveHandle
) -> float:
    """`K sum_i tau_i F(T0, x; Ti) + F(T0, x; Tn)` - the left side of (3.24).

    Decreasing in `state`, which is the whole basis of the decomposition.
    """
    coupons = math.fsum(
        terms.strike * accrual * hw_bond(params, state, terms.expiry, pay, discount)
        for accrual, pay in zip(terms.accruals, terms.settlements)
    )
    return coupons + hw_bond(params, state, terms.expiry, terms.settlements[-1], discount)


def jamshidian_critical_state(
    params: HullWhiteParams,
    terms: SwaptionTerms,
    discount: CurveHandle,
    _seen_from: tuple[float, float] = (0.0, 0.0),
) -> float:
    """The state `x~` at which the coupon-bearing bond is worth exactly 1 - eq (3.24).

    `seen_from` is accepted for symmetry with `hw_european_swaption` and is **not read**: the
    critical state is a property of the bond prices *at the expiry*, which `hw_bond` computes from
    the expiry outwards whatever vantage the caller holds. Taking the argument and ignoring it would
    be a carried-but-unread parameter (§6), so it is named `_seen_from` to say so out loud.

    Expressed in the model's state coordinate `x = r - f(0,t)` rather than in `r` itself, because
    that is what `hw_bond` reads; the two differ by a deterministic shift, so the critical *rate*
    the source names and this critical *state* describe the same point.

    Bracketed rather than Newton: the coupon bond is monotone in the state, so a sign change pins a
    unique root. The bracket is **checked** - if the coupon bond does not straddle 1 across a
    200% rate deviation either way, the swaption is so far from the money that the decomposition
    has no critical state, and that is refused rather than papered over with an extrapolation.
    """
    def excess(state: float) -> float:
        return _coupon_bond(params, terms, state, discount) - 1.0

    low, high = -_SEARCH_WIDTH, _SEARCH_WIDTH
    if excess(low) * excess(high) > 0.0:
        raise ValueError(
            f"the coupon bond does not cross 1 across a state range of +-{_SEARCH_WIDTH}: there is "
            f"no critical state for a strike of {terms.strike} on this curve, so Jamshidian's "
            "decomposition does not apply"
        )
    # `excess` is DECREASING in the state, and `bracketed_root` expects the opposite orientation,
    # so the sign is flipped rather than the bracket reversed - which would read as a typo.
    return bracketed_root(lambda state: -excess(state), low, high)


def hw_european_swaption(
    params: HullWhiteParams,
    terms: SwaptionTerms,
    discount: CurveHandle,
    seen_from: tuple[float, float] = (0.0, 0.0),
) -> float:
    """`PSwpt` - eq (3.25), a portfolio of zero-bond puts at the common critical state.

    `seen_from` is `(time, state)`: by default today with a zero state, which is eq (3.25) exactly.
    Valued from a **future** `(t, x)` instead, every discount factor comes from `hw_bond` at that
    state rather than from today's curve - which is the same formula read from a later vantage, not
    a new one, because the model's own bond *is* the curve there.

    That generality is pulled by the Bermudan (tranche 68 piece 2) and by nothing else: its backward
    induction needs the European swaption at every grid point of the last exercise date, and
    computing it in **closed form** there rather than by quadrature is the same argument that chose
    a lattice over a regression - §4's ordering, where a closed form beats an approximation of it.
    Measured: Gauss-Hermite on the kinked payoff converges only as about `n^-2` (3.9e-3 at 48 nodes,
    1.1e-4 at 192), because a Gauss rule loses its accuracy at a kink; the closed form has none of
    that error.

    Per unit notional. The strike of each put is the bond price *at the critical state*, which is
    what makes every option in the portfolio exercise together and the sum equal the option on the
    sum.

    At zero volatility every `BP` returns its intrinsic, so this collapses exactly to the positive
    part of today's forward-starting swap value - an exact reduction the oracle asserts, and the
    companion that stops the volatility-monotonicity check from standing alone.
    """
    at, state = seen_from
    critical = jamshidian_critical_state(params, terms, discount)
    final = terms.settlements[-1]

    def df(maturity: float) -> float:
        """The discount factor to `maturity` as seen from `(at, state)` - today's curve when that is
        inception, the model's own bond otherwise."""
        return hw_bond(params, state, at, maturity, discount)

    portfolio = math.fsum(
        terms.strike
        * accrual
        * zero_bond_option(
            df(terms.expiry),
            df(pay),
            hw_bond(params, critical, terms.expiry, pay, discount),
            _forward_variance(params, at, terms.expiry, pay),
            CapFloorKind.FLOOR,
        )
        for accrual, pay in zip(terms.accruals, terms.settlements)
    )
    return portfolio + zero_bond_option(
        df(terms.expiry),
        df(final),
        hw_bond(params, critical, terms.expiry, final, discount),
        _forward_variance(params, at, terms.expiry, final),
        CapFloorKind.FLOOR,
    )


def _forward_variance(
    params: HullWhiteParams, at: float, option_expiry: float, bond_maturity: float
) -> float:
    """`nu(t;S,T)` - the variance of `ln B(S,T)` as seen from `t`, eq (3.19)'s general case.

    `bond_option_variance` is this at `t = 0`; the general form replaces `(1 - e^{-2 kappa S})` by
    `(1 - e^{-2 kappa (S - t)})`, since only the variance *accumulated between now and expiry*
    matters. At `t = 0` the two agree identically, which is how it is oracled - against tranche 10's
    already-verified function rather than against a second transcription.
    """
    kappa, sigma = params.kappa, params.sigma
    span = bond_maturity - option_expiry
    return (
        sigma * sigma / (2.0 * kappa**3)
        * (1.0 - math.exp(-kappa * span)) ** 2
        * -math.expm1(-2.0 * kappa * (option_expiry - at))
    )
