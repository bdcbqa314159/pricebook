"""The base-correlation curve and base-tranche pricing (L4). `OKANE-2008` ch 20.2-20.3.

The market quotes index tranches by **base correlation**: the single `ρ_B(K)` that, applied to the
**equity `[0,K]`** tranche, reprices the quote. The curve is bootstrapped shortest-strike-first, and
any `[K₁,K₂]` tranche is the **difference of two base-correlation equity tranches**:

    EL([K₁,K₂]) = EL_{ρ_B(K₂)}([0,K₂]) − EL_{ρ_B(K₁)}([0,K₁])

**L4 because it inverts the engine.** Each strike's `ρ_B` is found by repricing `[0,K]` through
tranche 43's LHP pricer until it matches the quote — a consumer of `price()`, which §2 puts at L4
beside the Jarrow-Yildirim-to-floor-strip calibrator. The import direction confirms it.

**The no-arbitrage monotonicity is a theorem, not a check this module performs.**
`EL([K₁,K₂]) > 0 ⟺ Q < 1 and decreasing ⟺ protection > 0 ⟺ par > 0` — exactly. So every positive
quote set bootstraps to a monotone curve by construction, measured down to a 1bp mezzanine quote.
Asserting "refuse a non-monotone positive quote set" would reject something that cannot exist; what
*can* happen is the other side of that iff, and it is guarded:

**a quote with no base correlation at all.** Given `ρ_B(3%) = 0.25`, the `[3%,7%]` par spread
reachable over `ρ ∈ (0.02, 0.98)` is `[−729, +1467] bp`, so a 1600bp quote has no root. That is
refused here in the bootstrap's own vocabulary — `bracketed_root`'s docstring says a non-straddling
bracket is the *caller's* bug and that the caller should have refused an out-of-range target first,
so surfacing its message would be reporting our own failure in someone else's words.

**No interpolation.** Every tranche this module prices is bounded by *quoted* strikes, so
`correlation_at` refuses an unquoted one rather than picking a convention: interpolating `ρ_B` is a
modelling choice the quotes do not pin (§6, nothing consumes it) and doing it silently would be a
plausible wrong number (§3.4). The convention gets ruled when a bespoke tranche pulls it.

Provenance:
  stock:  none — the archived predecessor had no correlation products
  source: OKANE-2008 ch 20.2 (base correlation), ch 20.3 (building the curve); the equity tranches
          are priced by the LHP (tranche 43) through models/tranche_loss.py's composition
  oracle: tests/L4/test_base_correlation.py — the known-curve round trip from a non-flat skew
          (1e-12), the quotes repriced through the bootstrapped curve, the single-strike reduction
          onto tranche 43, and the unreachable-quote refusal
  crosscheck: quantlib BaseCorrelationLossModel / the base-correlation term structure — deferred
  slice:  tranche 44b — base correlation, closing the credit arc
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pricebook.foundation.money import Currency
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.solvers import bracketed_root
from pricebook.foundation.time_measure import is_historical
from pricebook.market.credit_legs import protection_leg_from_survival, risky_pv01
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity
from pricebook.models.lhp import LHPModel
from pricebook.products.synthetic_cdo_tranche import TrancheBounds, TrancheQuote

#: The correlation range the bootstrap searches. Open at both ends because `A(K)` divides by `β`
#: and the idiosyncratic weight vanishes at 1 — tranche 43 refuses both endpoints, so the search
#: must stay inside them.
_LOWEST, _HIGHEST = 0.02, 0.98

#: Steps in the base-tranche protection leg. The LHP's own committed count, so a base-correlation
#: tranche and a single-correlation one are integrated on one grid.
_PROTECTION_STEPS = 800


@dataclass(frozen=True)
class HomogeneousPool:
    """Which pool a base-correlation curve is on, and the curves it is priced against.

    Bundled because §7's limit compels it: the bootstrap needs the snapshot, the pool's entity, the
    recovery *and* the currency whose discount curve the legs use, which is five arguments before
    the quotes and the schedule. It is also the cohesive thing — everything here answers "which
    pool, on what curves", and the alternative is four loose arguments in a fixed order that a
    caller can silently transpose.

    It is `LHPModel` minus the correlation, which is exactly right: the correlation is what the
    bootstrap solves for, so it cannot be part of the pool's identity.
    """

    market: MarketSnapshot
    entity: ReferenceEntity
    recovery: float
    currency: Currency

    def model_with(self, correlation: float) -> LHPModel:
        """The LHP model for this pool at a trial correlation."""
        return LHPModel(self.market, self.entity, correlation, self.recovery)


@dataclass(frozen=True)
class BaseCorrelationCurve:
    """`ρ_B` at each quoted strike, with the pool it was bootstrapped against."""

    pool: HomogeneousPool
    strikes: tuple[float, ...]
    correlations: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.strikes) != len(self.correlations):
            raise ValueError("every strike needs a correlation and vice versa")
        if not self.strikes:
            raise ValueError("a base-correlation curve needs at least one strike")

    def correlation_at(self, strike: float) -> float:
        """`ρ_B(K)` at a **quoted** strike. Refuses anything else, deliberately."""
        for quoted, correlation in zip(self.strikes, self.correlations):
            if quoted == strike:
                return correlation
        raise KeyError(
            f"{strike} is not a quoted strike (quoted: {list(self.strikes)}). Interpolating "
            f"between strikes is a modelling choice the quotes do not pin, and nothing here needs "
            f"it — a bespoke tranche would, and its convention is a ruling rather than a default")

    def model_at(self, strike: float) -> LHPModel:
        """The LHP model for the equity tranche `[0, strike]`."""
        return self.pool.model_with(self.correlation_at(strike))


def base_tranche_expected_loss(
    curve: BaseCorrelationCurve, attachment: float, detachment: float, horizon: float
) -> float:
    """`EL_{ρ_B(K₂)}([0,K₂]) − EL_{ρ_B(K₁)}([0,K₁])` — the base-tranche difference.

    Two **different** correlations, which is why this is not the single-model composition in
    `models/tranche_loss.py`: that one caps a single model's loss twice, this one caps two models
    once each. The difference is the content of base correlation, not an implementation detail.
    """
    upper = curve.model_at(detachment).expected_loss_capped(detachment, horizon)
    if attachment <= 0.0:
        return upper
    return upper - curve.model_at(attachment).expected_loss_capped(attachment, horizon)


@dataclass(frozen=True)
class _BaseTrancheSurvival:
    """`Q(t)` for a base-correlation tranche — `credit_legs.SurvivalFunction` shaped."""

    curve: BaseCorrelationCurve
    bounds: TrancheBounds

    def survival(self, t: float) -> float:
        return 1.0 - base_tranche_expected_loss(
            self.curve, self.bounds.attachment, self.bounds.detachment, t
        ) / self.bounds.width


def base_tranche_par_spread(
    curve: BaseCorrelationCurve, bounds: TrancheBounds, schedule: CouponSchedule
) -> float:
    """The par spread of `[K₁,K₂]` through the curve — a CDS on the base-tranche survival.

    The legs are tranche 39's and tranche 42's, unchanged: the survival object above satisfies
    `credit_legs.SurvivalFunction`, so a base-correlation tranche reaches the same arithmetic as
    every other credit product in the library. `recovery = 0` on the protection leg because the
    `(1−R)` is already inside the expected loss (tranche 43's reasoning, unchanged).
    """
    market = curve.pool.market
    discount = market.curves.discount(curve.pool.currency)
    live = [
        period for period in schedule.periods()
        if not is_historical(period[1], market.valuation_date)
    ]
    if not live:
        raise ValueError(
            "a par spread needs a premium period still to pay; this schedule has matured")
    survival = _BaseTrancheSurvival(curve, bounds)
    annuity = risky_pv01(live, discount, survival)
    if annuity <= 0.0:
        raise ValueError("a par spread needs a positive risky annuity")
    horizon = discount.time_measure.year_fraction(live[-1][1])
    return protection_leg_from_survival(
        discount, survival, 0.0, horizon, _PROTECTION_STEPS) / annuity


def bootstrap_base_correlation(
    quotes: Sequence[TrancheQuote], pool: HomogeneousPool, schedule: CouponSchedule
) -> BaseCorrelationCurve:
    """Solve `ρ_B(K)` strike by strike, most-subordinated first — ch 20.3.

    Sequential and order-dependent by construction: each strike's correlation is found with the
    strikes below it already fixed, because the `[K₁,K₂]` quote prices as a *difference* of equity
    tranches and the lower one must already be known.

    **There is no separate strike-order check, because it would be unreachable.** A first draft had
    one; contiguity forces `attachment == previous_detachment`, and `TrancheBounds` forces
    `detachment > attachment`, so the detachments are increasing *already* and
    `detachment <= strikes[-1]` cannot hold. Two guards implying a third is a guard that can never
    fire — dead code by the T15 rule — so the contiguity check carries it and the suite asserts
    that it does.

    An **unreachable** quote is refused here, in this function's vocabulary, rather than left to
    `bracketed_root`: its contract says a non-straddling bracket is the caller's bug, so reporting
    its message would describe our failure in the root-finder's words.
    """
    if not quotes:
        raise ValueError("a base-correlation bootstrap needs at least one tranche quote")
    if quotes[0].bounds.attachment != 0.0:
        raise ValueError(
            f"the first quote must be an equity tranche attaching at 0, got "
            f"{quotes[0].bounds.attachment}: the bootstrap starts from `[0,K]` because there is "
            f"nothing below it to subtract")

    strikes: list[float] = []
    correlations: list[float] = []
    previous_detachment = 0.0
    for quote in quotes:
        bounds = quote.bounds
        if bounds.attachment != previous_detachment:
            raise ValueError(
                f"the quoted tranches must form a contiguous capital structure: "
                f"[{bounds.attachment}, {bounds.detachment}] does not attach where the previous "
                f"tranche detached ({previous_detachment}). A gap has no base correlation to "
                f"subtract, and treating it as absorbed loss would be silent and wrong")
        def spread_at(correlation: float, bounds: TrancheBounds = bounds) -> float:
            trial = BaseCorrelationCurve(
                pool, (*strikes, bounds.detachment), (*correlations, correlation))
            return base_tranche_par_spread(trial, bounds, schedule)

        widest, tightest = spread_at(_LOWEST), spread_at(_HIGHEST)
        low, high = min(widest, tightest), max(widest, tightest)
        if not low <= quote.spread <= high:
            raise ValueError(
                f"no base correlation reprices the [{bounds.attachment:.2%}, "
                f"{bounds.detachment:.2%}] quote of {quote.spread:.4%}: over the whole admissible "
                f"range ρ ∈ ({_LOWEST}, {_HIGHEST}) this tranche spans "
                f"[{low:.4%}, {high:.4%}]. The quote is inconsistent with the strikes below it — "
                f"there is no curve, not merely a hard root-find")

        correlations.append(
            bracketed_root(lambda rho: spread_at(rho) - quote.spread, _LOWEST, _HIGHEST))
        strikes.append(bounds.detachment)
        previous_detachment = bounds.detachment

    return BaseCorrelationCurve(pool, tuple(strikes), tuple(correlations))
