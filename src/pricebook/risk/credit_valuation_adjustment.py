"""Unilateral CVA and the expected-positive-exposure profile (L5) — the XVA spine's first unit.

**The adjustment, from Crepey's definition to the thing we integrate.** Crepey Prop 1 eq (17)
defines CVA as an expectation over the counterparty's default, `CVA = E*[(1-J)P+]` — the positive
part of the portfolio value at default, taken on the default event. Under the elementary
assumption this tranche makes, **exposure independent of default**, that expectation factorises
into the integral every desk writes:

    CVA = LGD * integral over [0,T] of DF(t) * EPE(t) * dPD(t),   dPD(t) = lambda(t) S(t) dt

so the three inputs are a discount curve, an exposure profile, and a survival curve. Independence
is the simplification, and it is the one worth naming: wrong-way risk (exposure correlated with
the counterparty's credit) is precisely what it discards, and recovering it needs a joint model
rather than a product of marginals.

**The functional takes an EPE profile, not an exposure model — deliberately.** `EPE(t) = E[V(t)+]`
arrives as numbers on a grid, so the integral neither knows nor cares whether they came from a
closed form or a simulation. That is not a speculative hook: `ExposureProfile.gaussian` is this
tranche's own consumer and the Monte-Carlo exposure engine (tranche 26) is the second, which is
rule-of-two with both consumers *present*, not hoped for. It also keeps the Gaussian assumption
out of the adjustment, where it does not belong.

**Trapezoid, not quadrature.** The integrand is sampled, not callable — a profile is data — so an
adaptive rule has nothing to subdivide. The grid is the caller's resolution choice and the oracle
asserts convergence under refinement rather than a fixed error.

Provenance:
  stock:  none - the archived predecessor's `risk/xva.py` was a phantom: its register rows named a
          CVA module that never existed (pruned by tranche 24)
  source: CREPEY-2026 (docs/references.md) ch I Prop 1 eq (17) and §3 - the CVA definition and its
          reduction under exposure/default independence; CAPINSKI-ZASTAWNIAK ch 2 for the survival
          side. The Bachelier positive-part expectation `E[(m+sZ)+] = m*Phi(m/s) + s*phi(m/s)` is
          standard and derived rather than transcribed (§4: a derived reference beats a copied one)
  oracle: tests/L5/test_credit_valuation_adjustment.py - the analytic EPE against the test's own
          numerical integral; the CVA against scipy quadrature of an independently built
          integrand; Crepey's two separate zeros (LGD = 0, lambda = 0), sign, monotonicity in both
          lambda and LGD, linearity in LGD, and the sandwich; the one-period step pinned
          bit-identically against a hand computation
  crosscheck: quantlib CreditDefaultSwap / CVA on a netting set - deferred to the optional
          QuantLib tier
  slice:  tranche 25 - unilateral CVA: the XVA spine opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.curve import CurveHandle
from pricebook.market.survival_curve import SurvivalCurve

_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


def _standard_normal_pdf(x: float) -> float:
    return _INV_SQRT_TWO_PI * math.exp(-0.5 * x * x)


def _standard_normal_cdf(x: float) -> float:
    # `erfc` rather than `(1+erf)/2`: tranche 20c measured the latter losing all relative accuracy
    # in the lower tail (exactly 0.0 below x ~ -8.3), and a deep-out-of-the-money exposure is
    # exactly where this is called.
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def expected_positive_exposure(mean: float, stdev: float) -> float:
    """`E[(mean + stdev*Z)+]` for a standard normal `Z` — the Bachelier positive part.

    `= mean*Phi(mean/stdev) + stdev*phi(mean/stdev)`. The two terms are the two ways a positive
    payoff arises: the forward itself, weighted by the probability of ending positive, plus the
    value of the optionality around it. At `stdev = 0` the exposure is deterministic and the
    formula is the plain positive part, which is also its limit.
    """
    if stdev < 0.0:
        raise ValueError("an exposure standard deviation may not be negative")
    if stdev == 0.0:
        return max(mean, 0.0)
    ratio = mean / stdev
    return mean * _standard_normal_cdf(ratio) + stdev * _standard_normal_pdf(ratio)


@dataclass(frozen=True)
class ExposureProfile:
    """Expected positive exposure sampled on a time grid: `epe[i] = E[V(times[i])+]`.

    Year fractions on the curves' own time measure, ascending. Two fields and no model: whatever
    produced the numbers — a closed form here, a simulation in tranche 26 — the adjustment reads
    the same shape.
    """

    times: tuple[float, ...]
    epe: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.epe):
            raise ValueError("times and epe must have equal length")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("times must be strictly ascending")
        if any(e < 0.0 for e in self.epe):
            raise ValueError("an expected POSITIVE exposure may not be negative")

    @classmethod
    def gaussian(
        cls,
        times: tuple[float, ...],
        means: tuple[float, ...],
        stdevs: tuple[float, ...],
    ) -> ExposureProfile:
        """The profile of a Gaussian mark `V(t) ~ N(means[i], stdevs[i]^2)`."""
        if not len(times) == len(means) == len(stdevs):
            raise ValueError("times, means and stdevs must have equal length")
        return cls(times, tuple(expected_positive_exposure(m, s) for m, s in zip(means, stdevs)))


def unilateral_cva_one_period(
    exposure: float,
    discount_factor: float,
    default_probability: float,
    loss_given_default: float,
) -> float:
    """CVA when default can only happen in one period: `LGD * DF * EPE * PD`.

    The integral's single term, as named surface rather than a step buried in a loop. §4: a bound
    on the integral cannot see the shape of the integrand, so the term a test can check by hand is
    extracted and checked by hand.
    """
    return loss_given_default * discount_factor * exposure * default_probability


def credit_adjustment(
    exposure: ExposureProfile,
    defaulting: SurvivalCurve,
    surviving: SurvivalCurve | None,
    discount: CurveHandle,
    loss_given_default: float,
) -> float:
    """`LGD * integral DF(t) exposure(t) S_surviving(t) lambda_def(t) S_def(t) dt`.

    **The one integral both credit legs are.** CVA and DVA differ only in which *side* of the
    exposure they weight and *whose* default triggers the loss - the shape is identical, so this is
    where it lives. Rule-of-two: tranche 25 was the first consumer and DVA is the second, which is
    when the charter says the generalisation gets decided (§6), not before.

    * **CVA** - `exposure` is the EPE, `defaulting` the counterparty, `surviving` us.
    * **DVA** - `exposure` is the ENE, `defaulting` us, `surviving` the counterparty.
    * **The unilateral case (tranche 25)** - `surviving=None`: nobody else can default, so there is
      no first-to-default conditioning to apply. `None` rather than a flat `S = 1` curve, because a
      multiplication that is *skipped* is cheaper to prove unchanged than one performed against a
      1.0: the operation order is then literally T25's, so the extraction is **bit-identical**
      rather than identical-to-a-few-ULP - §4's stronger proof, taken where it is available.
    """
    if not 0.0 <= loss_given_default <= 1.0:
        raise ValueError(f"loss given default must lie in [0, 1], got {loss_given_default}")
    if len(exposure.times) < 2:
        raise ValueError("integrating an exposure profile needs at least two grid points")
    for curve in (defaulting, surviving):
        if curve is not None and curve.time_measure != discount.time_measure:
            # The profile's `t` is read on EVERY curve here. Different anchors or day counts mean
            # the same `t` denotes different dates and the integral silently mixes them - C12, the
            # C7/C8 shared-clock family, guarded rather than recorded and hoped for (§7).
            raise ValueError(
                "the survival and discount curves must share one time measure: "
                f"{curve.time_measure} vs {discount.time_measure}"
            )

    integrand = []
    for t, side in zip(exposure.times, exposure.epe):
        term = discount.df_at(t) * side * defaulting.default_density(t)
        if surviving is not None:
            term *= surviving.survival(t)
        integrand.append(term)
    area = 0.0
    for i in range(len(integrand) - 1):
        area += 0.5 * (integrand[i] + integrand[i + 1]) * (exposure.times[i + 1] - exposure.times[i])
    return loss_given_default * area


def unilateral_cva(
    profile: ExposureProfile,
    survival: SurvivalCurve,
    discount: CurveHandle,
    loss_given_default: float,
) -> float:
    """`LGD * integral DF(t) EPE(t) lambda(t) S(t) dt` - tranche 25's adjustment, now a thin caller.

    Kept under its own name because "unilateral" names the *assumption* - that we cannot default -
    which a bare `credit_adjustment(..., None, ...)` at a call site would state far less clearly.
    """
    return credit_adjustment(profile, survival, None, discount, loss_given_default)


@dataclass(frozen=True)
class CreditParties:
    """Both sides' credit: whose curve and whose loss-given-default.

    Four cohesive fields rather than four loose arguments (§7's compelled bundling). It also removes
    the commonest bilateral error at a call site, which is passing the two curves - or the two LGDs -
    in the wrong order, since both orderings type-check and both produce a plausible number.
    """

    own: SurvivalCurve
    counterparty: SurvivalCurve
    own_lgd: float
    counterparty_lgd: float

    def __post_init__(self) -> None:
        for name, lgd in (("own_lgd", self.own_lgd), ("counterparty_lgd", self.counterparty_lgd)):
            if not 0.0 <= lgd <= 1.0:
                raise ValueError(f"{name}: loss given default must lie in [0, 1], got {lgd}")
        if self.own.time_measure != self.counterparty.time_measure:
            raise ValueError(
                "the two parties' survival curves must share one time measure: "
                f"{self.own.time_measure} vs {self.counterparty.time_measure}"
            )

    def swapped(self) -> CreditParties:
        """The same pair seen from the counterparty's side - our DVA is their CVA."""
        return CreditParties(self.counterparty, self.own, self.counterparty_lgd, self.own_lgd)


@dataclass(frozen=True)
class BilateralAdjustment:
    """Both legs, and the net - and both are reported, never only the net.

    `BCVA` alone hides which side it came from: the same small net can be two large offsetting legs
    or two negligible ones, and those are different risks. The net is a **property**, not a field, so
    it cannot drift from its parts.
    """

    cva: float
    dva: float

    @property
    def bcva(self) -> float:
        """`CVA - DVA` - Crepey Prop 1 eq (18), third line: `FV = CVA - DVA = CA - CL`."""
        return self.cva - self.dva


def bilateral_credit_adjustment(
    epe: ExposureProfile,
    ene: ExposureProfile,
    parties: CreditParties,
    discount: CurveHandle,
) -> BilateralAdjustment:
    """CVA and DVA, each conditioned on the *other* party still being alive.

    **The first-to-default conditioning is the whole difference from tranche 25**, and it has a
    direction: we cannot lose on the counterparty's default if we defaulted first, so the `S_own`
    factor in the CVA integrand can only *reduce* it. The unilateral spine dropped that term; here
    it is present, and the oracle asserts its **magnitude**, not only its sign - a term checked for
    presence and sign alone is §4's documented blind spot.

    **One convention, stated because the source differs.** Crepey conditions through the *bank
    survival measure* (`EY = (1-gamma)^-1 E*(Y')`, eq 7 p.37), so his conditioning carries a
    `(1-gamma)^-1` **normalisation**. This is the market-standard explicit-factor form - the natural
    extension of T25/T26, neither of which carries a normalisation. Both forms have the
    first-to-default term present and both satisfy `BCVA = CVA - DVA`; they differ by that factor.
    Raised as an `Ask:` for Cowork rather than ruled here.
    """
    if epe.times != ene.times:
        raise ValueError(
            "the positive and negative exposure must share the same time grid: "
            f"{len(epe.times)} vs {len(ene.times)} points"
        )
    return BilateralAdjustment(
        cva=credit_adjustment(
            epe, parties.counterparty, parties.own, discount, parties.counterparty_lgd
        ),
        dva=credit_adjustment(
            ene, parties.own, parties.counterparty, discount, parties.own_lgd
        ),
    )
