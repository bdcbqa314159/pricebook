"""The OIS / multi-curve bootstrap (L1) — tranche 54, PRE-REGISTERED. `LICHTERS-2015` ch 3.5.

The tree discounts every trade on **one curve per currency** — the pre-crisis world. A
*collateralised* trade discounts at the **collateral (OIS) rate** while each floating leg projects
its forwards off a **tenor-specific projection curve**, so a desk prices linear instruments off a
*set*: an OIS discount curve, and one projection curve per tenor calibrated **against** it.

**Half of this tranche already existed, which is the first thing the build established.** An OIS
par swap discounts its own cashflows, so its float leg telescopes and the bootstrap equation is
BERHAN 5.1.2 — exactly `nominal_curve_from_par_swaps` (T8). The OIS curve is therefore that
function, reused verbatim, not a reimplementation. The genuinely new content is the **projection**
bootstrap, where the discounting curve is a *different, already-known* curve and nothing telescopes.

**The source, read in both renderings and reconciled (§4).** Lichters §3.5 gives the stripping
relation as

    c · A^ON_{0,n} = Σ_{i=1..n} f^δ(t_{i−1}, t_i) · δ_i · df^ON_i

— *"where `A^ON` is the fixed side's annuity based on OIS discount factors which are all known"* —
with the projection curve's own recursion `df^δ_{s+δ} = df^δ_s / (1 + δ f^δ(s, s+δ))` and the
forward definition (1.1) `f(t,t+δ) = (df_t/df_{t+δ} − 1)/δ`. The `-layout` rendering splits (1.1)'s
fraction across three lines and the `-raw` splits it across six; they reconcile, and the
pre-substitution form at (1.1) is the independent check on §3.5's use of it.

**The discriminator between the two eras is a telescoping identity, and it is the companion the
reduction needs (§4/T39).** Eq (1.2) says `f(t,t+δ)·df_{t+δ}·δ = df_t − df_{t+δ}`, so a float leg
sums to `1 − df_n` **only when one curve both projects and discounts**. Measured on this fixture:
single-curve agrees exactly (0.0), multi-curve differs by **1.2e-03**. That is a structural
signature, not a tolerance — which is what makes the reduction non-vacuous.

**On the reduction's precision, stated rather than assumed.** The spec asked for the single-curve
case to reproduce T8 *"to the ULP"*. It does — worst **1 ULP** across the pillars — and that is the
honest claim rather than bit-identity: the two routes are the same quantity in ℝ (T8's rearranged
closed form against the projection recursion) computed in a **different operation order**, so §4's
op-order condition fails and bit-identity is unavailable. The committed assertion is therefore a
*relative* tolerance, with the measured ULP distance recorded here (the T49 lesson: bit-identity is
a property of a measurement, not of a committed cross-platform test).

**Premises (§4), because the degenerate case here is the one the tree already holds.** A **non-flat**
OIS curve, a **real basis** (OIS ≠ projection, so multi-curve-ness is visible at all — the
single-curve case is the degenerate fixture), **≥ 2 tenors** so a per-tenor bug shows, and the
projection bootstrap reading the OIS curve rather than a copy of itself.
"""

import math
import struct
from datetime import date

import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.multi_curve import projection_curve_from_par_swaps
from pricebook.market.nominal_bootstrap import (
    ParSwapQuote,
    nominal_curve_from_par_swaps,
    swap_annuity,
)

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
YEARS = (1, 2, 3, 4, 5)

#: A **non-flat** OIS curve: the reduction's own degenerate trap is a flat one, where a bootstrap
#: that ignored the earlier pillars would still land on the right answer.
OIS_RATES = {1: 0.0200, 2: 0.0215, 3: 0.0228, 4: 0.0237, 5: 0.0245}
#: The 6M tenor, quoted **above** OIS — a real basis, so the two curves genuinely differ.
SIX_MONTH = {1: 0.0225, 2: 0.0243, 3: 0.0258, 4: 0.0269, 5: 0.0278}
#: A second tenor, so a per-tenor bug is visible (§4: one tenor is the degenerate count).
THREE_MONTH = {1: 0.0210, 2: 0.0227, 3: 0.0241, 4: 0.0251, 5: 0.0260}


def _schedule(years: int) -> CouponSchedule:
    """Annual periods, one per quoted pillar — the nesting `nominal_bootstrap._validate` requires,
    since each quote's earlier payments must already be solved pillars."""
    return CouponSchedule(tuple(date(VAL.year + i, 1, 1) for i in range(years + 1)), DC)


def _quotes(rates: dict[int, float]) -> tuple[ParSwapQuote, ...]:
    return tuple(ParSwapQuote(_schedule(n), rates[n]) for n in YEARS)


def _ois():
    """The OIS discount curve — T8's bootstrap, reused rather than rebuilt."""
    return nominal_curve_from_par_swaps(MEASURE, _quotes(OIS_RATES))


def _forward(curve, start: date, pay: date, accrual: float) -> float:
    """Lichters (1.1), transcribed here: `f(t,t+δ) = (df_t/df_{t+δ} − 1)/δ`."""
    return (curve.df(start) / curve.df(pay) - 1.0) / accrual


def _floating_leg(projection, discount, schedule: CouponSchedule) -> float:
    """`Σ f^δ(t_{i−1},t_i)·δ_i·df^ON_i` — §3.5's right-hand side, composed in the test."""
    return sum(
        _forward(projection, start, pay, psi) * psi * discount.df(pay)
        for start, pay, psi in schedule.periods()
    )


class TestThePremises:
    """Asserted rather than believed — and here the degenerate case is the era the tree holds."""

    def test_the_ois_curve_is_not_flat(self) -> None:
        assert len(set(OIS_RATES.values())) == len(OIS_RATES)
        assert max(OIS_RATES.values()) - min(OIS_RATES.values()) > 0.004

    def test_there_is_a_REAL_basis_between_ois_and_each_tenor(self) -> None:
        for tenor in (SIX_MONTH, THREE_MONTH):
            for n in YEARS:
                assert tenor[n] > OIS_RATES[n], f"no basis at {n}y"
        # ... and the two tenors differ from each other, or one of them is decoration
        assert SIX_MONTH != THREE_MONTH

    def test_two_tenors_are_bootstrapped_so_a_per_tenor_bug_shows(self) -> None:
        ois = _ois()
        six = projection_curve_from_par_swaps(MEASURE, _quotes(SIX_MONTH), ois)
        three = projection_curve_from_par_swaps(MEASURE, _quotes(THREE_MONTH), ois)
        maturity = _schedule(5).dates[-1]
        assert six.df(maturity) != three.df(maturity)


class TestTheRoundTripIsTheCorrectnessAnchor:
    """§4's T8 pattern verbatim, now on a two-curve construction: a bootstrap must invert, so every
    calibrating instrument reprices to par — projecting off its own curve, discounting at OIS."""

    @pytest.mark.parametrize("rates", [SIX_MONTH, THREE_MONTH], ids=["6M", "3M"])
    @pytest.mark.parametrize("years", YEARS)
    def test_every_calibrating_swap_reprices_to_par(self, rates: dict, years: int) -> None:
        ois = _ois()
        projection = projection_curve_from_par_swaps(MEASURE, _quotes(rates), ois)
        schedule = _schedule(years)
        floating = _floating_leg(projection, ois, schedule)
        fixed = rates[years] * swap_annuity(ois, schedule)
        assert floating == pytest.approx(fixed, abs=1e-12)

    def test_the_OIS_swaps_reprice_off_the_OIS_curve(self) -> None:
        """The other half of the round trip, and the reuse claim's own check: the OIS curve is T8's
        bootstrap, so its own quotes must reprice — with the leg telescoping, since one curve both
        projects and discounts there."""
        ois = _ois()
        for years in YEARS:
            schedule = _schedule(years)
            floating = _floating_leg(ois, ois, schedule)
            fixed = OIS_RATES[years] * swap_annuity(ois, schedule)
            assert floating == pytest.approx(fixed, abs=1e-12)


class TestTheSingleCurveReduction:
    """The exact anchor to what we already hold — and the era invariant's other half: single-curve
    is not merely a limit of multi-curve, it is a mode the tree still prices in."""

    def test_ois_quotes_as_the_tenor_quotes_reproduce_T8(self) -> None:
        """When OIS = projection the float leg telescopes and §3.5's relation becomes BERHAN 5.1.1,
        so the projection bootstrap must land on T8's curve. Asserted as a **relative** tolerance
        with the ULP distance measured below — the two routes are one quantity in ℝ computed in
        different operation orders, so bit-identity is not available (§4) and committing it would
        be the T49 mistake."""
        ois = _ois()
        collapsed = projection_curve_from_par_swaps(MEASURE, _quotes(OIS_RATES), ois)
        for years in YEARS:
            maturity = _schedule(years).dates[-1]
            assert collapsed.df(maturity) == pytest.approx(ois.df(maturity), rel=1e-12)

    def test_THE_MEASURED_precision_of_that_reduction(self) -> None:
        """Recorded rather than claimed: worst **1 ULP** across the pillars, which is what "to the
        ULP" means here and why the committed assertion above is a relative tolerance."""
        ois = _ois()
        collapsed = projection_curve_from_par_swaps(MEASURE, _quotes(OIS_RATES), ois)

        def bits(value: float) -> int:
            return int.from_bytes(struct.pack("<d", value), "little")

        worst = max(
            abs(bits(collapsed.df(_schedule(n).dates[-1])) - bits(ois.df(_schedule(n).dates[-1])))
            for n in YEARS
        )
        assert worst <= 4, f"the reduction drifted {worst} ULP from T8 — more than a reordering"

    def test_THE_COMPANION_a_genuine_basis_makes_the_curves_DIFFER(self) -> None:
        """T39: the reduction is exact *because* it collapses the basis, so it is blind to
        everything multi-curve. A bootstrap that silently ignored the OIS curve and discounted on
        itself would pass the reduction and fail here."""
        ois = _ois()
        projection = projection_curve_from_par_swaps(MEASURE, _quotes(SIX_MONTH), ois)
        for years in YEARS:
            maturity = _schedule(years).dates[-1]
            basis = projection.df(maturity) / ois.df(maturity) - 1.0
            assert basis < -1e-4, f"no material basis at {years}y: {basis}"
        # the far pillar carries a basis of order 100bp, so this is not a rounding-scale difference
        far = _schedule(5).dates[-1]
        assert abs(projection.df(far) / ois.df(far) - 1.0) > 0.01


class TestTheTelescopingSignatureSeparatesTheEras:
    """Eq (1.2) as a structural discriminator rather than a tolerance, which is what makes the
    reduction above non-vacuous."""

    def test_a_single_curve_float_leg_TELESCOPES_and_a_multi_curve_one_does_NOT(self) -> None:
        """`Σ f·δ·df = df_0 − df_n` holds exactly when one curve both projects and discounts.
        Measured: single-curve 0.0, multi-curve 1.2e-03 — so the identity is an era test."""
        ois = _ois()
        schedule = _schedule(5)

        collapsed = projection_curve_from_par_swaps(MEASURE, _quotes(OIS_RATES), ois)
        single = _floating_leg(collapsed, ois, schedule)
        assert single == pytest.approx(1.0 - collapsed.df(schedule.dates[-1]), abs=1e-12)

        projection = projection_curve_from_par_swaps(MEASURE, _quotes(SIX_MONTH), ois)
        multi = _floating_leg(projection, ois, schedule)
        gap = abs(multi - (1.0 - projection.df(schedule.dates[-1])))
        assert gap > 1e-4, f"the multi-curve leg telescoped ({gap:.2e}) — it cannot"


class TestTheSequentialDependencyIsLoadBEARING:
    """The projection curve is bootstrapped *against* the OIS curve, which the layer box cannot
    show. Perturb OIS and the projection curve must move — a projection curve that ignored the
    discounting would be the old single-curve error wearing a new name."""

    def test_shifting_an_OIS_quote_moves_the_PROJECTION_curve(self) -> None:
        base = projection_curve_from_par_swaps(MEASURE, _quotes(SIX_MONTH), _ois())
        shifted_rates = {**OIS_RATES, 3: OIS_RATES[3] + 0.0025}
        shifted = projection_curve_from_par_swaps(
            MEASURE, _quotes(SIX_MONTH), nominal_curve_from_par_swaps(MEASURE, _quotes(shifted_rates))
        )
        maturity = _schedule(5).dates[-1]
        assert base.df(maturity) != shifted.df(maturity)
        assert abs(shifted.df(maturity) / base.df(maturity) - 1.0) > 1e-6

    def test_THE_PREMISE_the_tenor_quotes_are_UNCHANGED_by_that_shift(self) -> None:
        """So the movement above is attributable to the OIS curve and nothing else."""
        assert SIX_MONTH[3] == 0.0258
        assert {**OIS_RATES, 3: OIS_RATES[3] + 0.0025}[3] != OIS_RATES[3]


class TestTheGuardsAreReachable:
    def test_a_quote_set_that_is_not_nested_is_refused(self) -> None:
        """The projection bootstrap inherits T8's precondition: each quote's earlier payments must
        be solved pillars, or a difference subtracts two swaps that do not share their history."""
        gappy = (ParSwapQuote(_schedule(1), 0.02), ParSwapQuote(_schedule(3), 0.025))
        with pytest.raises(ValueError):
            projection_curve_from_par_swaps(MEASURE, gappy, _ois())

    def test_an_empty_quote_set_is_refused(self) -> None:
        with pytest.raises(ValueError):
            projection_curve_from_par_swaps(MEASURE, (), _ois())

    def test_a_quote_implying_a_non_positive_discount_factor_is_refused(self) -> None:
        absurd = (ParSwapQuote(_schedule(1), -5.0),)
        with pytest.raises(ValueError):
            projection_curve_from_par_swaps(MEASURE, absurd, _ois())
