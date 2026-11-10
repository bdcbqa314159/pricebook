"""Collateral / CSA (L5) — tranche 29, the last build of the XVA spine arc.

Under a **zero-threshold, zero-MTA variation-margin** CSA, collateral equals the mark at the last
margin call. What remains exposed is therefore not `V(t)` but the **gap over the margin period of
risk**:

    collateralized exposure(t) = (V(t) - V(t-delta))^+

`delta` is the margin period of risk — how stale the posted collateral is. Perfect margining
(`delta = 0`) leaves nothing exposed; no CSA leaves `V(t)` itself. Everything above this is reused
unchanged: the gap is assembled into a `SimulatedExposure`, and T25/T27's `credit_adjustment` prices
it with **no new CVA/DVA code**.

**The two anchors are exact reductions, one to zero and one to the prior tranche** — and each is
paired with the non-degenerate companion §4 demands, because a reduction a no-op also satisfies is
vacuous. The companion is the interesting case: a material `delta` on a volatile mark, where the
collateralized CVA sits **strictly between** zero and the uncollateralized value. Without it the
suite would test only the two endpoints and leave the middle — the actual product — a hole.

**The process is not re-anchored here, and that is deliberate.** The gap is a functional of the very
paths tranche 26's Leg B validated against the numéraire-relative martingale identity, so it inherits
that validation; inventing a fresh "independent" anchor over the same paths would add ceremony, not
evidence. What this file owes instead is a **closed form for the gap**, which a Gaussian mark
supplies: `V(t) - V(t-delta) ~ N(0, sigma^2 delta)` for a driftless arithmetic mark, so T25's
`expected_positive_exposure` on the gap's own moments is the reference — derived, not transcribed,
since no source equation is involved.
"""

import math
from datetime import date

import numpy as np
import pytest

from pricebook.foundation import Currency
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.risk.collateral import CollateralAgreement, collateralized_exposure
from pricebook.risk.credit_valuation_adjustment import expected_positive_exposure
from pricebook.risk.mc_exposure import SimulatedExposure, mc_exposure_cva

ANCHOR = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(ANCHOR, DC)
EUR = Currency("EUR")

SURVIVAL = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.020, 0.040, 0.060))
DISCOUNT = DiscountCurve(TM, (0.0, 5.0), (0.020, 0.035))
LGD = 0.6

INITIAL_MARK = 60.0
# A VOLATILE mark: the gap over the margin period must be material, else collateral is invisible.
SIGMA = 40.0
HORIZON = 4.0
DELTA = 0.25


def _simulated(delta: float, paths: int, seed: int = 17) -> SimulatedExposure:
    """`V(t) = V0 + sigma*W(t)` on a UNIFORM grid of spacing `delta` — the margin-call schedule."""
    steps = int(round(HORIZON / delta))
    times = tuple(delta * (k + 1) for k in range(steps))
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal((paths, steps)) * math.sqrt(delta) * SIGMA
    values = INITIAL_MARK + np.cumsum(increments, axis=1)
    return SimulatedExposure(times, tuple(tuple(row) for row in values))


def _agreement(delta: float = DELTA) -> CollateralAgreement:
    return CollateralAgreement(margin_period=delta, initial_mark=INITIAL_MARK)


def _cva(exposure: SimulatedExposure) -> float:
    return mc_exposure_cva(exposure, SURVIVAL, DISCOUNT, LGD, EUR).pv.amount


class TestTheFixtureIsNotDegenerate:
    def test_the_margin_period_is_MATERIAL_and_the_mark_VOLATILE(self) -> None:
        """Else the gap is ~0 and collateral is invisible — the §4 fixture trap."""
        assert DELTA > 0.05
        gap_sd = SIGMA * math.sqrt(DELTA)
        assert gap_sd > 5.0, f"the gap's own sd must be material, got {gap_sd}"

    def test_the_collateralized_and_UNCOLLATERALIZED_values_differ_MATERIALLY(self) -> None:
        """Else only the two reductions are tested and the middle — the product — is a hole."""
        raw = _simulated(DELTA, 20_000)
        collateralized = _cva(collateralized_exposure(raw, _agreement()))
        uncollateralized = _cva(raw)
        assert collateralized < 0.5 * uncollateralized, (
            f"collateral must bite: {collateralized} vs {uncollateralized}"
        )
        assert collateralized > 0.0


class TestTheTwoExactReductions:
    def test_PERFECT_margining_leaves_NOTHING_exposed(self) -> None:
        """`delta = 0` gives `V(t) - V(t) = 0` — exact by construction, not a limit."""
        raw = _simulated(DELTA, 5_000)
        perfect = collateralized_exposure(raw, CollateralAgreement(0.0, INITIAL_MARK))
        assert all(v == 0.0 for row in perfect.values for v in row)
        assert _cva(perfect) == 0.0

    def test_NO_CSA_is_the_uncollateralized_exposure_BIT_IDENTICALLY(self) -> None:
        """Uncollateralized is the `agreement=None` special case, so T26/T27's path must not move.

        The reduction is *trivially* exact — `None` short-circuits and the same object comes back —
        and saying so is the point: it is a structural guarantee rather than a numerical result, and
        the companion above is what makes the pair non-vacuous.
        """
        raw = _simulated(DELTA, 5_000)
        assert collateralized_exposure(raw, None) is raw
        assert _cva(collateralized_exposure(raw, None)) == _cva(raw)

    def test_a_SHRINKING_margin_period_drives_the_adjustment_down_as_ROOT_DELTA(self) -> None:
        """Not merely "it falls" — it falls at the **rate the finance predicts**.

        The gap's standard deviation is `sigma*sqrt(delta)`, so the gap EPE is proportional to
        `sqrt(delta)` and so is the adjustment: halving the margin period must multiply the CVA by
        `1/sqrt(2) = 0.7071`. Measured 0.772, 0.710, 0.706 over successive halvings — the first is
        the coarsest grid (only four margin dates, so the trapezoid itself is crude), the later two
        land on the predicted ratio. A rate oracle catches a wrong power of `delta` that a
        monotonicity check cannot see.
        """
        values = [_cva(collateralized_exposure(_simulated(d, 8_000), _agreement(d))) for d in (1.0, 0.5, 0.25, 0.125)]
        assert all(b < a for a, b in zip(values, values[1:])), f"must fall with delta: {values}"
        ratios = [b / a for a, b in zip(values, values[1:])]
        assert all(abs(r - 0.7071) < 0.08 for r in ratios[1:]), f"expected ~1/sqrt(2), got {ratios}"


class TestTheClosedFormAnchorOnAGaussianMark:
    """The gap of a driftless arithmetic mark is `N(0, sigma^2 delta)` — derived, not transcribed."""

    def test_the_simulated_gap_EPE_matches_T25s_closed_form_at_every_date(self) -> None:
        exposure = collateralized_exposure(_simulated(DELTA, 60_000), _agreement())
        reference = expected_positive_exposure(0.0, SIGMA * math.sqrt(DELTA))
        estimate = exposure.estimate()
        for t, epe in zip(estimate.times, estimate.epe):
            assert abs(epe.mean - reference) <= 4.0 * epe.standard_error, (
                f"t={t}: {epe.mean} vs {reference} +/- {epe.standard_error}"
            )

    def test_the_gap_EPE_is_FLAT_in_time_which_the_uncollateralized_EPE_is_not(self) -> None:
        """A structural signature: the gap's law does not depend on `t`, the mark's does.

        It separates the two quantities beyond their level, so a mutant computing `V(t)` instead of
        the gap fails on the SHAPE and not only the magnitude.
        """
        gap = collateralized_exposure(_simulated(DELTA, 40_000), _agreement()).estimate()
        raw = _simulated(DELTA, 40_000).estimate()
        gap_means = [e.mean for e in gap.epe]
        raw_means = [e.mean for e in raw.epe]
        gap_spread = max(gap_means) / min(gap_means) - 1.0
        raw_spread = max(raw_means) / min(raw_means) - 1.0
        # Measured: the gap varies 2.4% across the horizon (sampling noise on a law that does not
        # depend on `t`), the mark's EPE grows 17.4%. The assertion is the SEPARATION rather than
        # two absolute bounds, because the mark's growth depends on how far in the money it starts
        # and a bound tuned to this fixture would not be the structural claim.
        assert gap_spread < 0.05, f"the gap EPE must be ~flat in time: {gap_means[:4]}"
        assert raw_spread > 5.0 * gap_spread, (
            f"the mark's EPE must grow where the gap's does not: {raw_spread:.4f} vs {gap_spread:.4f}"
        )

    def test_the_error_falls_at_the_MONTE_CARLO_rate(self) -> None:
        reference = expected_positive_exposure(0.0, SIGMA * math.sqrt(DELTA))

        def rms(paths: int) -> float:
            est = collateralized_exposure(_simulated(DELTA, paths, seed=5), _agreement()).estimate()
            return math.sqrt(math.fsum((e.mean - reference) ** 2 for e in est.epe) / len(est.epe))

        coarse, fine = rms(2_500), rms(40_000)
        assert fine < coarse / 2.0, f"expected ~4x, got {coarse} -> {fine}"


class TestTheRequirementOraclesTheFinanceForces:
    def test_collateral_can_never_INCREASE_the_adjustment(self) -> None:
        for delta in (0.125, 0.5, 1.0, 2.0):
            raw = _simulated(delta, 8_000)
            assert _cva(collateralized_exposure(raw, _agreement(delta))) <= _cva(raw) + 1e-12

    def test_the_adjustment_is_MONOTONE_INCREASING_in_the_margin_period(self) -> None:
        """Staler collateral leaves more exposed. Compared on ONE grid so only delta varies."""
        fine = _simulated(0.125, 20_000)
        values = [
            _cva(collateralized_exposure(fine, CollateralAgreement(0.125, INITIAL_MARK))),
            _cva(collateralized_exposure(_simulated(0.5, 20_000), _agreement(0.5))),
            _cva(collateralized_exposure(_simulated(2.0, 20_000), _agreement(2.0))),
        ]
        assert all(b > a for a, b in zip(values, values[1:])), f"must rise with delta: {values}"

    def test_the_gap_exposure_is_never_negative(self) -> None:
        exposure = collateralized_exposure(_simulated(DELTA, 5_000), _agreement())
        assert all(v >= 0.0 for row in exposure.values for v in row)


class TestTheGuardsAreReachable:
    def test_a_NEGATIVE_margin_period_is_refused(self) -> None:
        with pytest.raises(ValueError, match="margin period"):
            CollateralAgreement(-0.1, INITIAL_MARK)

    def test_a_NON_UNIFORM_grid_is_refused(self) -> None:
        ragged = SimulatedExposure((0.25, 0.5, 1.4), ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
        with pytest.raises(ValueError, match="uniform"):
            collateralized_exposure(ragged, _agreement())

    def test_a_grid_whose_SPACING_is_not_the_margin_period_is_refused(self) -> None:
        """The collateral posted at `t-delta` has to be a point the simulation actually visited."""
        raw = _simulated(0.5, 100)
        with pytest.raises(ValueError, match="margin period"):
            collateralized_exposure(raw, _agreement(0.25))

    def test_a_NON_FINITE_initial_mark_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            CollateralAgreement(0.25, math.inf)


# ── the DRIFTING mark: what the driftless fixture could not see ─────────────────────────────────
DRIFT = 24.0


def _simulated_with_drift(delta: float, paths: int, seed: int = 29) -> SimulatedExposure:
    """`V(t) = V0 + mu*t + sigma*W(t)` — a mark with a trend, so the gap is ASYMMETRIC."""
    steps = int(round(HORIZON / delta))
    times = tuple(delta * (k + 1) for k in range(steps))
    rng = np.random.default_rng(seed)
    increments = DRIFT * delta + rng.standard_normal((paths, steps)) * math.sqrt(delta) * SIGMA
    values = INITIAL_MARK + np.cumsum(increments, axis=1)
    return SimulatedExposure(times, tuple(tuple(row) for row in values))


class TestTheGapsDIRECTIONIsPinnedWhichNeedsADriftingMark:
    """**A hole the mutation sweep found, and the fixture that closes it.**

    Reversing the gap to `(V(t-delta) - V(t))^+` survived all 986 tests. It is not an equivalent
    mutant — it is the *wrong* exposure, and it survived because the original fixture's mark is
    **driftless**: then `V(t) - V(t-delta) ~ N(0, sigma^2 delta)` is symmetric about zero, so the
    reversed gap has the *identical distribution* and every mean-based oracle is structurally blind
    to the direction. Measured: with no drift the two EPEs agree to the last digit; with a drift of
    24/year they differ by **87%**.

    This is §4's degenerate-fixture rule in a form the charter had not yet recorded: the degeneracy
    is a **distributional symmetry**, not a specific parameter value (`tau = 1`) nor a uniformity
    across a slice. The guard is therefore not "the values differ" but "the fixture breaks the
    symmetry the mutation exploits".
    """

    def test_the_drifting_fixture_BREAKS_the_symmetry_the_mutant_hides_behind(self) -> None:
        forward, sd = DRIFT * DELTA, SIGMA * math.sqrt(DELTA)
        assert forward / sd > 0.2, f"the drift must be material against the gap's sd, got {forward / sd}"
        up = expected_positive_exposure(forward, sd)
        down = expected_positive_exposure(-forward, sd)
        assert abs(up / down - 1.0) > 0.5, (
            f"the fixture must distinguish the two directions: {up} vs {down}"
        )

    def test_the_gap_EPE_matches_the_DRIFTING_closed_form(self) -> None:
        """`V(t) - V(t-delta) ~ N(mu*delta, sigma^2 delta)` — the reference carries the drift."""
        exposure = collateralized_exposure(_simulated_with_drift(DELTA, 60_000), _agreement())
        reference = expected_positive_exposure(DRIFT * DELTA, SIGMA * math.sqrt(DELTA))
        estimate = exposure.estimate()
        for t, epe in zip(estimate.times, estimate.epe):
            assert abs(epe.mean - reference) <= 4.0 * epe.standard_error, (
                f"t={t}: {epe.mean} vs {reference} +/- {epe.standard_error}"
            )

    def test_the_REVERSED_gap_is_rejected_by_that_same_oracle(self) -> None:
        """The companion proving the oracle has power: the mutant's value must fail the band.

        Asserted rather than assumed, because "the drifting fixture would catch it" is exactly the
        kind of claim that goes untested and turns out false.
        """
        exposure = collateralized_exposure(_simulated_with_drift(DELTA, 60_000), _agreement())
        reference = expected_positive_exposure(DRIFT * DELTA, SIGMA * math.sqrt(DELTA))
        reversed_reference = expected_positive_exposure(-DRIFT * DELTA, SIGMA * math.sqrt(DELTA))
        epe = exposure.estimate().epe[0]
        assert abs(epe.mean - reference) <= 4.0 * epe.standard_error
        assert abs(reversed_reference - reference) > 4.0 * epe.standard_error, (
            "the band must be tight enough to separate the two directions"
        )

    def test_a_drifting_mark_still_obeys_the_requirement_oracles(self) -> None:
        raw = _simulated_with_drift(DELTA, 20_000)
        collateralized = _cva(collateralized_exposure(raw, _agreement()))
        assert 0.0 < collateralized <= _cva(raw) + 1e-12
