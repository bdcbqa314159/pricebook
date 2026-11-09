"""Bilateral CVA + DVA (L5) — tranche 27, both parties can default.

T25/T26 priced the unilateral case: only the counterparty defaults. The bilateral adjustment adds
the mirror — the value to us of *our own* default while we are out of the money — and conditions
each leg on the other party still being alive:

    CVA = LGD_C · integral DF(t) · EPE(t) · S_own(t) · lambda_C(t) S_C(t) dt
    DVA = LGD_own · integral DF(t) · ENE(t) · S_C(t) · lambda_own(t) S_own(t) dt
    BCVA = CVA - DVA                                    (Crepey Prop 1 eq (18), third line)

**The two legs share a shape, so the integral is extracted (rule-of-two, §6).** T25 was the first
consumer; DVA is the second, and `credit_adjustment` is where the shared form lands — with T25's
`unilateral_cva` becoming the `surviving = None` special case. The reduction doubles as the purity
proof: routing T25 through the general integral must not move its number.

**The convention this file asserts, stated because the source differs.** Crepey conditions through
the **bank survival measure** (`EY = (1-gamma)^-1 E*(Y')`, eq (7) p.37), i.e. with a `(1-gamma)^-1`
*normalisation*; this builds the market-standard **explicit surviving-party survival factor**, which
is the natural extension of T25/T26 (neither carries a normalisation). Both have the
first-to-default term present and both satisfy `BCVA = CVA - DVA`; they differ by the normalisation.
Raised as an `Ask:` — the spec reserves the choice for Cowork — and asserted here as a *measured*
relationship rather than assumed away.
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
from pricebook.risk.credit_valuation_adjustment import (
    BilateralAdjustment,
    CreditParties,
    ExposureProfile,
    bilateral_credit_adjustment,
    credit_adjustment,
    expected_positive_exposure,
    unilateral_cva,
)
from pricebook.risk.mc_exposure import SimulatedExposure, mc_bilateral_adjustment

ANCHOR = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(ANCHOR, DC)
EUR = Currency("EUR")

DISCOUNT = DiscountCurve(TM, (0.0, 5.0), (0.020, 0.035))
# The two hazards DIFFER and the two LGDs differ: else CVA and DVA read the same survival and a
# swapped assignment is invisible (§4's degenerate-fixture rule).
OWN = SurvivalCurve(TM, (0.0, 2.0), (0.008, 0.015))
COUNTERPARTY = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.020, 0.040, 0.060))
PARTIES = CreditParties(own=OWN, counterparty=COUNTERPARTY, own_lgd=0.55, counterparty_lgd=0.70)

GRID = (0.25, 0.75, 1.5, 2.5, 4.0)
# A genuinely TWO-SIDED exposure: a small forward against a large vol, so EPE and ENE are both
# material. A one-sided exposure makes DVA ~ 0 and the whole bilateral character invisible.
V0, SIGMA = 5.0, 25.0


def _epe() -> ExposureProfile:
    return ExposureProfile(GRID, tuple(expected_positive_exposure(V0, SIGMA * math.sqrt(t)) for t in GRID))


def _ene() -> ExposureProfile:
    # E[V^-] = E[(-V)^+] with V ~ N(V0, s^2): the same closed form at a flipped mean.
    return ExposureProfile(GRID, tuple(expected_positive_exposure(-V0, SIGMA * math.sqrt(t)) for t in GRID))


def _flat(hazard: float) -> SurvivalCurve:
    return SurvivalCurve(TM, (0.0,), (hazard,))


def _simulated(paths: int, seed: int = 3) -> SimulatedExposure:
    rng = np.random.default_rng(seed)
    dt = np.diff((0.0,) + GRID)
    values = V0 + SIGMA * np.cumsum(rng.standard_normal((paths, len(GRID))) * np.sqrt(dt), axis=1)
    return SimulatedExposure(GRID, tuple(tuple(row) for row in values))


class TestTheFixtureIsNotDegenerate:
    def test_the_exposure_is_genuinely_TWO_SIDED(self) -> None:
        epe, ene = _epe().epe, _ene().epe
        assert min(epe) > 1.0 and min(ene) > 1.0, "DVA needs a material negative side"
        ratios = [n / p for p, n in zip(epe, ene)]
        assert all(0.3 < r < 3.0 for r in ratios), f"one-sided exposure hides DVA: {ratios}"

    def test_the_two_HAZARDS_differ(self) -> None:
        probe = 1.5
        assert abs(OWN.hazard(probe) - COUNTERPARTY.hazard(probe)) > 0.01

    def test_the_two_LGDs_differ(self) -> None:
        assert abs(PARTIES.own_lgd - PARTIES.counterparty_lgd) > 0.1


class TestTheReductionToTheUnilateralCase:
    """The preferred anchor (§4) — exact by construction, and the purity proof for the extraction."""

    def test_T25s_unilateral_cva_through_the_GENERAL_integral_is_BIT_IDENTICAL(self) -> None:
        general = credit_adjustment(_epe(), COUNTERPARTY, None, DISCOUNT, 0.70)
        assert general == unilateral_cva(_epe(), COUNTERPARTY, DISCOUNT, 0.70)

    def test_a_ZERO_own_hazard_gives_a_ZERO_DVA_and_the_unilateral_CVA(self) -> None:
        immortal = CreditParties(_flat(0.0), COUNTERPARTY, 0.55, 0.70)
        result = bilateral_credit_adjustment(_epe(), _ene(), immortal, DISCOUNT)
        assert result.dva == 0.0
        assert result.cva == pytest.approx(unilateral_cva(_epe(), COUNTERPARTY, DISCOUNT, 0.70), rel=1e-14)
        assert result.bcva == result.cva

    def test_a_ZERO_counterparty_hazard_gives_a_ZERO_CVA_and_BCVA_is_MINUS_DVA(self) -> None:
        safe = CreditParties(OWN, _flat(0.0), 0.55, 0.70)
        result = bilateral_credit_adjustment(_epe(), _ene(), safe, DISCOUNT)
        assert result.cva == 0.0
        assert result.bcva == -result.dva


class TestThePartySwapSymmetryTheIndependentAnchor:
    """Our DVA = the counterparty's CVA against us — a DIFFERENT composition of the same pieces.

    Non-telescoping where the reduction is not: it exercises the **positive**-part code path on a
    negated exposure, so an ENE computed wrongly (say as `-mean(V)` rather than `mean(V^-)`) or a
    swapped curve assignment shows up here and in no within-DVA check.
    """

    def test_our_DVA_equals_their_CVA_on_the_FLIPPED_exposure_with_curves_SWAPPED(self) -> None:
        ours = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT).dva
        # Their view: exposure sign flipped, so our ENE is their EPE; our default is their
        # counterparty's; our LGD is the one that applies.
        theirs = credit_adjustment(_ene(), OWN, COUNTERPARTY, DISCOUNT, PARTIES.own_lgd)
        assert ours == pytest.approx(theirs, rel=1e-14)

    def test_the_swap_holds_on_SIMULATED_paths_through_the_POSITIVE_part_path(self) -> None:
        exposure = _simulated(20_000)
        ene_from_negative_part = exposure.estimate().negative_profile()
        flipped = SimulatedExposure(GRID, tuple(tuple(-v for v in row) for row in exposure.values))
        epe_of_flipped = flipped.estimate().profile()
        for a, b in zip(ene_from_negative_part.epe, epe_of_flipped.epe):
            assert a == pytest.approx(b, rel=1e-14), "ENE(V) must equal EPE(-V)"


class TestTheRequirementOraclesCrepeyForces:
    def test_BCVA_is_EXACTLY_CVA_minus_DVA(self) -> None:
        r = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT)
        assert r.bcva == r.cva - r.dva

    def test_both_legs_are_NON_NEGATIVE_and_BCVA_may_take_either_sign(self) -> None:
        r = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT)
        assert r.cva > 0.0 and r.dva > 0.0
        riskier_us = CreditParties(_flat(0.40), _flat(0.001), 0.9, 0.9)
        assert bilateral_credit_adjustment(_epe(), _ene(), riskier_us, DISCOUNT).bcva < 0.0

    def test_the_CVA_rises_with_the_COUNTERPARTYS_hazard_and_LGD(self) -> None:
        base = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT).cva
        worse = CreditParties(OWN, SurvivalCurve(TM, COUNTERPARTY.times, tuple(h + 0.02 for h in COUNTERPARTY.hazards)), 0.55, 0.70)
        assert bilateral_credit_adjustment(_epe(), _ene(), worse, DISCOUNT).cva > base
        assert bilateral_credit_adjustment(_epe(), _ene(), CreditParties(OWN, COUNTERPARTY, 0.55, 0.95), DISCOUNT).cva > base

    def test_the_DVA_rises_with_OUR_hazard_and_OUR_LGD(self) -> None:
        base = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT).dva
        worse = CreditParties(SurvivalCurve(TM, OWN.times, tuple(h + 0.02 for h in OWN.hazards)), COUNTERPARTY, 0.55, 0.70)
        assert bilateral_credit_adjustment(_epe(), _ene(), worse, DISCOUNT).dva > base
        assert bilateral_credit_adjustment(_epe(), _ene(), CreditParties(OWN, COUNTERPARTY, 0.95, 0.70), DISCOUNT).dva > base


class TestTheFirstToDefaultTermIsPresentAndCorrectlySigned:
    """The term the unilateral case dropped, made visible — magnitude, not only presence (§4)."""

    def test_our_own_default_risk_STRICTLY_REDUCES_the_CVA(self) -> None:
        immortal = CreditParties(_flat(0.0), COUNTERPARTY, 0.55, 0.70)
        mortal = CreditParties(_flat(0.05), COUNTERPARTY, 0.55, 0.70)
        unconditioned = bilateral_credit_adjustment(_epe(), _ene(), immortal, DISCOUNT).cva
        conditioned = bilateral_credit_adjustment(_epe(), _ene(), mortal, DISCOUNT).cva
        assert conditioned < unconditioned, "we cannot lose on a counterparty default after our own"
        # Magnitude, not just sign: at a 5% flat own hazard over a 4y grid the survival weight is
        # materially below 1, so the term is worth percent, not epsilon.
        assert 0.02 < 1.0 - conditioned / unconditioned < 0.25

    def test_dropping_the_surviving_factor_gives_TWO_UNLINKED_unilaterals(self) -> None:
        """What the mutation would produce, asserted as a measured difference rather than assumed."""
        linked = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT)
        unlinked_cva = credit_adjustment(_epe(), COUNTERPARTY, None, DISCOUNT, PARTIES.counterparty_lgd)
        assert linked.cva < unlinked_cva
        assert abs(linked.cva / unlinked_cva - 1.0) > 0.005


class TestTheMonteCarloBilateralAdjustmentCarriesItsError:
    def test_the_pathwise_BCVA_carries_a_standard_error_that_falls_as_root_N(self) -> None:
        coarse = mc_bilateral_adjustment(_simulated(2_500), PARTIES, DISCOUNT, EUR)
        fine = mc_bilateral_adjustment(_simulated(40_000), PARTIES, DISCOUNT, EUR)
        assert coarse.standard_error > 0.0
        ratio = coarse.standard_error / fine.standard_error
        assert 2.5 < ratio < 5.5, f"expected ~4x, got {ratio}"

    def test_the_pathwise_mean_matches_the_PROFILE_form_on_the_same_paths(self) -> None:
        exposure = _simulated(20_000)
        estimate = exposure.estimate()
        profile_form = bilateral_credit_adjustment(
            estimate.profile(), estimate.negative_profile(), PARTIES, DISCOUNT
        )
        pathwise = mc_bilateral_adjustment(exposure, PARTIES, DISCOUNT, EUR)
        assert pathwise.pv.amount == pytest.approx(profile_form.bcva, rel=1e-12)

    def test_the_pathwise_SE_is_NOT_the_quadrature_sum_of_the_two_legs(self) -> None:
        """CVA and DVA are computed on the SAME paths, so the difference's error is not the sum.

        A reconstructed SE would assume independence and get this wrong; the pathwise estimator is
        required (T26's ratified choice) rather than merely convenient.
        """
        exposure = _simulated(20_000)
        joint = mc_bilateral_adjustment(exposure, PARTIES, DISCOUNT, EUR)
        cva_only = mc_bilateral_adjustment(exposure, CreditParties(_flat(0.0), COUNTERPARTY, PARTIES.own_lgd, PARTIES.counterparty_lgd), DISCOUNT, EUR)
        dva_only = mc_bilateral_adjustment(exposure, CreditParties(OWN, _flat(0.0), PARTIES.own_lgd, PARTIES.counterparty_lgd), DISCOUNT, EUR)
        naive = math.hypot(cva_only.standard_error, dva_only.standard_error)
        assert abs(joint.standard_error / naive - 1.0) > 0.02, (
            f"pathwise {joint.standard_error} vs independence-assuming {naive}"
        )


class TestTheGuardsAreReachable:
    def test_an_LGD_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="loss given default"):
            CreditParties(OWN, COUNTERPARTY, 1.4, 0.70)

    def test_two_curves_on_DIFFERENT_clocks_are_refused(self) -> None:
        elsewhere = SurvivalCurve(TimeMeasure(date(2027, 6, 1), DC), OWN.times, OWN.hazards)
        with pytest.raises(ValueError, match="share one time measure"):
            bilateral_credit_adjustment(_epe(), _ene(), CreditParties(elsewhere, COUNTERPARTY, 0.55, 0.70), DISCOUNT)

    def test_mismatched_exposure_grids_are_refused(self) -> None:
        shifted = ExposureProfile(tuple(t + 0.1 for t in GRID), _ene().epe)
        with pytest.raises(ValueError, match="same time grid"):
            bilateral_credit_adjustment(_epe(), shifted, PARTIES, DISCOUNT)

    def test_the_adjustment_exposes_both_legs_not_only_the_net(self) -> None:
        r = bilateral_credit_adjustment(_epe(), _ene(), PARTIES, DISCOUNT)
        assert isinstance(r, BilateralAdjustment) and r.cva > 0.0 and r.dva > 0.0
