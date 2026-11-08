"""Unilateral CVA and the expected-positive-exposure profile (L5) — tranche 25.

Two sources, split along §4's seam:

* **`CAPINSKI-ZASTAWNIAK` / Bachelier — pointwise transcription.** `E[(m+sZ)⁺] = m·Φ(m/s) +
  s·φ(m/s)`, checked against the test's OWN numerical integral `∫(m+sz)⁺φ(z)dz`. Two independent
  routes to one number.
* **`CREPEY-2026` Prop 1 eq (17) — requirement oracles.** `CVA = E[(1−J)P⁺]` reduces, under
  exposure ⊥ default, to `LGD·∫DF·EPE·dPD`. The zeros, the sign, the monotonicity and the sandwich
  are forced by the finance, so they survive a transcription error the pointwise check would share.

**Fixture degeneracies closed on purpose (§4, post-F-1.2/T23).** A flat hazard hides a cumulation
bug; `s = 0` collapses `EPE` to the forward and makes the `Φ/φ` split invisible; a flat discount
curve lets a mutant drop `DF(t)` entirely. All three are non-degenerate here and asserted as
premises.
"""

import math
from datetime import date

import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.risk.credit_valuation_adjustment import (
    ExposureProfile,
    expected_positive_exposure,
    unilateral_cva,
    unilateral_cva_one_period,
)

ANCHOR = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(ANCHOR, DC)

HORIZON = 5.0
LGD = 0.6

# Non-flat hazard and non-flat discount curve: see the module docstring.
SURVIVAL = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.010, 0.030, 0.055))
DISCOUNT = DiscountCurve(TM, (0.0, 5.0), (0.020, 0.035))


def _mean(t: float) -> float:
    return 100.0 * (1.0 - t / 10.0)


def _stdev(t: float) -> float:
    return 30.0 * math.sqrt(t)


def _epe_by_quadrature(mean: float, stdev: float) -> float:
    """`∫(mean+stdev·z)φ(z)dz` over the positive region — an independent ROUTE to the EPE.

    Different algebra from the analytic form under test, which is what makes it an oracle rather
    than a restatement. **Its accuracy argument, which this tranche had to learn (§4: a reference
    is code too):** the interval must be finite and must contain the mass. Integrating from
    `-mean/stdev` to `+∞` makes scipy's infinite-range transform miss the peak entirely when the
    ratio is large — it returned ~0 where the true EPE was 99.975 — and a fixed `lower + 60`
    truncates above the mass for the same reason. Clipping the lower limit at −8 and taking a
    16-wide window holds the whole distribution for any ratio. Trustworthy to ~1e-15 for
    `|mean/stdev| <= 8`; in a far negative tail the integral itself is ~1e-24 and this route is the
    weaker side, so the extreme regime is asserted against scipy instead.
    """
    if stdev == 0.0:
        return max(mean, 0.0)
    lower = max(-mean / stdev, -8.0)
    value, _ = quad(lambda z: (mean + stdev * z) * norm.pdf(z), lower, lower + 16.0, limit=400)
    return value


def _epe_by_scipy(mean: float, stdev: float) -> float:
    """The same algebra through scipy's `ndtr` — an independent IMPLEMENTATION, accurate across
    every ratio, so it is what the CVA integrand and the extreme-ratio checks use (the T20a
    arrangement: the oracle may reach for scipy while the library may not)."""
    if stdev == 0.0:
        return max(mean, 0.0)
    ratio = mean / stdev
    return float(mean * ndtr(ratio) + stdev * norm.pdf(ratio))


def _profile(points: int = 2001, horizon: float = HORIZON) -> ExposureProfile:
    times = tuple(horizon * k / (points - 1) for k in range(points))
    return ExposureProfile.gaussian(times, tuple(_mean(t) for t in times), tuple(_stdev(t) for t in times))


def _reference_cva(survival: SurvivalCurve = SURVIVAL, lgd: float = LGD) -> float:
    """`LGD·∫₀ᵀ DF(t)·EPE(t)·λ(t)S(t) dt` by scipy quadrature on an integrand the test builds itself.

    Every factor is recomputed here — the discount factor from the curve's zero rate, the EPE from
    the analytic Bachelier form, the default density as `λ·S` — so nothing is read back from
    `unilateral_cva` (§4 independence).
    """

    def integrand(t: float) -> float:
        df = math.exp(-DISCOUNT.zero_rate(t) * t)
        epe = _epe_by_scipy(_mean(t), _stdev(t))
        return df * epe * survival.hazard(t) * survival.survival(t)

    value, _ = quad(integrand, 0.0, HORIZON, limit=400)
    return lgd * value


class TestTheFixtureIsNotDegenerate:
    def test_the_hazard_is_NOT_flat(self) -> None:
        assert max(SURVIVAL.hazards) - min(SURVIVAL.hazards) > 0.04

    def test_the_discount_curve_is_NOT_flat(self) -> None:
        assert abs(DISCOUNT.zero_rates[-1] - DISCOUNT.zero_rates[0]) > 0.01

    def test_the_exposure_is_TWO_SIDED_so_the_Phi_phi_split_bites(self) -> None:
        ratios = [abs(_mean(t) / _stdev(t)) for t in (1.0, 2.5, 5.0)]
        assert all(0.3 < r < 4.0 for r in ratios), f"m/s must be O(1), got {ratios}"
        assert min(_stdev(t) for t in (1.0, 2.5, 5.0)) > 1.0


class TestExposureIsTranscribedPointwise:
    @pytest.mark.parametrize("mean,stdev", [(100.0, 30.0), (0.0, 25.0), (-40.0, 50.0), (80.0, 15.0), (-20.0, 100.0)])
    def test_the_analytic_EPE_matches_the_tests_OWN_quadrature(self, mean: float, stdev: float) -> None:
        """The independent route, inside the regime where the route itself is accurate."""
        assert abs(mean / stdev) <= 8.0, "outside the quadrature route's argued regime"
        assert expected_positive_exposure(mean, stdev) == pytest.approx(_epe_by_quadrature(mean, stdev), rel=1e-9)

    @pytest.mark.parametrize("mean,stdev", [(99.975, 1.5), (300.0, 2.0), (80.0, 5.0), (-100.0, 10.0)])
    def test_the_analytic_EPE_holds_RELATIVE_accuracy_at_EXTREME_ratios(self, mean: float, stdev: float) -> None:
        """Where the profile actually lives: `m/s` reaches 66 at the short end, and at `m/s = -10`
        the EPE is ~1e-24, so an absolute tolerance would be blind to any relative error (T20a)."""
        assert expected_positive_exposure(mean, stdev) == pytest.approx(_epe_by_scipy(mean, stdev), rel=1e-12)

    def test_the_TWO_references_agree_where_BOTH_are_valid(self) -> None:
        """Makes the regime split explicit rather than implicit in which test calls which."""
        for mean, stdev in [(100.0, 30.0), (0.0, 25.0), (99.975, 1.5), (-40.0, 50.0)]:
            assert _epe_by_quadrature(mean, stdev) == pytest.approx(_epe_by_scipy(mean, stdev), rel=1e-12)

    def test_at_a_zero_mean_the_EPE_is_stdev_over_root_two_pi(self) -> None:
        assert expected_positive_exposure(0.0, 40.0) == pytest.approx(40.0 / math.sqrt(2.0 * math.pi), rel=1e-14)

    def test_a_DETERMINISTIC_exposure_is_its_own_positive_part(self) -> None:
        assert expected_positive_exposure(70.0, 0.0) == 70.0
        assert expected_positive_exposure(-70.0, 0.0) == 0.0

    def test_EPE_is_never_below_the_forward_nor_below_zero(self) -> None:
        for mean, stdev in [(100.0, 30.0), (-40.0, 50.0), (0.0, 10.0)]:
            epe = expected_positive_exposure(mean, stdev)
            assert epe >= max(mean, 0.0) - 1e-12
            assert epe >= 0.0


class TestTheCvaIntegralAgainstAnIndependentQuadrature:
    def test_the_CVA_matches_scipy_quadrature_of_an_independently_built_integrand(self) -> None:
        """**The band, and what a green here rules out (§4).** The gap is the trapezoid's own
        discretisation error, not noise: measured 9.4e-05 relative at 8001 points, so the 2e-04
        band carries ~2x headroom and a green rules out any systematic defect above ~0.02% of the
        adjustment. The order test below is what pins the error as discretisation rather than a
        wrong integrand — a tolerance alone cannot tell those apart.
        """
        assert unilateral_cva(_profile(8001), SURVIVAL, DISCOUNT, LGD) == pytest.approx(
            _reference_cva(), rel=2e-4
        )

    def test_the_trapezoid_converges_at_the_SECOND_ORDER_rate(self) -> None:
        """A tolerance says the two numbers are close; the RATE says they are the same integral.

        Halving `h` must quarter the error for a trapezoid. A wrong integrand — a dropped factor, a
        mis-indexed hazard — converges to the WRONG limit, so its error ratio flattens towards 1
        instead of holding at 4. This tranche needed the distinction: a broken test reference
        plateaued the error at 3e-04 and only the failure to converge revealed it.
        """
        reference = _reference_cva()
        errors = [abs(unilateral_cva(_profile(n), SURVIVAL, DISCOUNT, LGD) - reference) for n in (501, 2001, 8001)]
        ratios = [a / b for a, b in zip(errors, errors[1:])]
        assert all(3.8 < r < 4.2 for r in ratios), f"second-order convergence required, got {ratios}"


class TestTheRequirementOraclesCrepeyForces:
    def test_a_ZERO_loss_given_default_gives_a_ZERO_CVA(self) -> None:
        assert unilateral_cva(_profile(), SURVIVAL, DISCOUNT, 0.0) == 0.0

    def test_a_COUNTERPARTY_THAT_NEVER_DEFAULTS_gives_a_ZERO_CVA(self) -> None:
        certain = SurvivalCurve(TM, (0.0,), (0.0,))
        assert unilateral_cva(_profile(), certain, DISCOUNT, LGD) == 0.0

    def test_the_CVA_is_never_negative(self) -> None:
        assert unilateral_cva(_profile(), SURVIVAL, DISCOUNT, LGD) > 0.0

    def test_the_CVA_increases_with_the_hazard_level(self) -> None:
        worse = SurvivalCurve(TM, SURVIVAL.times, tuple(h + 0.02 for h in SURVIVAL.hazards))
        assert unilateral_cva(_profile(), worse, DISCOUNT, LGD) > unilateral_cva(_profile(), SURVIVAL, DISCOUNT, LGD)

    def test_the_CVA_increases_with_the_loss_given_default(self) -> None:
        assert unilateral_cva(_profile(), SURVIVAL, DISCOUNT, 0.9) > unilateral_cva(_profile(), SURVIVAL, DISCOUNT, 0.4)

    def test_the_CVA_is_LINEAR_in_the_loss_given_default(self) -> None:
        half = unilateral_cva(_profile(), SURVIVAL, DISCOUNT, 0.5)
        assert unilateral_cva(_profile(), SURVIVAL, DISCOUNT, 1.0) == pytest.approx(2.0 * half, rel=1e-13)

    def test_the_CVA_is_SANDWICHED_by_the_undiscounted_worst_exposure(self) -> None:
        """`CVA ≤ LGD · max EPE · PD(T)` — every factor bounded above, discounting dropped."""
        profile = _profile()
        bound = LGD * max(profile.epe) * (1.0 - SURVIVAL.survival(HORIZON))
        assert 0.0 < unilateral_cva(profile, SURVIVAL, DISCOUNT, LGD) <= bound


class TestTheOnePeriodStepIsReachableAndPinned:
    """§4: extract the buried step into named surface so a test can reach it."""

    def test_the_one_period_CVA_is_BIT_IDENTICAL_to_a_hand_computation(self) -> None:
        # LGD 0.6, DF 0.95, EPE 40.0, PD 0.02 — multiplied in the documented order.
        assert unilateral_cva_one_period(40.0, 0.95, 0.02, 0.6) == 0.6 * 0.95 * 40.0 * 0.02

    def test_the_integral_REDUCES_to_the_one_period_form_on_a_single_short_period(self) -> None:
        flat = SurvivalCurve(TM, (0.0,), (0.02,))
        start, end = 0.0, 0.05
        times = tuple(start + (end - start) * k / 400 for k in range(401))
        epe = tuple(50.0 for _ in times)
        profile = ExposureProfile(times, epe)
        integral = unilateral_cva(profile, flat, DISCOUNT, LGD)
        one_period = unilateral_cva_one_period(
            50.0,
            math.exp(-DISCOUNT.zero_rate(end / 2) * end / 2),
            flat.survival(start) - flat.survival(end),
            LGD,
        )
        assert integral == pytest.approx(one_period, rel=1e-3)


class TestTheGuardsAreReachable:
    def test_a_profile_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            ExposureProfile((0.0, 1.0), (10.0,))

    def test_a_single_point_profile_cannot_be_integrated(self) -> None:
        with pytest.raises(ValueError, match="two"):
            unilateral_cva(ExposureProfile((0.0,), (10.0,)), SURVIVAL, DISCOUNT, LGD)

    def test_two_curves_on_DIFFERENT_clocks_are_refused(self) -> None:
        """C12. The profile's `t` is read on both curves, so one `t` must mean one date."""
        elsewhere = SurvivalCurve(TimeMeasure(date(2027, 6, 1), DC), SURVIVAL.times, SURVIVAL.hazards)
        with pytest.raises(ValueError, match="share one time measure"):
            unilateral_cva(_profile(), elsewhere, DISCOUNT, LGD)

    def test_a_loss_given_default_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="loss given default"):
            unilateral_cva(_profile(), SURVIVAL, DISCOUNT, 1.5)

    def test_a_NEGATIVE_exposure_standard_deviation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            expected_positive_exposure(10.0, -1.0)
