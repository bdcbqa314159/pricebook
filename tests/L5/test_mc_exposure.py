"""The Monte-Carlo exposure engine (L5) — tranche 26, the XVA spine's second unit.

T25 priced CVA from an exposure given in closed form. This prices it from an exposure produced by
**simulation**, which means T25's closed form stops being the whole oracle and becomes a *reduction
check*. Hence **two legs**, ruled on PR #34's review (thread 0127):

* **Leg A — the reduction to T25.** Feed the estimator a Gaussian mark `V(t) = V0 + sigma*W(t)`,
  whose `V(t) ~ N(V0, sigma^2 t)` is exactly T25's Bachelier case, and watch the simulated EPE
  converge to the closed form at the MC rate. This validates the **estimator** — the positive part,
  the averaging, the SE. It is **self-consistency**: both sides share the exposure convention, so
  §4's "self-consistency telescopes" applies and it cannot be the whole oracle.
* **Leg B — the martingale identity.** Under the terminal measure every traded value over the
  numeraire is a martingale, so `E[V(t)/B(t,T_end)] = V(0)/B(0,T_end)` **exactly, by construction —
  no model result and no T25**. This validates the **process**: a mis-specified drift or measure is
  invisible to Leg A (which only checks the `+` transform against a closed form computed on the same
  paths) and fatal here. It is the T19 free-control identity reused as an exposure oracle.

**The swap is observed only BEFORE its first reset, and that is deliberate, not a convenience.** The
martingale identity holds for a *fixed* claim; a swap that has already paid a coupon is not the same
claim it was at inception, so `E[V_remaining(t)/B_N(t)] = V(0)/B_N(0)` would be false and the oracle
would be asserting the wrong thing. Observing a forward-starting swap over `(0, T_first]` keeps the
claim fixed, so Leg B is **exact** rather than approximate. The exposure is still lognormal and
non-Gaussian, which is the whole point of the leg.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine.mc_path_engine import ObservationSchedule, simulate_forward_paths
from pricebook.foundation import Currency
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.results import PricingResult
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor
from pricebook.models.swap_rate_vol import forward_swap_rate, swap_annuity, tenor_discount_factors
from pricebook.products.swaption import SwapSpan
from pricebook.risk.credit_valuation_adjustment import expected_positive_exposure, unilateral_cva
from pricebook.risk.mc_exposure import (
    SimulatedExposure,
    live_exposure_times,
    mc_exposure_cva,
    swap_values_on_paths,
)

ANCHOR = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(ANCHOR, DC)
EUR = Currency("EUR")

SURVIVAL = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.010, 0.030, 0.055))
DISCOUNT = DiscountCurve(TM, (0.0, 5.0), (0.020, 0.035))
LGD = 0.6

# ── Leg A's fixture: an arithmetic Brownian mark, exactly T25's Bachelier case ──────────────────
V0, SIGMA = 40.0, 25.0
GRID = (0.25, 0.75, 1.5, 2.5)


def _brownian_exposure(paths: int, seed: int = 7) -> SimulatedExposure:
    """`V(t) = V0 + sigma*W(t)` on `GRID`, drawn by numpy — an independent generator, test-side."""
    rng = np.random.default_rng(seed)
    dt = np.diff((0.0,) + GRID)
    steps = rng.standard_normal((paths, len(GRID))) * np.sqrt(dt)
    values = V0 + SIGMA * np.cumsum(steps, axis=1)
    return SimulatedExposure(GRID, tuple(tuple(row) for row in values))


def _closed_form_epe(t: float) -> float:
    return expected_positive_exposure(V0, SIGMA * math.sqrt(t))


# ── Leg B's fixture: a forward-starting swap under the LMM ──────────────────────────────────────
TENOR = LmmTenor(
    accruals=(0.50, 0.25, 0.50, 0.75, 0.50, 0.25),
    forwards=(0.030, 0.034, 0.038, 0.042, 0.045, 0.047),
    volatilities=(0.22, 0.20, 0.18, 0.16, 0.15, 0.14),
)
TENOR_TIMES = tuple(math.fsum(TENOR.accruals[:i]) for i in range(len(TENOR) + 1))
RHO = LmmCorrelation(
    tuple(tuple(math.exp(-0.35 * abs(TENOR_TIMES[i] - TENOR_TIMES[j])) for j in range(6)) for i in range(6))
)
SPAN = SwapSpan(3, 3)
STRIKE = 0.040
OBSERVED = (1, 2, 3)  # tenor indices, all at or before T_first = T_3: the claim stays fixed
MODEL = LiborMarketModel(TENOR, RHO, numeraire_index=SPAN.first + SPAN.count)
CONFIG = MonteCarloConfig(paths=6_000, seed=20261108)


def _paths(config: MonteCarloConfig = CONFIG):
    schedule = ObservationSchedule(tuple(TENOR_TIMES[i] for i in OBSERVED), steps_per_segment=4)
    return simulate_forward_paths(MODEL, schedule, config)


def _initial_numeraire_relative_value() -> float:
    """`A(0)/B(0,T_end) * (S(0) - K)` — the martingale's time-0 level, from T25-independent surface."""
    bonds = tenor_discount_factors(TENOR)
    annuity = swap_annuity(TENOR, SPAN.first, SPAN.count)
    rate = forward_swap_rate(TENOR, SPAN.first, SPAN.count)
    return annuity / bonds[SPAN.first + SPAN.count] * (rate - STRIKE)


class TestTheFixturesAreNotDegenerate:
    def test_the_exposure_has_NON_ZERO_variance(self) -> None:
        """Else EPE = the forward and the positive part is invisible."""
        values = np.asarray(_brownian_exposure(2_000).values)
        assert values.std(axis=0).min() > 1.0

    def test_the_grid_has_MORE_THAN_ONE_future_date(self) -> None:
        """Else a single-date reduction hides the profile's time structure."""
        assert len(GRID) > 1 and len(OBSERVED) > 1

    def test_the_numeraire_is_NOT_flat_across_the_observed_dates(self) -> None:
        """Else `B(t,T_end)/B(0,T_end) = 1` and a dropped discounting survives Leg B."""
        bonds = tenor_discount_factors(TENOR)
        ratios = [bonds[SPAN.first + SPAN.count] / bonds[i] for i in OBSERVED]
        assert max(ratios) - min(ratios) > 0.01, f"numeraire too flat: {ratios}"

    def test_the_swap_exposure_is_NOT_Gaussian(self) -> None:
        """Leg A's closed form cannot reach it — which is why Leg B exists."""
        values = np.asarray(swap_values_on_paths(_paths(), TENOR, SPAN, STRIKE, OBSERVED).value)
        last = values[:, -1]
        skew = float(((last - last.mean()) ** 3).mean() / last.std() ** 3)
        assert abs(skew) > 0.05, f"a lognormal-driven swap value should be skewed, got {skew}"


class TestLegATheReductionToT25:
    """Validates the ESTIMATOR against a known closed form. Telescoping, hence necessary-not-whole."""

    def test_the_simulated_EPE_matches_T25s_closed_form_within_its_SE(self) -> None:
        estimate = _brownian_exposure(40_000).estimate()
        for t, epe in zip(estimate.times, estimate.epe):
            reference = _closed_form_epe(t)
            assert abs(epe.mean - reference) <= 4.0 * epe.standard_error, (
                f"t={t}: {epe.mean} vs {reference}, SE {epe.standard_error}"
            )

    def test_the_EPE_error_falls_at_the_MONTE_CARLO_rate(self) -> None:
        """16x the paths must roughly quarter the error — the `-1/2` rate, averaged over the grid
        because a single date's error is one noisy draw (T21: a null needs averaging too)."""
        def rms_error(paths: int) -> float:
            estimate = _brownian_exposure(paths, seed=11).estimate()
            return math.sqrt(
                math.fsum((e.mean - _closed_form_epe(t)) ** 2 for t, e in zip(estimate.times, estimate.epe))
                / len(estimate.times)
            )
        coarse, fine = rms_error(2_500), rms_error(40_000)
        assert fine < coarse / 2.0, f"expected ~4x improvement, got {coarse} -> {fine}"

    def test_the_MC_CVA_matches_T25s_CLOSED_FORM_CVA_on_the_Gaussian_mark(self) -> None:
        """A cross-product oracle: different exposure code each side, the same CVA integral."""
        exposure = _brownian_exposure(40_000)
        simulated = mc_exposure_cva(exposure, SURVIVAL, DISCOUNT, LGD, EUR)
        analytic = unilateral_cva(
            type(exposure.estimate().profile())(GRID, tuple(_closed_form_epe(t) for t in GRID)),
            SURVIVAL,
            DISCOUNT,
            LGD,
        )
        assert isinstance(simulated, PricingResult)
        assert abs(simulated.pv.amount - analytic) <= 4.0 * simulated.standard_error


class TestLegBTheMartingaleIdentity:
    """Validates the PROCESS. Consults T25 nowhere; exact by construction, not by model result."""

    def test_the_numeraire_relative_swap_value_is_a_MARTINGALE_at_every_date(self) -> None:
        values = np.asarray(swap_values_on_paths(_paths(), TENOR, SPAN, STRIKE, OBSERVED).numeraire_relative)
        expected = _initial_numeraire_relative_value()
        for column, index in zip(values.T, OBSERVED):
            mean = float(column.mean())
            se = float(column.std(ddof=1) / math.sqrt(len(column)))
            assert abs(mean - expected) <= 4.0 * se, (
                f"tenor date {index}: E[V/B_N] = {mean} +/- {se}, expected {expected}"
            )

    def test_the_martingale_error_IMPROVES_with_more_paths(self) -> None:
        expected = _initial_numeraire_relative_value()

        def error(paths: int) -> float:
            values = np.asarray(
                swap_values_on_paths(
                    _paths(MonteCarloConfig(paths=paths, seed=4242)), TENOR, SPAN, STRIKE, OBSERVED
                ).numeraire_relative
            )
            return abs(float(values[:, -1].mean()) - expected)

        assert error(24_000) < error(1_500)


class TestInvariantSevenTheEstimateCarriesItsError:
    def test_the_MC_CVA_returns_a_PricingResult_carrying_a_POSITIVE_standard_error(self) -> None:
        result = mc_exposure_cva(_brownian_exposure(5_000), SURVIVAL, DISCOUNT, LGD, EUR)
        assert result.standard_error is not None and result.standard_error > 0.0

    def test_the_standard_error_FALLS_as_root_N(self) -> None:
        coarse = mc_exposure_cva(_brownian_exposure(2_500), SURVIVAL, DISCOUNT, LGD, EUR)
        fine = mc_exposure_cva(_brownian_exposure(40_000), SURVIVAL, DISCOUNT, LGD, EUR)
        ratio = coarse.standard_error / fine.standard_error
        assert 2.5 < ratio < 5.5, f"expected ~4x, got {ratio}"

    def test_T25s_CLOSED_FORM_cva_stays_exact(self) -> None:
        """Invariant 7 rides where it is meaningful and is absent where it is not."""
        from pricebook.risk.credit_valuation_adjustment import ExposureProfile

        profile = ExposureProfile(GRID, tuple(_closed_form_epe(t) for t in GRID))
        assert isinstance(unilateral_cva(profile, SURVIVAL, DISCOUNT, LGD), float)


class TestTheRequirementOraclesCarryOver:
    def test_a_ZERO_loss_given_default_gives_a_ZERO_CVA(self) -> None:
        assert mc_exposure_cva(_brownian_exposure(2_000), SURVIVAL, DISCOUNT, 0.0, EUR).pv.amount == 0.0

    def test_a_COUNTERPARTY_THAT_NEVER_DEFAULTS_gives_a_ZERO_CVA(self) -> None:
        certain = SurvivalCurve(TM, (0.0,), (0.0,))
        assert mc_exposure_cva(_brownian_exposure(2_000), certain, DISCOUNT, LGD, EUR).pv.amount == 0.0

    def test_the_CVA_is_positive_and_increases_with_the_hazard_and_with_LGD(self) -> None:
        exposure = _brownian_exposure(5_000)
        base = mc_exposure_cva(exposure, SURVIVAL, DISCOUNT, LGD, EUR).pv.amount
        worse = SurvivalCurve(TM, SURVIVAL.times, tuple(h + 0.03 for h in SURVIVAL.hazards))
        assert base > 0.0
        assert mc_exposure_cva(exposure, worse, DISCOUNT, LGD, EUR).pv.amount > base
        assert mc_exposure_cva(exposure, SURVIVAL, DISCOUNT, 0.9, EUR).pv.amount > base


class TestTheHistoricalBoundaryIsReadThroughTheAtom:
    """C10's consumer: the exposure grid runs over FUTURE dates only."""

    def test_an_exposure_date_EXACTLY_on_the_valuation_date_is_excluded(self) -> None:
        dates = (ANCHOR, ANCHOR + timedelta(days=90), ANCHOR + timedelta(days=365))
        times = live_exposure_times(dates, ANCHOR, TM)
        assert len(times) == 2 and all(t > 0.0 for t in times)

    def test_a_PAST_exposure_date_is_excluded(self) -> None:
        dates = (ANCHOR - timedelta(days=30), ANCHOR + timedelta(days=90))
        assert live_exposure_times(dates, ANCHOR, TM) == (TM.year_fraction(ANCHOR + timedelta(days=90)),)

    def test_an_ALL_HISTORICAL_grid_leaves_nothing_to_integrate(self) -> None:
        assert live_exposure_times((ANCHOR - timedelta(days=1), ANCHOR), ANCHOR, TM) == ()


class TestTheGuardsAreReachable:
    def test_a_ragged_value_matrix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            SimulatedExposure((0.5, 1.0), ((1.0, 2.0), (3.0,)))

    def test_a_single_path_cannot_carry_a_standard_error(self) -> None:
        with pytest.raises(ValueError, match="two"):
            SimulatedExposure((0.5,), ((1.0,),)).estimate()

    def test_an_observed_index_beyond_the_span_start_is_refused(self) -> None:
        """Past `T_first` the swap has paid, the claim changes, and Leg B would be false."""
        with pytest.raises(ValueError, match="before the span"):
            swap_values_on_paths(_paths(), TENOR, SPAN, STRIKE, (1, 2, 5))
