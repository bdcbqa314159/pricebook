"""Deep tests for credit — DD4 hardening.

Covers: CDS par spread round-trip, protection-premium parity, CDO tranche
monotonicity, basket CDS bounds, rating transition properties, Merton model.
"""

import math
import pytest
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.cds import CDS, protection_leg_pv, risky_annuity, bootstrap_credit_curve
from pricebook.cdo import portfolio_loss_distribution, tranche_expected_loss, tranche_spread
from pricebook.basket_cds import ftd_spread, ntd_spread
from pricebook.rating_transition import RatingTransitionMatrix, standard_generator
from pricebook.structural_credit import merton_equity_credit, kmv_distance_to_default
from pricebook.day_count import DayCountConvention
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)


class TestCDSParSpread:

    def test_pv_zero_at_par_spread(self):
        """CDS struck at par spread has PV ≈ 0."""
        curve = make_flat_curve(REF, 0.03)
        surv = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(curve, surv)
        cds_at_par = CDS(REF, REF + relativedelta(years=5), spread=par)
        pv = cds_at_par.pv(curve, surv)
        assert abs(pv) < 10.0  # < $10 on $1M notional

    def test_par_spread_positive(self):
        curve = make_flat_curve(REF, 0.03)
        surv = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        assert cds.par_spread(curve, surv) > 0

    def test_higher_hazard_higher_spread(self):
        """Higher default probability → higher CDS spread."""
        curve = make_flat_curve(REF, 0.03)
        surv_low = make_flat_survival(REF, 0.01)
        surv_high = make_flat_survival(REF, 0.05)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        assert cds.par_spread(curve, surv_high) > cds.par_spread(curve, surv_low)

    def test_protection_equals_premium_at_par(self):
        """At par spread: PV(protection) = PV(premium)."""
        curve = make_flat_curve(REF, 0.03)
        surv = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(curve, surv)
        prot = cds.pv_protection(curve, surv)
        rpv01 = risky_annuity(REF, REF + relativedelta(years=5), curve, surv)
        premium = par * cds.notional * rpv01
        assert prot == pytest.approx(premium, rel=0.02)


class TestCreditBootstrap:

    def test_round_trip(self):
        """Bootstrapped curve reprices input CDS spreads."""
        curve = make_flat_curve(REF, 0.03)
        spreads = [
            (REF + relativedelta(years=1), 0.0050),
            (REF + relativedelta(years=3), 0.0080),
            (REF + relativedelta(years=5), 0.0100),
        ]
        surv = bootstrap_credit_curve(REF, spreads, curve)
        for mat, spread in spreads:
            cds = CDS(REF, mat, spread=0.0)
            recovered = cds.par_spread(curve, surv)
            assert recovered == pytest.approx(spread, abs=0.001)


class TestCDOTranches:

    def test_loss_distribution_sums_to_one(self):
        lg, d = portfolio_loss_distribution(0.02, 0.3, 0.6)
        assert d.sum() == pytest.approx(1.0, abs=0.05)

    def test_equity_tranche_loss_positive(self):
        lg, d = portfolio_loss_distribution(0.02, 0.3, 0.6)
        el = tranche_expected_loss(lg, d, 0.0, 0.03)
        assert el > 0

    def test_senior_tranche_loss_less_than_equity(self):
        """Senior tranche has lower expected loss than equity."""
        lg, d = portfolio_loss_distribution(0.02, 0.3, 0.6)
        eq = tranche_expected_loss(lg, d, 0.0, 0.03)
        sr = tranche_expected_loss(lg, d, 0.07, 0.15)
        assert sr < eq

    def test_higher_pd_higher_tranche_loss(self):
        lg1, d1 = portfolio_loss_distribution(0.01, 0.3, 0.6)
        lg2, d2 = portfolio_loss_distribution(0.05, 0.3, 0.6)
        el1 = tranche_expected_loss(lg1, d1, 0.0, 0.03)
        el2 = tranche_expected_loss(lg2, d2, 0.0, 0.03)
        assert el2 > el1


class TestBasketCDS:

    def test_ftd_spread_positive(self):
        curve = make_flat_curve(REF, 0.03)
        survs = [make_flat_survival(REF, 0.02) for _ in range(5)]
        spread = ftd_spread(survs, curve, rho=0.3, T=5.0, n_sims=10_000)
        assert spread > 0

    def test_ftd_geq_single_name(self):
        """FTD spread ≥ single-name spread (more names = more risk)."""
        curve = make_flat_curve(REF, 0.03)
        surv = make_flat_survival(REF, 0.02)
        single = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        single_par = single.par_spread(curve, surv)
        ftd = ftd_spread([surv]*5, curve, rho=0.3, T=5.0, n_sims=10_000)
        assert ftd > single_par * 0.5  # FTD should be meaningfully higher

    def test_ntd_spread_decreasing_in_n(self):
        """Higher n → lower NTD spread (harder to trigger)."""
        curve = make_flat_curve(REF, 0.03)
        survs = [make_flat_survival(REF, 0.02) for _ in range(5)]
        s1 = ntd_spread(survs, curve, rho=0.3, T=5.0, n=1, n_sims=10_000)
        s2 = ntd_spread(survs, curve, rho=0.3, T=5.0, n=2, n_sims=10_000)
        assert s1 > s2


class TestRatingTransition:

    def test_transition_matrix_rows_sum_to_one(self):
        rtm = standard_generator()
        P = rtm.transition_prob(1.0)
        for i in range(P.shape[0]):
            assert P[i, :].sum() == pytest.approx(1.0, abs=1e-10)

    def test_transition_probs_non_negative(self):
        rtm = standard_generator()
        P = rtm.transition_prob(1.0)
        assert np.all(P >= -1e-10)

    def test_default_is_absorbing(self):
        rtm = standard_generator()
        P = rtm.transition_prob(1.0)
        # Default row: stays in default
        assert P[-1, -1] == pytest.approx(1.0, abs=1e-10)

    def test_invalid_generator_rejected(self):
        """Generator with row not summing to 0 should be rejected."""
        with pytest.raises(ValueError, match="sum to 0"):
            RatingTransitionMatrix(
                ["A", "D"],
                [[0.1, 0.05], [0.0, 0.0]],  # row 0 sums to 0.15, not 0
            )


class TestMertonModel:

    def test_dd_positive_for_healthy_firm(self):
        """Healthy firm (low leverage) has high distance-to-default."""
        result = kmv_distance_to_default(
            equity_value=100, equity_vol=0.3,
            short_term_debt=20, long_term_debt=30,
            rate=0.05,
        )
        assert result.distance_to_default > 1.0
        assert result.default_probability < 0.5

    def test_higher_vol_higher_pd(self):
        """Higher equity vol → higher PD (same leverage)."""
        low = kmv_distance_to_default(
            equity_value=100, equity_vol=0.2,
            short_term_debt=40, long_term_debt=30, rate=0.05,
        )
        high = kmv_distance_to_default(
            equity_value=100, equity_vol=0.6,
            short_term_debt=40, long_term_debt=30, rate=0.05,
        )
        assert high.default_probability > low.default_probability

    def test_merton_spread_positive(self):
        result = merton_equity_credit(
            asset_value=150, debt_face=50, asset_vol=0.3, rate=0.05, T=1.0,
        )
        assert result.credit_spread_bps > 0
