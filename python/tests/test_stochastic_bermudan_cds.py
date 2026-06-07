"""Tests for pricebook.credit.stochastic_bermudan_cds."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.credit.stochastic_bermudan_cds import (
    cir_cds_pv,
    stochastic_bermudan_cds_swaption,
    stochastic_vs_deterministic,
)
from tests.conftest import make_flat_curve

REF = date(2024, 1, 15)
EXERCISE_DATES = [
    REF + relativedelta(years=1),
    REF + relativedelta(years=2),
    REF + relativedelta(years=3),
]
CDS_MATURITY = 5.0
STRIKE = 0.01         # 100 bps
HAZARD_RATE_0 = 0.02
KAPPA = 0.3
THETA = 0.02
SIGMA = 0.1
RECOVERY = 0.4
N_PATHS = 3000
SEED = 42

CURVE = make_flat_curve(REF, 0.04)


class TestCIRCDSPV:
    def test_returns_finite_float(self):
        pv = cir_cds_pv(
            hazard_rate=HAZARD_RATE_0,
            mean_reversion=KAPPA,
            long_run_hazard=THETA,
            hazard_vol=SIGMA,
            remaining_years=CDS_MATURITY,
            strike_spread=STRIKE,
            discount_rate=0.04,
            recovery=RECOVERY,
            n_coupons=20,
        )
        assert isinstance(float(pv), float)
        assert math.isfinite(float(pv))

    def test_itm_payer_positive(self):
        """Payer CDS with high hazard rate above strike spread should be +ve PV."""
        pv = cir_cds_pv(
            hazard_rate=0.05,           # hazard >> strike spread
            mean_reversion=KAPPA,
            long_run_hazard=0.05,
            hazard_vol=SIGMA,
            remaining_years=CDS_MATURITY,
            strike_spread=0.005,        # 50 bps strike
            discount_rate=0.04,
            recovery=RECOVERY,
            n_coupons=20,
        )
        assert float(pv) > 0


class TestStochasticBermudanCDSSwaption:
    def test_price_positive(self):
        res = stochastic_bermudan_cds_swaption(
            reference_date=REF,
            exercise_dates=EXERCISE_DATES,
            cds_maturity_years=CDS_MATURITY,
            strike_spread=STRIKE,
            discount_curve=CURVE,
            hazard_rate_0=HAZARD_RATE_0,
            mean_reversion=KAPPA,
            long_run_hazard=THETA,
            hazard_vol=SIGMA,
            recovery=RECOVERY,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.price > 0

    def test_spread_vol_contribution_nonneg(self):
        res = stochastic_bermudan_cds_swaption(
            reference_date=REF,
            exercise_dates=EXERCISE_DATES,
            cds_maturity_years=CDS_MATURITY,
            strike_spread=STRIKE,
            discount_curve=CURVE,
            hazard_rate_0=HAZARD_RATE_0,
            mean_reversion=KAPPA,
            long_run_hazard=THETA,
            hazard_vol=SIGMA,
            recovery=RECOVERY,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.spread_vol_contribution >= -1e-3  # allow small MC noise

    def test_bermudan_ge_european(self):
        res = stochastic_bermudan_cds_swaption(
            reference_date=REF,
            exercise_dates=EXERCISE_DATES,
            cds_maturity_years=CDS_MATURITY,
            strike_spread=STRIKE,
            discount_curve=CURVE,
            hazard_rate_0=HAZARD_RATE_0,
            mean_reversion=KAPPA,
            long_run_hazard=THETA,
            hazard_vol=SIGMA,
            recovery=RECOVERY,
            n_paths=N_PATHS,
            seed=SEED,
        )
        assert res.price >= res.european_price - 1e-6


class TestStochasticVsDeterministic:
    def test_returns_expected_keys(self):
        result = stochastic_vs_deterministic(
            reference_date=REF,
            exercise_dates=EXERCISE_DATES,
            cds_maturity_years=CDS_MATURITY,
            strike_spread=STRIKE,
            discount_curve=CURVE,
            hazard_rate_0=HAZARD_RATE_0,
            mean_reversion=KAPPA,
            long_run_hazard=THETA,
            hazard_vol=SIGMA,
            recovery=RECOVERY,
        )
        for key in ("stochastic_price", "deterministic_price", "difference",
                    "spread_vol_contribution"):
            assert key in result

    def test_stochastic_ge_deterministic(self):
        """Stochastic hazard vol should add value (optionality) vs deterministic."""
        result = stochastic_vs_deterministic(
            reference_date=REF,
            exercise_dates=EXERCISE_DATES,
            cds_maturity_years=CDS_MATURITY,
            strike_spread=STRIKE,
            discount_curve=CURVE,
            hazard_rate_0=HAZARD_RATE_0,
            mean_reversion=KAPPA,
            long_run_hazard=THETA,
            hazard_vol=SIGMA,
            recovery=RECOVERY,
        )
        # Allow small MC noise tolerance
        assert result["stochastic_price"] >= result["deterministic_price"] - 1e-4
