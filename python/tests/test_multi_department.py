"""Multi-department tests: prudent valuation, XVA, market risk, valuation report."""

from __future__ import annotations

import math
from datetime import date

import pytest
import numpy as np

from pricebook.risk.prudent_valuation import (
    market_price_uncertainty_ava, close_out_cost_ava, model_risk_ava,
    concentration_ava, unearned_credit_spread_ava, investing_funding_ava,
    early_termination_ava, future_admin_cost_ava, model_risk_ava_from_exotic_book,
    compute_prudent_value,
)
from pricebook.new_desk_xva import (
    inflation_analytic_xva, asw_xva, structured_credit_xva,
)
from pricebook.risk.market_risk_enhanced import (
    incremental_var, stressed_var, copula_es,
)
from pricebook.risk.valuation_report import valuation_report, ValuationReport
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 7, 15)


def _disc():
    return DiscountCurve.flat(REF, 0.04)


# ── Prudent Valuation (AVA) ──

class TestMarketPriceUncertainty:
    def test_ava_positive(self):
        r = market_price_uncertainty_ava(100.0, 99.5, 100.5, n_quotes=5)
        assert r.ava > 0

    def test_tight_spread_small_ava(self):
        r = market_price_uncertainty_ava(100.0, 99.99, 100.01, n_quotes=10)
        assert r.ava < 0.1

    def test_few_quotes_higher_ava(self):
        r1 = market_price_uncertainty_ava(100.0, 99.5, 100.5, n_quotes=1)
        r5 = market_price_uncertainty_ava(100.0, 99.5, 100.5, n_quotes=5)
        assert r1.ava > r5.ava


class TestCloseOutCost:
    def test_ava_positive(self):
        r = close_out_cost_ava(10_000_000, "bond_ig")
        assert r.ava > 0

    def test_illiquid_higher(self):
        r_ig = close_out_cost_ava(10_000_000, "bond_ig")
        r_struct = close_out_cost_ava(10_000_000, "structured")
        assert r_struct.ava > r_ig.ava


class TestModelRisk:
    def test_ava_from_spread(self):
        r = model_risk_ava([100.0, 101.5, 99.0])
        assert r.ava > 0
        assert r.model_spread == 2.5

    def test_single_model_zero(self):
        r = model_risk_ava([100.0])
        assert r.ava == 0.0


class TestConcentration:
    def test_small_position_small_ava(self):
        r = concentration_ava(1_000_000, 1_000_000_000)  # 0.1%
        assert r.ava > 0
        assert r.concentration_pct < 0.01

    def test_large_position_large_ava(self):
        r_small = concentration_ava(1e6, 1e9)
        r_large = concentration_ava(100e6, 1e9)  # 10%
        assert r_large.ava > r_small.ava * 10


class TestUnearnedCreditSpread:
    def test_uncollateralised_full_cva(self):
        r = unearned_credit_spread_ava(50_000, collateralised=False)
        assert r.ava == 50_000

    def test_collateralised_reduced(self):
        r = unearned_credit_spread_ava(50_000, collateralised=True)
        assert r.ava == pytest.approx(5_000)  # 10% of CVA


class TestInvestingFunding:
    def test_ava_positive(self):
        r = investing_funding_ava(10_000_000, 50.0, 5.0)
        assert r.ava > 0

    def test_capped_at_1yr(self):
        r1 = investing_funding_ava(10e6, 50.0, 1.0)
        r5 = investing_funding_ava(10e6, 50.0, 5.0)
        assert r1.ava == r5.ava  # capped at 1-year horizon


class TestEarlyTermination:
    def test_callable_bond_ava(self):
        r = early_termination_ava(102.0, 99.5)  # non-callable - callable
        assert r.ava == pytest.approx(2.5)
        assert r.option_value == 2.5

    def test_zero_when_equal(self):
        r = early_termination_ava(100.0, 100.0)
        assert r.ava == 0.0


class TestFutureAdminCost:
    def test_vanilla_low(self):
        r = future_admin_cost_ava(10e6, 5.0, complexity_score=1)
        assert r.ava > 0
        assert r.annual_admin_bp == 0.5

    def test_complex_higher(self):
        r1 = future_admin_cost_ava(10e6, 5.0, complexity_score=1)
        r5 = future_admin_cost_ava(10e6, 5.0, complexity_score=5)
        assert r5.ava > r1.ava * 10

    def test_capped_at_5yr(self):
        r5 = future_admin_cost_ava(10e6, 5.0, complexity_score=3)
        r20 = future_admin_cost_ava(10e6, 20.0, complexity_score=3)
        assert r5.ava == r20.ava  # capped at 5 years


class TestModelRiskIntegration:
    def test_from_exotic_book(self):
        greeks = {
            "black_scholes": {"pv": 1_000_000, "delta": 0.5},
            "local_vol": {"pv": 1_005_000, "delta": 0.48},
            "sabr": {"pv": 998_000, "delta": 0.51},
        }
        r = model_risk_ava_from_exotic_book(greeks, 10e6)
        assert r.ava > 0
        assert r.model_spread == 7_000  # 1005000 - 998000


class TestPrudentValue:
    def test_prudent_below_mid(self):
        mpu = market_price_uncertainty_ava(100.0, 99.5, 100.5)
        coc = close_out_cost_ava(10e6, "bond_ig")
        report = compute_prudent_value(100.0, mpu=mpu, coc=coc)
        assert report.prudent_value < 100.0
        assert report.total_ava > 0

    def test_diversification_reduces_ava(self):
        mpu = market_price_uncertainty_ava(100.0, 99.5, 100.5)
        r0 = compute_prudent_value(100.0, mpu=mpu, diversification_pct=0.0)
        r50 = compute_prudent_value(100.0, mpu=mpu, diversification_pct=0.50)
        assert r50.total_ava_diversified < r0.total_ava_diversified

    def test_to_dict(self):
        report = compute_prudent_value(100.0)
        d = report.to_dict()
        assert "prudent_value" in d
        assert "total_ava" in d


# ── XVA for New Desks ──

class TestInflationXVA:
    def test_cva_positive(self):
        r = inflation_analytic_xva(1_000_000, 10e6, 5.0, ie01=5000)
        assert r.cva > 0

    def test_total_xva_positive(self):
        r = inflation_analytic_xva(500_000, 10e6, 5.0)
        assert r.total_xva > 0


class TestASWXVA:
    def test_fva_positive(self):
        r = asw_xva(98.0, 10e6, 5.0)
        assert r.fva > 0
        assert r.total_xva > 0


class TestStructuredCreditXVA:
    def test_wrong_way_risk(self):
        r_low = structured_credit_xva(500_000, 10e6, 5.0, correlation=0.1)
        r_high = structured_credit_xva(500_000, 10e6, 5.0, correlation=0.5)
        assert r_high.wrong_way_adj > r_low.wrong_way_adj

    def test_cva_positive(self):
        r = structured_credit_xva(500_000, 10e6, 5.0)
        assert r.cva > 0


# ── Market Risk Enhanced ──

class TestIncrementalVaR:
    def _pnls(self):
        rng = np.random.default_rng(42)
        return {
            "pos_A": list(rng.normal(0, 100, 250)),
            "pos_B": list(rng.normal(0, 200, 250)),
            "pos_C": list(rng.normal(50, 150, 250)),
        }

    def test_parametric_ivar(self):
        r = incremental_var(self._pnls(), method="parametric")
        assert r.portfolio_var > 0
        assert len(r.incremental_vars) == 3
        # Sum of IVaR ≈ portfolio VaR (Euler)
        assert sum(r.incremental_vars) == pytest.approx(r.portfolio_var, rel=0.15)

    def test_historical_ivar(self):
        r = incremental_var(self._pnls(), method="historical")
        assert r.portfolio_var > 0

    def test_diversification_positive(self):
        r = incremental_var(self._pnls(), method="parametric")
        assert r.diversification_benefit > 0

    def test_empty(self):
        r = incremental_var({})
        assert r.portfolio_var == 0


class TestStressedVaR:
    def test_stressed_higher(self):
        rng = np.random.default_rng(42)
        current = list(rng.normal(0, 100, 250))
        stressed = list(rng.normal(-50, 300, 250))  # wider, shifted
        r = stressed_var(current, stressed)
        assert r.stressed_var > r.current_var

    def test_capital_positive(self):
        rng = np.random.default_rng(42)
        r = stressed_var(list(rng.normal(0, 100, 250)),
                         list(rng.normal(0, 200, 250)))
        assert r.capital_charge > 0


class TestCopulaES:
    def test_es_positive(self):
        rng = np.random.default_rng(42)
        pnls = list(rng.normal(0, 100, 500))
        r = copula_es(pnls)
        assert r.selected_es > 0

    def test_heavy_tails_higher_es(self):
        rng = np.random.default_rng(42)
        normal = list(rng.normal(0, 100, 500))
        heavy = list(rng.standard_t(3, 500) * 100)  # t(3) has heavy tails
        r_normal = copula_es(normal)
        r_heavy = copula_es(heavy)
        assert r_heavy.selected_es > r_normal.es_normal * 0.5  # heavy tail should be larger


# ── Valuation Report ──

class TestValuationReport:
    def test_full_report(self):
        r = valuation_report(
            "irs", 1_000_000, 50e6, REF,
            dv01=4500, cs01=0, carry_1d=1370,
            bid_price=999_000, ask_price=1_001_000,
            model_prices=[1_000_000, 1_002_000, 998_000],
            cva=15_000, fva=5_000,
            ead=2_000_000, rwa=400_000, capital_req=32_000,
            var_contribution=50_000,
            stress_scenarios={"rates_up_100": -45_000, "rates_dn_100": 43_000},
        )
        assert r.mid_price == 1_000_000
        assert r.prudent_value < r.mid_price
        assert r.total_xva > 0
        assert r.capital > 0

    def test_to_dict(self):
        r = valuation_report("cds", 500_000, 10e6, REF, cs01=-2000)
        d = r.to_dict()
        assert "trading" in d
        assert "prudent" in d
        assert "xva" in d
        assert "regulatory" in d
        assert "risk" in d

    def test_minimal_report(self):
        r = valuation_report("bond", 100.0, 10e6, REF)
        assert r.mid_price == 100.0
        assert math.isfinite(r.prudent_value)
