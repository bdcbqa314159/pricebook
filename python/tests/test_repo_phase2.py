"""Tests for repo Phase 2: CVA, dynamic haircuts, correlated XVA."""

import pytest
import math

from pricebook.risk.repo_cva import (
    repo_cva, repo_wrong_way_risk, repo_bilateral_cva, RepoCVAResult,
)
from pricebook.risk.dynamic_haircuts import (
    DynamicHaircutModel, haircut_stress_scenarios,
    credit_spread_to_haircut, rating_trigger_impact, DynamicHaircutResult,
)
from pricebook.risk.repo_xva_advanced import (
    repo_xva_correlated, repo_all_in_xva, RepoXVACorrelatedResult,
)


# ═══════════════════════════════════════════════════════════════
# 2.1: Repo CVA
# ═══════════════════════════════════════════════════════════════

class TestRepoCVA:
    def test_basic(self):
        r = repo_cva(1e6, 90, 0.05, 0.02, counterparty_hazard=0.02)
        assert isinstance(r, RepoCVAResult)
        assert r.cva >= 0

    def test_higher_hazard_higher_cva(self):
        r1 = repo_cva(1e6, 90, 0.05, 0.02, counterparty_hazard=0.01)
        r2 = repo_cva(1e6, 90, 0.05, 0.02, counterparty_hazard=0.10)
        assert r2.cva > r1.cva

    def test_higher_haircut_lower_cva(self):
        """More overcollateralisation → less exposure → lower CVA."""
        r1 = repo_cva(1e6, 90, 0.05, 0.01, counterparty_hazard=0.05)
        r2 = repo_cva(1e6, 90, 0.05, 0.10, counterparty_hazard=0.05)
        assert r2.cva <= r1.cva

    def test_longer_term_higher_cva(self):
        r30 = repo_cva(1e6, 30, 0.05, 0.02, counterparty_hazard=0.05)
        r180 = repo_cva(1e6, 180, 0.05, 0.02, counterparty_hazard=0.05)
        assert r180.cva > r30.cva

    def test_pd_reasonable(self):
        r = repo_cva(1e6, 365, 0.05, 0.02, counterparty_hazard=0.05)
        expected_pd = 1 - math.exp(-0.05)
        assert abs(r.counterparty_pd - expected_pd) < 0.001

    def test_to_dict(self):
        d = repo_cva(1e6, 90, 0.05, 0.02, counterparty_hazard=0.02).to_dict()
        assert "cva" in d
        assert "expected_exposure" in d


class TestWrongWayRisk:
    def test_issuer_channel(self):
        wwr = repo_wrong_way_risk(1000, "issuer", 0.80,
                                   collateral_hazard=0.05, counterparty_hazard=0.05)
        assert wwr > 0

    def test_sector_channel(self):
        wwr = repo_wrong_way_risk(1000, "sector", 0.50)
        assert wwr > 0

    def test_spiral_channel(self):
        wwr = repo_wrong_way_risk(1000, "spiral", 0.70)
        assert wwr > 0

    def test_zero_correlation_no_wwr(self):
        wwr = repo_wrong_way_risk(1000, "issuer", 0.0)
        assert wwr == 0.0


class TestBilateralCVA:
    def test_basic(self):
        r = repo_bilateral_cva(
            1e6, 90, 0.05, 0.02,
            counterparty_hazard=0.03, counterparty_recovery=0.40,
            own_hazard=0.01, own_recovery=0.40,
        )
        assert r.cva >= 0
        assert r.dva >= 0

    def test_with_correlation(self):
        r_zero = repo_bilateral_cva(
            1e6, 90, 0.05, 0.02, 0.05, 0.40, 0.01, 0.40,
            correlation=0.0, collateral_hazard=0.05)
        r_high = repo_bilateral_cva(
            1e6, 90, 0.05, 0.02, 0.05, 0.40, 0.01, 0.40,
            correlation=0.80, collateral_hazard=0.05)
        assert r_high.wrong_way_add_on > r_zero.wrong_way_add_on


# ═══════════════════════════════════════════════════════════════
# 2.2: Dynamic Haircuts
# ═══════════════════════════════════════════════════════════════

class TestDynamicHaircuts:
    @pytest.fixture
    def model(self):
        return DynamicHaircutModel(
            base_haircut=0.06, duration=5.0, current_spread_bp=150,
        )

    def test_base_only(self, model):
        r = model.compute()
        assert r.total_haircut > r.base_haircut  # vol add-on even at zero shock

    def test_spread_shock_increases(self, model):
        r_base = model.compute()
        r_shock = model.compute(spread_shock_bp=200)
        assert r_shock.total_haircut > r_base.total_haircut

    def test_vol_shock_increases(self, model):
        r_base = model.compute()
        r_vol = model.compute(vol_shock_pct=0.20)
        assert r_vol.total_haircut > r_base.total_haircut

    def test_downgrade_increases(self, model):
        r_base = model.compute()
        r_dn = model.compute(rating_downgrade_notches=3)
        assert r_dn.total_haircut > r_base.total_haircut
        assert r_dn.rating_trigger_add_on == 0.06  # 3 notches = +6%

    def test_procyclicality_buffer(self, model):
        r = model.compute()
        assert r.procyclicality_buffer > 0

    def test_capped_at_100pct(self, model):
        r = model.compute(spread_shock_bp=5000, rating_downgrade_notches=5)
        assert r.total_haircut <= 1.0

    def test_stress_scenarios(self, model):
        scenarios = haircut_stress_scenarios(model)
        assert len(scenarios) == 7
        assert scenarios[0]["scenario"] == "base"
        # Combined stress should be highest
        assert scenarios[-1]["total_haircut"] > scenarios[0]["total_haircut"]

    def test_credit_spread_to_haircut(self):
        h_low = credit_spread_to_haircut(50, 5.0)
        h_high = credit_spread_to_haircut(500, 5.0)
        assert h_high > h_low

    def test_rating_trigger_impact(self):
        h = rating_trigger_impact(0.06, 2)
        assert h == 0.06 + 0.03  # 2 notches = +3%

    def test_to_dict(self, model):
        d = model.to_dict()
        assert "base_haircut" in d


# ═══════════════════════════════════════════════════════════════
# 2.3: Correlated XVA
# ═══════════════════════════════════════════════════════════════

class TestCorrelatedXVA:
    def test_basic(self):
        r = repo_xva_correlated(
            1e6, 90, 0.05, 0.02,
            counterparty_hazard=0.03, counterparty_recovery=0.40,
            collateral_spread_bp=150, collateral_duration=5.0,
            n_paths=5000,
        )
        assert isinstance(r, RepoXVACorrelatedResult)
        assert r.total_xva >= 0
        assert r.cva >= 0
        assert r.fva >= 0

    def test_correlation_increases_cva(self):
        """Positive correlation → counterparty defaults when collateral drops → higher CVA."""
        r_zero = repo_xva_correlated(
            1e6, 90, 0.05, 0.02, 0.05, 0.40, 200, 5.0,
            correlation=0.0, n_paths=10000, seed=42)
        r_high = repo_xva_correlated(
            1e6, 90, 0.05, 0.02, 0.05, 0.40, 200, 5.0,
            correlation=0.80, n_paths=10000, seed=42)
        assert r_high.correlation_impact > r_zero.correlation_impact

    def test_all_in_xva(self):
        result = repo_all_in_xva(
            1e6, 90, 0.05, 0.02, 0.03, 0.40, 150, 5.0,
        )
        assert "interest_income" in result
        assert "total_xva" in result
        assert "net_income" in result
        assert "breakeven_rate" in result
        assert result["interest_income"] > 0

    def test_to_dict(self):
        r = repo_xva_correlated(1e6, 90, 0.05, 0.02, 0.03, 0.40, 150, 5.0, n_paths=1000)
        d = r.to_dict()
        assert "correlation_impact" in d
