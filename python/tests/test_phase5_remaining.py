"""Tests for Phase 5 remaining items: B3-B6, C5-C9, D7-D9."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve

# B3-B6: CLN advanced
from pricebook.credit.cln_advanced import (
    cln_xva_spread_driven, dynamic_funding_cost,
    wrong_way_risk_adjustment, collateral_haircut_stress,
    XVAResult, HaircutStressResult,
)

# C5: Covered bonds
from pricebook.fixed_income.covered_bond import (
    CoverPool, CoveredBondResult, price_covered_bond,
)

# C6: Bond forwards + credit
from pricebook.fixed_income.bond_forward import credit_adjusted_forward_price

# C9: Issuer spread curve
from pricebook.credit.issuer_curve import (
    IssuerSpreadCurve, fit_issuer_curve,
)

# D7: Sukuk
from pricebook.fixed_income.sukuk import (
    SukukType, get_sukuk_conventions, list_sukuk_types, price_sukuk_as_bond,
)

# D8: ESG
from pricebook.fixed_income.esg_bond import (
    ESGLabel, classify_esg_bond, list_esg_labels, list_use_of_proceeds,
)

# D9: Supranationals
from pricebook.fixed_income.supranational import (
    get_supranational, list_supranationals,
)


REF = date(2024, 1, 15)


# ═══════════════════════════════════════════════════════════════
# B3-B6: CLN Advanced
# ═══════════════════════════════════════════════════════════════


class TestCLNAdvanced:
    def test_xva_spread_driven(self):
        r = cln_xva_spread_driven(1e6, 5.0, 0.02, 0.01)
        assert isinstance(r, XVAResult)
        assert r.cva >= 0
        assert r.fva >= 0
        assert r.total_xva >= 0

    def test_higher_hazard_higher_cva(self):
        r1 = cln_xva_spread_driven(1e6, 5.0, 0.01, 0.01)
        r2 = cln_xva_spread_driven(1e6, 5.0, 0.01, 0.05)
        assert r2.cva > r1.cva

    def test_wrong_way_adjustment(self):
        r = cln_xva_spread_driven(1e6, 5.0, 0.02, 0.02, correlation=0.5)
        assert r.wrong_way_adjustment > 0

    def test_dynamic_funding(self):
        exposure = [100_000, 200_000, 150_000, 100_000]
        funding = [0.005, 0.005, 0.006, 0.006]
        cost = dynamic_funding_cost(1e6, 4.0, exposure, funding, threshold=50_000)
        assert cost > 0

    def test_dynamic_funding_threshold(self):
        exposure = [100_000] * 4
        funding = [0.005] * 4
        c1 = dynamic_funding_cost(1e6, 4.0, exposure, funding, threshold=0)
        c2 = dynamic_funding_cost(1e6, 4.0, exposure, funding, threshold=80_000)
        assert c2 < c1  # higher threshold → less uncollateralised

    def test_wwr_adjustment(self):
        adj = wrong_way_risk_adjustment(10_000, 0.5, 0.5)
        assert adj > 0
        adj_zero = wrong_way_risk_adjustment(10_000, 0.0)
        assert adj_zero == 0.0

    def test_haircut_stress(self):
        r = collateral_haircut_stress(1e6, 0.02, 200, 5.0)
        assert isinstance(r, HaircutStressResult)
        assert r.stressed_haircut_pct > r.base_haircut_pct
        assert r.additional_margin_call >= 0

    def test_xva_to_dict(self):
        d = cln_xva_spread_driven(1e6, 5.0, 0.02, 0.01).to_dict()
        assert "cva" in d and "fva" in d


# ═══════════════════════════════════════════════════════════════
# C5: Covered Bonds
# ═══════════════════════════════════════════════════════════════


class TestCoveredBonds:
    @pytest.fixture
    def flat_curve(self):
        return DiscountCurve.flat(REF, 0.04)

    def test_basic(self, flat_curve):
        pool = CoverPool(1e9, 1.15, 0.60, "mortgage", "DE")
        r = price_covered_bond(0.035, 10.0, flat_curve, 80, pool)
        assert isinstance(r, CoveredBondResult)
        assert r.price > 0
        assert r.spread_bp < 80  # tighter than issuer

    def test_higher_oc_lower_spread(self, flat_curve):
        pool1 = CoverPool(1e9, 1.05, 0.70, "mortgage")
        pool2 = CoverPool(1e9, 1.30, 0.70, "mortgage")
        r1 = price_covered_bond(0.035, 10.0, flat_curve, 80, pool1)
        r2 = price_covered_bond(0.035, 10.0, flat_curve, 80, pool2)
        assert r2.spread_bp <= r1.spread_bp

    def test_oc_cushion(self, flat_curve):
        pool = CoverPool(1e9, 1.20, 0.60, "mortgage")
        r = price_covered_bond(0.035, 10.0, flat_curve, 80, pool)
        assert r.oc_cushion_pct == pytest.approx(20.0)

    def test_to_dict(self, flat_curve):
        pool = CoverPool(1e9, 1.15, 0.60, "mortgage")
        d = price_covered_bond(0.035, 10.0, flat_curve, 80, pool).to_dict()
        assert "cover_pool_benefit_bp" in d


# ═══════════════════════════════════════════════════════════════
# C6: Credit-Adjusted Forward
# ═══════════════════════════════════════════════════════════════


class TestCreditForward:
    def test_basic(self):
        r = credit_adjusted_forward_price(98.0, 0.04, 0.5, 0.98)
        assert r["forward_price"] > 0
        assert r["credit_charge"] >= 0

    def test_no_default_no_charge(self):
        r = credit_adjusted_forward_price(100.0, 0.04, 0.5, 1.0)
        assert r["credit_charge"] == pytest.approx(0.0)
        assert r["forward_price"] == pytest.approx(r["riskfree_forward"])

    def test_higher_pd_lower_forward(self):
        r1 = credit_adjusted_forward_price(98.0, 0.04, 1.0, 0.99)
        r2 = credit_adjusted_forward_price(98.0, 0.04, 1.0, 0.90)
        assert r2["forward_price"] < r1["forward_price"]


# ═══════════════════════════════════════════════════════════════
# C9: Issuer Spread Curve
# ═══════════════════════════════════════════════════════════════


class TestIssuerCurve:
    def test_fit(self):
        tenors = [2, 3, 5, 7, 10]
        spreads = [80, 100, 130, 155, 180]
        curve = fit_issuer_curve(tenors, spreads, "ACME Corp")
        assert curve.issuer == "ACME Corp"
        # Should fit reasonably well
        for t, s in zip(tenors, spreads):
            assert abs(curve.spread_bp(t) - s) < 20

    def test_term_structure(self):
        curve = IssuerSpreadCurve(0.015, -0.005, 0.002, 2.0)
        ts = curve.term_structure([1, 5, 10])
        assert len(ts) == 3

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            fit_issuer_curve([5], [100])


# ═══════════════════════════════════════════════════════════════
# D7: Sukuk
# ═══════════════════════════════════════════════════════════════


class TestSukuk:
    def test_list_types(self):
        types = list_sukuk_types()
        assert "ijara" in types
        assert "mudaraba" in types
        assert len(types) == 7

    def test_ijara_conventions(self):
        c = get_sukuk_conventions("ijara")
        assert c.asset_requirement is True
        assert c.tradeable is True

    def test_murabaha_not_tradeable(self):
        c = get_sukuk_conventions(SukukType.MURABAHA)
        assert c.tradeable is False

    def test_price_as_bond(self):
        p = price_sukuk_as_bond(0.04, 5.0, 50, 0.04)
        assert 95 < p < 105  # near par

    def test_to_dict(self):
        c = get_sukuk_conventions("ijara")
        d = c.to_dict()
        assert d["type"] == "ijara"


# ═══════════════════════════════════════════════════════════════
# D8: ESG
# ═══════════════════════════════════════════════════════════════


class TestESG:
    def test_classify_green(self):
        c = classify_esg_bond("green", ["renewable_energy", "clean_transport"])
        assert c.label == ESGLabel.GREEN
        assert len(c.use_of_proceeds) == 2
        assert c.greenium_bp == 5.0

    def test_classify_slb(self):
        c = classify_esg_bond("sustainability_linked", kpi_targets=["Reduce CO2 30% by 2030"], step_up_bp=25)
        assert c.label == ESGLabel.SLB
        assert c.step_up_bp == 25

    def test_list_labels(self):
        labels = list_esg_labels()
        assert "green" in labels
        assert "conventional" in labels

    def test_list_uop(self):
        uop = list_use_of_proceeds()
        assert "renewable_energy" in uop
        assert len(uop) == 12

    def test_to_dict(self):
        c = classify_esg_bond("green")
        d = c.to_dict()
        assert d["label"] == "green"


# ═══════════════════════════════════════════════════════════════
# D9: Supranationals
# ═══════════════════════════════════════════════════════════════


class TestSupranationals:
    def test_eib(self):
        s = get_supranational("EIB")
        assert s.rating == "AAA"
        assert "EUR" in s.typical_currencies

    def test_ibrd(self):
        s = get_supranational("IBRD")
        assert s.rating == "AAA"

    def test_list(self):
        codes = list_supranationals()
        assert "EIB" in codes
        assert "IBRD" in codes
        assert "ADB" in codes
        assert len(codes) == 10

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_supranational("FAKE")

    def test_to_dict(self):
        d = get_supranational("EIB").to_dict()
        assert d["code"] == "EIB"
        assert d["rating"] == "AAA"
