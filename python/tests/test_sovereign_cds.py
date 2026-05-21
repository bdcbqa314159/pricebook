"""Tests for sovereign CDS conventions and hazard rate bootstrap."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.sovereign_cds import (
    get_sovereign_cds_conventions, list_sovereign_cds,
    bootstrap_sovereign_hazard, SovereignCDSConventions,
    RestructuringClause, SovereignHazardResult,
)


REF = date(2024, 1, 15)


@pytest.fixture
def usd_curve():
    return DiscountCurve.flat(REF, 0.05)


class TestConventions:
    def test_brazil(self):
        c = get_sovereign_cds_conventions("BR")
        assert c.country_name == "Brazil"
        assert c.currency == "USD"
        assert c.restructuring == RestructuringClause.CR
        assert c.recovery_rate == 0.25

    def test_italy_eur(self):
        c = get_sovereign_cds_conventions("IT")
        assert c.currency == "EUR"
        assert c.restructuring == RestructuringClause.MM
        assert c.recovery_rate == 0.40

    def test_korea_xr(self):
        c = get_sovereign_cds_conventions("KR")
        assert c.restructuring == RestructuringClause.XR

    def test_poland_mm(self):
        c = get_sovereign_cds_conventions("PL")
        assert c.restructuring == RestructuringClause.MM

    def test_long_tenors_it(self):
        c = get_sovereign_cds_conventions("IT")
        assert 30 in c.standard_tenors

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No sovereign CDS"):
            get_sovereign_cds_conventions("XX")

    def test_list_count(self):
        codes = list_sovereign_cds()
        assert len(codes) == 31

    def test_to_dict(self):
        c = get_sovereign_cds_conventions("BR")
        d = c.to_dict()
        assert d["restructuring"] == "CR"
        assert d["recovery_rate"] == 0.25


class TestBootstrap:
    def test_single_tenor(self, usd_curve):
        result = bootstrap_sovereign_hazard(
            REF, {5: 200}, usd_curve, "BR",
        )
        assert result.n_pillars == 1
        assert result.pillar_hazards[0] > 0
        assert result.country_code == "BR"

    def test_term_structure(self, usd_curve):
        spreads = {1: 80, 3: 120, 5: 180, 10: 250}
        result = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "TR")
        assert result.n_pillars == 4
        assert len(result.pillar_hazards) == 4
        # Increasing spreads → roughly increasing hazards
        assert result.pillar_hazards[-1] > result.pillar_hazards[0]

    def test_survivals_decreasing(self, usd_curve):
        spreads = {1: 100, 5: 200, 10: 300}
        result = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "ZA")
        for i in range(1, len(result.pillar_survivals)):
            assert result.pillar_survivals[i] < result.pillar_survivals[i - 1]

    def test_fitted_spreads_close(self, usd_curve):
        """Fitted spreads should be close to input (first-order approx)."""
        spreads = {5: 150}
        result = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "MX")
        assert abs(result.fitted_spreads_bp[0] - 150) < 20  # within 20bp

    def test_recovery_override(self, usd_curve):
        """Override recovery → different hazard."""
        spreads = {5: 200}
        r1 = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "BR")
        r2 = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "BR", recovery_override=0.40)
        assert r2.pillar_hazards[0] != r1.pillar_hazards[0]
        assert r2.recovery_rate == 0.40
        assert r1.recovery_rate == 0.25

    def test_high_spread_distressed(self, usd_curve):
        """Very high spread (distressed) still works."""
        spreads = {5: 2000}  # 20% spread (e.g. Argentina)
        result = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "AR")
        assert result.pillar_hazards[0] > 0.10  # high hazard

    def test_tight_spread_ig(self, usd_curve):
        """Tight spread (IG) → low hazard."""
        spreads = {5: 20}
        result = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "KR")
        assert result.pillar_hazards[0] < 0.01

    def test_survival_curve_usable(self, usd_curve):
        """Result's survival curve can be queried at any date."""
        spreads = {1: 100, 5: 200, 10: 300}
        result = bootstrap_sovereign_hazard(REF, spreads, usd_curve, "ID")
        q5 = result.survival_curve.survival(date(2029, 1, 15))
        assert 0 < q5 < 1

    def test_to_dict(self, usd_curve):
        result = bootstrap_sovereign_hazard(REF, {5: 150}, usd_curve, "BR")
        d = result.to_dict()
        assert "pillar_hazards" in d
        assert "country_code" in d
        assert d["restructuring"] == "CR"

    def test_multiple_em_countries(self, usd_curve):
        """Bootstrap works for a range of EM countries."""
        for code in ["BR", "TR", "ZA", "CN", "ID", "MX", "CO", "IT", "ES"]:
            result = bootstrap_sovereign_hazard(REF, {5: 150}, usd_curve, code)
            assert result.n_pillars == 1
            assert result.pillar_hazards[0] > 0
