"""Tests for mandate compliance."""
import pytest
from datetime import date
from pricebook.core.mandate import (
    Mandate, PortfolioHolding, MandateReport,
    check_mandate, rating_at_least,
    investment_grade_mandate, sovereign_only_mandate, balanced_mandate,
)

@pytest.fixture
def ig_portfolio():
    return [
        PortfolioHolding("T1", "corp_bond", "Apple", "AA", "tech", "US", "USD", 20, 1e6, 5, 4.5),
        PortfolioHolding("T2", "corp_bond", "Google", "A+", "fin", "US", "USD", 20, 1e6, 3, 2.8),
        PortfolioHolding("T3", "govt_bond", "UST", "AAA", "sovereign", "US", "USD", 20, 1e6, 10, 8.2),
        PortfolioHolding("T4", "corp_bond", "BMW", "A", "auto", "DE", "EUR", 15, 0.75e6, 7, 6.1),
        PortfolioHolding("T5", "corp_bond", "Shell", "BBB+", "energy", "UK", "GBP", 15, 0.75e6, 4, 3.5),
    ]

class TestRating:
    def test_aaa_above_bbb(self):
        assert rating_at_least("AAA", "BBB-")
    def test_bb_below_bbb(self):
        assert not rating_at_least("BB+", "BBB-")
    def test_same(self):
        assert rating_at_least("A", "A")
    def test_nr(self):
        assert not rating_at_least("NR", "BBB-")

class TestTemplates:
    def test_ig(self):
        m = investment_grade_mandate()
        assert m.min_rating == "BBB-"
    def test_sovereign(self):
        m = sovereign_only_mandate()
        assert m.eligible_asset_classes == ["govt_bond"]

class TestCheckMandate:
    def test_ig_passes(self, ig_portfolio):
        m = Mandate("IG", min_rating="BBB-", max_single_name_pct=0.25, max_sector_pct=0.50, max_duration=10.0)
        r = check_mandate(ig_portfolio, m, date(2024, 6, 15))
        assert r.is_compliant

    def test_rating_breach(self, ig_portfolio):
        ig_portfolio.append(PortfolioHolding("T6", "corp_bond", "Junk", "B-", "energy", "US", "USD", 10, 0.5e6))
        m = investment_grade_mandate()
        r = check_mandate(ig_portfolio, m, date(2024, 6, 15))
        assert not r.is_compliant
        failed = [x for x in r.results if x.rule_type == "min_rating"]
        assert len(failed) == 1 and not failed[0].passed

    def test_concentration_breach(self):
        holdings = [
            PortfolioHolding("T1", "corp_bond", "BigCo", "A", "tech", "US", "USD", 60, 3e6),
            PortfolioHolding("T2", "corp_bond", "SmallCo", "A", "tech", "US", "USD", 40, 2e6),
        ]
        m = Mandate("strict", max_single_name_pct=0.50)
        r = check_mandate(holdings, m)
        assert not r.is_compliant

    def test_currency_restriction(self, ig_portfolio):
        m = Mandate("usd_only", currency_restrictions=["USD"])
        r = check_mandate(ig_portfolio, m)
        assert not r.is_compliant  # BMW(EUR) and Shell(GBP) breach

    def test_duration_breach(self, ig_portfolio):
        m = Mandate("short_dur", max_duration=3.0)
        r = check_mandate(ig_portfolio, m)
        failed = [x for x in r.results if x.rule_type == "duration"]
        assert len(failed) == 1 and not failed[0].passed

    def test_all_pass(self):
        holdings = [PortfolioHolding("T1", "govt_bond", "UST", "AAA", "sovereign", "US", "USD", 100, 10e6)]
        m = Mandate("sovereign", eligible_asset_classes=["govt_bond"], min_rating="A-")
        r = check_mandate(holdings, m)
        assert r.is_compliant

    def test_to_dict(self, ig_portfolio):
        m = investment_grade_mandate()
        d = check_mandate(ig_portfolio, m).to_dict()
        assert "is_compliant" in d
        assert "results" in d

    def test_empty_portfolio(self):
        r = check_mandate([], investment_grade_mandate())
        assert r.is_compliant  # no violations on empty portfolio
