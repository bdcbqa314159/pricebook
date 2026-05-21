"""Tests for cross-market sovereign relative value framework."""

import pytest

from pricebook.fixed_income.sovereign_rv import (
    SovereignRVInput, SpreadDecomposition, RVScore,
    sovereign_spread_decomposition, cross_market_rv_scores,
)


def _make_market(code, ccy, spread, cds, debt_gdp, fiscal, ca, rating, vol=10, reserves=4):
    return SovereignRVInput(
        country_code=code, currency=ccy,
        spread_bp=spread, cds_spread_bp=cds,
        debt_to_gdp=debt_gdp, fiscal_balance_gdp=fiscal,
        current_account_gdp=ca, rating_notch=rating,
        fx_vol_3m=vol, reserves_months_imports=reserves,
    )


# Standard EM universe for cross-section tests
MARKETS = [
    _make_market("BR", "BRL", 250, 180, 75, -6.0, -3.0, 12, 15, 10),
    _make_market("MX", "MXN", 150, 100, 50, -3.5, -1.5, 10, 12, 5),
    _make_market("ZA", "ZAR", 300, 200, 70, -5.0, -2.0, 12, 18, 4),
    _make_market("TR", "TRY", 400, 350, 40, -3.0, -5.0, 14, 25, 2),
    _make_market("ID", "IDR", 120, 80, 40, -2.5, -1.0, 10, 10, 7),
    _make_market("PL", "PLN", 60, 30, 50, -4.0, 0.5, 6, 8, 5),
    _make_market("KR", "KRW", 30, 15, 45, -1.0, 4.0, 3, 7, 8),
]


class TestSpreadDecomposition:
    def test_basic(self):
        d = sovereign_spread_decomposition(MARKETS[0])  # Brazil
        assert d.country_code == "BR"
        assert d.total_spread_bp == 250
        assert d.credit_bp == 180
        assert d.liquidity_bp >= 0
        assert d.fundamental_bp >= 0

    def test_components_sum(self):
        """Components should roughly sum to total."""
        d = sovereign_spread_decomposition(MARKETS[0])
        total = d.credit_bp + d.liquidity_bp + d.fundamental_bp + d.technical_bp
        assert abs(total - d.total_spread_bp) < 0.01

    def test_low_risk_country(self):
        """Low risk (Korea) → small fundamental component."""
        d = sovereign_spread_decomposition(MARKETS[6])  # Korea
        assert d.fundamental_bp < 30

    def test_high_risk_country(self):
        """High risk (Turkey) → larger fundamental component."""
        d = sovereign_spread_decomposition(MARKETS[3])  # Turkey
        assert d.fundamental_bp > 10

    def test_to_dict(self):
        d = sovereign_spread_decomposition(MARKETS[0])
        out = d.to_dict()
        assert "fundamental_bp" in out
        assert "credit_bp" in out


class TestCrossMarketRV:
    def test_basic(self):
        scores = cross_market_rv_scores(MARKETS)
        assert len(scores) == 7
        # Sorted by z_score descending (cheapest first)
        assert scores[0].z_score >= scores[-1].z_score

    def test_signals(self):
        scores = cross_market_rv_scores(MARKETS)
        signals = {s.signal for s in scores}
        # With 7 markets spanning 30-400bp, should have variety
        assert len(signals) >= 2

    def test_cheapest_is_widest(self):
        """Widest spread should have highest z_score (cheapest)."""
        scores = cross_market_rv_scores(MARKETS)
        assert scores[0].country_code == "TR"  # 400bp, widest

    def test_richest_is_tightest(self):
        """Tightest spread should have lowest z_score (richest)."""
        scores = cross_market_rv_scores(MARKETS)
        assert scores[-1].country_code == "KR"  # 30bp, tightest

    def test_percentile_range(self):
        scores = cross_market_rv_scores(MARKETS)
        pcts = [s.percentile for s in scores]
        assert min(pcts) > 0
        assert max(pcts) <= 100

    def test_to_dict(self):
        scores = cross_market_rv_scores(MARKETS)
        d = scores[0].to_dict()
        assert "z_score" in d
        assert "signal" in d

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            cross_market_rv_scores([MARKETS[0]])

    def test_fundamental_z_scores(self):
        """Fundamental Z-scores should also have variety."""
        scores = cross_market_rv_scores(MARKETS)
        fz = [s.fundamental_z for s in scores]
        assert max(fz) > min(fz)

    def test_all_markets_in_output(self):
        scores = cross_market_rv_scores(MARKETS)
        codes = {s.country_code for s in scores}
        expected = {m.country_code for m in MARKETS}
        assert codes == expected
