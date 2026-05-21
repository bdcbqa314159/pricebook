"""Tests for NDF-implied discount curve construction.

Covers: CIP relationship, round-trip, curve construction, CIP basis,
edge cases, multiple EM currencies.
"""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.ndf_implied import (
    NDFQuote, NDFImpliedResult,
    build_ndf_implied_curve, ndf_from_curves, cip_basis,
    _add_months,
)


REF = date(2024, 1, 15)


@pytest.fixture
def usd_curve():
    return DiscountCurve.flat(REF, 0.05)


@pytest.fixture
def cny_ndfs():
    """Synthetic CNY NDFs consistent with 2.5% rate differential."""
    # If USD rate = 5%, CNY rate = 7.5%, spot = 7.18
    # NDF(T) = Spot × df_usd(T) / df_cny(T)
    # NDF(1M) = 7.18 × exp(-0.05/12) / exp(-0.075/12) ≈ 7.195
    spot = 7.18
    usd_r = 0.05
    cny_r = 0.075
    quotes = []
    for m in [1, 3, 6, 12]:
        t = m / 12.0
        ndf = spot * math.exp(-usd_r * t) / math.exp(-cny_r * t)
        quotes.append(NDFQuote(tenor_months=m, outright=ndf))
    return quotes


# ═══════════════════════════════════════════════════════════════
# Basic construction
# ═══════════════════════════════════════════════════════════════


class TestConstruction:
    def test_basic_build(self, usd_curve, cny_ndfs):
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve, "USD", "CNY")
        assert result.n_pillars == 4
        assert result.em_currency == "CNY"
        assert result.spot_rate == 7.18
        assert all(0 < df < 1 for df in result.implied_dfs)

    def test_implied_rates_positive(self, usd_curve, cny_ndfs):
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        assert all(r > 0 for r in result.implied_zero_rates)

    def test_implied_rate_above_base(self, usd_curve, cny_ndfs):
        """EM rate should be above USD rate (NDF at premium)."""
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        for r in result.implied_zero_rates:
            assert r > 0.05  # above USD rate

    def test_forward_points_positive(self, usd_curve, cny_ndfs):
        """NDF > spot when EM rate > base rate → positive forward points."""
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        assert all(fp > 0 for fp in result.ndf_forward_points)

    def test_pillar_dates_increasing(self, usd_curve, cny_ndfs):
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        for i in range(1, len(result.pillar_dates)):
            assert result.pillar_dates[i] > result.pillar_dates[i - 1]


# ═══════════════════════════════════════════════════════════════
# Round-trip: curves → NDF → curves
# ═══════════════════════════════════════════════════════════════


class TestRoundTrip:
    def test_ndf_from_curves_round_trip(self, usd_curve, cny_ndfs):
        """Build curve from NDFs, then regenerate NDFs → should match."""
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        regenerated = ndf_from_curves(
            REF, 7.18, usd_curve, result.em_curve,
            [q.tenor_months for q in cny_ndfs],
        )
        for original, regen in zip(cny_ndfs, regenerated):
            assert abs(original.outright - regen) < 0.001, \
                f"Tenor {original.tenor_months}M: orig={original.outright:.4f}, regen={regen:.4f}"

    def test_implied_rate_recovery(self, usd_curve):
        """Known rate differential should be recovered."""
        spot = 100.0
        usd_r = 0.05
        em_r = 0.10
        quotes = []
        for m in [3, 6, 12]:
            t = m / 12.0
            ndf = spot * math.exp(-usd_r * t) / math.exp(-em_r * t)
            quotes.append(NDFQuote(tenor_months=m, outright=ndf))

        result = build_ndf_implied_curve(REF, spot, quotes, usd_curve)
        for r in result.implied_zero_rates:
            assert abs(r - em_r) < 0.005  # within 50bp of 10%


# ═══════════════════════════════════════════════════════════════
# CIP Basis
# ═══════════════════════════════════════════════════════════════


class TestCIPBasis:
    def test_zero_basis_when_consistent(self, usd_curve, cny_ndfs):
        """If EM curve was built from NDFs, CIP basis should be ~0."""
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        basis = cip_basis(REF, 7.18, cny_ndfs, usd_curve, result.em_curve)
        for b in basis:
            assert abs(b["basis_bp"]) < 1.0, f"{b['tenor_months']}M basis = {b['basis_bp']:.2f}bp"

    def test_nonzero_basis_with_mismatch(self, usd_curve, cny_ndfs):
        """If EM curve doesn't match NDFs, basis should be non-zero."""
        # Use a flat EM curve that doesn't match the NDFs
        wrong_em = DiscountCurve.flat(REF, 0.04)  # Too low
        basis = cip_basis(REF, 7.18, cny_ndfs, usd_curve, wrong_em)
        assert any(abs(b["basis_bp"]) > 10 for b in basis)


# ═══════════════════════════════════════════════════════════════
# Multiple EM currencies
# ═══════════════════════════════════════════════════════════════


class TestMultipleCurrencies:
    def test_inr(self, usd_curve):
        """INR NDF construction."""
        spot = 83.0
        quotes = [
            NDFQuote(1, 83.15),
            NDFQuote(3, 83.50),
            NDFQuote(6, 84.10),
            NDFQuote(12, 85.30),
        ]
        result = build_ndf_implied_curve(REF, spot, quotes, usd_curve, "USD", "INR")
        assert result.em_currency == "INR"
        assert result.n_pillars == 4
        # INR rates should be higher than USD
        assert all(r > 0.04 for r in result.implied_zero_rates)

    def test_krw(self, usd_curve):
        """KRW NDF construction."""
        spot = 1300.0
        quotes = [
            NDFQuote(1, 1302.0),
            NDFQuote(3, 1306.0),
            NDFQuote(12, 1318.0),
        ]
        result = build_ndf_implied_curve(REF, spot, quotes, usd_curve, "USD", "KRW")
        assert result.n_pillars == 3

    def test_brl(self, usd_curve):
        """BRL NDF construction (high rate differential)."""
        spot = 4.90
        quotes = [
            NDFQuote(1, 4.94),
            NDFQuote(3, 5.02),
            NDFQuote(6, 5.15),
            NDFQuote(12, 5.40),
        ]
        result = build_ndf_implied_curve(REF, spot, quotes, usd_curve, "USD", "BRL")
        # BRL rates much higher than USD
        assert all(r > 0.08 for r in result.implied_zero_rates)


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_raises(self, usd_curve):
        with pytest.raises(ValueError, match="At least one"):
            build_ndf_implied_curve(REF, 7.18, [], usd_curve)

    def test_negative_spot_raises(self, usd_curve):
        with pytest.raises(ValueError, match="positive"):
            build_ndf_implied_curve(REF, -1.0, [NDFQuote(1, 7.20)], usd_curve)

    def test_single_tenor(self, usd_curve):
        """One NDF quote → one-pillar curve."""
        result = build_ndf_implied_curve(
            REF, 7.18, [NDFQuote(12, 7.50)], usd_curve,
        )
        assert result.n_pillars == 1

    def test_bid_ask_uses_mid(self, usd_curve):
        """When bid/ask provided, mid is used."""
        q = NDFQuote(12, outright=7.50, bid=7.48, ask=7.52)
        assert q.mid == 7.50
        result = build_ndf_implied_curve(REF, 7.18, [q], usd_curve)
        assert result.n_pillars == 1

    def test_to_dict(self, usd_curve, cny_ndfs):
        result = build_ndf_implied_curve(REF, 7.18, cny_ndfs, usd_curve)
        d = result.to_dict()
        assert "implied_dfs" in d
        assert "implied_zero_rates" in d
        assert d["n_pillars"] == 4

    def test_ndf_quote_to_dict(self):
        q = NDFQuote(3, 7.25, bid=7.24, ask=7.26)
        d = q.to_dict()
        assert d["tenor_months"] == 3
        assert d["outright"] == 7.25


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


class TestHelpers:
    def test_add_months(self):
        assert _add_months(date(2024, 1, 15), 1) == date(2024, 2, 15)
        assert _add_months(date(2024, 1, 15), 12) == date(2025, 1, 15)
        assert _add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)  # leap year
        assert _add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)  # non-leap
        assert _add_months(date(2024, 11, 15), 3) == date(2025, 2, 15)
