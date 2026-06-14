"""Regression for L2 T4 audit of `regulatory.liquidity._asf_factor`:

Pre-fix the ``maturity_days > 180`` branch came BEFORE the
retail-deposit check, so a 200-day retail/SME deposit returned 50% ASF
instead of the Basel LIQ40.5 retail factor (90%).  Per Basel LIQ40 the
retail-deposit category applies at all maturities — maturity is
irrelevant for stable/less-stable retail.
"""

from __future__ import annotations

import pytest

from pricebook.regulatory.liquidity import (
    LiquidityPosition, _asf_factor, calculate_portfolio_lcr,
)


class TestRetailDepositASF:
    @pytest.mark.parametrize("days", [30, 100, 200, 364])
    def test_retail_gets_90_at_all_short_maturities(self, days):
        """All retail deposits ≤1Y → 90% (less-stable retail proxy)."""
        pos = LiquidityPosition(
            position_id=f"R{days}", product_type="deposit",
            notional=1_000_000.0, maturity_days=days,
            counterparty_type="retail", is_asset=False,
        )
        assert _asf_factor(pos) == pytest.approx(0.90)

    def test_retail_long_term_still_high(self):
        """Retail deposit ≥1Y also gets retail factor (90%, not 100%).
        Code returns 90% for retail at all maturities — generic ≥1Y
        funding (100%) does NOT apply to retail."""
        pos = LiquidityPosition(
            position_id="R_long", product_type="deposit",
            notional=1_000_000.0, maturity_days=730,
            counterparty_type="retail", is_asset=False,
        )
        assert _asf_factor(pos) == pytest.approx(0.90)

    def test_wholesale_200_day_gets_50(self):
        """Wholesale deposit 200 days → 50% (the existing 6mo-1Y branch)."""
        pos = LiquidityPosition(
            position_id="W200", product_type="deposit",
            notional=1_000_000.0, maturity_days=200,
            counterparty_type="corporate", is_asset=False,
        )
        assert _asf_factor(pos) == pytest.approx(0.50)

    def test_wholesale_100_day_gets_50(self):
        """Wholesale deposit < 6mo → 50% (operational proxy)."""
        pos = LiquidityPosition(
            position_id="W100", product_type="deposit",
            notional=1_000_000.0, maturity_days=100,
            counterparty_type="corporate", is_asset=False,
        )
        assert _asf_factor(pos) == pytest.approx(0.50)


class TestPortfolioASFRetailCounted:
    """Portfolio-level: a retail deposit at 200 days must contribute
    0.90 × notional to ASF, not 0.50 × notional."""

    def test_retail_deposit_contributes_full_asf(self):
        positions = [
            LiquidityPosition(
                position_id="R", product_type="deposit",
                notional=100_000_000.0, maturity_days=200,
                counterparty_type="retail", is_asset=False,
            ),
            # Add some HQLA so LCR is computable.
            LiquidityPosition(
                position_id="A", product_type="cash",
                notional=10_000_000.0, is_asset=True,
            ),
        ]
        r = calculate_portfolio_lcr(positions)
        # ASF should be 100M × 0.9 = 90M (was 50M pre-fix).
        assert r.asf_total == pytest.approx(90_000_000.0, rel=1e-9)
