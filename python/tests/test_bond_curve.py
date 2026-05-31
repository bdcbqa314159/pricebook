"""Tests for yield curve bootstrapping from bond prices."""

import pytest
import math
from datetime import date

from pricebook.curves.bond_curve import (
    BondQuote, BondCurveResult, bootstrap_curve_from_bonds,
)

REF = date(2024, 6, 1)


def _treasury_quotes():
    """Realistic US Treasury quotes."""
    return [
        BondQuote(date(2025, 6, 1), 0.00,   97.5, frequency=2),   # 1Y bill
        BondQuote(date(2026, 6, 1), 0.04,   99.0, frequency=2),   # 2Y note
        BondQuote(date(2027, 6, 1), 0.0425, 98.5, frequency=2),   # 3Y note
        BondQuote(date(2029, 6, 1), 0.045,  97.0, frequency=2),   # 5Y note
        BondQuote(date(2034, 6, 1), 0.0475, 95.0, frequency=2),   # 10Y note
        BondQuote(date(2054, 6, 1), 0.05,   92.0, frequency=2),   # 30Y bond
    ]


# ═══════════════════════════════════════════════════════════════
# Sequential bootstrap
# ═══════════════════════════════════════════════════════════════

class TestSequentialBootstrap:
    def test_basic(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="sequential")
        assert result.converged
        assert result.n_bonds == 6
        assert result.method == "sequential"

    def test_discount_factors_decreasing(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="sequential")
        curve = result.discount_curve
        dfs = [curve.df(date(2024 + y, 6, 1)) for y in range(1, 11)]
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i - 1] + 0.001  # monotonically decreasing

    def test_zero_rates_positive(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="sequential")
        assert all(z > 0 for z in result.pillar_zeros)

    def test_single_bond(self):
        """Single bond should produce a flat curve."""
        quotes = [BondQuote(date(2029, 6, 1), 0.045, 97.0)]
        result = bootstrap_curve_from_bonds(REF, quotes, method="sequential")
        assert result.n_bonds == 1
        # Should have one pillar
        assert len(result.pillar_dates) == 1

    def test_zero_coupon(self):
        """Zero-coupon bond (T-Bill) should give exact DF."""
        quotes = [BondQuote(date(2025, 6, 1), 0.0, 95.24)]  # ~5% yield
        result = bootstrap_curve_from_bonds(REF, quotes, method="sequential")
        df = result.discount_curve.df(date(2025, 6, 1))
        assert df == pytest.approx(0.9524, abs=0.001)


# ═══════════════════════════════════════════════════════════════
# Global fit
# ═══════════════════════════════════════════════════════════════

class TestGlobalFit:
    def test_basic(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="global")
        assert result.converged
        assert result.method == "global"

    def test_fewer_pillars_than_bonds(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="global",
                                             n_pillars=3)
        assert len(result.pillar_dates) == 3
        assert result.rmse_bp < 500

    def test_on_the_run_higher_weight(self):
        """On-the-run bonds should be fitted more closely."""
        quotes = _treasury_quotes()
        quotes[3].is_on_the_run = True  # 5Y on-the-run
        result = bootstrap_curve_from_bonds(REF, quotes, method="global")
        # 5Y fitted price should be close to market
        assert abs(result.residuals_bp[3]) < abs(result.residuals_bp[0]) + 100

    def test_noisy_prices(self):
        """Global fit should handle noisy prices gracefully."""
        quotes = _treasury_quotes()
        # Add noise
        quotes[0].dirty_price += 0.5
        quotes[2].dirty_price -= 0.3
        result = bootstrap_curve_from_bonds(REF, quotes, method="global")
        assert result.converged


# ═══════════════════════════════════════════════════════════════
# Nelson-Siegel
# ═══════════════════════════════════════════════════════════════

class TestNelsonSiegel:
    def test_basic(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="nelson_siegel")
        assert result.converged
        assert result.method == "nelson_siegel"
        assert "beta0" in result.parameters
        assert "tau" in result.parameters

    def test_smooth_curve(self):
        """NS curve should produce smooth forward rates."""
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="nelson_siegel")
        curve = result.discount_curve
        zeros = [curve.zero_rate(date(2024 + y, 6, 1)) for y in range(1, 20)]
        # No wild oscillations (max difference between adjacent < 200bp)
        for i in range(1, len(zeros)):
            assert abs(zeros[i] - zeros[i - 1]) < 0.02

    def test_long_end_stable(self):
        """NS long-end converges to β0."""
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="nelson_siegel")
        beta0 = result.parameters["beta0"]
        long_zero = result.discount_curve.zero_rate(date(2074, 6, 1))
        assert long_zero == pytest.approx(beta0, abs=0.005)


# ═══════════════════════════════════════════════════════════════
# Svensson
# ═══════════════════════════════════════════════════════════════

class TestSvensson:
    def test_basic(self):
        result = bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="svensson")
        assert result.converged
        assert "beta3" in result.parameters
        assert "tau2" in result.parameters

    def test_fits_better_than_ns(self):
        """Svensson (6 params) should fit at least as well as NS (4 params)."""
        quotes = _treasury_quotes()
        ns = bootstrap_curve_from_bonds(REF, quotes, method="nelson_siegel")
        sv = bootstrap_curve_from_bonds(REF, quotes, method="svensson")
        assert sv.rmse_bp <= ns.rmse_bp + 10  # small tolerance


# ═══════════════════════════════════════════════════════════════
# Auto method selection
# ═══════════════════════════════════════════════════════════════

class TestAutoMethod:
    def test_few_bonds_sequential(self):
        quotes = _treasury_quotes()[:4]
        result = bootstrap_curve_from_bonds(REF, quotes, method="auto")
        assert result.method == "sequential"

    def test_many_bonds_global(self):
        # Duplicate some maturities to trigger global
        quotes = _treasury_quotes() + [BondQuote(date(2029, 6, 1), 0.04, 97.5)]
        result = bootstrap_curve_from_bonds(REF, quotes, method="auto")
        assert result.method == "global"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_curve_from_bonds(REF, _treasury_quotes(), method="unknown")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Need at least one"):
            bootstrap_curve_from_bonds(REF, [])


# ═══════════════════════════════════════════════════════════════
# Cross-method consistency
# ═══════════════════════════════════════════════════════════════

class TestCrossMethod:
    def test_all_methods_positive_zeros(self):
        """All methods should produce positive zero rates."""
        quotes = _treasury_quotes()
        for method in ["sequential", "global", "nelson_siegel", "svensson"]:
            result = bootstrap_curve_from_bonds(REF, quotes, method=method)
            assert all(z > 0 for z in result.pillar_zeros), f"{method} has negative zeros"

    def test_5y_rate_consistent(self):
        """5Y zero rate should be roughly similar across methods."""
        quotes = _treasury_quotes()
        zeros_5y = []
        for method in ["sequential", "global", "nelson_siegel"]:
            result = bootstrap_curve_from_bonds(REF, quotes, method=method)
            z5 = result.discount_curve.zero_rate(date(2029, 6, 1))
            zeros_5y.append(z5)
        # All within 200bp of each other
        assert max(zeros_5y) - min(zeros_5y) < 0.02


class TestBondQuote:
    def test_to_dict(self):
        q = BondQuote(date(2029, 6, 1), 0.045, 97.0)
        d = q.to_dict()
        assert "maturity" in d
        assert d["coupon"] == 0.045

    def test_on_the_run(self):
        q = BondQuote(date(2029, 6, 1), 0.045, 97.0, is_on_the_run=True)
        assert q.is_on_the_run
