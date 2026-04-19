"""Tests for CPI seasonality."""
import numpy as np, pytest
from pricebook.cpi_seasonality import estimate_seasonal_factors, deseasonalise_breakeven, seasonal_carry_signal

class TestSeasonalFactors:
    def test_basic(self):
        # 2 years of monthly CPI changes
        data = [0.3, 0.2, 0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.3, 0.4, 0.2, 0.3,
                0.4, 0.3, 0.5, 0.2, 0.6, 0.7, 0.1, 0.2, 0.4, 0.3, 0.1, 0.4]
        r = estimate_seasonal_factors(data)
        assert len(r.monthly_factors) == 12
        assert r.monthly_factors.mean() == pytest.approx(1.0, abs=0.01)
    def test_insufficient_data(self):
        r = estimate_seasonal_factors([0.3, 0.2])
        assert r.method == "insufficient_data"

class TestDeseasonalise:
    def test_basic(self):
        sf = estimate_seasonal_factors([0.3]*6 + [0.1]*6 + [0.3]*6 + [0.1]*6)
        r = deseasonalise_breakeven(2.5, sf, current_month=0, months_to_maturity=12)
        assert r.deseasonalised != r.raw_breakeven or r.seasonal_adjustment == pytest.approx(0)

class TestSeasonalCarry:
    def test_positive_signal(self):
        sf = estimate_seasonal_factors([0.5]*3 + [0.1]*9 + [0.5]*3 + [0.1]*9)
        r = seasonal_carry_signal(sf, current_month=0)
        # First 3 months have high CPI → positive signal
        assert r.signal > 0
    def test_negative_signal(self):
        sf = estimate_seasonal_factors([0.1]*3 + [0.5]*9 + [0.1]*3 + [0.5]*9)
        r = seasonal_carry_signal(sf, current_month=0)
        assert r.signal < 0
