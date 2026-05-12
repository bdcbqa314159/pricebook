"""Tests for extended equity exotics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.equity_exotic_extended import (
    forward_start_option,
    chooser_option,
    quanto_equity_option,
    himalaya_option,
    outperformance_option,
    equity_accumulator,
)


class TestForwardStart:
    def test_atm_positive(self):
        r = forward_start_option(100, 0.05, 0.02, 0.20, 0.5, 1.0)
        assert r.price > 0

    def test_otm_cheaper(self):
        atm = forward_start_option(100, 0.05, 0.02, 0.20, 0.5, 1.0, moneyness=1.0)
        otm = forward_start_option(100, 0.05, 0.02, 0.20, 0.5, 1.0, moneyness=1.10)
        assert otm.price < atm.price

    def test_invalid_dates_raises(self):
        with pytest.raises(ValueError):
            forward_start_option(100, 0.05, 0.02, 0.20, 1.0, 0.5)

    def test_put_positive(self):
        r = forward_start_option(100, 0.05, 0.02, 0.20, 0.5, 1.0, is_call=False)
        assert r.price > 0


class TestChooser:
    def test_worth_more_than_call(self):
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp(0.03 * 1.0)
        df = math.exp(-0.05 * 1.0)
        call = black76_price(F, 100, 0.20, 1.0, df, OptionType.CALL)
        ch = chooser_option(100, 100, 0.05, 0.02, 0.20, 0.5, 1.0)
        assert ch.price > call

    def test_late_choice_approaches_call_plus_put(self):
        early = chooser_option(100, 100, 0.05, 0.02, 0.20, 0.1, 1.0)
        late = chooser_option(100, 100, 0.05, 0.02, 0.20, 0.9, 1.0)
        assert late.price > early.price

    def test_components_positive(self):
        r = chooser_option(100, 100, 0.05, 0.02, 0.20, 0.5, 1.0)
        assert r.call_component > 0
        assert r.put_component > 0


class TestQuantoEquity:
    def test_positive_price(self):
        r = quanto_equity_option(100, 100, 0.05, 0.03, 0.02, 0.20, 0.10, -0.3, 1.0)
        assert r.price > 0

    def test_positive_corr_reduces_forward(self):
        # Positive corr: equity up → FX up → quanto forward reduced
        r_neg = quanto_equity_option(100, 100, 0.05, 0.03, 0.02, 0.20, 0.10, -0.5, 1.0)
        r_pos = quanto_equity_option(100, 100, 0.05, 0.03, 0.02, 0.20, 0.10, 0.5, 1.0)
        assert r_pos.quanto_forward < r_neg.quanto_forward

    def test_zero_fx_vol_no_adjustment(self):
        r = quanto_equity_option(100, 100, 0.05, 0.03, 0.02, 0.20, 0.0, -0.5, 1.0)
        assert r.quanto_adjustment == pytest.approx(0.0, abs=1e-10)


class TestHimalaya:
    def test_positive_price(self):
        corr = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
        r = himalaya_option([100, 100, 100], 0.05, [0.02]*3, [0.20]*3, corr, 1.0,
                             n_paths=5_000)
        assert r.price > 0
        assert r.n_assets == 3

    def test_more_assets_different_price(self):
        corr2 = np.eye(2)
        corr3 = np.eye(3)
        r2 = himalaya_option([100, 100], 0.05, [0.02]*2, [0.20]*2, corr2, 1.0, n_paths=5_000)
        r3 = himalaya_option([100]*3, 0.05, [0.02]*3, [0.20]*3, corr3, 1.0, n_paths=5_000)
        assert r2.price != r3.price


class TestOutperformance:
    def test_symmetric_zero(self):
        # Same assets → outperformance = 0
        r = outperformance_option(100, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 1.0, 1.0)
        assert r.price == pytest.approx(0, abs=0.5)

    def test_higher_div_cheaper(self):
        # Asset 1 with higher div → lower forward → less outperformance
        r_low = outperformance_option(100, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 0.5, 1.0)
        r_high = outperformance_option(100, 100, 0.05, 0.05, 0.02, 0.20, 0.20, 0.5, 1.0)
        assert r_high.price < r_low.price

    def test_low_correlation_more_valuable(self):
        r_high = outperformance_option(100, 100, 0.05, 0.02, 0.03, 0.20, 0.25, 0.9, 1.0)
        r_low = outperformance_option(100, 100, 0.05, 0.02, 0.03, 0.20, 0.25, 0.1, 1.0)
        assert r_low.price > r_high.price


class TestEquityAccumulator:
    def test_positive_ko_probability(self):
        r = equity_accumulator(100, 95, 110, 0.05, 0.02, 0.20, 1.0, n_paths=5_000)
        assert 0 < r.knockout_probability < 1

    def test_lower_strike_more_valuable(self):
        r_high = equity_accumulator(100, 98, 110, 0.05, 0.02, 0.20, 1.0, n_paths=5_000)
        r_low = equity_accumulator(100, 90, 110, 0.05, 0.02, 0.20, 1.0, n_paths=5_000)
        assert r_low.price > r_high.price

    def test_accumulation_positive(self):
        r = equity_accumulator(100, 95, 115, 0.05, 0.02, 0.20, 1.0, n_paths=5_000)
        assert r.expected_accumulation > 0
