"""Tests for FX hedging structures."""
from __future__ import annotations
import math
import pytest
from pricebook.fx.fx_hedging import (
    window_barrier_option, fader_option, participating_forward, seagull,
)


class TestWindowBarrier:
    def test_price_positive(self):
        r = window_barrier_option(1.10, 1.10, 1.20, 0.03, 0.01, 0.08, 1.0, 0.25, 0.75, n_paths=5_000)
        assert r.price > 0

    def test_narrower_window_cheaper_ko(self):
        full = window_barrier_option(1.10, 1.10, 1.20, 0.03, 0.01, 0.08, 1.0, 0.0, 1.0, n_paths=10_000)
        narrow = window_barrier_option(1.10, 1.10, 1.20, 0.03, 0.01, 0.08, 1.0, 0.4, 0.6, n_paths=10_000)
        assert narrow.price > full.price * 0.8  # narrower window = fewer knockouts = higher price

    def test_ko_probability_bounded(self):
        r = window_barrier_option(1.10, 1.10, 1.20, 0.03, 0.01, 0.08, 1.0, 0.0, 1.0, n_paths=5_000)
        assert 0 <= r.knockout_probability <= 1


class TestFader:
    def test_price_positive(self):
        r = fader_option(1.10, 1.10, 1.20, 0.03, 0.01, 0.08, 1.0, n_paths=5_000)
        assert r.price > 0

    def test_fading_bounded(self):
        r = fader_option(1.10, 1.10, 1.20, 0.03, 0.01, 0.08, 1.0, n_paths=5_000)
        assert 0 < r.average_fading_factor <= 1

    def test_low_barrier_more_fading(self):
        high = fader_option(1.10, 1.10, 1.50, 0.03, 0.01, 0.08, 1.0, n_paths=5_000)
        low = fader_option(1.10, 1.10, 1.15, 0.03, 0.01, 0.08, 1.0, n_paths=5_000)
        assert low.average_fading_factor < high.average_fading_factor


class TestParticipatingForward:
    def test_zero_cost_solve(self):
        r = participating_forward(1.10, 0.03, 0.01, 0.08, 1.0, floor_rate=1.08)
        assert r.zero_cost
        assert 0 < r.participation_rate < 1
        assert abs(r.price) < 100  # near zero cost

    def test_participation_bounded(self):
        r = participating_forward(1.10, 0.03, 0.01, 0.08, 1.0)
        assert 0 < r.participation_rate <= 1

    def test_deeper_otm_floor_lower_participation(self):
        # Floor below forward → cheap put → low participation needed to offset
        # Floor above forward → expensive put → high participation to offset
        otm = participating_forward(1.10, 0.03, 0.01, 0.08, 1.0, floor_rate=1.05)
        itm = participating_forward(1.10, 0.03, 0.01, 0.08, 1.0, floor_rate=1.12)
        assert otm.participation_rate < itm.participation_rate


class TestSeagull:
    def test_near_zero_cost(self):
        # Choose strikes for near-zero cost
        r = seagull(1.10, 0.03, 0.01, 0.08, 1.0, 1.08, 1.02, 1.15)
        assert math.isfinite(r.price)

    def test_wider_cap_more_expensive(self):
        tight = seagull(1.10, 0.03, 0.01, 0.08, 1.0, 1.08, 1.02, 1.13)
        wide = seagull(1.10, 0.03, 0.01, 0.08, 1.0, 1.08, 1.02, 1.20)
        # Higher cap = less premium received from sold call = higher cost
        assert wide.price > tight.price

    def test_forward_positive(self):
        r = seagull(1.10, 0.03, 0.01, 0.08, 1.0, 1.08, 1.02, 1.15)
        assert r.forward_rate > 0
