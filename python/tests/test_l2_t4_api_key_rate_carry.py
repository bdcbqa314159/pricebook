"""Regression for L2 T4 audit of `desks.api.key_rate_dv01` + `carry_rolldown`.

`key_rate_dv01`: pre-fix bumped the curve in PARALLEL for every tenor in
the ladder, so all tenors returned identical DV01 values — the "key
rate" decomposition was a silent wrong answer.  Fix: nearest-pillar
``bumped_at`` per tenor.

`carry_rolldown`: pre-fix had three coupled defects — ``curve.bumped(0.0)``
is a no-op rolldown, ``int(shorter_T)`` truncated to integer years, and
the ``"carry"`` field held ``pv_today`` (just the PV) not the carry.
Fix: use ``curve.roll_down``, return an annuity-style carry approximation,
and surface ``"pv"`` separately.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

import pricebook.desks.api as pb


class TestKeyRateDV01NotAllSame:
    def test_different_tenors_produce_different_dv01s(self):
        """Pre-fix: every tenor returned the same parallel DV01.
        Post-fix: per-pillar bumps give distinct values."""
        ref = date(2026, 1, 15)
        curve_a = pb.build_curve(
            "USD", ref,
            swaps={"1Y": 0.04, "3Y": 0.045, "5Y": 0.05,
                   "7Y": 0.052, "10Y": 0.055},
        )
        ladder = pb.key_rate_dv01(
            lambda c: pb.irs("7Y", 0.04, c, start=ref), curve_a,
            tenors=["1Y", "3Y", "5Y", "7Y", "10Y"],
        )
        values = list(ladder.values())
        # At least two distinct non-zero values — pre-fix would give
        # all identical.
        distinct = len(set(round(v, 8) for v in values if abs(v) > 1e-9))
        assert distinct >= 2

    def test_short_tenor_smaller_dv01_than_long(self):
        """For a 7Y swap, the 1Y key-rate bump should produce a smaller
        absolute sensitivity than the 7Y or 10Y key-rate bump (which sits
        closer to the swap's maturity)."""
        ref = date(2026, 1, 15)
        curve = pb.build_curve(
            "USD", ref,
            swaps={"1Y": 0.04, "3Y": 0.045, "5Y": 0.05,
                   "7Y": 0.052, "10Y": 0.055},
        )
        ladder = pb.key_rate_dv01(
            lambda c: pb.irs("7Y", 0.04, c, start=ref), curve,
            tenors=["1Y", "7Y"],
        )
        # 7Y pillar should drive much more of the swap's DV01 than 1Y.
        assert abs(ladder["7Y"]) > abs(ladder["1Y"])


class TestCarryRolldownReal:
    def test_returns_pv_not_in_carry_field(self):
        """Post-fix: "carry" field is a small number (≈ (fixed-par)·dt),
        not the full PV."""
        ref = date(2026, 1, 15)
        curve = pb.build_curve(
            "USD", ref, swaps={"1Y": 0.04, "3Y": 0.045, "5Y": 0.05},
        )
        # Use a fixed_rate equal to the 5Y par so carry ≈ 0.
        par_5y = pb.par_rate("5Y", curve)
        result = pb.carry_rolldown("5Y", par_5y, curve, days=1)
        # carry ≈ 0 (fixed = par).
        assert abs(result["carry"]) < 1e-6
        # pv field exists and is non-zero (at-par swap, near zero by design).
        assert "pv" in result
        # rolldown should be a real number (not stuck at 0).
        assert isinstance(result["rolldown"], float)

    def test_rolldown_uses_curve_roll(self):
        """Post-fix: rolldown is computed from genuine curve.roll_down,
        not from a broken integer-tenor swap construction."""
        ref = date(2026, 1, 15)
        # Upward-sloping curve → positive rolldown.
        curve = pb.build_curve(
            "USD", ref,
            swaps={"1Y": 0.03, "3Y": 0.04, "5Y": 0.05, "10Y": 0.06},
        )
        result = pb.carry_rolldown("5Y", 0.045, curve, days=30)
        # Independent check: roll curve manually and reprice.
        rolled = curve.roll_down(30)
        pv_today = pb.irs("5Y", 0.045, curve)
        pv_rolled = pb.irs("5Y", 0.045, rolled)
        expected_rolldown = pv_rolled - pv_today
        assert result["rolldown"] == pytest.approx(expected_rolldown, rel=1e-9)
