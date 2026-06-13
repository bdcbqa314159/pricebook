"""Regression for L2 phase-2 audit of `risk.ipv`:

Pre-fix `ipv_single_trade` had no concept of position direction.
``prudent_value = mid - ava`` was hardcoded for long positions.  Short
positions were silently mispriced: a short bond position's prudent
value should be ABOVE mid (the conservative liability is larger), not
below.  Pre-fix would have under-stated the liability by ``2·ava``.

Additionally, the AVA computation paths used the signed ``notional``
directly.  For ``notional=-1e6`` (short $1M), `close_out_cost_ava`
multiplies by notional, producing a NEGATIVE AVA — which then makes
``mid - ava`` *increase* (subtracting a negative).  Coincidentally
correct in *direction* for shorts but wrong in *magnitude* (because
the AVA sign flip is convolving with the direction).

Fix: take ``abs(notional)`` for AVA size, apply explicit ``direction``
parameter for the AVA-vs-mid sign convention.  Default `direction=+1`
preserves pre-fix long behaviour.
"""

from __future__ import annotations

import pytest

from pricebook.risk.ipv import ipv_single_trade


class TestIpvLongUnchanged:
    def test_long_default_matches_pre_fix(self):
        """Default direction=+1 → prudent_value < mid."""
        r = ipv_single_trade("T_long", "bond", 100.5, 1e6,
                              market_price=100.3, n_quotes=3)
        assert r.prudent_value < r.mid
        # AVA contribution is positive.
        assert r.total_ava > 0


class TestIpvShort:
    def test_short_prudent_above_mid(self):
        """Short position → prudent_value > mid (liability conservatively larger)."""
        r = ipv_single_trade("T_short", "bond", 100.5, 1e6,
                              market_price=100.3, n_quotes=3,
                              direction=-1)
        assert r.prudent_value > r.mid

    def test_short_ava_magnitude_matches_long(self):
        """Same trade, opposite direction: AVA size identical, sign flipped."""
        r_long = ipv_single_trade("T_L", "bond", 100.5, 1e6,
                                    market_price=100.3, n_quotes=3,
                                    direction=+1)
        r_short = ipv_single_trade("T_S", "bond", 100.5, 1e6,
                                     market_price=100.3, n_quotes=3,
                                     direction=-1)
        # |mid - prudent| should match (just opposite sign).
        long_adj = r_long.mid - r_long.prudent_value
        short_adj = r_short.prudent_value - r_short.mid
        assert long_adj == pytest.approx(short_adj, rel=1e-12)
        assert long_adj > 0  # both positive

    def test_negative_notional_treated_as_short_magnitude(self):
        """abs(notional) used for AVA size — negative notional doesn't flip AVA."""
        r_pos = ipv_single_trade("T_P", "bond", 100.0, 1e6,
                                   market_price=100.0, n_quotes=3,
                                   direction=-1)
        r_neg = ipv_single_trade("T_N", "bond", 100.0, -1e6,
                                   market_price=100.0, n_quotes=3,
                                   direction=-1)
        # |mid - prudent| should match — direction param drives sign, not notional.
        assert (r_pos.prudent_value - r_pos.mid) == pytest.approx(
            r_neg.prudent_value - r_neg.mid, rel=1e-12
        )


class TestIpvDirectionValidation:
    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            ipv_single_trade("T_bad", "bond", 100.0, 1e6, direction=0)

    def test_string_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            ipv_single_trade("T_bad", "bond", 100.0, 1e6, direction=+2)


class TestIpvVarianceWithNegativeNotional:
    def test_variance_bp_finite_for_negative_notional(self):
        """Pre-fix: variance_bp denominator used raw notional → negative for shorts.
        ``negative > threshold`` is always False → breach check silently disabled."""
        r = ipv_single_trade("T_neg", "bond", 100.5, -1e6,
                              market_price=100.0, n_quotes=3,
                              direction=-1,
                              variance_threshold_bp=0.001)  # very low threshold
        assert r.variance_to_model_bp >= 0  # never negative now
        # With model_price=100.5 and market_price=100.0 on 1M notional,
        # variance is |0.5|/1e6*10000 = 0.005 bp.  Above threshold 0.001.
        assert r.threshold_breach
