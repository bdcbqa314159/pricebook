"""Tests for FX exotic extensions: digitals, quantos, var swaps, compound, chooser."""

import pytest
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════
# FX1: Digital Options
# ═══════════════════════════════════════════════════════════════

class TestFXDigital:
    def test_call_atm(self):
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        r = fx_digital_option(1.10, 1.10, 0.04, 0.03, 0.08, 1.0, payout=1.0)
        # ATM digital ≈ 0.5 × df
        assert 0.3 < r.price < 0.7

    def test_deep_itm_call(self):
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        r = fx_digital_option(1.30, 1.10, 0.04, 0.03, 0.08, 1.0, payout=1.0)
        df = math.exp(-0.04)
        assert r.price > 0.8 * df  # deep ITM → near payout × df

    def test_put_call_parity(self):
        """digital_call + digital_put = df × payout."""
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        c = fx_digital_option(1.10, 1.10, 0.04, 0.03, 0.08, 1.0, option_type="call")
        p = fx_digital_option(1.10, 1.10, 0.04, 0.03, 0.08, 1.0, option_type="put")
        df = math.exp(-0.04)
        assert c.price + p.price == pytest.approx(df, rel=0.01)

    def test_overhedge(self):
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        base = fx_digital_option(1.10, 1.10, 0.04, 0.03, 0.08, 1.0)
        shifted = fx_digital_option(1.10, 1.10, 0.04, 0.03, 0.08, 1.0, overhedge_shift=0.005)
        # Overhedge shifts strike up for call → cheaper
        assert shifted.price < base.price

    def test_foreign_payout(self):
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        r = fx_digital_option(1.10, 1.10, 0.04, 0.03, 0.08, 1.0, payout_currency="foreign")
        assert r.price > 0


# ═══════════════════════════════════════════════════════════════
# FX2: Quanto Options
# ═══════════════════════════════════════════════════════════════

class TestFXQuanto:
    def test_positive_corr_higher_forward(self):
        """Positive ρ → lower r_quanto → higher forward → more expensive call."""
        from pricebook.fx.fx_exotic_extensions import fx_quanto_option
        no_corr = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, 0.0, 1.0)
        pos_corr = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, 0.5, 1.0)
        assert pos_corr.price > no_corr.price  # higher forward → more expensive call

    def test_negative_corr_cheaper_call(self):
        from pricebook.fx.fx_exotic_extensions import fx_quanto_option
        no_corr = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, 0.0, 1.0)
        neg_corr = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, -0.5, 1.0)
        assert neg_corr.price < no_corr.price

    def test_quanto_adjustment_sign(self):
        from pricebook.fx.fx_exotic_extensions import fx_quanto_option
        r = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, 0.5, 1.0)
        assert r.quanto_adjustment > 0  # positive corr → higher forward → positive adjustment

    def test_fx_rate_scaling(self):
        from pricebook.fx.fx_exotic_extensions import fx_quanto_option
        r1 = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, 0.3, 1.0, fx_rate=1.0)
        r2 = fx_quanto_option(100, 100, 0.04, 0.02, 0.20, 0.10, 0.3, 1.0, fx_rate=1.5)
        assert r2.price == pytest.approx(r1.price * 1.5, rel=0.01)


# ═══════════════════════════════════════════════════════════════
# FX3: Variance/Vol Swaps
# ═══════════════════════════════════════════════════════════════

class TestFXVarianceSwap:
    def test_flat_smile(self):
        """No butterfly → fair vol ≈ ATM vol."""
        from pricebook.fx.fx_exotic_extensions import fx_variance_swap
        r = fx_variance_swap(0.10, rr25=0.0, bf25=0.0)
        assert r.fair_vol == pytest.approx(10.0, abs=0.1)

    def test_smile_increases_fair_vol(self):
        """Positive butterfly → fair vol > ATM."""
        from pricebook.fx.fx_exotic_extensions import fx_variance_swap
        flat = fx_variance_swap(0.10, bf25=0.0)
        smile = fx_variance_swap(0.10, bf25=0.005)
        assert smile.fair_vol > flat.fair_vol

    def test_mtm_positive_when_realised_above(self):
        from pricebook.fx.fx_exotic_extensions import fx_variance_swap
        r = fx_variance_swap(0.10, realised_vol=0.15)
        assert r.pv > 0  # realised > fair → long variance profits


# ═══════════════════════════════════════════════════════════════
# FX4: Local Vol
# ═══════════════════════════════════════════════════════════════

class TestFXLocalVol:
    def test_build_surface(self):
        from pricebook.fx.fx_exotic_extensions import fx_local_vol
        strikes = [1.05, 1.08, 1.10, 1.12, 1.15]
        expiries = [0.25, 0.5, 1.0]
        vols = [[0.10, 0.09, 0.08, 0.09, 0.10]] * 3
        surf = fx_local_vol(1.10, 0.04, 0.03, strikes, expiries, vols)
        assert surf.local_vols.shape == (3, 5)

    def test_interpolation(self):
        from pricebook.fx.fx_exotic_extensions import fx_local_vol
        strikes = [1.05, 1.10, 1.15]
        expiries = [0.25, 1.0]
        vols = [[0.10, 0.08, 0.10], [0.09, 0.07, 0.09]]
        surf = fx_local_vol(1.10, 0.04, 0.03, strikes, expiries, vols)
        v = surf.vol(0.5, 1.10)
        assert 0.01 < v < 0.30


# ═══════════════════════════════════════════════════════════════
# FX5: Double-Barrier Options
# ═══════════════════════════════════════════════════════════════

class TestFXDoubleBarrier:
    def test_knock_out_cheaper(self):
        """Double knock-out < vanilla."""
        from pricebook.fx.fx_exotic_extensions import fx_double_barrier_option
        r = fx_double_barrier_option(1.10, 1.10, 1.00, 1.20, 0.04, 0.03, 0.08, 1.0)
        assert r.price < r.vanilla_price
        assert r.price >= 0

    def test_knock_in_parity(self):
        """knock-out + knock-in = vanilla."""
        from pricebook.fx.fx_exotic_extensions import fx_double_barrier_option
        ko = fx_double_barrier_option(1.10, 1.10, 1.00, 1.20, 0.04, 0.03, 0.08, 1.0, knock_type="out")
        ki = fx_double_barrier_option(1.10, 1.10, 1.00, 1.20, 0.04, 0.03, 0.08, 1.0, knock_type="in")
        assert ko.price + ki.price == pytest.approx(ko.vanilla_price, rel=0.05)

    def test_wide_barriers_near_vanilla(self):
        """Very wide barriers → price ≈ vanilla."""
        from pricebook.fx.fx_exotic_extensions import fx_double_barrier_option
        r = fx_double_barrier_option(1.10, 1.10, 0.50, 2.00, 0.04, 0.03, 0.08, 1.0)
        assert r.price == pytest.approx(r.vanilla_price, rel=0.10)

    def test_spot_outside_barriers(self):
        from pricebook.fx.fx_exotic_extensions import fx_double_barrier_option
        r = fx_double_barrier_option(0.95, 1.10, 1.00, 1.20, 0.04, 0.03, 0.08, 1.0, knock_type="out")
        assert r.price == 0  # already knocked out


# ═══════════════════════════════════════════════════════════════
# FX6: Compound Options
# ═══════════════════════════════════════════════════════════════

class TestFXCompound:
    def test_call_on_call(self):
        from pricebook.fx.fx_exotic_extensions import fx_compound_option
        r = fx_compound_option(1.10, 1.10, 0.01, 0.04, 0.03, 0.08, 0.5, 1.0,
                                n_sims=20_000)
        assert r.price > 0
        assert r.compound_type == "call_on_call"

    def test_compound_cheaper_than_underlying(self):
        """Compound call-on-call < underlying call value."""
        from pricebook.fx.fx_exotic_extensions import fx_compound_option
        r = fx_compound_option(1.10, 1.10, 0.01, 0.04, 0.03, 0.08, 0.5, 1.0,
                                n_sims=20_000)
        assert r.price < r.underlying_option_price

    def test_put_on_call(self):
        from pricebook.fx.fx_exotic_extensions import fx_compound_option
        r = fx_compound_option(1.10, 1.10, 0.01, 0.04, 0.03, 0.08, 0.5, 1.0,
                                outer_type="put", n_sims=20_000)
        assert r.price >= 0
        assert r.compound_type == "put_on_call"


# ═══════════════════════════════════════════════════════════════
# FX7: Chooser Options
# ═══════════════════════════════════════════════════════════════

class TestFXChooser:
    def test_chooser_more_than_call_or_put(self):
        """Chooser ≥ max(call, put) since holder picks the better one."""
        from pricebook.fx.fx_exotic_extensions import fx_chooser_option
        r = fx_chooser_option(1.10, 1.10, 0.04, 0.03, 0.08, 0.5, 1.0, n_sims=20_000)
        assert r.price >= max(r.call_value, r.put_value) * 0.95

    def test_prob_choose_call_atm(self):
        """ATM: roughly 50/50 call vs put choice."""
        from pricebook.fx.fx_exotic_extensions import fx_chooser_option
        r = fx_chooser_option(1.10, 1.10, 0.04, 0.03, 0.08, 0.5, 1.0, n_sims=20_000)
        assert 0.3 < r.prob_choose_call < 0.7

    def test_longer_choose_date_more_valuable(self):
        """Later choice → more optionality → higher price."""
        from pricebook.fx.fx_exotic_extensions import fx_chooser_option
        short = fx_chooser_option(1.10, 1.10, 0.04, 0.03, 0.08, 0.1, 1.0, n_sims=20_000)
        long = fx_chooser_option(1.10, 1.10, 0.04, 0.03, 0.08, 0.8, 1.0, n_sims=20_000)
        assert long.price >= short.price * 0.95

    def test_to_dict(self):
        from pricebook.fx.fx_exotic_extensions import fx_chooser_option
        r = fx_chooser_option(1.10, 1.10, 0.04, 0.03, 0.08, 0.5, 1.0, n_sims=10_000)
        d = r.to_dict()
        assert "prob_choose_call" in d
