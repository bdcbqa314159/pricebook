"""Tests for European swaption pricing."""

import pytest
from datetime import date

from pricebook.swaption import Swaption, SwaptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.vol_surface import FlatVol
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
import math


@pytest.fixture
def flat_curve():
    """Flat 3% zero-rate discount curve."""
    ref = date(2024, 1, 15)
    r = 0.03
    dates = [date(2024 + i, 1, 15) for i in range(1, 21)]
    dfs = [math.exp(-r * i) for i in range(1, 21)]
    return DiscountCurve(
        reference_date=ref,
        dates=dates,
        dfs=dfs,
    )


@pytest.fixture
def flat_vol():
    """Flat 20% vol surface."""
    return FlatVol(0.20)


class TestSwaptionConstruction:
    def test_basic_construction(self):
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        assert swn.expiry == date(2025, 1, 15)
        assert swn.swap_end == date(2030, 1, 15)
        assert swn.strike == 0.03
        assert swn.swaption_type == SwaptionType.PAYER
        assert swn.notional == 1_000_000.0

    def test_receiver_swaption(self):
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
            swaption_type=SwaptionType.RECEIVER,
        )
        assert swn.swaption_type == SwaptionType.RECEIVER

    def test_expiry_after_swap_end_raises(self):
        with pytest.raises(ValueError, match="expiry must be before swap_end"):
            Swaption(
                expiry=date(2030, 1, 15),
                swap_end=date(2025, 1, 15),
                strike=0.03,
            )

    def test_expiry_equals_swap_end_raises(self):
        with pytest.raises(ValueError, match="expiry must be before swap_end"):
            Swaption(
                expiry=date(2025, 1, 15),
                swap_end=date(2025, 1, 15),
                strike=0.03,
            )


class TestSwaptionForwardRate:
    def test_forward_swap_rate_flat_curve(self, flat_curve):
        """On a flat curve, forward swap rate ≈ the zero rate."""
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        fwd = swn.forward_swap_rate(flat_curve)
        # On a flat 3% curve, par rate should be close to 3%
        assert fwd == pytest.approx(0.03, abs=0.002)

    def test_annuity_positive(self, flat_curve):
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        ann = swn.annuity(flat_curve)
        assert ann > 0


class TestSwaptionPricing:
    def test_payer_price_positive(self, flat_curve, flat_vol):
        """Payer swaption on a flat curve with ATM strike has positive value."""
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        pv = swn.pv(flat_curve, flat_vol)
        assert pv > 0

    def test_receiver_price_positive(self, flat_curve, flat_vol):
        """Receiver swaption also has positive value."""
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
            swaption_type=SwaptionType.RECEIVER,
        )
        pv = swn.pv(flat_curve, flat_vol)
        assert pv > 0

    def test_payer_receiver_parity(self, flat_curve, flat_vol):
        """Payer - Receiver = PV of the underlying swap (forward swap PV)."""
        strike = 0.03
        payer = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=strike,
            swaption_type=SwaptionType.PAYER,
        )
        receiver = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=strike,
            swaption_type=SwaptionType.RECEIVER,
        )

        pv_payer = payer.pv(flat_curve, flat_vol)
        pv_receiver = receiver.pv(flat_curve, flat_vol)

        # Parity: payer - receiver = notional * annuity * (fwd - strike)
        fwd = payer.forward_swap_rate(flat_curve)
        ann = payer.annuity(flat_curve)
        expected_diff = payer.notional * ann * (fwd - strike)

        assert pv_payer - pv_receiver == pytest.approx(expected_diff, abs=1.0)

    def test_higher_vol_higher_price(self, flat_curve):
        """Higher vol means higher option price."""
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        pv_low = swn.pv(flat_curve, FlatVol(0.10))
        pv_high = swn.pv(flat_curve, FlatVol(0.30))
        assert pv_high > pv_low

    def test_deep_itm_payer_approx_intrinsic(self, flat_curve):
        """Deep ITM payer swaption ≈ intrinsic value."""
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.01,  # well below the ~3% forward rate
        )
        fwd = swn.forward_swap_rate(flat_curve)
        ann = swn.annuity(flat_curve)
        intrinsic = swn.notional * ann * max(fwd - 0.01, 0)

        pv = swn.pv(flat_curve, FlatVol(0.01))  # very low vol
        assert pv == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_payer_near_zero(self, flat_curve):
        """Deep OTM payer swaption ≈ 0."""
        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.10,  # well above the ~3% forward rate
        )
        pv = swn.pv(flat_curve, FlatVol(0.01))  # low vol
        assert pv < 100  # essentially zero on a 1M notional

    def test_longer_expiry_higher_price(self, flat_curve, flat_vol):
        """Longer expiry = more time value = higher price (ATM)."""
        short = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        long = Swaption(
            expiry=date(2026, 1, 15),
            swap_end=date(2031, 1, 15),
            strike=0.03,
        )
        pv_short = short.pv(flat_curve, flat_vol)
        pv_long = long.pv(flat_curve, flat_vol)
        assert pv_long > pv_short
