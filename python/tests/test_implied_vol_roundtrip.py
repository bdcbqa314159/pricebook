"""
Slice 11 round-trip validation: implied vol + smile.

1. Implied vol recovers input vol across strikes and expiries
2. Smile-adjusted prices differ from flat-vol prices
3. Put-call parity holds with smile vols
4. ATM implied vol matches flat vol when smile is symmetric
5. Swaption and cap/floor pricing work with smile surface
"""

import pytest
import math
from datetime import date

from pricebook.implied_vol import implied_vol_black76, implied_vol_bachelier
from pricebook.black76 import OptionType, black76_price, bachelier_price
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike
from pricebook.equity_option import equity_option_price
from pricebook.capfloor import CapFloor
from pricebook.swaption import Swaption
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
F, K, T = 100.0, 100.0, 1.0
DF = math.exp(-0.05)


class TestImpliedVolRoundTrip:
    @pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    def test_black76_across_strikes(self, strike, opt):
        vol = 0.25
        price = black76_price(F, strike, vol, T, DF, opt)
        recovered = implied_vol_black76(price, F, strike, T, DF, opt)
        assert recovered == pytest.approx(vol, abs=1e-8)

    @pytest.mark.parametrize("expiry", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_black76_across_expiries(self, expiry):
        vol = 0.20
        price = black76_price(F, K, vol, expiry, DF, OptionType.CALL)
        recovered = implied_vol_black76(price, F, K, expiry, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol, abs=1e-7)

    @pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
    def test_bachelier_across_strikes(self, strike):
        vol_n = 10.0
        price = bachelier_price(F, strike, vol_n, T, DF, OptionType.CALL)
        recovered = implied_vol_bachelier(price, F, strike, T, DF, OptionType.CALL)
        assert recovered == pytest.approx(vol_n, abs=1e-5)


class TestSmileImpact:
    def test_smile_prices_differ_from_flat(self):
        """OTM prices with smile differ from flat vol."""
        flat_vol = 0.20
        smile = VolSmile(
            [90, 95, 100, 105, 110],
            [0.25, 0.22, 0.20, 0.22, 0.25],
        )

        # OTM call at 110: smile vol > flat vol → higher price
        price_flat = black76_price(F, 110, flat_vol, T, DF, OptionType.CALL)
        price_smile = black76_price(F, 110, smile.vol(110), T, DF, OptionType.CALL)
        assert price_smile > price_flat

    def test_atm_matches_when_symmetric(self):
        """ATM vol from a symmetric smile matches the ATM pillar."""
        smile = VolSmile(
            [90, 95, 100, 105, 110],
            [0.25, 0.22, 0.20, 0.22, 0.25],
        )
        assert smile.vol(100.0) == pytest.approx(0.20, abs=1e-10)


class TestPutCallParityWithSmile:
    def test_parity_holds_with_smile_vol(self):
        """Put-call parity depends on forward, not vol. Holds for any vol."""
        smile = VolSmile(
            [90, 95, 100, 105, 110],
            [0.25, 0.22, 0.20, 0.22, 0.25],
        )
        for strike in [90, 95, 100, 105, 110]:
            v = smile.vol(strike)
            c = black76_price(F, strike, v, T, DF, OptionType.CALL)
            p = black76_price(F, strike, v, T, DF, OptionType.PUT)
            expected = DF * (F - strike)
            assert c - p == pytest.approx(expected, abs=1e-10)


class TestSmileSurfaceWithPricing:
    def test_swaption_with_smile_surface(self):
        """Swaption prices with a strike-dependent vol surface."""
        curve = make_flat_curve(REF, 0.03)
        smile_6m = VolSmile([0.01, 0.03, 0.05], [0.22, 0.20, 0.22])
        smile_1y = VolSmile([0.01, 0.03, 0.05], [0.24, 0.22, 0.24])
        surface = VolSurfaceStrike(
            REF,
            [date(2024, 7, 15), date(2025, 1, 15)],
            [smile_6m, smile_1y],
        )

        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.03,
        )
        pv = swn.pv(curve, surface)
        assert pv > 0

    def test_capfloor_with_smile_surface(self):
        """Cap prices with a strike-dependent vol surface."""
        curve = make_flat_curve(REF, 0.03)
        smile = VolSmile([0.01, 0.03, 0.05], [0.22, 0.20, 0.22])
        surface = VolSurfaceStrike(REF, [date(2025, 1, 15)], [smile])

        cap = CapFloor(
            start=REF,
            end=date(2026, 1, 15),
            strike=0.03,
            option_type=OptionType.CALL,
            frequency=Frequency.QUARTERLY,
        )
        pv = cap.pv(curve, surface)
        assert pv > 0

    def test_otm_swaption_smile_vs_flat(self):
        """OTM swaption with smile should differ from flat vol pricing."""
        curve = make_flat_curve(REF, 0.03)

        from pricebook.vol_surface import FlatVol
        flat = FlatVol(0.20)
        smile = VolSmile([0.01, 0.03, 0.05, 0.07], [0.28, 0.20, 0.20, 0.28])
        surface = VolSurfaceStrike(REF, [date(2025, 1, 15)], [smile])

        swn = Swaption(
            expiry=date(2025, 1, 15),
            swap_end=date(2030, 1, 15),
            strike=0.06,  # OTM
        )
        pv_flat = swn.pv(curve, flat)
        pv_smile = swn.pv(curve, surface)
        # Smile has higher vol at OTM strikes → higher price
        assert pv_smile > pv_flat
