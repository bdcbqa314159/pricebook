"""Tests for AAD swaptions and caplets."""

import math
import pytest

from pricebook.aad import Number, Tape
from pricebook.aad_pricing import aad_swaption_pv, aad_caplet_pv
from pricebook.black76 import black76_price, OptionType


class TestAADSwaption:
    def test_payer_positive(self):
        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            ann = Number(4.0)
            pv = aad_swaption_pv(fwd, 0.05, vol, 1.0, ann)
            assert pv.value > 0

    def test_all_greeks_one_pass(self):
        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            ann = Number(4.0)
            pv = aad_swaption_pv(fwd, 0.05, vol, 1.0, ann)
            pv.propagate_to_start()

            delta = fwd.adjoint
            vega = vol.adjoint
            ann_sens = ann.adjoint

            assert delta > 0  # payer delta positive
            assert vega > 0   # long vega
            assert ann_sens > 0  # annuity sensitivity positive

    def test_delta_fd_check(self):
        eps = 1e-5
        K, vol_val, T, ann_val = 0.05, 0.20, 1.0, 4.0

        with Tape():
            fwd = Number(0.05)
            vol = Number(vol_val)
            ann = Number(ann_val)
            pv = aad_swaption_pv(fwd, K, vol, T, ann)
            pv.propagate_to_start()
            aad_delta = fwd.adjoint

        with Tape():
            pv_up = aad_swaption_pv(Number(0.05 + eps), K, Number(vol_val), T, Number(ann_val)).value
        with Tape():
            pv_dn = aad_swaption_pv(Number(0.05 - eps), K, Number(vol_val), T, Number(ann_val)).value

        fd_delta = (pv_up - pv_dn) / (2 * eps)
        assert aad_delta == pytest.approx(fd_delta, rel=1e-3)

    def test_vega_fd_check(self):
        eps = 1e-5

        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            ann = Number(4.0)
            pv = aad_swaption_pv(fwd, 0.05, vol, 1.0, ann)
            pv.propagate_to_start()
            aad_vega = vol.adjoint

        with Tape():
            pv_up = aad_swaption_pv(Number(0.05), 0.05, Number(0.20 + eps), 1.0, Number(4.0)).value
        with Tape():
            pv_dn = aad_swaption_pv(Number(0.05), 0.05, Number(0.20 - eps), 1.0, Number(4.0)).value

        fd_vega = (pv_up - pv_dn) / (2 * eps)
        assert aad_vega == pytest.approx(fd_vega, rel=1e-3)

    def test_receiver(self):
        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            ann = Number(4.0)
            pv = aad_swaption_pv(fwd, 0.05, vol, 1.0, ann, is_payer=False)
            assert pv.value > 0


class TestAADCaplet:
    def test_caplet_positive(self):
        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            df = Number(0.95)
            pv = aad_caplet_pv(fwd, 0.05, vol, 1.0, 0.25, df)
            assert pv.value > 0

    def test_all_greeks(self):
        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            df = Number(0.95)
            pv = aad_caplet_pv(fwd, 0.05, vol, 1.0, 0.25, df, notional=1e6)
            pv.propagate_to_start()

            assert fwd.adjoint > 0  # caplet delta
            assert vol.adjoint > 0  # caplet vega
            assert df.adjoint > 0   # df sensitivity

    def test_vega_fd_check(self):
        eps = 1e-5

        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            df = Number(0.95)
            pv = aad_caplet_pv(fwd, 0.05, vol, 1.0, 0.25, df)
            pv.propagate_to_start()
            aad_vega = vol.adjoint

        with Tape():
            pv_up = aad_caplet_pv(Number(0.05), 0.05, Number(0.20 + eps), 1.0, 0.25, Number(0.95)).value
        with Tape():
            pv_dn = aad_caplet_pv(Number(0.05), 0.05, Number(0.20 - eps), 1.0, 0.25, Number(0.95)).value

        fd = (pv_up - pv_dn) / (2 * eps)
        assert aad_vega == pytest.approx(fd, rel=1e-3)

    def test_floorlet(self):
        with Tape():
            fwd = Number(0.05)
            vol = Number(0.20)
            df = Number(0.95)
            pv = aad_caplet_pv(fwd, 0.05, vol, 1.0, 0.25, df, is_cap=False)
            assert pv.value > 0
