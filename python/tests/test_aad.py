"""Tests for AAD (Adjoint Algorithmic Differentiation)."""

import pytest
import math

from pricebook.aad import Number, Tape, exp, log, sqrt, maximum, minimum, norm_cdf, norm_pdf


@pytest.fixture(autouse=True)
def fresh_tape():
    """Each test gets a fresh tape."""
    tape = Tape()
    Number.tape = tape
    yield tape


class TestBasicArithmetic:
    def test_addition(self):
        x = Number(3.0)
        y = Number(4.0)
        z = x + y
        z.propagate_to_start()
        assert z.value == 7.0
        assert x.adjoint == pytest.approx(1.0)
        assert y.adjoint == pytest.approx(1.0)

    def test_multiplication(self):
        x = Number(3.0)
        y = Number(4.0)
        z = x * y
        z.propagate_to_start()
        assert z.value == 12.0
        assert x.adjoint == pytest.approx(4.0)  # dz/dx = y
        assert y.adjoint == pytest.approx(3.0)  # dz/dy = x

    def test_subtraction(self):
        x = Number(5.0)
        y = Number(3.0)
        z = x - y
        z.propagate_to_start()
        assert z.value == 2.0
        assert x.adjoint == pytest.approx(1.0)
        assert y.adjoint == pytest.approx(-1.0)

    def test_division(self):
        x = Number(6.0)
        y = Number(3.0)
        z = x / y
        z.propagate_to_start()
        assert z.value == 2.0
        assert x.adjoint == pytest.approx(1.0 / 3.0)  # 1/y
        assert y.adjoint == pytest.approx(-6.0 / 9.0)  # -x/y^2

    def test_negation(self):
        x = Number(5.0)
        z = -x
        z.propagate_to_start()
        assert z.value == -5.0
        assert x.adjoint == pytest.approx(-1.0)

    def test_power(self):
        x = Number(3.0)
        z = x ** 2
        z.propagate_to_start()
        assert z.value == 9.0
        assert x.adjoint == pytest.approx(6.0)  # d(x^2)/dx = 2x

    def test_mixed_number_float(self):
        x = Number(3.0)
        z = x * 2.0 + 1.0
        z.propagate_to_start()
        assert z.value == 7.0
        assert x.adjoint == pytest.approx(2.0)

    def test_float_minus_number(self):
        x = Number(3.0)
        z = 10.0 - x
        z.propagate_to_start()
        assert z.value == 7.0
        assert x.adjoint == pytest.approx(-1.0)

    def test_float_divided_by_number(self):
        x = Number(4.0)
        z = 1.0 / x
        z.propagate_to_start()
        assert z.value == 0.25
        assert x.adjoint == pytest.approx(-1.0 / 16.0)


class TestMathFunctions:
    def test_exp(self):
        x = Number(2.0)
        z = exp(x)
        z.propagate_to_start()
        assert z.value == pytest.approx(math.exp(2.0))
        assert x.adjoint == pytest.approx(math.exp(2.0))

    def test_log(self):
        x = Number(3.0)
        z = log(x)
        z.propagate_to_start()
        assert z.value == pytest.approx(math.log(3.0))
        assert x.adjoint == pytest.approx(1.0 / 3.0)

    def test_sqrt(self):
        x = Number(4.0)
        z = sqrt(x)
        z.propagate_to_start()
        assert z.value == pytest.approx(2.0)
        assert x.adjoint == pytest.approx(0.25)  # 1/(2*sqrt(4))

    def test_norm_cdf(self):
        x = Number(0.0)
        z = norm_cdf(x)
        z.propagate_to_start()
        assert z.value == pytest.approx(0.5)
        pdf_at_0 = 1.0 / math.sqrt(2 * math.pi)
        assert x.adjoint == pytest.approx(pdf_at_0)

    def test_maximum(self):
        x = Number(5.0)
        z = maximum(x, 3.0)
        z.propagate_to_start()
        assert z.value == 5.0
        assert x.adjoint == pytest.approx(1.0)

    def test_maximum_below(self):
        x = Number(1.0)
        z = maximum(x, 3.0)
        z.propagate_to_start()
        assert z.value == 3.0
        assert x.adjoint == pytest.approx(0.0)


class TestCompositions:
    def test_chain_rule(self):
        """d/dx exp(x^2) = 2x * exp(x^2)."""
        x = Number(1.5)
        z = exp(x ** 2)
        z.propagate_to_start()
        expected = 2 * 1.5 * math.exp(1.5**2)
        assert x.adjoint == pytest.approx(expected)

    def test_product_of_functions(self):
        """d/dx [x * sin(x)] via AAD vs analytical."""
        x = Number(1.0)
        # sin(x) ≈ x - x^3/6 ... use exp(i*x) trick or just test with known
        # Simpler: d/dx [x * exp(x)] = exp(x) + x*exp(x) = (1+x)*exp(x)
        z = x * exp(x)
        z.propagate_to_start()
        expected = (1 + 1.0) * math.exp(1.0)
        assert x.adjoint == pytest.approx(expected)

    def test_multiple_uses_of_same_variable(self):
        """d/dx [x*x] = 2x (both uses contribute)."""
        x = Number(3.0)
        z = x * x
        z.propagate_to_start()
        assert x.adjoint == pytest.approx(6.0)

    def test_complex_expression(self):
        """d/dx [(x^2 + 1) / (x - 1)] at x=3."""
        x = Number(3.0)
        z = (x ** 2 + 1.0) / (x - 1.0)
        z.propagate_to_start()
        # f(x) = (x^2+1)/(x-1), f'(x) = (2x(x-1) - (x^2+1)) / (x-1)^2
        # = (x^2 - 2x - 1) / (x-1)^2
        expected = (9 - 6 - 1) / 4.0  # = 0.5
        assert x.adjoint == pytest.approx(expected)


class TestBlackScholesGreeks:
    """Compute BS Greeks via AAD and compare to analytical."""

    def _bs_call(self, S, K, r, vol, T):
        """Black-Scholes call using AAD Numbers."""
        d1 = (log(S / K) + (r + vol * vol * 0.5) * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

    def test_delta(self):
        S = Number(100.0)
        K = Number(100.0)
        r = Number(0.05)
        vol = Number(0.20)
        T = Number(1.0)

        price = self._bs_call(S, K, r, vol, T)
        price.propagate_to_start()

        from pricebook.equity_option import equity_delta
        from pricebook.black76 import OptionType
        analytical = equity_delta(100, 100, 0.05, 0.20, 1.0, OptionType.CALL)

        assert S.adjoint == pytest.approx(analytical, rel=0.01)

    def test_vega(self):
        S = Number(100.0)
        K = Number(100.0)
        r = Number(0.05)
        vol = Number(0.20)
        T = Number(1.0)

        price = self._bs_call(S, K, r, vol, T)
        price.propagate_to_start()

        from pricebook.equity_option import equity_vega
        analytical = equity_vega(100, 100, 0.05, 0.20, 1.0)

        assert vol.adjoint == pytest.approx(analytical, rel=0.01)

    def test_rho(self):
        S = Number(100.0)
        K = Number(100.0)
        r = Number(0.05)
        vol = Number(0.20)
        T = Number(1.0)

        price = self._bs_call(S, K, r, vol, T)
        price.propagate_to_start()

        from pricebook.equity_option import equity_rho
        from pricebook.black76 import OptionType
        analytical = equity_rho(100, 100, 0.05, 0.20, 1.0, OptionType.CALL)

        assert r.adjoint == pytest.approx(analytical, rel=0.01)

    def test_all_greeks_one_pass(self):
        """All Greeks computed in a single backward pass."""
        S = Number(100.0)
        K = Number(100.0)
        r = Number(0.05)
        vol = Number(0.20)
        T = Number(1.0)

        price = self._bs_call(S, K, r, vol, T)
        price.propagate_to_start()

        # All adjoints are available simultaneously
        assert S.adjoint != 0  # delta
        assert vol.adjoint != 0  # vega
        assert r.adjoint != 0  # rho
        assert T.adjoint != 0  # theta-related


class TestTapeOperations:
    def test_clear(self):
        x = Number(1.0)
        _ = x + x
        assert Number.tape.size > 0
        Number.tape.clear()
        assert Number.tape.size == 0

    def test_mark_rewind(self):
        x = Number(1.0)
        Number.tape.mark()
        y = x + x
        assert Number.tape.size > 1
        Number.tape.rewind_to_mark()
        assert Number.tape.size == 1  # only x remains

    def test_reset_adjoints(self):
        x = Number(3.0)
        z = x * x
        z.propagate_to_start()
        assert x.adjoint != 0
        Number.tape.reset_adjoints()
        assert x.adjoint == 0

    def test_mark_rewind_mc_pattern(self):
        """Simulate the MC Greeks pattern: mark, loop, propagate."""
        spot = Number(100.0)
        vol = Number(0.20)
        Number.tape.mark()

        total_delta = 0.0
        total_vega = 0.0

        for z_val in [0.5, -0.3, 1.2]:
            Number.tape.rewind_to_mark()
            Number.tape.reset_adjoints()

            # Simplified "path": S_T = spot * exp((r-0.5*vol^2)*T + vol*sqrt(T)*z)
            r_val = 0.05
            T_val = 1.0
            drift = (r_val - 0.5 * vol * vol) * T_val
            diffusion = vol * sqrt(Number(T_val)) * z_val
            s_T = spot * exp(drift + diffusion)
            payoff = maximum(s_T - 100.0, 0.0)

            payoff.propagate_to_mark()
            total_delta += spot.adjoint
            total_vega += vol.adjoint

        # Average Greeks over paths
        n = 3
        avg_delta = total_delta / n
        avg_vega = total_vega / n

        # Delta should be positive for a call
        assert avg_delta > 0
        # Vega should be positive (higher vol = higher option value)
        assert avg_vega > 0


class TestCompoundAssignment:
    def test_iadd(self):
        x = Number(3.0)
        y = Number(2.0)
        z = Number(0.0)
        z += x
        z += y
        z.propagate_to_start()
        assert z.value == 5.0
        assert x.adjoint == pytest.approx(1.0)
        assert y.adjoint == pytest.approx(1.0)

    def test_imul(self):
        x = Number(3.0)
        z = Number(1.0)
        z *= x
        z *= x
        z.propagate_to_start()
        assert z.value == 9.0

    def test_isub(self):
        x = Number(10.0)
        y = Number(3.0)
        z = Number(0.0)
        z += x
        z -= y
        z.propagate_to_start()
        assert z.value == 7.0
        assert y.adjoint == pytest.approx(-1.0)

    def test_itruediv(self):
        x = Number(6.0)
        y = Number(2.0)
        z = x
        z /= y
        z.propagate_to_start()
        assert z.value == 3.0


class TestNewMathFunctions:
    def test_norm_pdf(self):
        x = Number(0.0)
        z = norm_pdf(x)
        z.propagate_to_start()
        expected_val = 1.0 / math.sqrt(2 * math.pi)
        assert z.value == pytest.approx(expected_val)
        assert x.adjoint == pytest.approx(0.0, abs=1e-10)  # -x * pdf(x) at x=0

    def test_minimum(self):
        x = Number(3.0)
        y = Number(5.0)
        z = minimum(x, y)
        z.propagate_to_start()
        assert z.value == pytest.approx(3.0)

    def test_pos(self):
        x = Number(5.0)
        z = +x
        assert z.value == 5.0


class TestContextManager:
    def test_tape_context(self):
        with Tape() as tape:
            x = Number(2.0)
            y = x * x
            y.propagate_to_start()
            assert x.adjoint == pytest.approx(4.0)
        assert tape.size == 0  # cleared on exit


class TestPutOnTape:
    def test_re_register(self):
        x = Number(5.0)
        z = x * 2.0
        z.propagate_to_start()
        assert x.adjoint == pytest.approx(2.0)

        Number.tape.reset_adjoints()
        x.put_on_tape()
        z2 = x * 3.0
        z2.propagate_to_start()
        assert x.adjoint == pytest.approx(3.0)


class TestHashable:
    def test_in_dict(self):
        x = Number(1.0)
        y = Number(2.0)
        d = {x: "a", y: "b"}
        assert d[x] == "a"

    def test_in_set(self):
        x = Number(1.0)
        y = Number(2.0)
        s = {x, y}
        assert len(s) == 2


class TestThreadSafety:
    def test_thread_local_tapes(self):
        """Different threads get independent tapes."""
        import threading
        results = {}

        def worker(val, name):
            tape = Tape()
            Number.tape = tape
            x = Number(val)
            z = x * x
            z.propagate_to_start()
            results[name] = x.adjoint

        t1 = threading.Thread(target=worker, args=(3.0, "t1"))
        t2 = threading.Thread(target=worker, args=(5.0, "t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == pytest.approx(6.0)
        assert results["t2"] == pytest.approx(10.0)
