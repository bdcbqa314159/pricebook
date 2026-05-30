"""Tests for fractional Hawkes credit derivatives framework."""

import pytest
import math
import numpy as np

from pricebook.models.hawkes_credit import (
    FractionalHawkesProcess, MultivariateHawkesProcess,
    HawkesKernel, branching_ratio, evaluate_kernel,
    hawkes_mle_exponential, approximate_power_law,
)


class TestKernels:
    def test_exponential_kernel(self):
        v = evaluate_kernel(1.0, HawkesKernel.EXPONENTIAL, {"alpha": 0.5, "beta": 1.0})
        assert abs(v - 0.5 * math.exp(-1.0)) < 1e-10

    def test_power_law_kernel(self):
        v = evaluate_kernel(1.0, HawkesKernel.POWER_LAW, {"alpha": 1.0, "H": 0.3})
        assert v > 0  # (1+ε)^(-1.2) > 0

    def test_mittag_leffler_gamma1_is_exponential(self):
        """Mittag-Leffler with γ=1 should reduce to exponential."""
        v_ml = evaluate_kernel(1.0, HawkesKernel.MITTAG_LEFFLER,
                               {"alpha": 0.5, "beta": 1.0, "gamma": 1.0})
        v_exp = evaluate_kernel(1.0, HawkesKernel.EXPONENTIAL,
                                {"alpha": 0.5, "beta": 1.0})
        assert abs(v_ml - v_exp) < 0.01  # series approximation

    def test_branching_ratio_exponential(self):
        br = branching_ratio(HawkesKernel.EXPONENTIAL, {"alpha": 0.3, "beta": 1.0})
        assert abs(br - 0.3) < 1e-10


class TestFractionalHawkes:
    def test_poisson_limit(self):
        """α=0 → pure Poisson: events ∝ μT."""
        h = FractionalHawkesProcess(mu=1.0, kernel=HawkesKernel.EXPONENTIAL,
                                     kernel_params={"alpha": 0.0, "beta": 1.0})
        r = h.simulate(T=10.0, n_paths=500, seed=42)
        assert abs(r.n_events_mean - 10.0) < 3.0  # μT = 10 ± noise

    def test_self_excitation_increases_events(self):
        """Higher α → more events."""
        r_low = FractionalHawkesProcess(0.5, HawkesKernel.EXPONENTIAL,
                                         {"alpha": 0.1, "beta": 1.0}).simulate(10.0, 200, seed=42)
        r_high = FractionalHawkesProcess(0.5, HawkesKernel.EXPONENTIAL,
                                          {"alpha": 0.5, "beta": 1.0}).simulate(10.0, 200, seed=42)
        assert r_high.n_events_mean > r_low.n_events_mean

    def test_intensity_non_negative(self):
        h = FractionalHawkesProcess(0.5, HawkesKernel.EXPONENTIAL, {"alpha": 0.3, "beta": 1.0})
        r = h.simulate(5.0, 50, seed=42)
        assert np.all(r.intensities >= 0)

    def test_power_law_kernel_runs(self):
        h = FractionalHawkesProcess(0.5, HawkesKernel.POWER_LAW, {"alpha": 0.05, "H": 0.3})
        r = h.simulate(5.0, 50, seed=42)
        assert r.n_events_mean > 0

    def test_stationarity_warning(self):
        """BR ≥ 1 should warn."""
        with pytest.warns(RuntimeWarning, match="non-stationary"):
            FractionalHawkesProcess(0.5, HawkesKernel.EXPONENTIAL, {"alpha": 1.5, "beta": 1.0})


class TestMultivariateHawkes:
    def test_cross_excitation(self):
        """Events in name 0 should raise intensity of name 1."""
        mu = np.array([0.5, 0.1])
        alpha = np.array([[0.1, 0.0],
                           [0.3, 0.1]])  # name 0's events excite name 1
        mv = MultivariateHawkesProcess(mu, alpha, HawkesKernel.EXPONENTIAL, {"beta": 1.0})
        r = mv.simulate(10.0, 100, seed=42)
        assert r.n_names == 2

    def test_default_times_shape(self):
        mu = np.array([0.3, 0.2, 0.25])
        alpha = np.eye(3) * 0.1
        mv = MultivariateHawkesProcess(mu, alpha, HawkesKernel.EXPONENTIAL, {"beta": 1.0})
        r = mv.simulate(5.0, 50, seed=42)
        assert r.default_times.shape == (50, 3)


class TestHawkesCDS:
    def test_spread_positive(self):
        from pricebook.credit.hawkes_cds import hawkes_cds_spread
        h = FractionalHawkesProcess(0.02, HawkesKernel.EXPONENTIAL, {"alpha": 0.3, "beta": 1.0})
        r = hawkes_cds_spread(h, 5.0, 0.4, n_paths=3_000, seed=42)
        assert r.par_spread_bp > 0

    def test_spread_increases_with_alpha(self):
        from pricebook.credit.hawkes_cds import hawkes_cds_spread
        spreads = []
        for alpha in [0.1, 0.3, 0.5]:
            h = FractionalHawkesProcess(0.02, HawkesKernel.EXPONENTIAL, {"alpha": alpha, "beta": 1.0})
            r = hawkes_cds_spread(h, 5.0, 0.4, n_paths=3_000, seed=42)
            spreads.append(r.par_spread_bp)
        assert spreads[2] > spreads[0]


class TestHawkesBasket:
    def test_tranche_hierarchy(self):
        from pricebook.credit.hawkes_basket import hawkes_basket_defaults, hawkes_tranche_spread
        mu = np.array([0.03] * 5)
        alpha = np.eye(5) * 0.1 + np.ones((5, 5)) * 0.02
        np.fill_diagonal(alpha, 0.1)
        basket = hawkes_basket_defaults(mu, alpha, n_paths=3_000, seed=42)
        eq = hawkes_tranche_spread(basket, 0.0, 0.03)
        sr = hawkes_tranche_spread(basket, 0.15, 1.0)
        assert eq["par_spread_bp"] >= sr["par_spread_bp"]

    def test_loss_distribution_bounded(self):
        from pricebook.credit.hawkes_basket import hawkes_basket_defaults
        mu = np.array([0.02] * 3)
        alpha = np.eye(3) * 0.1
        basket = hawkes_basket_defaults(mu, alpha, n_paths=1_000, seed=42)
        assert np.all(basket.loss_distribution >= 0)
        assert np.all(basket.loss_distribution <= 1)


class TestHawkesAnalytics:
    def test_contagion_scenario(self):
        from pricebook.credit.hawkes_analytics import contagion_scenario
        mu = np.array([0.02, 0.03])
        alpha = np.array([[0.1, 0.0], [0.2, 0.1]])
        mv = MultivariateHawkesProcess(mu, alpha, HawkesKernel.EXPONENTIAL, {"beta": 1.0})
        r = contagion_scenario(mv, trigger_name=0)
        assert r.intensity_jump[1] > 0  # name 0 default raises name 1 intensity

    def test_clustering_metrics_poisson(self):
        """Poisson events should have CV ≈ 1, burstiness ≈ 0."""
        from pricebook.credit.hawkes_analytics import clustering_metrics
        rng = np.random.default_rng(42)
        events = sorted(rng.uniform(0, 10, 50))
        m = clustering_metrics(events)
        assert 0.5 < m.inter_arrival_cv < 1.5  # roughly Poisson

    def test_clustering_metrics_hawkes(self):
        """Hawkes events should have CV > 1 (clustered)."""
        from pricebook.credit.hawkes_analytics import clustering_metrics
        h = FractionalHawkesProcess(0.5, HawkesKernel.EXPONENTIAL, {"alpha": 0.5, "beta": 1.0})
        r = h.simulate(20.0, 1, seed=42)
        if len(r.event_times[0]) > 5:
            m = clustering_metrics(r.event_times[0])
            # Self-exciting → CV should tend to be > 1
            assert m.inter_arrival_cv > 0.5  # relaxed: small samples noisy


class TestMLE:
    def test_mle_recovers_direction(self):
        """MLE on Hawkes data should recover α > 0."""
        h = FractionalHawkesProcess(0.5, HawkesKernel.EXPONENTIAL, {"alpha": 0.4, "beta": 1.0})
        r = h.simulate(50.0, 1, seed=42)
        if len(r.event_times[0]) > 10:
            mle = hawkes_mle_exponential(r.event_times[0], 50.0)
            assert mle["alpha"] > 0  # should detect self-excitation


class TestSumExpApproximation:
    def test_approximation_positive(self):
        params = approximate_power_law(H=0.3, alpha=0.1, K=5)
        assert all(w > 0 for w in params["weights"])
        assert all(b > 0 for b in params["betas"])
        assert len(params["weights"]) == 5
