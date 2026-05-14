"""Kalman filter: state-space models, dynamic beta, signal extraction.

    from pricebook.kalman import KalmanFilter, dynamic_beta, dynamic_hedge_ratio

References:
    Kalman (1960). A New Approach to Linear Filtering and Prediction Problems.
    Harvey (1990). Forecasting, Structural Time Series Models and the Kalman Filter.
    Rauch, Tung & Striebel (1965). Maximum Likelihood Estimates of Linear Dynamic Systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    """Kalman filter/smoother output."""
    filtered_states: np.ndarray    # (T, n_states)
    filtered_covariances: np.ndarray  # (T, n_states, n_states)
    predicted_states: np.ndarray   # (T, n_states)
    log_likelihood: float
    n_obs: int

    def to_dict(self) -> dict:
        return {"log_likelihood": self.log_likelihood, "n_obs": self.n_obs}


class KalmanFilter:
    """Linear Gaussian state-space model.

    State equation:    x_t = F x_{t-1} + w_t,    w ~ N(0, Q)
    Observation:       y_t = H x_t + v_t,        v ~ N(0, R)

    Args:
        F: state transition matrix (n_states × n_states).
        H: observation matrix (n_obs × n_states).
        Q: state noise covariance (n_states × n_states).
        R: observation noise covariance (n_obs × n_obs).
        x0: initial state mean (n_states,). Default: zeros.
        P0: initial state covariance (n_states × n_states). Default: identity.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ):
        self.F = np.atleast_2d(F)
        self.H = np.atleast_2d(H)
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        n = self.F.shape[0]
        self.x = x0 if x0 is not None else np.zeros(n)
        self.P = P0 if P0 is not None else np.eye(n)

    def predict(self):
        """Prediction step: x_{t|t-1} = F x_{t-1|t-1}."""
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray):
        """Update step: incorporate observation y_t."""
        y = np.atleast_1d(y)
        innovation = y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R  # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = x_pred + K @ innovation
        self.P = (np.eye(len(x_pred)) - K @ self.H) @ P_pred
        innov_scalar = float(np.squeeze(innovation))
        S_scalar = float(np.squeeze(S))
        return self.x.copy(), self.P.copy(), innov_scalar, S_scalar

    def filter(self, observations: np.ndarray) -> KalmanResult:
        """Run forward Kalman filter on a sequence of observations.

        Args:
            observations: (T,) or (T, n_obs) array.

        Returns:
            KalmanResult with filtered states and covariances.
        """
        obs = np.atleast_2d(observations)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T  # make column
        T = obs.shape[0]
        n = self.F.shape[0]

        filtered_x = np.zeros((T, n))
        filtered_P = np.zeros((T, n, n))
        predicted_x = np.zeros((T, n))
        log_lik = 0.0

        for t in range(T):
            x_pred, P_pred = self.predict()
            predicted_x[t] = x_pred
            x_filt, P_filt, innov, S = self.update(obs[t], x_pred, P_pred)
            filtered_x[t] = x_filt
            filtered_P[t] = P_filt
            # Log-likelihood contribution
            if S > 0:
                log_lik += -0.5 * (np.log(2 * np.pi * S) + innov ** 2 / S)

        return KalmanResult(
            filtered_states=filtered_x,
            filtered_covariances=filtered_P,
            predicted_states=predicted_x,
            log_likelihood=log_lik,
            n_obs=T,
        )

    def smooth(self, observations: np.ndarray) -> KalmanResult:
        """Rauch-Tung-Striebel (RTS) smoother.

        Forward filter + backward smoothing pass.
        """
        # Forward pass
        obs = np.atleast_2d(observations)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        T = obs.shape[0]
        n = self.F.shape[0]

        # Reset state
        x_save = self.x.copy()
        P_save = self.P.copy()

        filt_x = np.zeros((T, n))
        filt_P = np.zeros((T, n, n))
        pred_x = np.zeros((T, n))
        pred_P = np.zeros((T, n, n))

        for t in range(T):
            x_p, P_p = self.predict()
            pred_x[t] = x_p
            pred_P[t] = P_p
            x_f, P_f, _, _ = self.update(obs[t], x_p, P_p)
            filt_x[t] = x_f
            filt_P[t] = P_f

        # Backward pass
        smooth_x = filt_x.copy()
        smooth_P = filt_P.copy()

        for t in range(T - 2, -1, -1):
            try:
                L = filt_P[t] @ self.F.T @ np.linalg.inv(pred_P[t + 1])
            except np.linalg.LinAlgError:
                continue
            smooth_x[t] = filt_x[t] + L @ (smooth_x[t + 1] - pred_x[t + 1])
            smooth_P[t] = filt_P[t] + L @ (smooth_P[t + 1] - pred_P[t + 1]) @ L.T

        # Restore
        self.x = x_save
        self.P = P_save

        return KalmanResult(
            filtered_states=smooth_x,
            filtered_covariances=smooth_P,
            predicted_states=pred_x,
            log_likelihood=0.0,  # use filter log-lik
            n_obs=T,
        )


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════

def dynamic_beta(
    returns: np.ndarray,
    benchmark: np.ndarray,
    state_noise: float = 0.001,
    obs_noise: float | None = None,
) -> np.ndarray:
    """Time-varying beta via Kalman filter.

    State: beta_t = beta_{t-1} + w_t (random walk)
    Observation: r_t = beta_t × b_t + v_t

    Args:
        returns: portfolio returns.
        benchmark: benchmark returns.
        state_noise: variance of state noise (smaller = smoother beta).
        obs_noise: variance of observation noise. If None, estimated from residuals.

    Returns:
        Array of time-varying betas (same length as returns).
    """
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = len(r)

    if obs_noise is None:
        # Estimate from OLS residuals
        beta_ols = np.cov(r, b)[0, 1] / max(np.var(b), 1e-15)
        resid = r - beta_ols * b
        obs_noise = float(np.var(resid))

    betas = np.zeros(n)
    x = np.cov(r, b)[0, 1] / max(np.var(b), 1e-15)  # OLS initial
    P = 1.0

    for t in range(n):
        # Predict
        x_pred = x
        P_pred = P + state_noise

        # Update
        H_t = b[t]
        S = H_t ** 2 * P_pred + obs_noise
        if abs(S) < 1e-15:
            betas[t] = x_pred
            continue
        K = P_pred * H_t / S
        innovation = r[t] - H_t * x_pred
        x = x_pred + K * innovation
        P = (1 - K * H_t) * P_pred

        betas[t] = x

    return betas


def dynamic_hedge_ratio(
    y: np.ndarray,
    x: np.ndarray,
    state_noise: float = 0.001,
    obs_noise: float | None = None,
) -> np.ndarray:
    """Time-varying hedge ratio for pairs trading via Kalman filter.

    State: h_t = h_{t-1} + w_t (random walk)
    Observation: y_t = h_t × x_t + v_t

    Returns array of time-varying hedge ratios.
    """
    return dynamic_beta(y, x, state_noise, obs_noise)


def trend_extraction(
    series: np.ndarray,
    signal_noise_ratio: float = 0.1,
) -> np.ndarray:
    """Extract trend via local level model (Kalman filter).

    State: mu_t = mu_{t-1} + w_t
    Observation: y_t = mu_t + v_t

    signal_noise_ratio = var(w) / var(v) controls smoothness.
    Small ratio → smoother trend.

    Returns smoothed trend (same length as series).
    """
    y = np.asarray(series, dtype=float)
    obs_var = float(np.var(y))
    state_var = obs_var * signal_noise_ratio

    kf = KalmanFilter(
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[state_var]]),
        R=np.array([[obs_var]]),
        x0=np.array([y[0]]),
        P0=np.array([[obs_var]]),
    )

    result = kf.smooth(y)
    return result.filtered_states[:, 0]
