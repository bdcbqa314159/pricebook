"""2D recombining trinomial lattice for the G2++ two-factor model.

Supports backward induction for path-dependent products such as callable bonds
and Bermudan swaptions.  Each factor (x, y) is independently discretised on a
trinomial tree following Hull-White / Brigo-Mercurio conventions.  The full 2D
state space is the Cartesian product of the two 1D grids; joint transition
probabilities are assembled from the independent marginals with a first-order
correlation correction.

* :class:`G2PPTreeResult` — pricing result container.
* :class:`G2PPTree` — 2D recombining trinomial tree for G2++.
* :func:`g2pp_european_swaption_tree` — European swaption on the 2D tree
  (used to cross-validate the analytical Jamshidian formula).

References:
    Brigo, D. & Mercurio, F., *Interest Rate Models — Theory and Practice*,
    2nd ed., Springer, 2006, Ch. 4.3 (G2++ discretisation).
    Hull, J., *Options, Futures, and Other Derivatives*, 11th ed., Ch. 33.
    Hull, J. & White, A., "Numerical Procedures for Implementing Term
    Structure Models II: Two-Factor Models", *Journal of Derivatives*, 1994.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pricebook.models.vasicek import G2PlusPlus
from pricebook.core.day_count import date_from_year_fraction


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class G2PPTreeResult:
    """Container for a G2++ tree pricing result."""

    price: float
    n_x_nodes: int       # total x-axis nodes = 2 * j_max_x + 1
    n_y_nodes: int       # total y-axis nodes = 2 * j_max_y + 1
    n_time_steps: int

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "n_x_nodes": self.n_x_nodes,
            "n_y_nodes": self.n_y_nodes,
            "n_time_steps": self.n_time_steps,
        }


# ---------------------------------------------------------------------------
# G2PPTree
# ---------------------------------------------------------------------------

class G2PPTree:
    """2D recombining trinomial lattice for the G2++ model.

    The two OU factors share the same time grid but independent spatial grids:

        x-grid:  nodes  x_j = j * dx,  j in [−j_max_x, +j_max_x]
        y-grid:  nodes  y_k = k * dy,  k in [−j_max_y, +j_max_y]

    where  dx = sigma1 * sqrt(3 * dt)  (standard Hull-White trinomial
    spacing) and likewise for dy.  j_max is chosen via the criterion
    0.1835 / (a * dt) (Hull & White, 1994) capped by n_std standard
    deviations of the stationary distribution — whichever is larger.

    Transition probabilities for each 1D tree are the standard
    mean-reversion trinomial formula.  At the boundary nodes the
    branching is reflected so that node indices never leave the grid.

    Correlation between the two factors is incorporated via a first-order
    correction to the joint transition probabilities (Brigo-Mercurio 4.3.3):

        p_joint(u, v) ≈ p_x(u) * p_y(v) + ρ * δ(u) * δ(v) * sqrt(...) * dt

    where δ(up)=+1, δ(mid)=0, δ(down)=−1.

    Parameters
    ----------
    g2pp:
        Calibrated G2PlusPlus instance (carries a, b, sigma1, sigma2, rho,
        and a DiscountCurve for phi).
    T:
        Tree horizon in years.
    n_steps:
        Number of time steps.
    n_std:
        Grid half-width in stationary standard deviations (default 4).
    """

    def __init__(
        self,
        g2pp: G2PlusPlus,
        T: float,
        n_steps: int,
        n_std: float = 4.0,
    ) -> None:
        self.g2pp = g2pp
        self.T = T
        self.n_steps = n_steps
        self.n_std = n_std
        self.dt = T / n_steps
        self.times = np.linspace(0.0, T, n_steps + 1)

        self._build_grids()
        self._build_transition_tables()

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grids(self) -> None:
        a, b_ = self.g2pp.a, self.g2pp.b
        s1, s2 = self.g2pp.sigma1, self.g2pp.sigma2
        dt = self.dt

        # Trinomial spacing: Hull-White standard choice
        self.dx = s1 * math.sqrt(3.0 * dt)
        self.dy = s2 * math.sqrt(3.0 * dt)

        # j_max: hull-white criterion and n_std cap
        hw_jmax_x = max(1, int(math.ceil(0.1835 / (a * dt)))) if a > 0 else 10
        hw_jmax_y = max(1, int(math.ceil(0.1835 / (b_ * dt)))) if b_ > 0 else 10

        stat_sx = s1 / math.sqrt(2.0 * a) if a > 0 else s1 * math.sqrt(self.T)
        stat_sy = s2 / math.sqrt(2.0 * b_) if b_ > 0 else s2 * math.sqrt(self.T)
        nstd_jmax_x = max(1, math.ceil(self.n_std * stat_sx / self.dx))
        nstd_jmax_y = max(1, math.ceil(self.n_std * stat_sy / self.dy))

        self.j_max_x = max(hw_jmax_x, nstd_jmax_x)
        self.j_max_y = max(hw_jmax_y, nstd_jmax_y)

        self.n_x = 2 * self.j_max_x + 1
        self.n_y = 2 * self.j_max_y + 1

        self.x_nodes = np.arange(-self.j_max_x, self.j_max_x + 1) * self.dx
        self.y_nodes = np.arange(-self.j_max_y, self.j_max_y + 1) * self.dy

    # ------------------------------------------------------------------
    # Trinomial probabilities
    # ------------------------------------------------------------------

    @staticmethod
    def _trinomial_probs_1d(
        j: int,
        a_factor: float,
        dt: float,
        j_max: int,
    ) -> tuple[float, float, float]:
        """(p_up, p_mid, p_down) for OU factor at signed index j.

        Uses the Hull-White formula with boundary reflection: when the
        natural central destination lies outside [−j_max, +j_max] the
        branching pattern is shifted.
        """
        j2 = j * j
        a_dt = a_factor * dt

        p_up = 1.0 / 6.0 + (j2 * a_dt * a_dt - j * a_dt) / 6.0
        p_mid = 2.0 / 3.0 - j2 * a_dt * a_dt / 3.0
        p_dn = 1.0 / 6.0 + (j2 * a_dt * a_dt + j * a_dt) / 6.0

        # Clamp negative probabilities (occurs near boundary)
        p_up = max(0.0, p_up)
        p_mid = max(0.0, p_mid)
        p_dn = max(0.0, p_dn)
        total = p_up + p_mid + p_dn
        if total > 0.0:
            inv = 1.0 / total
            p_up *= inv
            p_mid *= inv
            p_dn *= inv
        return p_up, p_mid, p_dn

    def _build_transition_tables(self) -> None:
        """Pre-compute 1D trinomial tables for both factors at every step."""
        a, b_ = self.g2pp.a, self.g2pp.b
        dt = self.dt

        # _dest_x[t, xi, branch] = destination xi index (0=down, 1=mid, 2=up)
        # _probs_x[t, xi, branch] = probability
        def _build(j_max, a_factor, n_nodes):
            dest = np.zeros((self.n_steps, n_nodes, 3), dtype=np.int32)
            probs = np.zeros((self.n_steps, n_nodes, 3), dtype=np.float64)
            for t_idx in range(self.n_steps):
                for xi in range(n_nodes):
                    j = xi - j_max
                    p_up, p_mid, p_dn = self._trinomial_probs_1d(
                        j, a_factor, dt, j_max)
                    probs[t_idx, xi, 0] = p_dn
                    probs[t_idx, xi, 1] = p_mid
                    probs[t_idx, xi, 2] = p_up
                    dest[t_idx, xi, 0] = max(0, xi - 1)
                    dest[t_idx, xi, 1] = xi
                    dest[t_idx, xi, 2] = min(n_nodes - 1, xi + 1)
            return dest, probs

        self._dest_x, self._probs_x = _build(self.j_max_x, a, self.n_x)
        self._dest_y, self._probs_y = _build(self.j_max_y, b_, self.n_y)

    def _transition_probs(self, t_idx: int) -> tuple:
        """Return pre-computed transition tables for time step t_idx.

        Returns
        -------
        (dest_x, probs_x, dest_y, probs_y)
            Each *_x array has shape (n_x_nodes, 3); likewise for y.
            Branch ordering: 0 = down, 1 = mid, 2 = up.
        """
        return (
            self._dest_x[t_idx],
            self._probs_x[t_idx],
            self._dest_y[t_idx],
            self._probs_y[t_idx],
        )

    # ------------------------------------------------------------------
    # Instantaneous forward rate helper
    # ------------------------------------------------------------------

    def _fwd_rate(self, t: float) -> float:
        """Instantaneous forward rate from the market curve at time t.

        Fix T4-G2T1: a finite-difference with ``eps = 1e-5`` years (≈ 8 s)
        is destroyed by ``date_from_year_fraction``'s day rounding — both
        ``t±eps`` round to the same date (so fwd = 0) or to one-day-apart
        dates (so fwd is over-stated by (1/365) / (2·eps) ≈ 137×).  For
        a flat 4% curve at 30-step T=5y this gave phi(t) ≈ 0 at 27 of
        31 grid times and phi(t) ≈ 5.48 at the other 4 — catastrophically
        over-discounting bond prices (P(0,5) tree ≈ 0.026 vs market 0.82).
        Delegate to :meth:`DiscountCurve.instantaneous_forward` which uses
        a stable one-day step.
        """
        return self.g2pp.curve.instantaneous_forward(t)

    # ------------------------------------------------------------------
    # phi(t): G2++ deterministic shift
    # ------------------------------------------------------------------

    def _phi(self, t: float) -> float:
        """G2++ deterministic shift phi(t) = f_mkt(0,t) + correction."""
        g = self.g2pp
        a, b_ = g.a, g.b
        s1, s2, rho = g.sigma1, g.sigma2, g.rho

        ea = (1.0 - math.exp(-a * t)) if a > 1e-12 else t
        eb = (1.0 - math.exp(-b_ * t)) if b_ > 1e-12 else t
        ca = ea**2 / (2.0 * a**2) if a > 1e-12 else t**2 / 2.0
        cb = eb**2 / (2.0 * b_**2) if b_ > 1e-12 else t**2 / 2.0
        cab = ea * eb / (a * b_) if a > 1e-12 and b_ > 1e-12 else t**2
        corr = s1**2 * ca + s2**2 * cb + rho * s1 * s2 * cab
        return self._fwd_rate(t) + corr

    # ------------------------------------------------------------------
    # ZCB price at a lattice node (analytical)
    # ------------------------------------------------------------------

    def zcb_price(
        self,
        t_idx: int,
        x_idx: int,
        y_idx: int,
        T_maturity: float,
    ) -> float:
        """Analytical G2++ ZCB price at a grid node.

        Computes P(x, y; t, T) using the closed-form G2++ formula
        (Brigo & Mercurio eq. 4.3.16), adjusted for the fact that the
        tree starts at (x, y) at time t rather than at (0, 0) at time 0.

        Parameters
        ----------
        t_idx:
            Time-step index on the tree (0 = today).
        x_idx:
            Index into x_nodes array.
        y_idx:
            Index into y_nodes array.
        T_maturity:
            Absolute maturity in years (must be > times[t_idx]).

        Returns
        -------
        float
            ZCB price P(x, y; t, T).
        """
        t = self.times[t_idx]
        tau = T_maturity - t
        if tau <= 0.0:
            return 1.0 if tau >= 0.0 else 0.0

        x_val = self.x_nodes[x_idx]
        y_val = self.y_nodes[y_idx]
        g = self.g2pp
        a, b_ = g.a, g.b
        s1, s2, rho = g.sigma1, g.sigma2, g.rho

        Bx = (1.0 - math.exp(-a * tau)) / a if a > 0 else tau
        By = (1.0 - math.exp(-b_ * tau)) / b_ if b_ > 0 else tau

        def Bk(k, tt):
            return (1.0 - math.exp(-k * tt)) / k if k > 0 else tt

        def _V(tt: float) -> float:
            return (
                s1**2 / a**2 * (tt - 2.0 * Bk(a, tt) + Bk(2.0 * a, tt))
                + s2**2 / b_**2 * (tt - 2.0 * Bk(b_, tt) + Bk(2.0 * b_, tt))
                + 2.0 * rho * s1 * s2 / (a * b_) * (
                    tt - Bk(a, tt) - Bk(b_, tt) + Bk(a + b_, tt))
            )

        V_T = _V(T_maturity)
        V_t = _V(t)
        half_delta_V = 0.5 * (V_T - V_t)

        ref = g.curve.reference_date
        P_T = g.curve.df(date_from_year_fraction(ref, T_maturity))
        P_t = (g.curve.df(date_from_year_fraction(ref, t))
               if t > 1e-10 else 1.0)

        return (P_T / P_t) * math.exp(
            -Bx * x_val - By * y_val + half_delta_V)

    # ------------------------------------------------------------------
    # Backward induction
    # ------------------------------------------------------------------

    def backward_induction(
        self,
        terminal_values: np.ndarray,
        option_func: Optional[Callable] = None,
    ) -> float:
        """Generic backward induction on the 2D G2++ grid.

        Parameters
        ----------
        terminal_values:
            Array of shape (n_x, n_y) — payoff / value at the final
            time step.
        option_func:
            Optional callable with signature
            ``(t_idx, x_idx, y_idx, continuation) -> float``.
            Applied at every node before storing to enforce exercise
            constraints (callable bond, Bermudan swaption).
            Pass ``None`` for European (no constraint).

        Returns
        -------
        float
            Price at the root node (x = 0, y = 0, t = 0).
        """
        V = terminal_values.astype(float).copy()
        dt = self.dt
        rho = self.g2pp.rho

        for t_idx in range(self.n_steps - 1, -1, -1):
            t = self.times[t_idx]
            dest_x, probs_x, dest_y, probs_y = self._transition_probs(t_idx)

            V_new = np.empty((self.n_x, self.n_y))

            for xi in range(self.n_x):
                px = probs_x[xi]       # (3,) [p_dn, p_mid, p_up]
                dx_d = dest_x[xi]     # (3,) destination indices

                for yi in range(self.n_y):
                    py = probs_y[yi]
                    dy_d = dest_y[yi]

                    # Expected continuation over 9 joint branches
                    # Collect probabilities, apply correlation correction, renormalize
                    p_list = []
                    v_list = []
                    for bx in range(3):
                        xi2 = dx_d[bx]
                        sign_x = bx - 1   # -1, 0, +1
                        for by in range(3):
                            yi2 = dy_d[by]
                            sign_y = by - 1
                            # Hull-White (1994): corner branches get rho/4 correction
                            p_j = px[bx] * py[by]
                            if sign_x != 0 and sign_y != 0:
                                p_j += rho * sign_x * sign_y * 0.25
                            p_list.append(max(p_j, 0.0))
                            v_list.append(V[xi2, yi2])
                    # Renormalize to ensure sum = 1
                    p_sum = sum(p_list)
                    ev = sum(p * v for p, v in zip(p_list, v_list)) / p_sum if p_sum > 0 else 0.0

                    # Discount: use x+y+phi as short rate approximation
                    x_val = self.x_nodes[xi]
                    y_val = self.y_nodes[yi]
                    phi_t = self._phi(t)
                    r_node = x_val + y_val + phi_t
                    disc = math.exp(-r_node * dt)
                    cont = disc * ev

                    if option_func is not None:
                        cont = option_func(t_idx, xi, yi, cont)

                    V_new[xi, yi] = cont

            V = V_new

        # Root: centre node where x = 0, y = 0
        return float(V[self.j_max_x, self.j_max_y])


# ---------------------------------------------------------------------------
# European swaption on the 2D tree
# ---------------------------------------------------------------------------

def g2pp_european_swaption_tree(
    g2pp: G2PlusPlus,
    expiry_years: float,
    swap_end_years: float,
    strike: float,
    is_payer: bool = True,
    n_steps: int = 50,
    swap_freq: int = 2,
) -> G2PPTreeResult:
    """Price a European swaption on the 2D G2++ trinomial tree.

    This function is primarily used to cross-validate the analytical
    Jamshidian / G2++ swaption formula.  The tree value should converge
    to the analytical price as n_steps increases.

    Parameters
    ----------
    g2pp:
        Calibrated G2PlusPlus model.
    expiry_years:
        Option expiry in years.
    swap_end_years:
        Swap maturity in years (must be > expiry_years).
    strike:
        Swap fixed rate.
    is_payer:
        True = payer swaption (right to pay fixed, receive float).
    n_steps:
        Number of time steps to expiry.
    swap_freq:
        Swap payment frequency per year (default 2 = semi-annual).

    Returns
    -------
    G2PPTreeResult
    """
    tree = G2PPTree(g2pp, expiry_years, n_steps)

    # Build payment schedule
    coupon_dt = 1.0 / swap_freq
    pay_times: list[float] = []
    t = expiry_years + coupon_dt
    while t <= swap_end_years + 1e-9:
        pay_times.append(round(t, 10))
        t += coupon_dt
    if not pay_times:
        raise ValueError(
            f"No swap payment dates found after expiry ({expiry_years}y)."
        )

    alpha = coupon_dt   # accrual fraction (30/360 simplified)
    last_step = n_steps

    # Terminal payoff at expiry: intrinsic swaption value on the 2D grid
    terminal = np.zeros((tree.n_x, tree.n_y))
    for xi in range(tree.n_x):
        for yi in range(tree.n_y):
            annuity = sum(
                alpha * tree.zcb_price(last_step, xi, yi, tp)
                for tp in pay_times
            )
            P_start = tree.zcb_price(last_step, xi, yi, expiry_years)
            P_end = tree.zcb_price(last_step, xi, yi, pay_times[-1])
            if annuity > 1e-12:
                swap_rate = (P_start - P_end) / annuity
            else:
                swap_rate = 0.0
            swap_pv = annuity * (swap_rate - strike)
            if not is_payer:
                swap_pv = -swap_pv
            terminal[xi, yi] = max(swap_pv, 0.0)

    price = tree.backward_induction(terminal, option_func=None)

    return G2PPTreeResult(
        price=price,
        n_x_nodes=tree.n_x,
        n_y_nodes=tree.n_y,
        n_time_steps=n_steps,
    )
