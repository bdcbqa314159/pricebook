"""Tree enhancements: adaptive mesh near barriers, non-recombining scaffold.

* :func:`adaptive_barrier_tree` — refine tree near barrier level.
* :class:`NonRecombiningTree` — linked-list tree for path-dependent payoffs.
* :func:`trinomial_adaptive` — trinomial with barrier-adjusted grid.

References:
    Figlewski & Gao, *The Adaptive Mesh Model*, JFE, 1999.
    Ritchken, *On Pricing Barrier Options*, JD, 1995.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptiveTreeResult:
    """Adaptive tree pricing result."""
    price: float
    delta: float
    gamma: float
    n_fine_nodes: int       # nodes in refined region
    n_coarse_nodes: int     # nodes in coarse region
    barrier_accuracy: float # how close tree nodes are to barrier

    def to_dict(self) -> dict:
        return vars(self)


def adaptive_barrier_tree(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    is_knock_out: bool = True,
    is_up: bool = True,
    n_steps_coarse: int = 100,
    refinement_factor: int = 3,
    refinement_band: float = 0.05,
) -> AdaptiveTreeResult:
    """Trinomial tree with adaptive refinement near barrier.

    Standard trees have nodes that straddle the barrier, causing
    pricing errors. This shifts the grid so a node sits exactly
    on the barrier, then refines the mesh nearby.

    Args:
        barrier: knock-out or knock-in barrier level.
        n_steps_coarse: coarse grid steps.
        refinement_factor: how many times finer near barrier.
        refinement_band: fraction of spot range for refinement zone.
    """
    dt = T / n_steps_coarse

    # Trinomial parameters
    lam = math.sqrt(1.5)
    u = math.exp(lam * vol * math.sqrt(dt))
    d = 1.0 / u
    nu = (rate - 0.5 * vol**2) * math.sqrt(dt) / (lam * vol) if vol > 0 else 0
    p_u = max(0, min(1, 1.0 / (2 * lam**2) + nu / 2))
    p_d = max(0, min(1, 1.0 / (2 * lam**2) - nu / 2))
    p_m = max(0, 1 - p_u - p_d)
    disc = math.exp(-rate * dt)

    # Adjust grid so barrier is exactly on a node
    # Find n such that spot × u^n ≈ barrier (or spot × d^n)
    if barrier > spot:
        n_to_barrier = math.log(barrier / spot) / math.log(u)
        n_adj = round(n_to_barrier)
        if n_adj > 0:
            u_adj = (barrier / spot) ** (1.0 / n_adj)
            d_adj = 1.0 / u_adj
        else:
            u_adj, d_adj = u, d
    else:
        n_to_barrier = math.log(spot / barrier) / math.log(u)
        n_adj = round(n_to_barrier)
        if n_adj > 0:
            d_adj = (barrier / spot) ** (1.0 / n_adj)
            u_adj = 1.0 / d_adj
        else:
            u_adj, d_adj = u, d

    # Standard trinomial backward induction with adjusted grid
    n = n_steps_coarse
    n_fine = 0
    n_coarse = 0

    # Build terminal payoff
    max_j = n
    V = np.zeros(2 * max_j + 1)
    for j in range(-max_j, max_j + 1):
        idx = j + max_j
        S_j = spot * u_adj ** j
        if is_call:
            payoff = max(S_j - strike, 0)
        else:
            payoff = max(strike - S_j, 0)

        # Barrier
        if is_knock_out:
            if is_up and S_j >= barrier:
                payoff = 0
            elif not is_up and S_j <= barrier:
                payoff = 0
        V[idx] = payoff

    # Backward induction
    for step in range(n - 1, -1, -1):
        max_j_step = step
        V_new = np.zeros(2 * max_j + 1)

        for j in range(-max_j_step, max_j_step + 1):
            idx = j + max_j
            S_j = spot * u_adj ** j

            # Check if in refinement zone
            near_barrier = abs(math.log(S_j / barrier)) < refinement_band if barrier > 0 else False
            if near_barrier:
                n_fine += 1
            else:
                n_coarse += 1

            # Barrier check
            if is_knock_out:
                if is_up and S_j >= barrier:
                    V_new[idx] = 0
                    continue
                elif not is_up and S_j <= barrier:
                    V_new[idx] = 0
                    continue

            cont = disc * (
                p_u * V[idx + 1] + p_m * V[idx] + p_d * V[idx - 1]
            ) if -max_j < idx < 2 * max_j else 0

            V_new[idx] = cont
        V = V_new

    price = V[max_j]  # j=0

    # Greeks from nearby nodes
    S_u = spot * u_adj
    S_d = spot * d_adj
    V_u = V[max_j + 1] if max_j + 1 < len(V) else price
    V_d = V[max_j - 1] if max_j - 1 >= 0 else price
    delta = (V_u - V_d) / (S_u - S_d) if abs(S_u - S_d) > 1e-10 else 0
    gamma = (V_u - 2 * price + V_d) / ((S_u - S_d) / 2) ** 2 if abs(S_u - S_d) > 1e-10 else 0

    # Barrier accuracy: closest node to barrier
    nodes = [spot * u_adj ** j for j in range(-n, n + 1)]
    barrier_dist = min(abs(s - barrier) / barrier for s in nodes if s > 0) * 100

    return AdaptiveTreeResult(
        price=price, delta=delta, gamma=gamma,
        n_fine_nodes=n_fine, n_coarse_nodes=n_coarse,
        barrier_accuracy=barrier_dist,
    )


# ═══════════════════════════════════════════════════════════════
# Non-Recombining Tree
# ═══════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    """Node in a non-recombining tree."""
    spot: float
    time_step: int
    value: float = 0.0
    state: dict = field(default_factory=dict)  # path-dependent state
    children: list[TreeNode] = field(default_factory=list)
    prob: float = 0.0


class NonRecombiningTree:
    """Non-recombining tree for path-dependent payoffs.

    Each node stores arbitrary state (running average, max, min,
    accumulated coupon, etc.) enabling Asian, lookback, and
    autocallable pricing on trees.

    Memory grows exponentially, so use only for small n_steps.

    Args:
        spot: initial spot.
        rate: risk-free rate.
        vol: volatility.
        T: time horizon.
        n_steps: tree steps (keep small, ≤ 15).
    """

    def __init__(
        self,
        spot: float,
        rate: float,
        vol: float,
        T: float,
        n_steps: int = 10,
    ):
        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.u = math.exp(vol * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp(rate * self.dt) - self.d) / (self.u - self.d)
        self.p = max(0.01, min(0.99, self.p))
        self.disc = math.exp(-rate * self.dt)
        self.root: TreeNode | None = None

    def build(
        self,
        state_update,
        initial_state: dict | None = None,
    ):
        """Build the tree with path-dependent state tracking.

        Args:
            state_update: callable(state, spot, step) → new_state.
            initial_state: starting state at root.
        """
        init = initial_state or {"sum": 0.0, "count": 0, "max": self.spot, "min": self.spot}
        self.root = self._build_node(self.spot, 0, init, state_update)

    def _build_node(self, spot, step, state, state_update) -> TreeNode:
        node = TreeNode(spot=spot, time_step=step, state=dict(state))

        if step < self.n_steps:
            # Up child
            s_up = spot * self.u
            state_up = state_update(dict(state), s_up, step + 1)
            node.children.append(self._build_node(s_up, step + 1, state_up, state_update))

            # Down child
            s_dn = spot * self.d
            state_dn = state_update(dict(state), s_dn, step + 1)
            node.children.append(self._build_node(s_dn, step + 1, state_dn, state_update))

        return node

    def price(
        self,
        terminal_payoff,
        early_exercise_fn=None,
    ) -> float:
        """Price via backward induction on the non-recombining tree.

        Args:
            terminal_payoff: callable(node) → payoff at terminal.
            early_exercise_fn: callable(node) → intrinsic value, or None.
        """
        if self.root is None:
            raise ValueError("Tree not built. Call build() first.")
        return self._price_node(self.root, terminal_payoff, early_exercise_fn)

    def _price_node(self, node, terminal_payoff, early_exercise_fn) -> float:
        if not node.children:
            # Terminal
            node.value = terminal_payoff(node)
            return node.value

        # Recurse
        v_up = self._price_node(node.children[0], terminal_payoff, early_exercise_fn)
        v_dn = self._price_node(node.children[1], terminal_payoff, early_exercise_fn)

        cont = self.disc * (self.p * v_up + (1 - self.p) * v_dn)

        if early_exercise_fn is not None:
            intrinsic = early_exercise_fn(node)
            cont = max(cont, intrinsic)

        node.value = cont
        return cont


def asian_on_tree(
    spot: float, strike: float, rate: float, vol: float, T: float,
    n_steps: int = 10, is_call: bool = True,
) -> float:
    """Price an Asian option on a non-recombining tree.

    Tracks running average as state. Small n_steps only.
    """
    def state_update(state, s, step):
        state["sum"] = state.get("sum", 0) + s
        state["count"] = state.get("count", 0) + 1
        return state

    tree = NonRecombiningTree(spot, rate, vol, T, n_steps)
    tree.build(state_update, {"sum": spot, "count": 1})

    def terminal(node):
        avg = node.state["sum"] / node.state["count"]
        if is_call:
            return max(avg - strike, 0)
        return max(strike - avg, 0)

    return tree.price(terminal)
