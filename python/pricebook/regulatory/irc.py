"""Incremental Risk Charge (IRC): rating migration Monte Carlo.

Vectorised IRC calculation with 7 transition matrices (global, EU, EM,
financials, sovereign, recession, benign), Gaussian copula simulation,
and 99.9% 1-year loss percentile.

    from pricebook.regulatory.irc import (
        IRCPosition, IRCConfig, calculate_irc, quick_irc,
        TRANSITION_MATRICES, get_transition_matrix,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.regulatory.ratings import (
    RATING_TO_PD, get_rating_from_pd, normalize_rating,
)


# ---- Rating categories ----

RATING_CATEGORIES = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


# ---- Transition matrices ----

# Global / US Corporate (S&P historical)
TRANSITION_MATRIX_GLOBAL = {
    "AAA": {"AAA": 0.9081, "AA": 0.0833, "A": 0.0068, "BBB": 0.0006, "BB": 0.0008, "B": 0.0003, "CCC": 0.0001, "D": 0.0000},
    "AA":  {"AAA": 0.0070, "AA": 0.9065, "A": 0.0779, "BBB": 0.0064, "BB": 0.0006, "B": 0.0010, "CCC": 0.0004, "D": 0.0002},
    "A":   {"AAA": 0.0009, "AA": 0.0227, "A": 0.9105, "BBB": 0.0552, "BB": 0.0074, "B": 0.0021, "CCC": 0.0006, "D": 0.0006},
    "BBB": {"AAA": 0.0002, "AA": 0.0033, "A": 0.0595, "BBB": 0.8693, "BB": 0.0530, "B": 0.0102, "CCC": 0.0027, "D": 0.0018},
    "BB":  {"AAA": 0.0003, "AA": 0.0014, "A": 0.0067, "BBB": 0.0773, "BB": 0.8053, "B": 0.0804, "CCC": 0.0180, "D": 0.0106},
    "B":   {"AAA": 0.0000, "AA": 0.0011, "A": 0.0024, "BBB": 0.0043, "BB": 0.0648, "B": 0.8297, "CCC": 0.0456, "D": 0.0521},
    "CCC": {"AAA": 0.0022, "AA": 0.0000, "A": 0.0022, "BBB": 0.0130, "BB": 0.0238, "B": 0.1124, "CCC": 0.6486, "D": 0.1978},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRIX_EUROPE = {
    "AAA": {"AAA": 0.9150, "AA": 0.0770, "A": 0.0060, "BBB": 0.0008, "BB": 0.0006, "B": 0.0004, "CCC": 0.0002, "D": 0.0000},
    "AA":  {"AAA": 0.0080, "AA": 0.9120, "A": 0.0720, "BBB": 0.0058, "BB": 0.0008, "B": 0.0008, "CCC": 0.0004, "D": 0.0002},
    "A":   {"AAA": 0.0012, "AA": 0.0250, "A": 0.9150, "BBB": 0.0500, "BB": 0.0060, "B": 0.0018, "CCC": 0.0005, "D": 0.0005},
    "BBB": {"AAA": 0.0003, "AA": 0.0040, "A": 0.0620, "BBB": 0.8750, "BB": 0.0460, "B": 0.0085, "CCC": 0.0025, "D": 0.0017},
    "BB":  {"AAA": 0.0004, "AA": 0.0016, "A": 0.0075, "BBB": 0.0820, "BB": 0.8100, "B": 0.0750, "CCC": 0.0150, "D": 0.0085},
    "B":   {"AAA": 0.0000, "AA": 0.0012, "A": 0.0028, "BBB": 0.0050, "BB": 0.0700, "B": 0.8350, "CCC": 0.0420, "D": 0.0440},
    "CCC": {"AAA": 0.0020, "AA": 0.0000, "A": 0.0025, "BBB": 0.0150, "BB": 0.0280, "B": 0.1200, "CCC": 0.6525, "D": 0.1800},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRIX_EM = {
    "AAA": {"AAA": 0.8800, "AA": 0.1000, "A": 0.0140, "BBB": 0.0030, "BB": 0.0015, "B": 0.0010, "CCC": 0.0005, "D": 0.0000},
    "AA":  {"AAA": 0.0050, "AA": 0.8850, "A": 0.0900, "BBB": 0.0120, "BB": 0.0040, "B": 0.0020, "CCC": 0.0012, "D": 0.0008},
    "A":   {"AAA": 0.0006, "AA": 0.0180, "A": 0.8900, "BBB": 0.0700, "BB": 0.0130, "B": 0.0050, "CCC": 0.0020, "D": 0.0014},
    "BBB": {"AAA": 0.0001, "AA": 0.0020, "A": 0.0480, "BBB": 0.8400, "BB": 0.0750, "B": 0.0200, "CCC": 0.0080, "D": 0.0069},
    "BB":  {"AAA": 0.0002, "AA": 0.0010, "A": 0.0050, "BBB": 0.0650, "BB": 0.7700, "B": 0.1050, "CCC": 0.0300, "D": 0.0238},
    "B":   {"AAA": 0.0000, "AA": 0.0008, "A": 0.0018, "BBB": 0.0035, "BB": 0.0550, "B": 0.7900, "CCC": 0.0650, "D": 0.0839},
    "CCC": {"AAA": 0.0015, "AA": 0.0000, "A": 0.0015, "BBB": 0.0100, "BB": 0.0200, "B": 0.0900, "CCC": 0.5770, "D": 0.3000},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRIX_FINANCIALS = {
    "AAA": {"AAA": 0.9000, "AA": 0.0880, "A": 0.0085, "BBB": 0.0015, "BB": 0.0010, "B": 0.0006, "CCC": 0.0003, "D": 0.0001},
    "AA":  {"AAA": 0.0060, "AA": 0.9000, "A": 0.0820, "BBB": 0.0080, "BB": 0.0018, "B": 0.0012, "CCC": 0.0006, "D": 0.0004},
    "A":   {"AAA": 0.0007, "AA": 0.0200, "A": 0.9050, "BBB": 0.0600, "BB": 0.0090, "B": 0.0030, "CCC": 0.0012, "D": 0.0011},
    "BBB": {"AAA": 0.0001, "AA": 0.0025, "A": 0.0550, "BBB": 0.8600, "BB": 0.0580, "B": 0.0140, "CCC": 0.0050, "D": 0.0054},
    "BB":  {"AAA": 0.0002, "AA": 0.0010, "A": 0.0055, "BBB": 0.0700, "BB": 0.7900, "B": 0.0900, "CCC": 0.0250, "D": 0.0183},
    "B":   {"AAA": 0.0000, "AA": 0.0008, "A": 0.0020, "BBB": 0.0040, "BB": 0.0600, "B": 0.8100, "CCC": 0.0550, "D": 0.0682},
    "CCC": {"AAA": 0.0018, "AA": 0.0000, "A": 0.0020, "BBB": 0.0120, "BB": 0.0220, "B": 0.1050, "CCC": 0.6072, "D": 0.2500},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRIX_SOVEREIGN = {
    "AAA": {"AAA": 0.9500, "AA": 0.0450, "A": 0.0040, "BBB": 0.0005, "BB": 0.0003, "B": 0.0001, "CCC": 0.0001, "D": 0.0000},
    "AA":  {"AAA": 0.0100, "AA": 0.9400, "A": 0.0450, "BBB": 0.0035, "BB": 0.0008, "B": 0.0004, "CCC": 0.0002, "D": 0.0001},
    "A":   {"AAA": 0.0015, "AA": 0.0300, "A": 0.9300, "BBB": 0.0320, "BB": 0.0045, "B": 0.0012, "CCC": 0.0005, "D": 0.0003},
    "BBB": {"AAA": 0.0003, "AA": 0.0050, "A": 0.0700, "BBB": 0.8800, "BB": 0.0350, "B": 0.0060, "CCC": 0.0022, "D": 0.0015},
    "BB":  {"AAA": 0.0005, "AA": 0.0020, "A": 0.0100, "BBB": 0.0900, "BB": 0.8200, "B": 0.0550, "CCC": 0.0130, "D": 0.0095},
    "B":   {"AAA": 0.0000, "AA": 0.0015, "A": 0.0030, "BBB": 0.0060, "BB": 0.0800, "B": 0.8300, "CCC": 0.0380, "D": 0.0415},
    "CCC": {"AAA": 0.0025, "AA": 0.0000, "A": 0.0030, "BBB": 0.0180, "BB": 0.0350, "B": 0.1300, "CCC": 0.6315, "D": 0.1800},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRIX_RECESSION = {
    "AAA": {"AAA": 0.8500, "AA": 0.1200, "A": 0.0200, "BBB": 0.0050, "BB": 0.0030, "B": 0.0012, "CCC": 0.0006, "D": 0.0002},
    "AA":  {"AAA": 0.0040, "AA": 0.8600, "A": 0.1050, "BBB": 0.0180, "BB": 0.0060, "B": 0.0035, "CCC": 0.0020, "D": 0.0015},
    "A":   {"AAA": 0.0005, "AA": 0.0150, "A": 0.8700, "BBB": 0.0800, "BB": 0.0200, "B": 0.0080, "CCC": 0.0035, "D": 0.0030},
    "BBB": {"AAA": 0.0001, "AA": 0.0020, "A": 0.0400, "BBB": 0.8200, "BB": 0.0850, "B": 0.0300, "CCC": 0.0120, "D": 0.0109},
    "BB":  {"AAA": 0.0001, "AA": 0.0008, "A": 0.0040, "BBB": 0.0550, "BB": 0.7400, "B": 0.1200, "CCC": 0.0450, "D": 0.0351},
    "B":   {"AAA": 0.0000, "AA": 0.0005, "A": 0.0015, "BBB": 0.0030, "BB": 0.0450, "B": 0.7600, "CCC": 0.0800, "D": 0.1100},
    "CCC": {"AAA": 0.0010, "AA": 0.0000, "A": 0.0010, "BBB": 0.0080, "BB": 0.0150, "B": 0.0800, "CCC": 0.5050, "D": 0.3900},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRIX_BENIGN = {
    "AAA": {"AAA": 0.9300, "AA": 0.0640, "A": 0.0045, "BBB": 0.0008, "BB": 0.0004, "B": 0.0002, "CCC": 0.0001, "D": 0.0000},
    "AA":  {"AAA": 0.0100, "AA": 0.9250, "A": 0.0580, "BBB": 0.0050, "BB": 0.0010, "B": 0.0006, "CCC": 0.0003, "D": 0.0001},
    "A":   {"AAA": 0.0015, "AA": 0.0300, "A": 0.9300, "BBB": 0.0330, "BB": 0.0040, "B": 0.0010, "CCC": 0.0003, "D": 0.0002},
    "BBB": {"AAA": 0.0005, "AA": 0.0050, "A": 0.0750, "BBB": 0.8900, "BB": 0.0230, "B": 0.0045, "CCC": 0.0012, "D": 0.0008},
    "BB":  {"AAA": 0.0005, "AA": 0.0020, "A": 0.0100, "BBB": 0.0950, "BB": 0.8400, "B": 0.0400, "CCC": 0.0080, "D": 0.0045},
    "B":   {"AAA": 0.0000, "AA": 0.0015, "A": 0.0035, "BBB": 0.0060, "BB": 0.0850, "B": 0.8600, "CCC": 0.0250, "D": 0.0190},
    "CCC": {"AAA": 0.0030, "AA": 0.0000, "A": 0.0035, "BBB": 0.0200, "BB": 0.0400, "B": 0.1500, "CCC": 0.6835, "D": 0.1000},
    "D":   {"AAA": 0.0, "AA": 0.0, "A": 0.0, "BBB": 0.0, "BB": 0.0, "B": 0.0, "CCC": 0.0, "D": 1.0},
}

TRANSITION_MATRICES: dict[str, dict] = {
    "global": TRANSITION_MATRIX_GLOBAL,
    "us_corporate": TRANSITION_MATRIX_GLOBAL,
    "default": TRANSITION_MATRIX_GLOBAL,
    "europe": TRANSITION_MATRIX_EUROPE,
    "eu": TRANSITION_MATRIX_EUROPE,
    "em": TRANSITION_MATRIX_EM,
    "emerging_markets": TRANSITION_MATRIX_EM,
    "financials": TRANSITION_MATRIX_FINANCIALS,
    "financial": TRANSITION_MATRIX_FINANCIALS,
    "banks": TRANSITION_MATRIX_FINANCIALS,
    "sovereign": TRANSITION_MATRIX_SOVEREIGN,
    "sovereigns": TRANSITION_MATRIX_SOVEREIGN,
    "recession": TRANSITION_MATRIX_RECESSION,
    "stressed": TRANSITION_MATRIX_RECESSION,
    "downturn": TRANSITION_MATRIX_RECESSION,
    "benign": TRANSITION_MATRIX_BENIGN,
    "expansion": TRANSITION_MATRIX_BENIGN,
}


def get_transition_matrix(name: str = "global") -> dict:
    """Get transition matrix by name."""
    key = name.lower()
    if key not in TRANSITION_MATRICES:
        raise ValueError(f"Unknown matrix: {name}. Available: {list(TRANSITION_MATRICES.keys())}")
    return TRANSITION_MATRICES[key]


def list_transition_matrices() -> list[str]:
    """List unique matrix names."""
    return ["global", "europe", "em", "financials", "sovereign", "recession", "benign"]


# ---- Credit spreads (basis points) ----

CREDIT_SPREADS: dict[str, dict[float, float]] = {
    "AAA": {1: 15, 2: 18, 3: 20, 5: 25, 7: 30, 10: 35},
    "AA":  {1: 25, 2: 30, 3: 35, 5: 45, 7: 55, 10: 65},
    "A":   {1: 45, 2: 55, 3: 65, 5: 80, 7: 95, 10: 110},
    "BBB": {1: 90, 2: 105, 3: 120, 5: 150, 7: 175, 10: 200},
    "BB":  {1: 200, 2: 240, 3: 280, 5: 350, 7: 400, 10: 450},
    "B":   {1: 400, 2: 480, 3: 550, 5: 650, 7: 720, 10: 800},
    "CCC": {1: 1000, 2: 1100, 3: 1200, 5: 1350, 7: 1450, 10: 1550},
    "D":   {1: 5000, 2: 5000, 3: 5000, 5: 5000, 7: 5000, 10: 5000},
}


def get_credit_spread(rating: str, tenor_years: float) -> float:
    """Linear-interpolated credit spread in basis points."""
    spreads = CREDIT_SPREADS.get(rating, CREDIT_SPREADS["B"])
    tenors = sorted(spreads.keys())
    if tenor_years <= tenors[0]:
        return spreads[tenors[0]]
    if tenor_years >= tenors[-1]:
        return spreads[tenors[-1]]
    for i in range(len(tenors) - 1):
        if tenors[i] <= tenor_years <= tenors[i + 1]:
            t1, t2 = tenors[i], tenors[i + 1]
            s1, s2 = spreads[t1], spreads[t2]
            return s1 + (s2 - s1) * (tenor_years - t1) / (t2 - t1)
    return spreads[tenors[-1]]


# ---- LGD ----

LGD_BY_SENIORITY = {
    "senior_secured": 0.25,
    "senior_unsecured": 0.45,
    "subordinated": 0.75,
    "equity": 1.00,
}


# ---- Position dataclass ----

@dataclass
class IRCPosition:
    """A single position for IRC calculation."""
    position_id: str
    issuer: str
    notional: float
    market_value: float
    rating: str
    tenor_years: float
    seniority: str = "senior_unsecured"
    sector: str = "corporate"
    liquidity_horizon_months: int = 3
    is_long: bool = True
    coupon_rate: float = 0.0
    lgd: float | None = None


def get_lgd(pos: IRCPosition) -> float:
    """LGD with override priority."""
    if pos.lgd is not None:
        return pos.lgd
    return LGD_BY_SENIORITY.get(pos.seniority, 0.45)


@dataclass
class IRCConfig:
    """IRC simulation configuration."""
    num_simulations: int = 100_000
    confidence_level: float = 0.999
    horizon_years: float = 1.0
    systematic_correlation: float = 0.50
    sector_correlation: float = 0.25
    seed: int = 42
    transition_matrix: str | dict = "global"

    def get_matrix(self) -> dict:
        if isinstance(self.transition_matrix, dict):
            return self.transition_matrix
        return get_transition_matrix(self.transition_matrix)


# ---- Duration helpers ----

def calculate_modified_duration(
    tenor_years: float,
    coupon_rate: float = 0.05,
    yield_rate: float = 0.05,
) -> float:
    """Modified duration."""
    if tenor_years <= 0:
        return 0.0
    if coupon_rate <= 0:
        return tenor_years / (1 + yield_rate)
    mac = (1 - (1 + yield_rate) ** (-tenor_years)) / yield_rate
    mod = mac / (1 + yield_rate)
    return min(mod, tenor_years)


def calculate_spread_pv01(notional: float, tenor_years: float, coupon_rate: float = 0.05) -> float:
    """Spread PV01: notional × duration × 0.0001."""
    return notional * calculate_modified_duration(tenor_years, coupon_rate) * 0.0001


# ---- Vectorised simulation ----

def _build_transition_arrays(matrix: dict) -> tuple[dict, dict, dict]:
    """Build NumPy-friendly transition arrays."""
    rating_to_idx = {r: i for i, r in enumerate(RATING_CATEGORIES)}
    thresholds: dict[str, list[float]] = {}
    targets: dict[str, list[int]] = {}
    for from_rating in RATING_CATEGORIES[:-1]:
        probs = matrix[from_rating]
        cum: list[float] = []
        tgt: list[int] = []
        running = 0.0
        for to_rating in RATING_CATEGORIES:
            running += probs[to_rating]
            cum.append(running)
            tgt.append(rating_to_idx[to_rating])
        thresholds[from_rating] = cum
        targets[from_rating] = tgt
    return thresholds, targets, rating_to_idx


def simulate_irc_portfolio(
    positions: list[IRCPosition],
    config: IRCConfig | None = None,
) -> np.ndarray:
    """Vectorised IRC Monte Carlo via Gaussian copula."""
    if config is None:
        config = IRCConfig()
    n_sims = config.num_simulations
    rng = np.random.default_rng(config.seed)

    matrix = config.get_matrix()
    trans_thresholds, trans_targets, rating_to_idx = _build_transition_arrays(matrix)

    # Group by issuer
    issuer_pos: dict[str, list[IRCPosition]] = {}
    for pos in positions:
        issuer_pos.setdefault(pos.issuer, []).append(pos)

    issuers = list(issuer_pos.keys())
    n_issuers = len(issuers)
    issuer_to_idx = {iss: i for i, iss in enumerate(issuers)}

    issuer_ratings = [issuer_pos[iss][0].rating for iss in issuers]
    issuer_rhos = np.full(n_issuers, config.systematic_correlation)

    # Position-level params
    n_pos = len(positions)
    pos_issuer_idx = np.zeros(n_pos, dtype=np.int32)
    pos_lgd = np.zeros(n_pos)
    pos_pv01 = np.zeros(n_pos)
    pos_curr_spread = np.zeros(n_pos)
    pos_lh = np.zeros(n_pos)
    pos_dir = np.zeros(n_pos)
    pos_notional = np.zeros(n_pos)

    for i, pos in enumerate(positions):
        pos_issuer_idx[i] = issuer_to_idx[pos.issuer]
        pos_lgd[i] = get_lgd(pos)
        pos_pv01[i] = calculate_spread_pv01(pos.notional, pos.tenor_years, pos.coupon_rate)
        pos_curr_spread[i] = get_credit_spread(pos.rating, pos.tenor_years)
        pos_lh[i] = math.sqrt(pos.liquidity_horizon_months / 12.0)
        pos_dir[i] = 1.0 if pos.is_long else -1.0
        pos_notional[i] = abs(pos.notional)

    # Spread lookup per (position, target rating)
    spread_lookup: dict[tuple[int, str], float] = {}
    for i, pos in enumerate(positions):
        for to_r in RATING_CATEGORIES:
            spread_lookup[(i, to_r)] = get_credit_spread(to_r, pos.tenor_years)

    # Random draws
    systematic = rng.standard_normal(n_sims)
    idio = rng.standard_normal((n_sims, n_issuers))
    sqrt_one_minus_rho2 = np.sqrt(1.0 - issuer_rhos ** 2)
    z = issuer_rhos * systematic[:, np.newaxis] + sqrt_one_minus_rho2 * idio
    u = norm.cdf(z)

    # Migration: vectorised threshold lookup per issuer
    new_rating_idx = np.zeros((n_sims, n_issuers), dtype=np.int32)
    for j, iss in enumerate(issuers):
        cur = issuer_ratings[j]
        if cur == "D":
            new_rating_idx[:, j] = rating_to_idx["D"]
            continue
        thr = np.array(trans_thresholds[cur])
        tgt = np.array(trans_targets[cur])
        idx = np.searchsorted(thr, u[:, j], side="left")
        idx = np.clip(idx, 0, len(tgt) - 1)
        new_rating_idx[:, j] = tgt[idx]

    # Compute losses
    losses_matrix = np.zeros((n_sims, n_pos))
    default_idx = rating_to_idx["D"]
    for i, pos in enumerate(positions):
        new_r_for_pos = new_rating_idx[:, pos_issuer_idx[i]]
        is_default = new_r_for_pos == default_idx
        default_loss = pos_lgd[i] * pos_notional[i]
        migration_loss = np.zeros(n_sims)
        for r_idx, r in enumerate(RATING_CATEGORIES):
            if r == "D":
                continue
            mask = new_r_for_pos == r_idx
            if mask.any():
                new_spread = spread_lookup[(i, r)]
                spread_change = new_spread - pos_curr_spread[i]
                migration_loss[mask] = spread_change * pos_pv01[i]
        loss = np.where(is_default, default_loss, migration_loss)
        losses_matrix[:, i] = loss * pos_lh[i] * pos_dir[i]

    # Aggregate by issuer (netting within issuer only)
    issuer_pnl = np.zeros((n_sims, n_issuers))
    for i in range(n_pos):
        issuer_pnl[:, pos_issuer_idx[i]] += losses_matrix[:, i]

    # Portfolio loss = sum of positive issuer P&Ls
    return np.sum(np.maximum(issuer_pnl, 0.0), axis=1)


# ---- Main IRC calculation ----

def calculate_irc(positions: list[IRCPosition], config: IRCConfig | None = None) -> dict:
    """Calculate IRC: 99.9% 1-year loss percentile via MC."""
    if config is None:
        config = IRCConfig()
    if not positions:
        return {"irc": 0.0, "mean_loss": 0.0, "num_simulations": 0}

    # Normalize ratings
    for pos in positions:
        pos.rating = normalize_rating(pos.rating)

    losses = simulate_irc_portfolio(positions, config)
    n = len(losses)
    sorted_losses = np.sort(losses)
    idx_999 = min(int(n * config.confidence_level), n - 1)
    irc = float(sorted_losses[idx_999])

    tail = sorted_losses[idx_999:]
    es_999 = float(tail.mean()) if len(tail) > 0 else irc

    return {
        "irc": irc,
        "mean_loss": float(losses.mean()),
        "median_loss": float(sorted_losses[n // 2]),
        "percentile_95": float(sorted_losses[int(n * 0.95)]),
        "percentile_99": float(sorted_losses[int(n * 0.99)]),
        "percentile_999": irc,
        "expected_shortfall_999": es_999,
        "max_loss": float(sorted_losses[-1]),
        "num_simulations": n,
        "num_positions": len(positions),
        "num_issuers": len({p.issuer for p in positions}),
        "matrix": config.transition_matrix if isinstance(config.transition_matrix, str) else "custom",
    }


# ---- Convenience ----

def quick_irc(positions: list[dict], num_simulations: int = 50_000, matrix: str = "global") -> dict:
    """Quick IRC from a list of dicts."""
    irc_pos = []
    for i, p in enumerate(positions):
        irc_pos.append(IRCPosition(
            position_id=p.get("position_id", f"pos_{i}"),
            issuer=p.get("issuer", f"issuer_{i}"),
            notional=p.get("notional", 0),
            market_value=p.get("market_value", p.get("notional", 0)),
            rating=p.get("rating", "BBB"),
            tenor_years=p.get("tenor_years", 5),
            seniority=p.get("seniority", "senior_unsecured"),
            sector=p.get("sector", "corporate"),
            is_long=p.get("is_long", True),
            coupon_rate=p.get("coupon_rate", 0.05),
        ))
    config = IRCConfig(num_simulations=num_simulations, transition_matrix=matrix)
    return calculate_irc(irc_pos, config)


def calculate_irc_by_issuer(positions: list[IRCPosition], config: IRCConfig | None = None) -> dict[str, float]:
    """IRC contribution per issuer (stand-alone for each name)."""
    if config is None:
        config = IRCConfig()
    issuer_pos: dict[str, list[IRCPosition]] = {}
    for p in positions:
        issuer_pos.setdefault(p.issuer, []).append(p)
    return {iss: calculate_irc(pos, config)["irc"] for iss, pos in issuer_pos.items()}
