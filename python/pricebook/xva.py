"""XVA: CVA, DVA, bilateral CVA, FVA, MVA, KVA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import date_from_year_fraction
from pricebook.pricing_context import PricingContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_to_date(ref: date, t: float) -> date:
    return date_from_year_fraction(ref, t)


def _time_grid_dts(time_grid: list[float]) -> list[float]:
    return [time_grid[0]] + [
        time_grid[i] - time_grid[i - 1] for i in range(1, len(time_grid))
    ]


def _default_leg(
    exposure: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    lgd: float,
) -> float:
    """Shared computation for CVA and DVA: sum_i exposure_i * df_i * delta_PD_i * lgd."""
    ref = discount_curve.reference_date
    result = 0.0
    sp_prev = 1.0

    for i, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        sp = survival_curve.survival(d)
        delta_pd = sp_prev - sp
        result += exposure[i] * df * delta_pd * lgd
        sp_prev = sp

    return result


def _discounted_integral(
    profile: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    spread: float,
) -> float:
    """Shared computation for FVA, MVA, KVA: sum_i profile_i * spread * dt_i * df_i."""
    ref = discount_curve.reference_date
    dts = _time_grid_dts(time_grid)
    result = 0.0

    for i, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        result += profile[i] * spread * dts[i] * df

    return result


# ---------------------------------------------------------------------------
# Exposure simulation
# ---------------------------------------------------------------------------


def simulate_exposures(
    pricer,
    ctx: PricingContext,
    time_grid: list[float],
    n_paths: int = 1000,
    rate_vol: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Simulate portfolio PV at future dates under diffused rates.

    Returns an (n_paths, n_times) array of mark-to-market values.

    Simple model: parallel rate shifts drawn from N(0, rate_vol * sqrt(dt)).
    """
    rng = np.random.default_rng(seed)
    n_times = len(time_grid)
    pvs = np.zeros((n_paths, n_times))

    for j, t in enumerate(time_grid):
        shifts = rng.normal(0, rate_vol * math.sqrt(t), n_paths)
        for i, shift in enumerate(shifts):
            bumped_curve = ctx.discount_curve.bumped(shift)
            bumped_ctx = ctx.replace(discount_curve=bumped_curve)
            pvs[i, j] = pricer(bumped_ctx)

    return pvs


def expected_positive_exposure(pvs: np.ndarray) -> np.ndarray:
    """EPE at each time point: E[max(V, 0)]."""
    return np.maximum(pvs, 0).mean(axis=0)


def expected_negative_exposure(pvs: np.ndarray) -> np.ndarray:
    """ENE at each time point: E[max(-V, 0)]."""
    return np.maximum(-pvs, 0).mean(axis=0)


def expected_exposure(pvs: np.ndarray) -> np.ndarray:
    """EE at each time point: E[V]."""
    return pvs.mean(axis=0)


# ---------------------------------------------------------------------------
# CVA / DVA
# ---------------------------------------------------------------------------


def cva(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    """Unilateral CVA = sum_i EPE(t_i) * df(t_i) * delta_PD(t_i) * (1-R)."""
    return _default_leg(epe, time_grid, discount_curve, survival_curve, 1.0 - recovery)


def dva(
    ene: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    own_survival: SurvivalCurve,
    own_recovery: float = 0.4,
) -> float:
    """DVA = sum_i ENE(t_i) * df(t_i) * delta_PD_own(t_i) * (1-R_own)."""
    return _default_leg(ene, time_grid, discount_curve, own_survival, 1.0 - own_recovery)


def bilateral_cva(cva_val: float, dva_val: float) -> float:
    """BCVA = CVA - DVA."""
    return cva_val - dva_val


# ---- Switching effective discount rate (Lou 2018, Eq 5) ----

def effective_discount_rate(
    mu: float,
    r: float,
    r_b: float,
    r_c: float,
    exposure_sign: float,
) -> float:
    """Switching effective discount rate (Lou 2018, Eq 5).

    re = μr + (1-μ) rw  where rw = rb if W ≤ 0, rc if W > 0.

    Args:
        mu: collateralisation ratio (1 = full CSA, 0 = uncollateralised).
        r: OIS rate.
        r_b: bank's unsecured rate.
        r_c: customer's unsecured rate.
        exposure_sign: sign of W = V - L (negative = bank owes, positive = customer owes).
    """
    rw = r_b if exposure_sign <= 0 else r_c
    return mu * r + (1 - mu) * rw


def xva_spread_decomposition(
    total_xva: float,
    s_b: float,
    s_c: float,
    mu_b: float = 0.0,
    mu_c: float = 0.0,
) -> dict[str, float]:
    """Decompose total XVA into CVA, DVA, CFA, DFA (Lou 2018, Eq 15-18).

    U = CVA - DVA + CFA - DFA

    Proportional split based on CDS spreads (s) and bond/CDS basis (μ).
    The sign convention follows Lou: positive U means bank is exposed.

    Args:
        s_b, s_c: CDS premiums of bank and customer.
        mu_b, mu_c: bond/CDS basis of bank and customer.
    """
    total = s_b + s_c + mu_b + mu_c
    if abs(total) < 1e-15:
        return {"cva": 0.0, "dva": 0.0, "cfa": 0.0, "dfa": 0.0}

    absU = abs(total_xva)
    return {
        "cva": absU * s_c / total,
        "dva": absU * s_b / total,
        "cfa": absU * mu_c / total,
        "dfa": absU * mu_b / total,
    }


# ---------------------------------------------------------------------------
# FVA / MVA / KVA
# ---------------------------------------------------------------------------


def fva(
    ee: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    funding_spread: float,
) -> float:
    """FVA = sum_i EE(t_i) * funding_spread * dt_i * df(t_i)."""
    return _discounted_integral(ee, time_grid, discount_curve, funding_spread)


def mva(
    im_profile: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    funding_spread: float,
) -> float:
    """MVA = sum_i IM(t_i) * funding_spread * dt_i * df(t_i)."""
    return _discounted_integral(im_profile, time_grid, discount_curve, funding_spread)


def kva(
    capital_profile: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    hurdle_rate: float,
) -> float:
    """KVA = sum_i K(t_i) * hurdle_rate * dt_i * df(t_i)."""
    return _discounted_integral(capital_profile, time_grid, discount_curve, hurdle_rate)


# ---------------------------------------------------------------------------
# Total XVA
# ---------------------------------------------------------------------------


@dataclass
class XVAResult:
    """Aggregated XVA components."""

    cva: float = 0.0
    dva: float = 0.0
    fva: float = 0.0
    mva: float = 0.0
    kva: float = 0.0

    @property
    def bcva(self) -> float:
        return self.cva - self.dva

    @property
    def total(self) -> float:
        return self.cva - self.dva + self.fva + self.mva + self.kva


# ---------------------------------------------------------------------------
# Wrong-way risk
# ---------------------------------------------------------------------------


def simulate_wwr_exposures(
    pricer,
    ctx: PricingContext,
    time_grid: list[float],
    hazard_rate: float,
    rate_credit_corr: float = 0.0,
    n_paths: int = 1000,
    rate_vol: float = 0.01,
    hazard_vol: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated exposure and default intensity paths.

    Joint simulation of rate shifts and hazard rate shocks,
    correlated via a Gaussian copula.

    Returns:
        (pvs, hazards) each of shape (n_paths, n_times).
    """
    rng = np.random.default_rng(seed)
    n_times = len(time_grid)
    pvs = np.zeros((n_paths, n_times))
    hazards = np.zeros((n_paths, n_times))

    for j, t in enumerate(time_grid):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # Correlated normals
        w_rate = z1
        w_hazard = rate_credit_corr * z1 + math.sqrt(1.0 - rate_credit_corr**2) * z2

        rate_shifts = w_rate * rate_vol * math.sqrt(t)
        hazard_shocks = np.exp(w_hazard * hazard_vol * math.sqrt(t) - 0.5 * hazard_vol**2 * t)
        hazards[:, j] = hazard_rate * hazard_shocks

        for i in range(n_paths):
            bumped_curve = ctx.discount_curve.bumped(rate_shifts[i])
            bumped_ctx = ctx.replace(discount_curve=bumped_curve)
            pvs[i, j] = pricer(bumped_ctx)

    return pvs, hazards


def cva_wrong_way(
    pvs: np.ndarray,
    hazards: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    recovery: float = 0.4,
) -> float:
    """CVA with wrong-way risk from correlated simulation.

    Uses path-level EPE × default probability instead of
    separating exposure and credit.
    """
    ref = discount_curve.reference_date
    lgd = 1.0 - recovery
    dts = _time_grid_dts(time_grid)
    n_paths = pvs.shape[0]
    result = 0.0

    for j, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        dt = dts[j]

        # Per-path: max(V, 0) * hazard * dt * lgd * df
        epe_x_pd = np.maximum(pvs[:, j], 0.0) * hazards[:, j] * dt
        result += epe_x_pd.mean() * df * lgd

    return result


# ---------------------------------------------------------------------------
# Collateralised XVA
# ---------------------------------------------------------------------------


def cva_collateralised(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    threshold: float = 0.0,
    mpr_days: int = 10,
) -> float:
    """CVA with collateral (margin period of risk).

    Collateral reduces exposure to the threshold plus the change
    in exposure during the margin period of risk (MPR).

    Args:
        epe: expected positive exposure at each time point.
        threshold: CSA threshold (uncollateralised amount).
        mpr_days: margin period of risk in days (~10 for bilateral CSA).
    """
    mpr_fraction = mpr_days / 365.0
    ref = discount_curve.reference_date
    lgd = 1.0 - recovery
    result = 0.0
    sp_prev = 1.0

    for i, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        sp = survival_curve.survival(d)
        delta_pd = sp_prev - sp

        # Collateralised exposure: threshold + MPR exposure change
        if i > 0:
            dt = t - time_grid[i - 1]
            mpr_exposure = abs(epe[i] - epe[i - 1]) * mpr_fraction / max(dt, 1e-10)
        else:
            mpr_exposure = epe[i] * mpr_fraction / max(t, 1e-10)

        collateralised_epe = min(epe[i], threshold + mpr_exposure)
        result += collateralised_epe * df * delta_pd * lgd
        sp_prev = sp

    return result


def fva_collateralised(
    ee: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    funding_spread: float,
    threshold: float = 0.0,
) -> float:
    """FVA with CSA: only uncollateralised portion incurs funding cost.

    FVA = 0 for fully collateralised (threshold=0).
    """
    dts = _time_grid_dts(time_grid)
    ref = discount_curve.reference_date
    result = 0.0

    for i, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        uncoll = min(ee[i], threshold)
        result += uncoll * funding_spread * dts[i] * df

    return result


# ---------------------------------------------------------------------------
# Lou Papers Framework (2015-2017)
# ---------------------------------------------------------------------------


@dataclass
class TotalXVAResult:
    """Complete XVA decomposition (Lou 2015 liability-side pricing).

    Lou (2015) Eq 3:  U = CVA - DVA + CFA - DFA + ColVA + FVA + MVA + KVA
    Lou (2015) Eq 1:  r - r_N = s + μ  (spread decomposition)

    where s = CDS spread, μ = bond/CDS basis.

    CFA/DFA split uses Lou (2015) Eq 15-18 via xva_spread_decomposition().
    """
    cva: float = 0.0
    dva: float = 0.0
    cfa: float = 0.0   # credit funding adjustment (bank)
    dfa: float = 0.0   # debit funding adjustment (customer)
    colva: float = 0.0
    fva_val: float = 0.0
    mva_val: float = 0.0
    kva_val: float = 0.0

    @property
    def bilateral_cva(self) -> float:
        return self.cva - self.dva

    @property
    def total_funding(self) -> float:
        return self.cfa - self.dfa + self.fva_val

    @property
    def total(self) -> float:
        return (self.cva - self.dva
                + self.cfa - self.dfa
                + self.colva
                + self.fva_val + self.mva_val + self.kva_val)

    def to_dict(self) -> dict[str, float]:
        return {
            "cva": self.cva, "dva": self.dva,
            "cfa": self.cfa, "dfa": self.dfa,
            "colva": self.colva, "fva": self.fva_val,
            "mva": self.mva_val, "kva": self.kva_val,
            "bilateral_cva": self.bilateral_cva,
            "total_funding": self.total_funding,
            "total": self.total,
        }


def total_xva_decomposition(
    epe: np.ndarray,
    ene: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    cpty_survival: SurvivalCurve,
    own_survival: SurvivalCurve,
    cpty_recovery: float = 0.4,
    own_recovery: float = 0.4,
    s_b: float = 0.01,
    s_c: float = 0.01,
    mu_b: float = 0.0,
    mu_c: float = 0.0,
    funding_spread: float = 0.005,
    im_profile: np.ndarray | None = None,
    capital_profile: np.ndarray | None = None,
    hurdle_rate: float = 0.10,
    collateral_rate: float = 0.0,
    discount_rate: float = 0.0,
    collateral_profile: list[float] | None = None,
) -> TotalXVAResult:
    """Complete XVA decomposition (Lou 2015 Eq 3).

    Computes all XVA components from exposure profiles:
    U = CVA - DVA + CFA - DFA + ColVA + FVA + MVA + KVA

    Args:
        epe, ene: expected positive/negative exposure profiles.
        cpty_survival, own_survival: survival curves.
        s_b, s_c: CDS spreads of bank and counterparty.
        mu_b, mu_c: bond/CDS basis of bank and counterparty.
        funding_spread: unsecured - secured rate.
        im_profile: initial margin profile (for MVA).
        capital_profile: regulatory capital profile (for KVA).
        collateral_rate, discount_rate: for ColVA computation.
        collateral_profile: collateral posted at each time (for ColVA).
    """
    # CVA
    cva_val = cva(epe, time_grid, discount_curve, cpty_survival, cpty_recovery)

    # DVA
    dva_val = dva(ene, time_grid, discount_curve, own_survival, own_recovery)

    # CFA/DFA: from spread decomposition (Lou 2015 Eq 15-18)
    # The decomposition splits the total adjustment |U|, not just credit.
    # At this point FVA is not yet computed, so use CVA + DVA + FVA estimate.
    fva_est = fva(epe, time_grid, discount_curve, funding_spread)
    total_adjustment = cva_val + dva_val + abs(fva_est)
    decomp = xva_spread_decomposition(total_adjustment, s_b, s_c, mu_b, mu_c)
    cfa_val = decomp["cfa"]
    dfa_val = decomp["dfa"]

    # FVA (reuse estimate from above)
    fva_val = fva_est

    # ColVA
    colva_val = 0.0
    if collateral_profile is not None and len(collateral_profile) == len(time_grid):
        from pricebook.csa import colva as colva_fn
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        colva_val = colva_fn(
            list(epe), collateral_profile,
            collateral_rate, discount_rate, dt,
        )

    # MVA
    mva_val = 0.0
    if im_profile is not None:
        mva_val = mva(im_profile, time_grid, discount_curve, funding_spread)

    # KVA
    kva_val = 0.0
    if capital_profile is not None:
        kva_val = kva(capital_profile, time_grid, discount_curve, hurdle_rate)

    return TotalXVAResult(
        cva=cva_val, dva=dva_val,
        cfa=cfa_val, dfa=dfa_val,
        colva=colva_val, fva_val=fva_val,
        mva_val=mva_val, kva_val=kva_val,
    )


# ---------------------------------------------------------------------------
# IRS-specific XVA (Lou 2016a)
# ---------------------------------------------------------------------------


def irs_xva(
    swap_pv: float,
    swap_dv01: float,
    time_to_maturity: float,
    discount_curve: DiscountCurve,
    cpty_survival: SurvivalCurve,
    own_survival: SurvivalCurve,
    cpty_recovery: float = 0.4,
    own_recovery: float = 0.4,
    funding_spread: float = 0.005,
    rate_vol: float = 0.01,
    n_steps: int = 20,
) -> TotalXVAResult:
    """XVA for interest rate swaps (Lou 2016a §3).

    Uses swap DV01 and rate volatility to build approximate exposure profiles.
    Under a normal model, V(t) ~ N(μ, σ²) where:
        μ = swap_pv (drift neglected for simplicity)
        σ = |DV01| × rate_vol × √t × 100  (bps → currency)

    EPE = E[max(V, 0)] = μ Φ(μ/σ) + σ φ(μ/σ)   (standard normal call formula)
    ENE = E[max(-V, 0)] = EPE - μ               (put-call parity for expectations)

    Derivation: for X ~ N(μ,σ²),
        E[max(X,0)] = μ Φ(μ/σ) + σ φ(μ/σ)
    follows from integrating x × φ((x-μ)/σ)/σ from 0 to ∞.

    Limitations:
    - Assumes constant DV01 (ignores pull-to-par, amortisation).
    - No PV drift from carry (swap PV changes as rates evolve).
    - Single-factor model (no curve shape risk).
    For production use, replace with full simulation via simulate_exposures().

    Args:
        swap_pv: current swap PV.
        swap_dv01: DV01 of the swap.
        time_to_maturity: remaining life.
        rate_vol: interest rate volatility (annualised).
        n_steps: number of time steps.
    """
    from scipy.stats import norm as norm_dist

    dt = time_to_maturity / n_steps
    time_grid = [i * dt for i in range(1, n_steps + 1)]

    # Approximate exposure profiles from DV01
    epe = np.zeros(n_steps)
    ene = np.zeros(n_steps)

    for i, t in enumerate(time_grid):
        # Volatility of PV at time t
        sigma_t = abs(swap_dv01) * rate_vol * math.sqrt(t) * 100  # bps → currency
        mu = swap_pv

        if sigma_t < 1e-10:
            epe[i] = max(mu, 0.0)
            ene[i] = max(-mu, 0.0)
        else:
            d = mu / sigma_t
            # EPE = μ Φ(d) + σ φ(d)  (exact under normal distribution)
            epe[i] = mu * norm_dist.cdf(d) + sigma_t * norm_dist.pdf(d)
            # ENE = EPE - μ
            ene[i] = epe[i] - mu

    return total_xva_decomposition(
        epe=epe, ene=ene, time_grid=time_grid,
        discount_curve=discount_curve,
        cpty_survival=cpty_survival,
        own_survival=own_survival,
        cpty_recovery=cpty_recovery,
        own_recovery=own_recovery,
        funding_spread=funding_spread,
    )


# ---------------------------------------------------------------------------
# Repo gap risk estimator (Lou 2016b)
# ---------------------------------------------------------------------------


def repo_gap_risk(
    position_value: float,
    repo_rate: float,
    funding_rate: float,
    collateral_coverage: float,
    time_horizon: float = 1.0,
) -> float:
    """Analytical repo rate estimator when no liquid quote exists (Lou 2016b Eq 8).

    rs = r + (r_N - r) × (1 - h)

    where h = collateral_coverage, r_N = unsecured funding rate.
    The gap risk arises when collateral coverage < 1:
    - The unfunded portion must be financed at the unsecured rate.
    - The gap cost = (1 - h) × position × (r_N - r) × T.

    Args:
        position_value: market value of the position.
        repo_rate: observable GC repo rate.
        funding_rate: unsecured funding rate.
        collateral_coverage: fraction of position covered by collateral (0 to 1).
        time_horizon: risk horizon in years.

    Returns:
        Estimated repo gap cost.
    """
    if not 0.0 <= collateral_coverage <= 1.0:
        raise ValueError(f"collateral_coverage must be in [0, 1], got {collateral_coverage}")
    funding_premium = funding_rate - repo_rate
    gap = (1.0 - collateral_coverage) * position_value * funding_premium * time_horizon
    return gap


def implied_repo_rate_from_gap(
    repo_rate: float,
    funding_rate: float,
    collateral_coverage: float,
) -> float:
    """Implied all-in repo rate accounting for gap risk (Lou 2016b Eq 8).

    rs_implied = r + (r_N - r) × (1 - h)
    """
    if not 0.0 <= collateral_coverage <= 1.0:
        raise ValueError(f"collateral_coverage must be in [0, 1], got {collateral_coverage}")
    return repo_rate + (funding_rate - repo_rate) * (1.0 - collateral_coverage)

from pricebook.serialisable import _register as _reg_result

TotalXVAResult._SERIAL_TYPE = "total_xva_result"

_orig_xva_to_dict = TotalXVAResult.to_dict

def _xva_to_dict_wrapped(self):
    d = _orig_xva_to_dict(self)
    return {"type": "total_xva_result", "params": d}

@classmethod
def _xva_from_dict(cls, d):
    p = d["params"]
    return cls(cva=p["cva"], dva=p["dva"], cfa=p.get("cfa", 0.0),
               dfa=p.get("dfa", 0.0), colva=p.get("colva", 0.0),
               fva_val=p.get("fva", 0.0), mva_val=p.get("mva", 0.0),
               kva_val=p.get("kva", 0.0))

TotalXVAResult.to_dict = _xva_to_dict_wrapped
TotalXVAResult.from_dict = _xva_from_dict
_reg_result(TotalXVAResult)
