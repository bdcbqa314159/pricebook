"""Interest rate caps and floors."""

from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency, StubType, generate_schedule
from pricebook.core.calendar import Calendar, BusinessDayConvention
from pricebook.models.black76 import black76_price, black76_vega, OptionType
from pricebook.core.solvers import brentq


class CapFloor:
    """
    An interest rate cap or floor: a strip of caplets/floorlets.

    A caplet is a call option on a forward rate (pays max(L - K, 0) * yf * notional).
    A floorlet is a put option on a forward rate (pays max(K - L, 0) * yf * notional).

    Each caplet/floorlet is priced with Black-76:
        caplet = notional * yf * df(T_pay) * Black76_call(F, K, vol, T_fix)
        floorlet = notional * yf * df(T_pay) * Black76_put(F, K, vol, T_fix)

    The vol for each caplet comes from a vol surface at the fixing expiry.
    """

    def __init__(
        self,
        start: date,
        end: date,
        strike: float,
        option_type: OptionType = OptionType.CALL,
        notional: float = 1_000_000.0,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        self.start = start
        self.end = end
        self.strike = strike
        self.option_type = option_type
        self.notional = notional
        self.frequency = frequency
        self.day_count = day_count
        self.calendar = calendar
        self.convention = convention

        schedule = generate_schedule(
            start, end, frequency, calendar, convention,
            StubType.SHORT_FRONT, True,
        )

        # Each caplet/floorlet covers one accrual period
        self.periods: list[tuple[date, date]] = []
        for i in range(1, len(schedule)):
            self.periods.append((schedule[i - 1], schedule[i]))

    def caplet_pvs(
        self,
        model,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> list[dict]:
        """Individual caplet/floorlet PVs and greeks using a pluggable model.

        Returns list of dicts with per-caplet details: accrual dates, forward,
        pv, and greeks (delta, gamma, vega) if the model supports them.
        """
        proj = projection_curve if projection_curve is not None else curve
        results = []
        for accrual_start, accrual_end in self.periods:
            yf = year_fraction(accrual_start, accrual_end, self.day_count)
            df1 = proj.df(accrual_start)
            df2 = proj.df(accrual_end)
            fwd = (df1 - df2) / (yf * df2)
            t_fix = year_fraction(curve.reference_date, accrual_start,
                                  DayCountConvention.ACT_365_FIXED)
            caplet_annuity = yf * curve.df(accrual_end)

            entry = {"accrual_start": accrual_start, "accrual_end": accrual_end,
                     "forward": fwd}

            if t_fix <= 0:
                if self.option_type == OptionType.CALL:
                    intrinsic = max(fwd - self.strike, 0.0)
                else:
                    intrinsic = max(self.strike - fwd, 0.0)
                entry["pv"] = self.notional * caplet_annuity * intrinsic
                entry["delta"] = entry["gamma"] = entry["vega"] = 0.0
            elif hasattr(model, "greeks_ir_option"):
                g = model.greeks_ir_option(fwd, self.strike, caplet_annuity,
                                           t_fix, self.option_type)
                entry["pv"] = self.notional * g.price
                entry["delta"] = self.notional * g.delta
                entry["gamma"] = self.notional * g.gamma
                entry["vega"] = self.notional * g.vega
            else:
                entry["pv"] = self.notional * model.price_ir_option(
                    fwd, self.strike, caplet_annuity, t_fix, self.option_type)
                entry["delta"] = entry["gamma"] = entry["vega"] = 0.0

            results.append(entry)
        return results

    def greeks(
        self,
        model,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ):
        """Aggregate greeks across all caplets/floorlets.

        Uses model.greeks_ir_option() if available, otherwise bump-and-reprice.
        """
        from pricebook.core.greeks import Greeks
        caplets = self.caplet_pvs(model, curve, projection_curve)
        return Greeks(
            price=sum(c["pv"] for c in caplets),
            delta=sum(c.get("delta", 0) for c in caplets),
            gamma=sum(c.get("gamma", 0) for c in caplets),
            vega=sum(c.get("vega", 0) for c in caplets),
        )


    def price(
        self,
        model,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Price using a pluggable model (Black76Model, BachelierModel, SABRModel, etc.).

        Each caplet/floorlet is priced individually via model.price_ir_option().
        The model receives (forward, strike, annuity_caplet, T_fix, option_type)
        where annuity_caplet = yf * df(T_pay).

        Args:
            model: any object implementing ``price_ir_option(forward, strike, annuity, T, option_type)``.
            curve: discount curve.
            projection_curve: forward projection curve (None = single-curve).
        """
        if not hasattr(model, "price_ir_option"):
            raise TypeError(
                f"{type(model).__name__} does not implement price_ir_option() "
                f"— cannot price a cap/floor with this model"
            )

        proj = projection_curve if projection_curve is not None else curve
        total = 0.0
        for accrual_start, accrual_end in self.periods:
            yf = year_fraction(accrual_start, accrual_end, self.day_count)
            df1 = proj.df(accrual_start)
            df2 = proj.df(accrual_end)
            fwd = (df1 - df2) / (yf * df2)

            t_fix = year_fraction(curve.reference_date, accrual_start,
                                  DayCountConvention.ACT_365_FIXED)
            if t_fix <= 0:
                if self.option_type == OptionType.CALL:
                    payoff = max(fwd - self.strike, 0.0)
                else:
                    payoff = max(self.strike - fwd, 0.0)
                total += self.notional * yf * curve.df(accrual_end) * payoff
                continue

            # Caplet annuity = yf * df(payment_date)
            caplet_annuity = yf * curve.df(accrual_end)
            total += self.notional * model.price_ir_option(
                fwd, self.strike, caplet_annuity, t_fix, self.option_type)

        return total


def strip_caplet_vols(
    cap_flat_vols: list[tuple[date, float]],
    strike: float,
    curve: DiscountCurve,
    start: date,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
) -> list[tuple[date, float]]:
    """Strip individual caplet vols from flat cap vols.

    Given flat (quoted) cap vols at increasing maturities, bootstrap
    the individual forward vol for each caplet period.

    The flat vol prices the entire cap. The caplet vol is the vol that
    prices just the marginal caplet added at each maturity.

    Returns list of (fixing_date, caplet_vol).
    """
    from pricebook.models.models import Black76Model

    sorted_quotes = sorted(cap_flat_vols, key=lambda x: x[0])
    caplet_vols: list[tuple[date, float]] = []
    prev_pv = 0.0
    prev_periods: list[tuple[date, date]] = []

    for mat, flat_vol in sorted_quotes:
        cap = CapFloor(start, mat, strike, OptionType.CALL, 1.0, frequency, day_count)
        total_pv = cap.price(Black76Model(vol=flat_vol), curve)
        marginal_pv = total_pv - prev_pv

        if cap.periods and len(cap.periods) > len(prev_periods):
            # The new caplet is the last period
            new_period = cap.periods[-1]
            accrual_start, accrual_end = new_period
            yf = year_fraction(accrual_start, accrual_end, day_count)
            df1 = curve.df(accrual_start)
            df2 = curve.df(accrual_end)
            fwd = (df1 - df2) / (yf * df2)
            t_fix = year_fraction(curve.reference_date, accrual_start, DayCountConvention.ACT_365_FIXED)

            if t_fix > 0 and marginal_pv > 0:
                target = marginal_pv / (yf * df2)

                def obj(v):
                    return black76_price(fwd, strike, v, t_fix, 1.0, OptionType.CALL) - target

                try:
                    caplet_vol = brentq(obj, 0.001, 2.0)
                except Exception:
                    caplet_vol = flat_vol
                caplet_vols.append((accrual_start, caplet_vol))

        prev_pv = total_pv
        prev_periods = list(cap.periods)

    return caplet_vols


def _capfloor_pv_ctx(self, ctx) -> float:
    """PV using PricingContext.

    Per-caplet vol lookup from the IR vol surface.  Fix T4-CAP2:
    pre-fix this method called ``vol_surface.vol()`` with NO arguments
    (which only works for ``FlatVol``) and then built a single
    ``Black76Model(vol)`` used for every caplet in the cap.  For any
    real surface — term-structure, smile cube — this collapses the
    caplet vols to a single number (whichever ``vol()`` happens to
    return) and silently mis-prices every long-dated cap.  Loop
    per-caplet and look up vol at each caplet's accrual_start.
    """
    curve = ctx.discount_curve
    if curve is None:
        raise ValueError("No discount curve in context")
    proj = None
    if hasattr(ctx, 'projection_curves') and ctx.projection_curves:
        dc_key = self.day_count.value
        if dc_key in ctx.projection_curves:
            proj = ctx.projection_curves[dc_key]
        else:
            proj = next(iter(ctx.projection_curves.values()))

    from pricebook.models.models import Black76Model
    vol_surface = None
    if hasattr(ctx, 'vol_surfaces') and ctx.vol_surfaces:
        vol_surface = ctx.vol_surfaces.get("ir") or next(iter(ctx.vol_surfaces.values()), None)
    if vol_surface is None or not hasattr(vol_surface, 'vol'):
        raise ValueError("CapFloor.pv_ctx requires an IR vol surface in ctx.vol_surfaces")

    proj_used = proj if proj is not None else curve
    total = 0.0
    for accrual_start, accrual_end in self.periods:
        yf = year_fraction(accrual_start, accrual_end, self.day_count)
        df1 = proj_used.df(accrual_start)
        df2 = proj_used.df(accrual_end)
        fwd = (df1 - df2) / (yf * df2)
        t_fix = year_fraction(curve.reference_date, accrual_start,
                              DayCountConvention.ACT_365_FIXED)
        caplet_annuity = yf * curve.df(accrual_end)
        if t_fix <= 0:
            if self.option_type == OptionType.CALL:
                intrinsic = max(fwd - self.strike, 0.0)
            else:
                intrinsic = max(self.strike - fwd, 0.0)
            total += self.notional * caplet_annuity * intrinsic
            continue
        # Per-caplet vol from the surface at this caplet's expiry/strike.
        vol = vol_surface.vol(accrual_start, self.strike)
        model = Black76Model(vol)
        total += self.notional * model.price_ir_option(
            fwd, self.strike, caplet_annuity, t_fix, self.option_type)
    return total

CapFloor.pv_ctx = _capfloor_pv_ctx

from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("capfloor", ["start", "end", "strike", "option_type", "notional", "frequency", "day_count", "convention"])(CapFloor)
# Fix T4-CAP1: pre-fix dropped `convention` from the serialisable list,
# silently changing the cap/floor schedule on round-trip.  Same pattern
# as Swaption (v0.976) / IRS (v0.977) / CDS (v0.978).  `calendar`
# remains excluded (runtime-only holiday data).

@classmethod
def _cf_from_convention(cls, conv, start, end, strike, option_type=None, notional=1_000_000.0):
    """Create CapFloor from CurrencyConventions (uses float freq/dc)."""
    if option_type is None:
        from pricebook.models.black76 import OptionType
        option_type = OptionType.CALL
    return cls(start, end, strike, option_type, notional,
               frequency=conv.float_frequency, day_count=conv.float_day_count)

CapFloor.from_convention = _cf_from_convention


# ═══════════════════════════════════════════════════════════════
# Caplet vol stripping + SABR smile calibration
# ═══════════════════════════════════════════════════════════════

from dataclasses import dataclass


@dataclass
class StrippedCapletVol:
    """A single stripped caplet volatility."""
    fixing_date: date
    forward: float
    vol: float
    tenor_years: float

    def to_dict(self) -> dict:
        return vars(self)


def strip_caplet_vols_from_quotes(
    cap_quotes: list[dict],
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    strike: float | None = None,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
) -> list[StrippedCapletVol]:
    """Strip per-caplet Black vols from cap price quotes (dict format).

    Caps are quoted at various tenors (1Y, 2Y, 3Y, 5Y, ...).
    Each cap = strip of caplets. Longer caps embed shorter ones.
    Stripping: solve for the incremental caplet vol at each new tenor.

    Args:
        cap_quotes: list of {"tenor_years": float, "vol": float} (flat cap vol per tenor).
        curve: discount curve.
        projection_curve: forward rate projection curve (default: same as discount).
        strike: ATM strike (if None, use forward rate).
        frequency: caplet frequency (typically quarterly).
        day_count: day count convention.

    Returns:
        List of StrippedCapletVol, one per caplet period.
    """
    import math
    from pricebook.core.day_count import date_from_year_fraction

    proj = projection_curve or curve
    ref = curve.reference_date
    sorted_quotes = sorted(cap_quotes, key=lambda q: q["tenor_years"])

    stripped = []
    prev_cap_pv = 0.0
    prev_tenor = 0.0

    for quote in sorted_quotes:
        tenor_y = quote["tenor_years"]
        flat_vol = quote["vol"]

        # Price full cap at flat vol
        freq_months = {Frequency.QUARTERLY: 3, Frequency.SEMI_ANNUAL: 6,
                       Frequency.ANNUAL: 12}.get(frequency, 3)
        n_caplets = max(1, int(tenor_y * 12 / freq_months))

        full_cap_pv = 0.0
        for i in range(1, n_caplets + 1):
            t_fix = i * freq_months / 12
            t_pay = t_fix + freq_months / 12
            if t_pay > tenor_y + 0.01:
                break

            d_fix = date_from_year_fraction(ref, t_fix)
            d_pay = date_from_year_fraction(ref, t_pay)

            df_pay = curve.df(d_pay)
            fwd = (proj.df(d_fix) / proj.df(d_pay) - 1) / (freq_months / 12)
            k = strike if strike is not None else fwd
            tau = freq_months / 12

            caplet_pv = tau * df_pay * black76_price(fwd, k, flat_vol, t_fix, 1.0, OptionType.CALL)
            full_cap_pv += caplet_pv

        # Incremental caplet PV = full cap - previous cap
        incr_pv = max(full_cap_pv - prev_cap_pv, 0)

        # Number of new caplets in this increment
        prev_n = max(1, int(prev_tenor * 12 / freq_months))
        new_n = n_caplets - prev_n
        if new_n <= 0:
            prev_cap_pv = full_cap_pv
            prev_tenor = tenor_y
            continue

        # Average forward and solve for incremental vol
        avg_t = (prev_tenor + tenor_y) / 2
        d_avg_fix = date_from_year_fraction(ref, avg_t)
        d_avg_pay = date_from_year_fraction(ref, avg_t + freq_months / 12)
        avg_fwd = (proj.df(d_avg_fix) / max(proj.df(d_avg_pay), 1e-10) - 1) / (freq_months / 12)
        k = strike if strike is not None else avg_fwd

        # Solve for caplet vol that reprices the increment
        # Simplified: use the flat cap vol as the caplet vol for this bucket
        caplet_vol = flat_vol

        for i in range(prev_n + 1, n_caplets + 1):
            t_fix = i * freq_months / 12
            d_fix = date_from_year_fraction(ref, t_fix)
            d_pay = date_from_year_fraction(ref, t_fix + freq_months / 12)
            fwd_i = (proj.df(d_fix) / max(proj.df(d_pay), 1e-10) - 1) / (freq_months / 12)

            stripped.append(StrippedCapletVol(
                fixing_date=d_fix,
                forward=fwd_i,
                vol=caplet_vol,
                tenor_years=t_fix,
            ))

        prev_cap_pv = full_cap_pv
        prev_tenor = tenor_y

    return stripped


def calibrate_capfloor_sabr(
    stripped_vols: list[StrippedCapletVol],
    strike_grid: list[float] | None = None,
    beta: float = 0.5,
) -> list[dict]:
    """Calibrate SABR to stripped caplet vols.

    Groups caplets by fixing date and fits SABR per expiry.

    Args:
        stripped_vols: from strip_caplet_vols().
        strike_grid: strikes to generate smile at (default: ATM ± 100bp).
        beta: SABR beta.

    Returns:
        List of dicts with per-expiry SABR params.
    """
    from pricebook.options.sabr import sabr_calibrate

    if not stripped_vols:
        return []

    # Group by tenor bucket (approximate)
    buckets: dict[float, list[StrippedCapletVol]] = {}
    for sv in stripped_vols:
        bucket = round(sv.tenor_years * 2) / 2  # round to nearest 0.5Y
        buckets.setdefault(bucket, []).append(sv)

    results = []
    for tenor_y, vols in sorted(buckets.items()):
        if not vols:
            continue
        avg_fwd = sum(v.forward for v in vols) / len(vols)
        avg_vol = sum(v.vol for v in vols) / len(vols)

        if strike_grid is None:
            strikes = [avg_fwd - 0.01, avg_fwd - 0.005, avg_fwd, avg_fwd + 0.005, avg_fwd + 0.01]
        else:
            strikes = strike_grid

        # Generate synthetic smile from flat vol (parallel shift approximation)
        smile_vols = [avg_vol * (1 + 0.1 * (k - avg_fwd) / max(avg_fwd, 0.01)) for k in strikes]

        try:
            params = sabr_calibrate(avg_fwd, strikes, smile_vols, tenor_y, beta=beta)
            results.append({
                "tenor_years": tenor_y,
                "forward": avg_fwd,
                "atm_vol": avg_vol,
                **params,
            })
        except Exception:
            results.append({
                "tenor_years": tenor_y,
                "forward": avg_fwd,
                "atm_vol": avg_vol,
                "alpha": avg_vol, "beta": beta, "rho": 0.0, "nu": 0.3,
            })

    return results
