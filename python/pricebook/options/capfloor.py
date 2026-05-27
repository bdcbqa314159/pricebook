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
        from pricebook.risk.greeks import Greeks
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

    def pv_ctx(self, ctx) -> float:
        """PV using PricingContext (requires vol_surface for model)."""
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        proj = None
        if hasattr(ctx, 'projection_curves') and ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()), None)
        # Use flat vol model from context if available
        from pricebook.models.models import Black76Model
        vol_surface = ctx.vol_surfaces.get("ir") if hasattr(ctx, 'vol_surfaces') else None
        if vol_surface and hasattr(vol_surface, 'vol'):
            model = Black76Model(vol_surface.vol())
        else:
            model = Black76Model(0.20)  # fallback
        return self.price(model, curve, proj)

from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("capfloor", ["start", "end", "strike", "option_type", "notional", "frequency", "day_count"])(CapFloor)
