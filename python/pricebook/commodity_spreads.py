"""Inter-commodity spreads: crack, spark, dark, crush.

Standard refining, power-generation, and agricultural processing
spreads that express the margin between an input commodity and its
processed output.

* :class:`CrackSpread` — crude oil → refined products (3-2-1, 5-3-2, or custom).
* :class:`SparkSpread` — power − heat_rate × natural gas.
* :class:`DarkSpread` — power − heat_rate × coal.
* :class:`CrushSpread` — soybeans → soybean meal + soybean oil.
* :class:`GenericSpread` — weighted multi-leg inter-commodity spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---- Generic multi-leg spread ----

@dataclass
class SpreadLeg:
    """One leg of an inter-commodity spread.

    ``weight > 0`` is a buy (input/output depends on convention);
    ``weight < 0`` is a sell.
    """
    commodity: str
    weight: float
    unit: str = "unit"


@dataclass
class GenericSpread:
    """Weighted multi-leg inter-commodity spread.

    PV = direction × quantity × Σ (weight_i × price_i)

    A balanced spread has ``Σ weights = 0`` in output-normalised units,
    ensuring zero residual exposure to a uniform price shift.
    """
    name: str
    legs: list[SpreadLeg]
    quantity: float = 1.0
    direction: int = 1

    def spread_value(self, prices: dict[str, float]) -> float:
        """Spread level: Σ weight_i × price_i."""
        return sum(
            leg.weight * prices.get(leg.commodity, 0.0)
            for leg in self.legs
        )

    def pv(self, prices: dict[str, float]) -> float:
        """PV = direction × quantity × spread_value."""
        return self.direction * self.quantity * self.spread_value(prices)

    def residual_exposure(self) -> float:
        """Sum of weights — zero for a balanced spread.

        When all legs are denominated in the same currency-per-unit,
        a zero weight-sum means a uniform +1 price shift in all
        commodities leaves the spread PV unchanged.
        """
        return sum(leg.weight for leg in self.legs)


# ---- Crack spreads (refining) ----

def crack_spread_321(quantity: float = 1.0, direction: int = 1) -> GenericSpread:
    """Standard 3-2-1 crack spread.

    3 barrels crude → 2 barrels gasoline + 1 barrel distillate.
    Margin per barrel of crude = (2×gasoline + 1×distillate − 3×crude) / 3.
    """
    return GenericSpread(
        name="3-2-1 crack",
        legs=[
            SpreadLeg("crude", weight=-3.0, unit="bbl"),
            SpreadLeg("gasoline", weight=2.0, unit="bbl"),
            SpreadLeg("distillate", weight=1.0, unit="bbl"),
        ],
        quantity=quantity,
        direction=direction,
    )


def crack_spread_532(quantity: float = 1.0, direction: int = 1) -> GenericSpread:
    """5-3-2 crack spread.

    5 barrels crude → 3 barrels gasoline + 2 barrels heating oil.
    """
    return GenericSpread(
        name="5-3-2 crack",
        legs=[
            SpreadLeg("crude", weight=-5.0, unit="bbl"),
            SpreadLeg("gasoline", weight=3.0, unit="bbl"),
            SpreadLeg("heating_oil", weight=2.0, unit="bbl"),
        ],
        quantity=quantity,
        direction=direction,
    )


# ---- Spark / dark spreads (power generation) ----

@dataclass
class SparkSpread:
    """Power − heat_rate × natural gas price.

    The heat rate is in MMBtu per MWh (typically 7–10 for a gas plant).
    Positive spread → profitable generation.

    Attributes:
        heat_rate: fuel consumption per unit of power (MMBtu/MWh).
        quantity: number of MWh.
        direction: +1 long spark (buy gas, sell power); -1 short.
    """
    heat_rate: float = 7.0
    quantity: float = 1.0
    direction: int = 1

    def spread_value(
        self,
        power_price: float,
        gas_price: float,
    ) -> float:
        """Spark spread = power − heat_rate × gas."""
        return power_price - self.heat_rate * gas_price

    def pv(self, power_price: float, gas_price: float) -> float:
        return self.direction * self.quantity * self.spread_value(power_price, gas_price)

    def implied_generation_margin(
        self,
        power_price: float,
        gas_price: float,
        variable_om: float = 0.0,
    ) -> float:
        """Net generation margin after fuel and variable O&M."""
        return self.spread_value(power_price, gas_price) - variable_om


@dataclass
class DarkSpread:
    """Power − heat_rate × coal price.

    Same structure as SparkSpread but for coal-fired generation.
    Heat rate is in tonnes per MWh (or a thermal equivalent).
    """
    heat_rate: float = 0.4
    quantity: float = 1.0
    direction: int = 1

    def spread_value(
        self,
        power_price: float,
        coal_price: float,
    ) -> float:
        """Dark spread = power − heat_rate × coal."""
        return power_price - self.heat_rate * coal_price

    def pv(self, power_price: float, coal_price: float) -> float:
        return self.direction * self.quantity * self.spread_value(power_price, coal_price)


# ---- Crush spread (agriculture) ----

def crush_spread(
    soy_to_meal: float = 0.80,
    soy_to_oil: float = 0.18,
    quantity: float = 1.0,
    direction: int = 1,
) -> GenericSpread:
    """Soybean crush: 1 unit soybean → meal + oil.

    Standard ratios: 1 bushel soybeans → 0.80 short tons meal + 0.18 lbs oil
    (simplified to weight fractions for pricing purposes).

    A positive crush spread = profitable processing.
    """
    return GenericSpread(
        name="soybean crush",
        legs=[
            SpreadLeg("soybean", weight=-1.0, unit="bushel"),
            SpreadLeg("soybean_meal", weight=soy_to_meal, unit="short_ton"),
            SpreadLeg("soybean_oil", weight=soy_to_oil, unit="lb"),
        ],
        quantity=quantity,
        direction=direction,
    )


def reverse_crush(
    soy_to_meal: float = 0.80,
    soy_to_oil: float = 0.18,
    quantity: float = 1.0,
) -> GenericSpread:
    """Reverse crush = short the crush spread."""
    return crush_spread(soy_to_meal, soy_to_oil, quantity, direction=-1)
