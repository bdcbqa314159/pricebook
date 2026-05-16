"""Market conventions for equity, commodity, and inflation products.

* Equity: exchange calendars, settlement, index specs, dividend handling.
* Commodity: contract specs (CME, ICE, LME), delivery, contract months.
* Inflation: linker conventions per country (TIPS, ILG, OATi, BTPei).

References:
    CME Group, *Product Specifications*, 2024.
    ICE, *Contract Specifications*, 2024.
    LME, *Ring Trading and Contract Rules*, 2024.
    US Treasury, *TIPS Information*, TreasuryDirect.gov.
    UK DMO, *Index-Linked Gilts Information*, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- Equity index conventions ----

@dataclass
class EquityIndexSpec:
    """Equity index specification."""
    ticker: str
    name: str
    exchange: str
    currency: str
    settlement_lag: int         # T+N business days
    option_style: str           # "european" or "american"
    option_multiplier: float    # contract multiplier (e.g., $100 per point for SPX)
    dividend_frequency: str     # "quarterly", "semi-annual", "annual"
    ex_date_rule: str           # "T-1" (US), "T-2" (some EU), "record_date" (varies)


EQUITY_INDICES = {
    "SPX": EquityIndexSpec("SPX", "S&P 500", "CBOE", "USD", 2, "european", 100.0, "quarterly", "T-1"),
    "NDX": EquityIndexSpec("NDX", "Nasdaq 100", "CBOE", "USD", 2, "european", 100.0, "quarterly", "T-1"),
    "SX5E": EquityIndexSpec("SX5E", "Euro Stoxx 50", "Eurex", "EUR", 2, "european", 10.0, "annual", "T-2"),
    "DAX": EquityIndexSpec("DAX", "DAX 40", "Eurex", "EUR", 2, "european", 5.0, "annual", "T-2"),
    "UKX": EquityIndexSpec("UKX", "FTSE 100", "ICE", "GBP", 2, "european", 10.0, "quarterly", "T-1"),
    "NKY": EquityIndexSpec("NKY", "Nikkei 225", "OSE", "JPY", 2, "european", 1000.0, "semi-annual", "T-2"),
    "HSI": EquityIndexSpec("HSI", "Hang Seng", "HKEX", "HKD", 2, "european", 50.0, "semi-annual", "T-1"),
    "AS51": EquityIndexSpec("AS51", "ASX 200", "ASX", "AUD", 2, "european", 25.0, "semi-annual", "T-2"),
    "SPTSX": EquityIndexSpec("SPTSX", "S&P/TSX 60", "TMX", "CAD", 2, "european", 200.0, "quarterly", "T-1"),
}


def get_equity_index(ticker: str) -> EquityIndexSpec:
    """Look up equity index spec by ticker."""
    key = ticker.upper()
    if key in EQUITY_INDICES:
        return EQUITY_INDICES[key]
    raise ValueError(f"Unknown index: {ticker}. Available: {list(EQUITY_INDICES.keys())}")


# ---- Commodity contract conventions ----

@dataclass
class CommodityContractSpec:
    """Commodity futures contract specification."""
    symbol: str
    name: str
    exchange: str
    currency: str
    unit: str                   # e.g., "barrels", "MMBtu", "troy oz"
    contract_size: float        # units per contract
    tick_size: float            # minimum price increment
    tick_value: float           # dollar value of one tick
    contract_months: str        # e.g., "FGHJKMNQUVXZ" (all months)
    settlement_type: str        # "physical" or "cash"
    price_quotation: str        # e.g., "$/barrel", "cents/bushel"


COMMODITY_CONTRACTS = {
    "CL": CommodityContractSpec("CL", "WTI Crude Oil", "NYMEX/CME", "USD",
        "barrels", 1000, 0.01, 10.0, "FGHJKMNQUVXZ", "physical", "$/barrel"),
    "BRN": CommodityContractSpec("BRN", "Brent Crude", "ICE", "USD",
        "barrels", 1000, 0.01, 10.0, "FGHJKMNQUVXZ", "cash", "$/barrel"),
    "NG": CommodityContractSpec("NG", "Henry Hub Natural Gas", "NYMEX/CME", "USD",
        "MMBtu", 10000, 0.001, 10.0, "FGHJKMNQUVXZ", "physical", "$/MMBtu"),
    "GC": CommodityContractSpec("GC", "Gold", "COMEX/CME", "USD",
        "troy oz", 100, 0.10, 10.0, "GJMQVZ", "physical", "$/troy oz"),
    "SI": CommodityContractSpec("SI", "Silver", "COMEX/CME", "USD",
        "troy oz", 5000, 0.005, 25.0, "HKNUZ", "physical", "$/troy oz"),
    "HG": CommodityContractSpec("HG", "Copper", "COMEX/CME", "USD",
        "lbs", 25000, 0.0005, 12.50, "HKNUZ", "physical", "cents/lb"),
    "ZC": CommodityContractSpec("ZC", "Corn", "CBOT/CME", "USD",
        "bushels", 5000, 0.25, 12.50, "HKNUZ", "physical", "cents/bushel"),
    "ZW": CommodityContractSpec("ZW", "Wheat", "CBOT/CME", "USD",
        "bushels", 5000, 0.25, 12.50, "HKNUZ", "physical", "cents/bushel"),
    "ZS": CommodityContractSpec("ZS", "Soybeans", "CBOT/CME", "USD",
        "bushels", 5000, 0.25, 12.50, "FHKNQUX", "physical", "cents/bushel"),
    "CO": CommodityContractSpec("CO", "Cocoa", "ICE", "USD",
        "tonnes", 10, 1.0, 10.0, "HKNUZ", "physical", "$/tonne"),
}

# LME metals use prompt date system (not standard monthly delivery)
LME_METALS = {
    "LCU": CommodityContractSpec("LCU", "LME Copper", "LME", "USD",
        "tonnes", 25, 0.50, 12.50, "prompt", "physical", "$/tonne"),
    "LAH": CommodityContractSpec("LAH", "LME Aluminium", "LME", "USD",
        "tonnes", 25, 0.50, 12.50, "prompt", "physical", "$/tonne"),
    "LZS": CommodityContractSpec("LZS", "LME Zinc", "LME", "USD",
        "tonnes", 25, 0.50, 12.50, "prompt", "physical", "$/tonne"),
}


def get_commodity_contract(symbol: str) -> CommodityContractSpec:
    """Look up commodity contract spec by symbol."""
    key = symbol.upper()
    if key in COMMODITY_CONTRACTS:
        return COMMODITY_CONTRACTS[key]
    if key in LME_METALS:
        return LME_METALS[key]
    raise ValueError(f"Unknown commodity: {symbol}. Available: "
                     f"{list(COMMODITY_CONTRACTS.keys()) + list(LME_METALS.keys())}")


# ---- Inflation linker conventions ----

@dataclass
class LinkerConvention:
    """Inflation-linked bond convention per country."""
    country: str
    index_name: str             # CPI index used
    lag_months: int             # publication lag (3 = 3-month lag)
    coupon_frequency: str       # "semi-annual" or "annual"
    day_count: str              # day count convention
    deflation_floor: bool       # whether principal has floor at par
    index_ratio_method: str     # "daily_linear" (US/UK) or "monthly" (some)


LINKER_CONVENTIONS = {
    "US": LinkerConvention("US", "CPI-U (All Urban)", 3, "semi-annual",
        "ACT/ACT", True, "daily_linear"),
    "UK": LinkerConvention("UK", "RPI", 8, "semi-annual",
        "ACT/ACT", False, "daily_linear"),
    "FR": LinkerConvention("FR", "HICP ex-tobacco", 3, "annual",
        "ACT/ACT", False, "daily_linear"),
    "IT": LinkerConvention("IT", "HICP ex-tobacco", 3, "annual",
        "ACT/ACT", False, "daily_linear"),
    "DE": LinkerConvention("DE", "HICP ex-tobacco", 3, "annual",
        "ACT/ACT", False, "daily_linear"),
    "CA": LinkerConvention("CA", "CPI (All Items)", 3, "semi-annual",
        "ACT/365F", True, "daily_linear"),
    "AU": LinkerConvention("AU", "CPI (Weighted Average)", 6, "quarterly",
        "ACT/ACT", False, "quarterly"),
    "JP": LinkerConvention("JP", "CPI (ex-fresh food)", 3, "semi-annual",
        "ACT/365F", True, "daily_linear"),
}


def get_linker_convention(country: str) -> LinkerConvention:
    """Look up linker convention by country code."""
    key = country.upper()
    if key in LINKER_CONVENTIONS:
        return LINKER_CONVENTIONS[key]
    raise ValueError(f"Unknown country: {country}. Available: {list(LINKER_CONVENTIONS.keys())}")


def index_ratio(
    base_cpi: float,
    ref_cpi: float,
    daily_cpi_start: float | None = None,
    daily_cpi_end: float | None = None,
    day_of_month: int = 1,
    days_in_month: int = 30,
) -> float:
    """Compute inflation index ratio for a linker.

    For daily linear interpolation (US TIPS, UK ILG):
        CPI_ref(d) = CPI(m-1) + (d-1)/(D) × (CPI(m) − CPI(m-1))

    where m = settlement month minus lag, D = days in month.

    For simple monthly (Australian):
        index_ratio = ref_cpi / base_cpi

    Args:
        base_cpi: CPI at bond issue date.
        ref_cpi: CPI at current reference date.
        daily_cpi_start: CPI at start of interpolation month (for daily linear).
        daily_cpi_end: CPI at end of interpolation month.
        day_of_month: current day within the month.
        days_in_month: total days in the current month.
    """
    if daily_cpi_start is not None and daily_cpi_end is not None:
        # Daily linear interpolation
        interp_cpi = daily_cpi_start + (day_of_month - 1) / days_in_month * (daily_cpi_end - daily_cpi_start)
        return interp_cpi / base_cpi

    # Simple ratio
    return ref_cpi / base_cpi
