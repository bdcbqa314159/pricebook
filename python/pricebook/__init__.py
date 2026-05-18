__version__ = "0.558.0"

# ── Core ──
from pricebook.core.pricing_context import PricingContext
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.trade import Trade, Portfolio
from pricebook.core.book import Book, Desk, BookLimits, Position
from pricebook.core.daily_pnl import compute_daily_pnl, attribute_pnl, DailyPnL, BookAttribution

# ── Instruments ──
from pricebook.fixed_income.swap import InterestRateSwap
from pricebook.fixed_income.bond import FixedRateBond
from pricebook.fixed_income.fra import FRA
from pricebook.credit.cds import CDS
from pricebook.credit.cln import CreditLinkedNote
from pricebook.options.swaption import Swaption
from pricebook.options.capfloor import CapFloor
from pricebook.fixed_income.frn import FloatingRateNote
from pricebook.fx.fx_forward import FXForward
from pricebook.fx.fx_option import fx_option_price
from pricebook.options.equity_option import equity_option_price
from pricebook.equity.trs import TotalReturnSwap

# ── Models ──
from pricebook.models.models import (
    Black76Model, BachelierModel, SABRModel, SABRParams,
    HullWhiteModel, BSModel, HestonModel, MCEquityModel,
    IROptionModel, EquityOptionModel,
    price_european,
)
from pricebook.options.slv import HestonParams

# ── Vol ──
from pricebook.options.vol_surface import FlatVol

# ── MC engine ──
from pricebook.models.mc_engine import MCEngine, TimeGrid, MCResult

# ── Database ──
from pricebook.db.db import PricebookDB

# ── Recovery ──
from pricebook.credit.recovery_surface import RecoverySurface

# ── Serialisation ──
from pricebook.core.serialization import (
    to_json,
    from_json,
    instrument_to_dict,
    instrument_from_dict,
    load_trade,
    load_portfolio,
    get_instrument_class,
    list_instruments,
)
from pricebook.registry import get_solver, get_tree_european
