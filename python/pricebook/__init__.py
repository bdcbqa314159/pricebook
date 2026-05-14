__version__ = "0.550.0"

# ── Core infrastructure ──
from pricebook.pricing_context import PricingContext
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.trade import Trade, Portfolio
from pricebook.book import Book, Desk, BookLimits, Position
from pricebook.daily_pnl import compute_daily_pnl, attribute_pnl, DailyPnL, BookAttribution

# ── Instruments ──
from pricebook.swap import InterestRateSwap
from pricebook.bond import FixedRateBond
from pricebook.fra import FRA
from pricebook.cds import CDS
from pricebook.cln import CreditLinkedNote
from pricebook.swaption import Swaption
from pricebook.capfloor import CapFloor
from pricebook.frn import FloatingRateNote
from pricebook.fx_forward import FXForward
from pricebook.fx_option import fx_option_price
from pricebook.equity_option import equity_option_price
from pricebook.trs import TotalReturnSwap, TRSResult, FundingLegSpec
from pricebook.repo_desk import RepoTrade, RepoBook, RepoDirection
from pricebook.autocallable import Autocallable
from pricebook.barrier_option import BarrierOption
from pricebook.asian_option import AsianOption
from pricebook.cliquet import Cliquet
from pricebook.tarf import TARF
from pricebook.convertible_bond import (
    ConvertibleBond, ConvertibleResult,
    convertible_delta_hedge, convertible_soft_call,
    contingent_convertible, exchangeable_bond, mandatory_convertible,
)
from pricebook.convertible_bond_desk import (
    cb_risk_metrics, CBRiskMetrics,
    CBBook, CBBookEntry,
    cb_carry_decomposition, cb_daily_pnl, cb_dashboard,
    cb_stress_suite, cb_capital, cb_hedge_recommendations,
    CBLifecycle,
)

# ── Private Equity ──
from pricebook.lbo import LBOModel, LBOResult, SourcesAndUses, ExitAnalysis
from pricebook.dcf import DCFModel, WACCInputs, DCFResult, FootballField, EVBridge
from pricebook.pe_performance import (
    kaplan_schoar_pme, direct_alpha, long_nickels_pme,
    vintage_cohort, commitment_pacing, gp_economics,
    PMEResult, GPEconomics,
)
from pricebook.fund_participation import (
    FundParticipation, FundMetrics, PEFundParticipation, WaterfallConfig,
)
from pricebook.pe_desk import (
    pe_risk_metrics, PERiskMetrics,
    PEBook, PEBookEntry,
    pe_carry_decomposition, pe_daily_pnl, pe_dashboard,
    pe_stress_suite, pe_capital, pe_hedge_recommendations,
    PELifecycle,
)

# ── Models ──
from pricebook.models import (
    Black76Model, BachelierModel, SABRModel, SABRParams,
    HullWhiteModel, BSModel, HestonModel, MCEquityModel,
    IROptionModel, EquityOptionModel,
    price_european,
)
from pricebook.slv import HestonParams

# ── Vol ──
from pricebook.vol_surface import FlatVol

# ── MC engine ──
from pricebook.mc_engine import MCEngine, TimeGrid, MCResult

# ── Database ──
from pricebook.db import PricebookDB

# ── Recovery ──
from pricebook.recovery_surface import RecoverySurface

# ── Trader API ──
from pricebook.api_desk import analyse

# ── Serialisation ──
from pricebook.serialization import (
    to_json,
    from_json,
    instrument_to_dict,
    instrument_from_dict,
    load_trade,
    load_portfolio,
    get_instrument_class,
    list_instruments,
)
from pricebook.registry import (
    get_solver,
    get_tree_european,
    get_tree_american,
    get_pde_pricer,
    get_mc_pricer,
    get_optimizer,
    get_ode_solver,
    get_pricer,
    get_greek_engine,
)
