"""Event-driven backtesting: simulated execution, kept separate from historical data."""

from qrp.backtest.accounting import AccountingError, Fill, Ledger
from qrp.backtest.engine import BacktestConfig, BacktestResult, run_backtest, threshold_target
from qrp.backtest.execution import BookTimeline, ClockViolation, CostConfig, MarketView

__all__ = [
    "AccountingError",
    "BacktestConfig",
    "BacktestResult",
    "BookTimeline",
    "ClockViolation",
    "CostConfig",
    "Fill",
    "Ledger",
    "MarketView",
    "run_backtest",
    "threshold_target",
]
