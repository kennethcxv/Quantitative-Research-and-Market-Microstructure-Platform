"""Cash/inventory ledger for simulated fills, with invariants checked after every fill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class AccountingError(AssertionError):
    pass


@dataclass(frozen=True)
class Fill:
    """A SIMULATED execution. Never a historical trade; ``simulated`` is always True."""

    order_id: int
    decision_ns: int  # local time the strategy decided
    arrival_ns: int  # exchange time the order arrived and matched
    side: int  # +1 buy, -1 sell
    qty: float
    price: float  # after slippage
    touch: float  # best opposite price at arrival (ask for a buy)
    mid_decision: float  # mid the strategy saw when deciding
    mid_arrival: float  # exchange-side mid when the order arrived
    fee: float
    simulated: bool = True


@dataclass
class Ledger:
    initial_cash: float
    max_position: float
    cash: float = field(init=False)
    position: float = field(init=False)
    fees: float = field(init=False, default=0.0)
    fills: list[Fill] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash
        self.position = 0.0

    def apply(self, fill: Fill) -> None:
        if fill.qty <= 0 or not math.isfinite(fill.price):
            raise AccountingError(f"invalid fill {fill}")
        if fill.side * (fill.price - fill.touch) < -1e-12:
            raise AccountingError(f"fill better than the touch: {fill}")
        self.position += fill.side * fill.qty
        self.cash -= fill.side * fill.qty * fill.price + fill.fee
        self.fees += fill.fee
        self.fills.append(fill)
        if abs(self.position) > self.max_position * (1 + 1e-12):
            raise AccountingError(f"position {self.position} breaches limit {self.max_position}")

    def equity(self, mid: float) -> float:
        return self.cash + self.position * mid

    def liquidation_value(self, bid: float, ask: float) -> float:
        """Conservative mark: longs at the bid, shorts at the ask."""
        return self.cash + self.position * (bid if self.position > 0 else ask)

    def check(self) -> None:
        """Recompute position and cash from the fill list; they must match the running state.

        O(fills), so the engine calls it once at the end of a run; every fill is still checked
        against the touch and the position limit as it is applied."""
        pos = sum(f.side * f.qty for f in self.fills)
        cash = self.initial_cash - sum(f.side * f.qty * f.price + f.fee for f in self.fills)
        scale = max(1.0, abs(self.initial_cash))
        if abs(pos - self.position) > 1e-9 * max(1.0, abs(pos)):
            raise AccountingError(f"position {self.position} != sum of fills {pos}")
        if abs(cash - self.cash) > 1e-9 * scale:
            raise AccountingError(f"cash {self.cash} != initial - fills {cash}")
