"""Book timelines on two clocks, and the conservative fill model.

The strategy sees the book on the availability clock. Its orders match against the book on the
exchange clock at arrival time, which is later by the configured latencies. Both views come
from the same reconstructed messages; the historical data itself is never modified.

Fill model (marketable IOC orders only):

* reject if the exchange-side book is unsynced, crossed, one-sided or older than
  ``max_book_age`` at arrival;
* walk the visible levels from the touch; whatever the visible depth cannot fill is cancelled;
* liquidity our earlier orders took from the same book message is not offered again (our orders
  do not otherwise change later historical prices: impact persistence is not modelled);
* every fill pays ``slippage_bps + impact_bps_per_l1 * qty / L1 size`` adversely, plus taker fees.

No passive fills are simulated: L2 data cannot tell where a resting order would sit in the queue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel

from qrp.book import ladder_columns

FloatArray = npt.NDArray[np.float64]


class ClockViolation(RuntimeError):
    """Something asked for market data from the future."""


class CostConfig(BaseModel):
    taker_fee_bps: float = 10.0
    slippage_bps: float = 0.0
    impact_bps_per_l1: float = 0.0


@dataclass(frozen=True)
class BookTimeline:
    avail_ns: npt.NDArray[np.int64]  # availability clock, sorted
    exch_ns: npt.NDArray[np.int64]  # exchange clock, running max in message order
    bid_px: FloatArray  # (messages, levels)
    bid_sz: FloatArray
    ask_px: FloatArray
    ask_sz: FloatArray
    usable: npt.NDArray[np.bool_]

    @classmethod
    def from_quotes(cls, quotes: pl.DataFrame) -> BookTimeline:
        """``quotes`` from ``book.reconstruct`` plus ``available_time_ns``, in message order."""
        avail = quotes["available_time_ns"].to_numpy()
        if np.any(np.diff(avail) < 0):
            raise ClockViolation("quotes are not in availability order")
        levels = sum(c.startswith("bid_px_") for c in quotes.columns)
        cols = ladder_columns(levels)

        def side(prefix: str) -> FloatArray:
            names = [c for c in cols if c.startswith(prefix)]
            return quotes.select(names).to_numpy().astype(np.float64)

        exch = quotes.select(pl.coalesce("event_time_ns", "recv_time_ns")).to_series().to_numpy()
        bid_px, ask_px = side("bid_px_"), side("ask_px_")
        usable = (
            quotes["synced"].to_numpy()
            & ~quotes["crossed"].to_numpy()
            & np.isfinite(bid_px[:, 0])
            & np.isfinite(ask_px[:, 0])
        )
        return cls(
            avail_ns=avail,
            exch_ns=np.maximum.accumulate(exch),
            bid_px=bid_px,
            bid_sz=side("bid_sz_"),
            ask_px=ask_px,
            ask_sz=side("ask_sz_"),
            usable=usable,
        )

    def index(self, clock: npt.NDArray[np.int64], t: int) -> int:
        return int(np.searchsorted(clock, t, side="right")) - 1

    def mid(self, i: int) -> float:
        return float((self.bid_px[i, 0] + self.ask_px[i, 0]) / 2)


@dataclass
class MarketView:
    """What the strategy may look at: the availability-clock book, never past ``now``."""

    book: BookTimeline
    max_age_ns: int
    now: int = -(2**62)

    def mid(self, t: int) -> float | None:
        if t > self.now:
            raise ClockViolation(f"asked for t={t} at now={self.now}")
        i = self.book.index(self.book.avail_ns, t)
        if i < 0 or not self.book.usable[i] or t - self.book.avail_ns[i] > self.max_age_ns:
            return None
        return self.book.mid(i)


@dataclass(frozen=True)
class Execution:
    price: float
    qty: float
    touch: float
    mid_arrival: float
    fee: float


def match(
    book: BookTimeline,
    arrival_ns: int,
    side: int,
    qty: float,
    costs: CostConfig,
    max_age_ns: int,
    consumed: dict[tuple[int, int, float], float],
) -> list[Execution]:
    """Fill a marketable IOC order against the exchange-side book at ``arrival_ns``."""
    i = book.index(book.exch_ns, arrival_ns)
    if i < 0 or not book.usable[i] or arrival_ns - book.exch_ns[i] > max_age_ns:
        return []
    prices, sizes = (
        (book.ask_px[i], book.ask_sz[i]) if side > 0 else (book.bid_px[i], book.bid_sz[i])
    )
    touch, l1 = float(prices[0]), float(sizes[0])
    mid = book.mid(i)
    impact = costs.impact_bps_per_l1 * qty / l1
    out = []
    remaining = qty
    for price, size in zip(prices, sizes, strict=True):
        if remaining <= 0 or not np.isfinite(price):
            break
        key = (i, side, float(price))
        available = float(size) - consumed.get(key, 0.0)
        take = min(remaining, available)
        if take <= 0:
            continue
        consumed[key] = consumed.get(key, 0.0) + take
        remaining -= take
        paid = float(price) * (1 + side * (costs.slippage_bps + impact) / 1e4)
        fee = abs(take * paid) * costs.taker_fee_bps / 1e4
        out.append(Execution(price=paid, qty=take, touch=touch, mid_arrival=mid, fee=fee))
    return out
