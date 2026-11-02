"""Deterministic discrete-event backtester.

Events, processed strictly in time order from one priority queue:

* ``SIGNAL`` (availability clock): a forecast becomes available; the strategy picks a target.
* ``ARRIVAL`` (exchange clock): the order reaches the venue and is matched (execution.py).
* ``ACK`` (local clock): the strategy learns what filled. Until then it has one working order and
  does not send another, so it can never act on fills it has not been told about.

Ties are broken by event kind (arrivals, then acks, then signals) and then by insertion order,
so a run is a pure function of its inputs. Clocks are assumed synchronised; all latencies are
configured assumptions, not measurements.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel

from qrp.backtest.accounting import Fill, Ledger
from qrp.backtest.execution import BookTimeline, ClockViolation, CostConfig, MarketView, match
from qrp.stats import Interval, block_bootstrap

MS = 1_000_000
ARRIVAL, ACK, SIGNAL = 0, 1, 2  # tie-break priority at equal timestamps


class BacktestConfig(BaseModel):
    position_size: float = 100.0  # |target| when the signal clears the threshold
    max_position: float = 100.0  # inventory limit
    max_notional: float | None = None  # exposure limit in quote currency
    decision_latency_ms: float = 1.0
    order_latency_ms: float = 5.0
    ack_latency_ms: float = 5.0
    max_book_age_s: float = 5.0
    flatten_before_end_s: float = 60.0
    initial_cash: float = 1_000_000.0
    costs: CostConfig = CostConfig()


def threshold_target(threshold: float, size: float) -> Callable[[float], float]:
    """Long ``size`` above +threshold, short below -threshold, flat in between."""

    def target(pred: float) -> float:
        if pred > threshold:
            return size
        if pred < -threshold:
            return -size
        return 0.0

    return target


@dataclass(frozen=True)
class _Order:
    order_id: int
    decision_ns: int
    side: int
    qty: float
    mid_decision: float


@dataclass
class BacktestResult:
    fills: pl.DataFrame
    equity: pl.DataFrame
    summary: dict[str, Any]


def run_backtest(
    book: BookTimeline,
    signals: pl.DataFrame,
    cfg: BacktestConfig,
    target: Callable[[float], float],
    seed: int = 0,
) -> BacktestResult:
    """``signals``: columns ``t_ns`` (availability time), ``session``, ``pred``; sorted by time."""
    t_sig = signals["t_ns"].to_numpy()
    if np.any(np.diff(t_sig) < 0):
        raise ClockViolation("signals must be sorted by availability time")
    sessions = signals["session"].to_list()
    preds = signals["pred"].to_numpy()
    flatten_at = (
        signals.group_by("session")
        .agg(pl.col("t_ns").max() - int(cfg.flatten_before_end_s * 1e9))
        .rows()
    )
    flatten_from = dict(flatten_at)

    max_age = int(cfg.max_book_age_s * 1e9)
    view = MarketView(book, max_age)
    ledger = Ledger(cfg.initial_cash, cfg.max_position)
    consumed: dict[tuple[int, int, float], float] = {}
    queue: list[tuple[int, int, int, Any]] = []
    counter = itertools.count()
    for i, t in enumerate(t_sig):
        queue.append((int(t), SIGNAL, next(counter), i))
    heapq.heapify(queue)

    known_position = 0.0
    working = False
    order_ids = itertools.count(1)
    counts = dict.fromkeys(("orders", "rejected", "partial", "skipped_working", "no_mid"), 0)
    equity_rows: list[tuple[int, float, float, float]] = []
    last_time = -(2**62)

    while queue:
        now, kind, _, payload = heapq.heappop(queue)
        if now < last_time:
            raise ClockViolation("event queue went backwards")  # engine invariant
        last_time = now
        view.now = now

        if kind == SIGNAL:
            i = payload
            mid = view.mid(now)
            if mid is not None:
                equity_rows.append((now, ledger.position, ledger.cash, ledger.equity(mid)))
            if working:
                counts["skipped_working"] += 1
                continue
            goal = 0.0 if now >= flatten_from[sessions[i]] else target(float(preds[i]))
            goal = float(np.clip(goal, -cfg.max_position, cfg.max_position))
            if cfg.max_notional is not None and mid is not None:
                cap = cfg.max_notional / mid
                goal = float(np.clip(goal, -cap, cap))
            change = goal - known_position
            if abs(change) < 1e-12:
                continue
            if mid is None:
                counts["no_mid"] += 1
                continue
            order = _Order(next(order_ids), now, 1 if change > 0 else -1, abs(change), mid)
            send = now + int(cfg.decision_latency_ms * MS)
            heapq.heappush(
                queue, (send + int(cfg.order_latency_ms * MS), ARRIVAL, next(counter), order)
            )
            working = True
            counts["orders"] += 1

        elif kind == ARRIVAL:
            order = payload
            executions = match(book, now, order.side, order.qty, cfg.costs, max_age, consumed)
            filled = 0.0
            for ex in executions:
                ledger.apply(
                    Fill(
                        order_id=order.order_id,
                        decision_ns=order.decision_ns,
                        arrival_ns=now,
                        side=order.side,
                        qty=ex.qty,
                        price=ex.price,
                        touch=ex.touch,
                        mid_decision=order.mid_decision,
                        mid_arrival=ex.mid_arrival,
                        fee=ex.fee,
                    )
                )
                filled += ex.qty
            if not executions:
                counts["rejected"] += 1
            elif filled < order.qty - 1e-12:
                counts["partial"] += 1
            ack = (order, filled)
            heapq.heappush(queue, (now + int(cfg.ack_latency_ms * MS), ACK, next(counter), ack))

        else:  # ACK
            order, filled = payload
            known_position += order.side * filled
            working = False

    ledger.check()
    # Final mark after the last event: open inventory (a flatten order that could not fill) is
    # valued at the liquidation side, not the mid.
    last = book.index(book.avail_ns, last_time)
    final_value = (
        ledger.liquidation_value(float(book.bid_px[last, 0]), float(book.ask_px[last, 0]))
        if ledger.position and last >= 0
        else ledger.cash
    )
    fills = pl.DataFrame(
        [f.__dict__ for f in ledger.fills],
        schema={
            "order_id": pl.Int64,
            "decision_ns": pl.Int64,
            "arrival_ns": pl.Int64,
            "side": pl.Int8,
            "qty": pl.Float64,
            "price": pl.Float64,
            "touch": pl.Float64,
            "mid_decision": pl.Float64,
            "mid_arrival": pl.Float64,
            "fee": pl.Float64,
            "simulated": pl.Boolean,
        },
        orient="row",
    )
    equity = pl.DataFrame(
        equity_rows,
        schema={"t_ns": pl.Int64, "position": pl.Float64, "cash": pl.Float64, "equity": pl.Float64},
        orient="row",
    )
    summary = _summarize(ledger, fills, equity, final_value, cfg, counts, seed)
    return BacktestResult(fills, equity, summary)


def _summarize(
    ledger: Ledger,
    fills: pl.DataFrame,
    equity: pl.DataFrame,
    final_value: float,
    cfg: BacktestConfig,
    counts: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    # P&L identity, all in quote currency (s = side, q = qty, p = fill price):
    #   total = edge_at_decision - latency_cost - spread_cost - fees
    #   latency_cost = sum s*q*(mid_arrival - mid_decision)   price drift while the order flew
    #   spread_cost  = sum s*q*(p - mid_arrival)              half-spread, book walk, slippage
    # edge_at_decision is what the signal earned if we could trade at the decision mid for free
    # (when flat at the end it equals -sum s*q*mid_decision).
    total = final_value - ledger.initial_cash
    sq = fills["side"].cast(pl.Float64) * fills["qty"]
    latency_cost = float((sq * (fills["mid_arrival"] - fills["mid_decision"])).sum())
    spread_cost = float((sq * (fills["price"] - fills["mid_arrival"])).sum())
    fees = float(fills["fee"].sum())
    per_minute = (
        equity.with_columns(minute=pl.col("t_ns") // (60 * 10**9))
        .group_by("minute", maintain_order=True)
        .agg(pl.col("equity").last())
        .with_columns(pnl=pl.col("equity").diff())
        .drop_nulls("pnl")["pnl"]
        .to_numpy()
    )
    pnl_ci: Interval | None = None
    if len(per_minute) > 30:
        pnl_ci = block_bootstrap(
            lambda i: float(per_minute[i].mean()), len(per_minute), 30, 500, seed
        )
    curve = equity["equity"].to_numpy()
    drawdown = float((np.maximum.accumulate(curve) - curve).max()) if len(curve) else 0.0
    return {
        **counts,
        "fills": fills.height,
        "traded_qty": float(fills["qty"].sum()),
        "traded_notional": float((fills["qty"] * fills["price"]).sum()),
        "pnl_total": total,
        "edge_at_decision": total + latency_cost + spread_cost + fees,
        "latency_cost": latency_cost,
        "spread_cost": spread_cost,
        "fees": fees,
        "open_position_at_end": ledger.position,
        "max_drawdown": drawdown,
        "pnl_per_minute_mean_ci": pnl_ci.model_dump() if pnl_ci else None,
        "costs": cfg.costs.model_dump(),
    }
