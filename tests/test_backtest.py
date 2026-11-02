from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from qrp.backtest import (
    AccountingError,
    BacktestConfig,
    BookTimeline,
    ClockViolation,
    CostConfig,
    Fill,
    Ledger,
    MarketView,
    run_backtest,
    threshold_target,
)
from qrp.book import reconstruct
from qrp.features import FeatureConfig, TimingConfig, available_time, build_features
from qrp.synth import SynthConfig, generate, synthetic_capabilities

MS = 1_000_000
NAN = math.nan
DAY = dt.date(2024, 1, 2)


def timeline(rows: list[dict[str, float]]) -> BookTimeline:
    """rows: t_ms (exchange time; available 2 ms later) + two-level ladder."""
    base = {"bid_px_1": NAN, "bid_sz_1": NAN, "ask_px_1": NAN, "ask_sz_1": NAN}
    data = [
        {
            "event_time_ns": int(r["t_ms"] * MS),
            "recv_time_ns": int(r["t_ms"] * MS) + 2 * MS,
            "available_time_ns": int(r["t_ms"] * MS) + 2 * MS,
            **base,
            **{k: v for k, v in r.items() if k != "t_ms"},
            "synced": bool(r.get("synced", True)),
            "crossed": False,
        }
        for r in rows
    ]
    order = [
        "bid_px_0",
        "bid_sz_0",
        "ask_px_0",
        "ask_sz_0",
        "bid_px_1",
        "bid_sz_1",
        "ask_px_1",
        "ask_sz_1",
    ]
    df = pl.DataFrame(data).select(
        "event_time_ns", "recv_time_ns", "available_time_ns", *order, "synced", "crossed"
    )
    return BookTimeline.from_quotes(df)


BOOK = timeline(
    [
        {
            "t_ms": 0,
            "bid_px_0": 99,
            "bid_sz_0": 10,
            "ask_px_0": 101,
            "ask_sz_0": 5,
            "ask_px_1": 102,
            "ask_sz_1": 5,
        },
        {
            "t_ms": 10,
            "bid_px_0": 99,
            "bid_sz_0": 10,
            "ask_px_0": 100,
            "ask_sz_0": 5,
            "ask_px_1": 101,
            "ask_sz_1": 5,
        },
    ]
)
CFG = BacktestConfig(
    position_size=8,
    max_position=8,
    decision_latency_ms=1,
    order_latency_ms=5,
    ack_latency_ms=5,
    flatten_before_end_s=0,
    initial_cash=10_000,
    costs=CostConfig(taker_fee_bps=10),
)


def signals(times_ms: list[float], preds: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"t_ns": [int(t * MS) for t in times_ms], "session": [DAY] * len(preds), "pred": preds}
    )


def go(sig: pl.DataFrame, cfg: BacktestConfig = CFG, book: BookTimeline = BOOK):  # type: ignore[no-untyped-def]
    return run_backtest(book, sig, cfg, threshold_target(0.0, cfg.position_size))


def test_order_walks_the_book_and_pays_fees() -> None:
    # Decide at 3 ms, arrive at 9 ms: matched against message 0 (ask 101x5, 102x5).
    res = go(signals([3, 20], [1.0, 1.0]))  # the last signal flattens the session
    f = res.fills.filter(pl.col("order_id") == 1)
    assert f["price"].to_list() == [101, 102] and f["qty"].to_list() == [5, 3]
    assert res.fills["simulated"].all()
    assert f["fee"].sum() == pytest.approx((5 * 101 + 3 * 102) * 10 / 1e4)
    spread = (f["qty"] * (f["price"] - f["mid_arrival"])).sum()
    assert spread == pytest.approx(5 * 1 + 3 * 2)  # paid vs the arrival mid of 100
    assert (res.fills["side"] * (res.fills["price"] - res.fills["mid_arrival"]) > 0).all()


def test_fill_uses_the_book_at_arrival_not_at_decision() -> None:
    # Decided at 6 ms seeing ask 101; arrives at 12 ms after the ask improved to 100.
    res = go(signals([6, 30], [1.0, 1.0]))
    f = res.fills.row(0, named=True)
    assert f["price"] == 100 and f["mid_decision"] == 100.0 and f["mid_arrival"] == 99.5
    assert res.summary["latency_cost"] == pytest.approx(5 * -0.5 + 3 * -0.5)


def test_visible_depth_caps_the_fill() -> None:
    cfg = CFG.model_copy(update={"position_size": 50, "max_position": 50})
    res = go(signals([3, 40], [1.0, 0.0]), cfg)
    assert res.fills.filter(pl.col("order_id") == 1)["qty"].sum() == 10  # 5 + 5 visible
    assert res.summary["partial"] >= 1


def test_liquidity_taken_is_not_offered_again_within_one_book_state() -> None:
    fast = {"decision_latency_ms": 0, "order_latency_ms": 1, "ack_latency_ms": 0}
    cfg = CFG.model_copy(update={"position_size": 5, "max_position": 10, **fast})
    # Buy 5 (takes 101x5), go flat, buy again while message 0 is still the exchange book.
    res = go(signals([2, 3.5, 5, 9], [1.0, 0.0, 1.0, 0.0]), cfg)
    buys = res.fills.filter(pl.col("side") == 1)
    assert buys["price"].to_list() == [101, 102]


def test_stale_or_unsynced_book_rejects_orders() -> None:
    unsynced = timeline(
        [
            {
                "t_ms": 0,
                "bid_px_0": 99,
                "bid_sz_0": 1,
                "ask_px_0": 101,
                "ask_sz_0": 1,
                "synced": False,
            }
        ]
    )
    res = go(signals([3, 50], [1.0, 1.0]), book=unsynced)
    assert res.fills.is_empty() and res.summary["no_mid"] == 1
    stale = CFG.model_copy(update={"max_book_age_s": 0.003})  # book is 9 ms old on arrival
    res = go(signals([3, 50], [1.0, 1.0]), stale)
    assert res.fills.is_empty() and res.summary["rejected"] == 1


def test_position_and_exposure_limits() -> None:
    def bought(cfg: BacktestConfig) -> float:
        fills = go(signals([3, 30], [1.0, 1.0]), cfg).fills
        return float(fills.filter(pl.col("side") == 1)["qty"].sum())

    assert bought(CFG.model_copy(update={"position_size": 100, "max_position": 4})) == 4
    assert bought(CFG.model_copy(update={"max_notional": 300.0})) == pytest.approx(3.0)


def test_one_working_order_at_a_time() -> None:
    res = go(signals([3, 4, 5], [1.0, -1.0, 1.0]))
    assert res.summary["orders"] == 1 and res.summary["skipped_working"] == 2


def test_null_strategy_has_exactly_zero_pnl() -> None:
    res = run_backtest(BOOK, signals([3, 20], [5.0, -5.0]), CFG, lambda _: 0.0)
    assert res.fills.is_empty() and res.summary["pnl_total"] == 0.0


def test_pnl_identity_and_flat_end() -> None:
    res = go(signals([3, 20, 30, 40], [1.0, -1.0, -1.0, 0.0]))
    s = res.summary
    assert s["open_position_at_end"] == 0
    assert s["pnl_total"] == pytest.approx(
        s["edge_at_decision"] - s["latency_cost"] - s["spread_cost"] - s["fees"]
    )
    f = res.fills
    edge = -(f["side"].cast(pl.Float64) * f["qty"] * f["mid_decision"]).sum()
    assert s["edge_at_decision"] == pytest.approx(edge)


def test_signals_must_be_time_ordered_and_view_cannot_see_the_future() -> None:
    with pytest.raises(ClockViolation):
        go(signals([20, 3], [1.0, 1.0]))
    view = MarketView(BOOK, max_age_ns=10**12, now=5 * MS)
    assert view.mid(5 * MS) == 100.0
    with pytest.raises(ClockViolation):
        view.mid(6 * MS)


def test_ledger_rejects_impossible_fills_and_detects_corruption() -> None:
    ledger = Ledger(initial_cash=1000, max_position=10)
    good = Fill(1, 0, 1, 1, 2.0, 101.0, 101.0, 100.0, 100.0, 0.1)
    ledger.apply(good)
    with pytest.raises(AccountingError, match="better than the touch"):
        ledger.apply(Fill(2, 0, 1, 1, 1.0, 100.5, 101.0, 100.0, 100.0, 0.0))
    with pytest.raises(AccountingError, match="limit"):
        ledger.apply(Fill(3, 0, 1, 1, 9.0, 101.0, 101.0, 100.0, 100.0, 0.0))
    ledger = Ledger(initial_cash=1000, max_position=10)
    ledger.apply(good)
    ledger.cash += 1.0
    with pytest.raises(AccountingError, match="cash"):
        ledger.check()


@pytest.fixture(scope="module")
def synthetic_run() -> tuple[BookTimeline, pl.DataFrame]:
    feed = generate(SynthConfig(n_days=1, session_seconds=1800, seed=21))
    caps = synthetic_capabilities()
    quotes, _ = reconstruct(feed, {"feed": "contiguous"})
    quotes = quotes.with_columns(available_time(caps, TimingConfig()))
    feats = build_features(feed, caps, FeatureConfig(), TimingConfig())
    sig = feats.filter("valid").select(
        "t_ns", "session", pred=pl.col("imbalance_l1").fill_null(0.0) * 1e-4
    )
    return BookTimeline.from_quotes(quotes), sig


def test_backtest_is_deterministic(synthetic_run: tuple[BookTimeline, pl.DataFrame]) -> None:
    book, sig = synthetic_run
    cfg = BacktestConfig(position_size=10, max_position=10, threshold=2e-5)
    a = run_backtest(book, sig, cfg, threshold_target(2e-5, 10))
    b = run_backtest(book, sig, cfg, threshold_target(2e-5, 10))
    assert a.fills.height > 50
    assert a.fills.equals(b.fills) and a.summary == b.summary


def test_decisions_before_t_ignore_data_after_t(
    synthetic_run: tuple[BookTimeline, pl.DataFrame],
) -> None:
    book, sig = synthetic_run
    cfg = BacktestConfig(position_size=10, max_position=10, flatten_before_end_s=0)
    full = run_backtest(book, sig, cfg, threshold_target(2e-5, 10)).fills
    cut = int(sig["t_ns"][sig.height // 2])
    keep = int(np.searchsorted(book.avail_ns, cut, side="right"))
    truncated_book = BookTimeline(
        *(np.asarray(getattr(book, f))[:keep] for f in BookTimeline.__dataclass_fields__)
    )
    part = run_backtest(
        truncated_book, sig.filter(pl.col("t_ns") <= cut), cfg, threshold_target(2e-5, 10)
    ).fills
    settled = full.filter(pl.col("arrival_ns") <= cut - 100 * MS)
    assert settled.height > 10
    assert settled.equals(part.head(settled.height))
