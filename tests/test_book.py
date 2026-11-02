from __future__ import annotations

import math

import numpy as np
import polars as pl
from conftest import events
from hypothesis import given, settings
from hypothesis import strategies as st

from qrp.book import reconstruct, trades_inside_stale_quote
from qrp.schemas import Sequencing
from qrp.synth import SynthConfig, generate

CONTIGUOUS: dict[str, Sequencing] = {"feed": "contiguous"}


def top(quotes: pl.DataFrame, i: int) -> tuple[float, ...]:
    r = quotes.row(i, named=True)
    return (r["bid_px_0"], r["bid_sz_0"], r["ask_px_0"], r["ask_sz_0"])


def test_hand_checked_ladder(clean_events: pl.DataFrame) -> None:
    quotes, stats = reconstruct(clean_events, CONTIGUOUS, levels=2)
    assert quotes["seq"].to_list() == [1, 2, 3, 4]
    assert top(quotes, 0) == (99.99, 5, 100.01, 3)
    assert quotes.row(1, named=True)["bid_px_1"] == 99.98  # second bid level appears
    assert top(quotes, 2) == (99.99, 5, 100.01, 2)  # the trade depleted the ask
    assert quotes.row(3, named=True)["ask_px_1"] == 100.02
    assert quotes["synced"].all() and not quotes["crossed"].any()
    assert stats.gaps == 0 and stats.snapshots == 1


def test_gap_invalidates_until_next_snapshot() -> None:
    df = events(
        [
            (1, 0, "snapshot", 1, 99.0, 1),
            (1, 0, "snapshot", -1, 101.0, 1),
            (2, 1, "book", 1, 99.5, 2),
            (5, 2, "book", 1, 99.6, 2),  # seq 3-4 lost
            (6, 3, "book", -1, 100.5, 2),
            (7, 4, "snapshot", 1, 99.6, 2),
            (7, 4, "snapshot", -1, 100.4, 1),
            (8, 5, "book", -1, 100.3, 1),
        ]
    )
    quotes, stats = reconstruct(df, CONTIGUOUS)
    assert quotes["synced"].to_list() == [True, True, False, False, True, True]
    assert stats.gaps == 1 and stats.unsynced_messages == 2
    # The recovery snapshot replaced the book: the stale 100.5 ask is gone.
    assert quotes.row(4, named=True)["ask_px_1"] != quotes.row(4, named=True)["ask_px_1"]


def test_derived_sequencing_cannot_detect_loss() -> None:
    df = events([(1, 0, "snapshot", 1, 99.0, 1), (9, 1, "book", 1, 99.5, 2)])
    quotes, stats = reconstruct(df, {"feed": "derived"})
    assert quotes["synced"].all() and stats.gaps == 0


def test_trade_only_message_is_not_a_gap() -> None:
    df = events(
        [
            (1, 0, "snapshot", 1, 99.0, 1),
            (1, 0, "snapshot", -1, 101.0, 1),
            (2, 1, "trade", 1, 101.0, 1),  # e.g. a hidden-order execution: no book change
            (3, 2, "book", 1, 99.5, 2),
        ]
    )
    quotes, stats = reconstruct(df, CONTIGUOUS)
    assert stats.gaps == 0 and quotes["seq"].to_list() == [1, 3]


def test_updates_before_first_snapshot_are_unsynced_and_discarded() -> None:
    df = events(
        [
            (1, 0, "book", 1, 98.0, 1),
            (2, 1, "snapshot", 1, 99.0, 1),
            (2, 1, "snapshot", -1, 101.0, 1),
        ]
    )
    quotes, _ = reconstruct(df, CONTIGUOUS)
    assert quotes["synced"].to_list() == [False, True]
    assert math.isnan(quotes.row(1, named=True)["bid_px_1"])  # 98.0 was cleared


def test_crossed_book_and_unknown_delete_are_reported() -> None:
    df = events(
        [
            (1, 0, "snapshot", 1, 99.0, 1),
            (1, 0, "snapshot", -1, 101.0, 1),
            (2, 1, "book", 1, 101.5, 1),
            (3, 2, "book", -1, 105.0, 0),
        ]
    )
    quotes, stats = reconstruct(df, CONTIGUOUS)
    assert quotes["crossed"].to_list() == [False, True, True]
    assert stats.crossed_messages == 2 and stats.deletes_of_unknown_level == 1


def test_trades_inside_stale_quote() -> None:
    quotes = pl.DataFrame(
        {"available_time_ns": [0, 10], "bid_px_0": [99.0, 99.0], "ask_px_0": [101.0, 100.0]}
    ).with_columns(synced=pl.lit(True), crossed=pl.lit(False))
    trades = pl.DataFrame(
        {
            "available_time_ns": [5, 10, 11, 12],
            "side": [1, 1, 1, -1],
            "price": [100.5, 100.0, 100.0, 99.5],
        }
    )
    # t=5 buy at 100.5 < ask 101 (stale); t=10 sees the t=0 quote only (strictly before);
    # t=11 buy at the ask is fine; t=12 sell at 99.5 > bid 99 is stale.
    counts = trades_inside_stale_quote(quotes, trades)
    assert counts == {"trades_checked": 4, "trades_inside_stale_quote": 3}


def test_synthetic_feed_gaps_are_detected_and_recovered() -> None:
    feed = generate(SynthConfig(n_days=2, session_seconds=900, seed=11))
    quotes, stats = reconstruct(feed, CONTIGUOUS)
    assert stats.gaps == 2 and stats.snapshots == 4  # 2 session starts + 2 recoveries
    # Applying updates to a book that lost messages produces crossed quotes; a synced book never
    # does. This is why unsynced rows are excluded downstream rather than "repaired".
    assert stats.crossed_messages > 0
    assert quotes.filter(pl.col("synced") & pl.col("crossed")).is_empty()
    assert 0 < stats.unsynced_messages < 60  # only between loss and the recovery snapshot
    assert quotes["synced"][-1]


def _reference_top(rows: list[tuple[int, float, float]], levels: int) -> list[float]:
    book: dict[int, dict[float, float]] = {1: {}, -1: {}}
    for s, p, q in rows:
        if q > 0:
            book[s][p] = q
        else:
            book[s].pop(p, None)
    bids = sorted(book[1].items(), reverse=True)[:levels]
    asks = sorted(book[-1].items())[:levels]
    out = [math.nan] * (4 * levels)
    for i, (p, q) in enumerate(bids):
        out[4 * i : 4 * i + 2] = [p, q]
    for i, (p, q) in enumerate(asks):
        out[4 * i + 2 : 4 * i + 4] = [p, q]
    return out


update = st.tuples(
    st.sampled_from([1, -1]),
    st.sampled_from([99.0, 99.5, 100.0, 100.5, 101.0, 101.5]),
    st.sampled_from([0.0, 1.0, 2.0, 7.5]),
)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.lists(update, min_size=1, max_size=4, unique_by=lambda u: (u[0], u[1])),
        min_size=1,
        max_size=30,
    )
)
def test_matches_naive_reference(messages: list[list[tuple[int, float, float]]]) -> None:
    rows = [(i + 2, i, "book", s, p, q) for i, msg in enumerate(messages) for s, p, q in msg]
    df = events([(1, 0, "snapshot", 1, 1.0, 1.0), *rows])  # a far-away bid keeps the book synced
    quotes, _ = reconstruct(df, CONTIGUOUS, levels=3)
    applied: list[tuple[int, float, float]] = [(1, 1.0, 1.0)]
    for i, msg in enumerate(messages):
        applied.extend(msg)
        got = np.array(quotes.row(i + 1)[5 : 5 + 12], dtype=float)
        np.testing.assert_array_equal(got, _reference_top(applied, 3))
