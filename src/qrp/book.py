"""L2 book reconstruction: canonical level updates -> one top-of-book ladder row per message.

Validity is tracked causally, i.e. using only what the consumer knew at that message:

* ``synced`` becomes true when a snapshot message is applied and false when a ``contiguous``
  stream skips a sequence number (we know updates were lost the moment the next message
  arrives). Incremental updates keep being applied while unsynced, but rows are marked so that
  features and the backtester ignore them until the next snapshot.
* ``crossed`` marks best bid >= best ask, which a consistent book never shows.

Prices are used as dictionary keys. That is exact because every price for a level comes from
the same source representation; it would not be safe for prices computed by arithmetic.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping

import numpy as np
import polars as pl
from pydantic import BaseModel

from qrp.schemas import SNAPSHOT, TRADE, Sequencing


class BookStats(BaseModel):
    messages: int
    snapshots: int
    gaps: int  # contiguous-stream sequence jumps (each invalidates the book)
    unsynced_messages: int
    crossed_messages: int
    deletes_of_unknown_level: int  # benign on some venues, a sign of loss on others


def ladder_columns(levels: int) -> list[str]:
    return [
        f"{side}_{what}_{i}"
        for i in range(levels)
        for side in ("bid", "ask")
        for what in ("px", "sz")
    ]


def reconstruct(
    events: pl.DataFrame, sequencing: Mapping[str, Sequencing], levels: int = 5
) -> tuple[pl.DataFrame, BookStats]:
    """Rebuild the book for one symbol from rows in stored (stream, seq) order.

    Returns one row per book message with the message's timestamps, ``levels`` price/size pairs
    per side (NaN where the side is shallower), ``synced`` and ``crossed``.
    """
    book_streams = events.filter(pl.col("kind") != TRADE)["stream"].unique()
    if events["symbol"].n_unique() != 1 or len(book_streams) != 1:
        raise ValueError("reconstruct() takes one symbol with exactly one book stream")
    stream = book_streams[0]
    # Trade rows on the book stream stay in: they occupy sequence numbers, so dropping them
    # would turn every trade-only message into a false gap.
    rows = events.filter(pl.col("stream") == stream)
    check_gaps = sequencing[stream] == "contiguous"

    seq_np = rows["seq"].to_numpy()
    kind = rows["kind"].to_numpy()
    # Message boundaries: first row of every run of equal seq.
    starts_np = np.flatnonzero(np.r_[True, seq_np[1:] != seq_np[:-1]])
    # Per-message flags computed once, vectorised: calling numpy on millions of tiny slices
    # inside the loop cost ~2.5 s per 1.3 M messages in profiling.
    all_trade = np.logical_and.reduceat(kind == TRADE, starts_np).tolist()
    has_snapshot = np.logical_or.reduceat(kind == SNAPSHOT, starts_np).tolist()
    # The loop reads plain Python lists: indexing numpy arrays element by element allocates a
    # numpy scalar per access and was the other half of the cost.
    starts = starts_np.tolist()
    ends = [*starts[1:], len(seq_np)]
    seq = seq_np.tolist()
    is_trade = (kind == TRADE).tolist()
    side = rows["side"].to_list()
    price = rows["price"].to_list()
    size = rows["size"].to_list()

    size_at: dict[int, dict[float, float]] = {1: {}, -1: {}}
    prices: dict[int, list[float]] = {1: [], -1: []}  # ascending
    nan = float("nan")
    out: list[list[float]] = []
    synced_out: list[bool] = []
    crossed_out: list[bool] = []
    emitted: list[int] = []
    synced = False
    stats = dict.fromkeys(BookStats.model_fields, 0)
    prev_seq: int | None = None

    for m in range(len(starts)):
        lo, hi = starts[m], ends[m]
        if check_gaps and prev_seq is not None and seq[lo] != prev_seq + 1:
            synced = False
            stats["gaps"] += 1
        prev_seq = seq[lo]
        if all_trade[m]:
            continue
        snapshot = has_snapshot[m]
        if snapshot:
            for s in (1, -1):
                size_at[s].clear()
                prices[s].clear()
            synced = True
            stats["snapshots"] += 1

        for r in range(lo, hi):
            if is_trade[r]:
                continue
            s, p, q = side[r], price[r], size[r]
            levels_s, prices_s = size_at[s], prices[s]
            if q > 0:
                if p not in levels_s:
                    bisect.insort(prices_s, p)
                levels_s[p] = q
            elif p in levels_s:
                del levels_s[p]
                del prices_s[bisect.bisect_left(prices_s, p)]
            elif not snapshot:
                stats["deletes_of_unknown_level"] += 1

        bids, asks = prices[1], prices[-1]
        bid_sizes, ask_sizes = size_at[1], size_at[-1]
        row = [nan] * (4 * levels)
        for i in range(min(levels, len(bids))):
            p = bids[-1 - i]
            row[4 * i] = p
            row[4 * i + 1] = bid_sizes[p]
        for i in range(min(levels, len(asks))):
            p = asks[i]
            row[4 * i + 2] = p
            row[4 * i + 3] = ask_sizes[p]
        out.append(row)
        synced_out.append(synced)
        crossed_out.append(bool(bids and asks and bids[-1] >= asks[0]))
        emitted.append(lo)

    stats["messages"] = len(emitted)
    stats["unsynced_messages"] = len(synced_out) - sum(synced_out)
    stats["crossed_messages"] = sum(crossed_out)

    # A message is complete only when its last row has arrived, so it carries the latest
    # timestamps of its rows (a snapshot can span several arrivals).
    message = np.cumsum(np.r_[True, seq_np[1:] != seq_np[:-1]])
    heads = (
        rows.with_columns(_message=pl.Series(message))
        .with_columns(pl.col("event_time_ns", "recv_time_ns").max().over("_message"))[emitted]
        .select("symbol", "stream", "seq", "event_time_ns", "recv_time_ns")
    )
    ladder = pl.DataFrame(
        np.array(out, dtype=np.float64).reshape(len(out), 4 * levels),
        schema=ladder_columns(levels),
        orient="row",
    )
    quotes = pl.concat([heads, ladder], how="horizontal").with_columns(
        synced=pl.Series(synced_out), crossed=pl.Series(crossed_out)
    )
    return quotes, BookStats(**stats)


def trades_inside_stale_quote(
    quotes: pl.DataFrame,
    trades: pl.DataFrame,
    time_col: str = "available_time_ns",
    tol: float = 0.0,
) -> dict[str, int]:
    """Count prints that contradict the quote we held just before them.

    A buy that prints below our best ask (or a sell above our best bid) means liquidity existed
    that our book did not show: the quote was stale, e.g. because the depth channel is batched
    while trades are real-time. Both frames must carry ``time_col`` and be sorted by it.
    """
    joined = trades.join_asof(
        quotes.filter(pl.col("synced") & ~pl.col("crossed")).select(
            time_col, "bid_px_0", "ask_px_0"
        ),
        on=time_col,
        strategy="backward",
        allow_exact_matches=False,
    ).filter(pl.col("ask_px_0").is_not_null())
    stale = ((pl.col("side") == 1) & (pl.col("price") < pl.col("ask_px_0") - tol)) | (
        (pl.col("side") == -1) & (pl.col("price") > pl.col("bid_px_0") + tol)
    )
    return {
        "trades_checked": joined.height,
        "trades_inside_stale_quote": joined.filter(stale).height,
    }
