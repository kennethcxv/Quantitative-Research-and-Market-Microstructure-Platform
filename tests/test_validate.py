from __future__ import annotations

import polars as pl
import pytest
from conftest import MS, events
from hypothesis import given
from hypothesis import strategies as st

from qrp.schemas import SchemaError, Sequencing, conform
from qrp.validate import deduplicate, validate_events

CONTIGUOUS: dict[str, Sequencing] = {"feed": "contiguous"}


def violations(df: pl.DataFrame, seq: dict[str, Sequencing] | None = None) -> dict[str, int]:
    report = validate_events(df, seq or CONTIGUOUS)
    return {k: v for k, v in report.summary()["violations"].items() if v}


def test_clean_data_passes(clean_events: pl.DataFrame) -> None:
    assert validate_events(clean_events, CONTIGUOUS).ok
    assert violations(clean_events) == {}


def test_conform_rejects_missing_and_lossy_columns(clean_events: pl.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing"):
        conform(clean_events.drop("stream"))
    with pytest.raises(SchemaError):
        conform(clean_events.with_columns(pl.lit("x").alias("price")))


def test_redelivered_message_is_removed_by_deduplicate(clean_events: pl.DataFrame) -> None:
    redelivered = pl.concat([clean_events.head(5), clean_events.slice(3, 2), clean_events.tail(1)])
    assert violations(redelivered)["duplicate_rows"] == 4  # both copies of two rows
    deduped, removed = deduplicate(redelivered)
    assert removed == 2
    assert deduped.equals(clean_events)


def test_conflicting_update_differs_from_duplicate() -> None:
    df = events([(1, 0, "book", 1, 99.0, 5), (1, 0, "book", 1, 99.0, 6)])
    assert violations(df) == {"conflicting_update": 2}


def test_snapshot_and_incremental_rows_cannot_share_a_message() -> None:
    df = events([(1, 0, "snapshot", 1, 99.0, 5), (1, 0, "book", -1, 101.0, 6)])
    assert violations(df) == {"mixed_snapshot_message": 2}


@pytest.mark.parametrize(
    ("mode", "expect_gap"), [("contiguous", 1), ("monotonic", 0), ("derived", 0)]
)
def test_seq_gap_meaning_depends_on_stream_sequencing(mode: Sequencing, expect_gap: int) -> None:
    df = events([(1, 0, "book", 1, 99.0, 5), (5, 10, "book", 1, 99.0, 6)])
    assert violations(df, {"feed": mode}).get("seq_gap", 0) == expect_gap


def test_streams_have_independent_sequences() -> None:
    book = events([(1, 0, "book", 1, 99.0, 5), (2, 20, "book", 1, 99.0, 6)], stream="depth")
    trades = events([(70001, 10, "trade", 1, 99.0, 1), (70002, 30, "trade", -1, 99.0, 1)], "trades")
    interleaved = pl.concat([book.head(1), trades.head(1), book.tail(1), trades.tail(1)])
    seq: dict[str, Sequencing] = {"depth": "derived", "trades": "contiguous"}
    assert violations(interleaved, seq) == {}


def test_undeclared_stream_is_an_error(clean_events: pl.DataFrame) -> None:
    assert violations(clean_events, {"other": "derived"}) == {"undeclared_stream": 6}


def test_out_of_order_sequence_and_receive_time(clean_events: pl.DataFrame) -> None:
    swapped = pl.concat([clean_events.slice(2, 1), clean_events.head(2), clean_events.tail(3)])
    v = violations(swapped)
    assert v["seq_out_of_order"] == 1
    assert v["recv_time_out_of_order"] == 1


def test_missing_receive_time_is_allowed_but_partial_is_flagged(clean_events: pl.DataFrame) -> None:
    no_recv = clean_events.with_columns(pl.lit(None, pl.Int64).alias("recv_time_ns"))
    assert violations(no_recv) == {}
    partial = clean_events.with_columns(
        pl.when(pl.col("seq") == 2)
        .then(None)
        .otherwise(pl.col("recv_time_ns"))
        .alias("recv_time_ns")
    )
    assert violations(partial) == {"partial_recv_time": 1}
    neither = no_recv.with_columns(pl.lit(None, pl.Int64).alias("event_time_ns"))
    assert violations(neither)["no_timestamp"] == 6


def test_timestamp_anomalies(clean_events: pl.DataFrame) -> None:
    early = clean_events.with_columns(
        pl.when(pl.col("seq") == 2)
        .then(pl.col("event_time_ns") + 5 * MS)
        .otherwise(pl.col("event_time_ns"))
        .alias("event_time_ns")
    )
    assert violations(early)["recv_before_event"] == 1
    slow = clean_events.with_columns(pl.col("recv_time_ns") + 5_000 * MS)
    assert violations(slow) == {"latency_outlier": 6}
    future = validate_events(clean_events, CONTIGUOUS, now_ns=0)
    assert [c.name for c in future.failed("error")] == ["timestamp_out_of_range"]


def test_bad_values_and_domains() -> None:
    df = events(
        [(1, 0, "book", 2, 99.0, 1), (2, 1, "trade", 1, 0.0, 1), (3, 2, "quote", 1, 1.0, 1)]
    )
    v = violations(df)
    assert v["domain"] == 2
    assert v["nonpositive_price_or_size"] == 1


def test_empty_snapshot_level_is_a_warning() -> None:
    df = events([(1, 0, "snapshot", 1, 99.0, 0), (1, 0, "snapshot", -1, 101.0, 2)])
    assert violations(df) == {"empty_snapshot_level": 1}
    assert validate_events(df, CONTIGUOUS).ok


def test_stale_feed_gap_within_session() -> None:
    df = events([(1, 0, "book", 1, 99.0, 1), (2, 61_000, "book", 1, 99.0, 2)])
    assert violations(df) == {"stale_feed_gap": 1}


@given(st.sets(st.integers(min_value=2, max_value=40)))
def test_gap_count_equals_number_of_missing_runs(removed: set[int]) -> None:
    kept = [s for s in range(1, 42) if s not in removed]  # seq 1 and 41 always survive
    df = events([(s, s, "book", 1, 99.0, s) for s in kept])
    runs = sum(1 for s in removed if s - 1 not in removed)
    assert violations(df).get("seq_gap", 0) == runs
