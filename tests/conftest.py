from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from qrp.schemas import conform

T0 = 1_704_205_800 * 10**9  # 2024-01-02 14:30:00 UTC
MS = 1_000_000


def events(rows: list[tuple[Any, ...]], stream: str = "feed", recv: bool = True) -> pl.DataFrame:
    """rows: (seq, t_ms, kind, side, price, size). recv = event + 1 ms unless recv=False."""
    event = [T0 + r[1] * MS for r in rows]
    return conform(
        pl.DataFrame(
            {
                "symbol": ["TST"] * len(rows),
                "stream": [stream] * len(rows),
                "seq": [r[0] for r in rows],
                "event_time_ns": event,
                "recv_time_ns": [t + MS for t in event] if recv else [None] * len(rows),
                "kind": [r[2] for r in rows],
                "side": [r[3] for r in rows],
                "price": [float(r[4]) for r in rows],
                "size": [float(r[5]) for r in rows],
            },
            schema_overrides={"recv_time_ns": pl.Int64},
        )
    )


@pytest.fixture
def clean_events() -> pl.DataFrame:
    return events(
        [
            (1, 0, "snapshot", 1, 99.99, 5),
            (1, 0, "snapshot", -1, 100.01, 3),
            (2, 10, "book", 1, 99.98, 7),
            (3, 20, "trade", 1, 100.01, 1),
            (3, 20, "book", -1, 100.01, 2),
            (4, 30, "book", -1, 100.02, 4),
        ]
    )
