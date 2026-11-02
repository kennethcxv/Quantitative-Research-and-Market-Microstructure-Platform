"""Integrity checks for canonical event data.

Each check counts violations and keeps a few example rows. ``error`` checks block ingestion;
``warning`` checks are stored in the manifest and surfaced in reports.

Sequence semantics are per ``(symbol, stream)`` and depend on the stream's declared
``Sequencing``: only a ``contiguous`` stream can prove that data is missing.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

from qrp.schemas import BOOK, KINDS, NULLABLE, SNAPSHOT, TRADE, Sequencing

Severity = Literal["error", "warning"]
_NS_2000 = 946_684_800 * 10**9
_STREAM = ["symbol", "stream"]


class ValidationConfig(BaseModel):
    max_latency_ms: float = 1_000.0  # recv - event above this is suspicious
    max_feed_gap_s: float = 60.0  # silence on a stream within a UTC day above this is stale
    n_examples: int = 3


class CheckResult(BaseModel):
    name: str
    severity: Severity
    description: str
    violations: int
    examples: list[dict[str, Any]] = []


class ValidationReport(BaseModel):
    n_rows: int
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(c.violations == 0 for c in self.checks if c.severity == "error")

    def failed(self, severity: Severity | None = None) -> list[CheckResult]:
        return [c for c in self.checks if c.violations and severity in (None, c.severity)]

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "violations": {c.name: c.violations for c in self.checks},
        }


def deduplicate(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop rows repeated verbatim (re-delivered messages), keeping the first copy."""
    deduped = df.unique(maintain_order=True)
    return deduped, df.height - deduped.height


def validate_events(
    df: pl.DataFrame,
    sequencing: Mapping[str, Sequencing],
    cfg: ValidationConfig | None = None,
    now_ns: int | None = None,
) -> ValidationReport:
    """Run all checks on a conformed frame that is still in feed (capture) order."""
    cfg = cfg or ValidationConfig()
    now_ns = now_ns if now_ns is not None else time.time_ns()
    contiguous = [s for s, mode in sequencing.items() if mode == "contiguous"]

    rows = df.with_row_index("_row").with_columns(
        _t=pl.coalesce("recv_time_ns", "event_time_ns"),
    )
    rows = rows.with_columns(_date=pl.from_epoch("_t", time_unit="ns").dt.date())
    # Message-level view: first row of each (symbol, stream, seq), in feed order.
    msgs = rows.unique([*_STREAM, "seq"], keep="first", maintain_order=True)

    def prev(column: str) -> pl.Expr:
        return pl.col(column).shift(1).over(_STREAM)

    def check(
        frame: pl.DataFrame, mask: pl.Expr, name: str, sev: Severity, desc: str
    ) -> CheckResult:
        bad = frame.filter(mask)
        examples = bad.head(cfg.n_examples).drop("_row", "_t", "_date", strict=False).to_dicts()
        return CheckResult(
            name=name, severity=sev, description=desc, violations=bad.height, examples=examples
        )

    level_key = pl.struct(*_STREAM, "seq", "kind", "side", "price")
    all_columns = pl.struct(*df.columns)
    book_kinds_in_msg = (
        pl.col("kind").filter(pl.col("kind") != TRADE).n_unique().over(*_STREAM, "seq")
    )
    non_null = [c for c in df.columns if c not in NULLABLE]

    checks = [
        check(
            rows,
            pl.any_horizontal(pl.col(c).is_null() for c in non_null)
            | ~pl.col("price").is_finite()
            | ~pl.col("size").is_finite(),
            "null_or_nonfinite",
            "error",
            "required value is null or non-finite",
        ),
        check(
            rows,
            pl.col("_t").is_null(),
            "no_timestamp",
            "error",
            "row has neither an exchange nor a receive timestamp",
        ),
        check(
            rows,
            ~pl.col("kind").is_in(KINDS)
            | (pl.col("kind").is_in([BOOK, SNAPSHOT]) & ~pl.col("side").is_in([1, -1]))
            | ((pl.col("kind") == TRADE) & ~pl.col("side").is_in([1, -1, 0])),
            "domain",
            "error",
            "unknown kind or side code",
        ),
        check(
            rows,
            ~pl.col("symbol").str.contains(r"^[A-Za-z0-9._-]+$"),
            "unsafe_symbol",
            "error",
            "symbol must match [A-Za-z0-9._-]+ (it becomes a partition directory name)",
        ),
        check(
            rows,
            ~pl.col("stream").is_in(list(sequencing)),
            "undeclared_stream",
            "error",
            "stream has no declared sequencing semantics",
        ),
        check(
            rows,
            (pl.col("price") <= 0)
            | (pl.col("size") < 0)
            | ((pl.col("kind") == TRADE) & (pl.col("size") <= 0)),
            "nonpositive_price_or_size",
            "error",
            "price <= 0, negative size, or an empty trade",
        ),
        check(
            rows,
            (pl.col("kind") == SNAPSHOT) & (pl.col("size") == 0),
            "empty_snapshot_level",
            "warning",
            "snapshot lists a level with zero size (treated as absent)",
        ),
        check(
            rows,
            all_columns.is_duplicated(),
            "duplicate_rows",
            "error",
            "row repeated verbatim (run deduplicate() first)",
        ),
        check(
            rows,
            level_key.is_duplicated() & ~all_columns.is_duplicated(),
            "conflicting_update",
            "error",
            "two different rows for one (kind, side, price) in a message; prints at the same "
            "price in one message must be aggregated by the producer",
        ),
        check(
            rows,
            book_kinds_in_msg > 1,
            "mixed_snapshot_message",
            "error",
            "message mixes snapshot and incremental rows",
        ),
        check(
            rows,
            pl.col("seq") < prev("seq"),
            "seq_out_of_order",
            "error",
            "seq decreases within a stream in feed order",
        ),
        check(
            msgs,
            pl.col("stream").is_in(contiguous) & ((pl.col("seq") - prev("seq")) > 1),
            "seq_gap",
            "warning",
            "contiguous stream skipped sequence numbers: messages were lost",
        ),
        check(
            rows,
            pl.col("recv_time_ns") < prev("recv_time_ns"),
            "recv_time_out_of_order",
            "error",
            "receive time decreases within a stream",
        ),
        check(
            msgs,
            pl.col("event_time_ns") < prev("event_time_ns"),
            "event_time_out_of_order",
            "warning",
            "exchange time decreases across messages of one stream",
        ),
        check(
            rows,
            pl.col("recv_time_ns") < pl.col("event_time_ns"),
            "recv_before_event",
            "error",
            "received before it happened: clock error",
        ),
        check(
            rows,
            (pl.col("recv_time_ns") - pl.col("event_time_ns")) > cfg.max_latency_ms * 1e6,
            "latency_outlier",
            "warning",
            f"recv - event > {cfg.max_latency_ms} ms",
        ),
        check(
            rows,
            (pl.col("event_time_ns") < _NS_2000)
            | (pl.col("recv_time_ns") < _NS_2000)
            | (pl.col("_t") > now_ns),
            "timestamp_out_of_range",
            "error",
            "before 2000-01-01 or in the future",
        ),
        check(
            rows,
            pl.col("recv_time_ns").is_null()
            & pl.col("recv_time_ns").is_not_null().any().over(_STREAM),
            "partial_recv_time",
            "warning",
            "stream has receive times on some rows only",
        ),
        check(
            msgs,
            (pl.col("_t") - pl.col("_t").shift(1).over(*_STREAM, "_date"))
            > cfg.max_feed_gap_s * 1e9,
            "stale_feed_gap",
            "warning",
            f"no messages on a stream for > {cfg.max_feed_gap_s}s within a UTC day",
        ),
    ]
    return ValidationReport(n_rows=df.height, checks=checks)
