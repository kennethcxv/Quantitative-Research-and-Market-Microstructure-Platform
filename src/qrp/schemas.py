"""Canonical event schema (``qrp.events.v1``) and dataset metadata.

docs/event_interface.md is the authoritative description of every column.
"""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

EVENTS_SCHEMA_VERSION = "qrp.events.v1"

EVENT_SCHEMA: dict[str, type[pl.DataType]] = {
    "symbol": pl.String,
    "stream": pl.String,
    "seq": pl.Int64,
    "event_time_ns": pl.Int64,  # nullable: exchange did not provide a timestamp
    "recv_time_ns": pl.Int64,  # nullable: nobody captured an arrival time
    "kind": pl.String,
    "side": pl.Int8,
    "price": pl.Float64,
    "size": pl.Float64,
}
NULLABLE = frozenset({"event_time_ns", "recv_time_ns"})

BOOK = "book"  # incremental: new total size at one price level (0 removes the level)
SNAPSHOT = "snapshot"  # all rows of a snapshot message together are the complete book
TRADE = "trade"
KINDS = (BOOK, SNAPSHOT, TRADE)

# How a stream's seq numbers behave, which decides what a jump in seq means.
#   contiguous: source ids increase by exactly 1 per message; a jump is lost data.
#   monotonic:  source ids increase but may skip (ids shared with other channels); a jump is normal.
#   derived:    the adapter numbered messages in file order; loss is undetectable from seq.
Sequencing = Literal["contiguous", "monotonic", "derived"]


class SchemaError(ValueError):
    """Input cannot be represented in the canonical schema."""


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Select and strictly cast the canonical columns, in canonical order.

    Extra columns are dropped. A missing column or a lossy cast raises; nothing is filled in.
    """
    missing = [c for c in EVENT_SCHEMA if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns for {EVENTS_SCHEMA_VERSION}: {missing}")
    try:
        return df.select(pl.col(c).cast(t, strict=True) for c, t in EVENT_SCHEMA.items())
    except pl.exceptions.InvalidOperationError as exc:
        raise SchemaError(f"cannot cast to {EVENTS_SCHEMA_VERSION}: {exc}") from exc


class SourceInfo(BaseModel):
    name: str
    license: str
    synthetic: bool
    url: str | None = None
    notes: str = ""


class StreamInfo(BaseModel):
    sequencing: Sequencing
    description: str = ""


class Capabilities(BaseModel):
    """What the feed actually exposes. Features and the backtester check these."""

    book_depth: int | None  # price levels per side in the feed; None = full depth
    has_trades: bool
    recv_time: Literal["measured", "absent"]
    event_time: Literal["exchange", "partial"]  # partial: some rows lack an exchange timestamp
    streams: dict[str, StreamInfo]


class FileEntry(BaseModel):
    path: str
    sha256: str
    rows: int


class DatasetManifest(BaseModel):
    name: str
    version: str
    schema_version: str = EVENTS_SCHEMA_VERSION
    created_at_ns: int  # knowledge time: when this version entered the catalog
    parent_version: str | None = None
    revision_note: str | None = None
    source: SourceInfo
    capabilities: Capabilities
    adapter: str
    adapter_params: dict[str, Any]
    input_sha256: list[str]
    n_rows: int
    duplicates_removed: int
    symbols: list[str]
    time_min_ns: int
    time_max_ns: int
    files: list[FileEntry]
    validation: dict[str, Any]
