"""Source adapters: raw files -> canonical ``qrp.events.v1`` frame plus metadata.

Every adapter returns an ``Imported`` bundle; the catalog does validation and storage. Adapters
translate representation only. They never fill a missing value with an assumed one.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from qrp.catalog import sha256_file
from qrp.schemas import (
    BOOK,
    EVENTS_SCHEMA_VERSION,
    SNAPSHOT,
    TRADE,
    Capabilities,
    SchemaError,
    SourceInfo,
    StreamInfo,
    conform,
)


@dataclass(frozen=True)
class Imported:
    events: pl.DataFrame
    source: SourceInfo
    capabilities: Capabilities
    adapter: str
    params: dict[str, Any]
    input_sha256: list[str]


# --- qrp.events.v1 files (Project 1 / Project 2 exports, synthetic fixtures) -----------------


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".qrp.json")


def read_qrp_v1(path: Path) -> Imported:
    """Read a conforming export: a Parquet/CSV/NDJSON file and its ``.qrp.json`` sidecar."""
    meta_path = sidecar_path(path)
    if not meta_path.exists():
        raise SchemaError(f"missing sidecar {meta_path.name} (see docs/event_interface.md)")
    meta = json.loads(meta_path.read_text())
    version = meta.get("schema_version", "")
    if version != EVENTS_SCHEMA_VERSION and not version.startswith(EVENTS_SCHEMA_VERSION + "."):
        raise SchemaError(f"unsupported schema_version {version!r}")
    try:
        if path.suffix == ".parquet":
            raw = pl.read_parquet(path)
        elif path.suffix == ".csv":
            raw = pl.read_csv(path)
        elif path.suffix == ".ndjson":
            raw = pl.read_ndjson(path)
        else:
            raise SchemaError(f"unsupported file type {path.suffix!r}")
    except (pl.exceptions.ComputeError, OSError) as exc:
        raise SchemaError(f"cannot parse {path.name}: {exc}") from exc
    return Imported(
        events=conform(raw),
        source=SourceInfo.model_validate(meta["source"]),
        capabilities=Capabilities.model_validate(meta["capabilities"]),
        adapter="qrp-v1",
        params={"file": path.name, "schema_version": version},
        input_sha256=[sha256_file(path), sha256_file(meta_path)],
    )


def write_qrp_v1(events: pl.DataFrame, path: Path, source: SourceInfo, caps: Capabilities) -> None:
    """Write a conforming export (used for fixtures and by tests of the import path)."""
    events = conform(events)
    if path.suffix == ".ndjson":
        events.write_ndjson(path)
    elif path.suffix == ".csv":
        events.write_csv(path)
    else:
        events.write_parquet(path)
    sidecar = {
        "schema_version": EVENTS_SCHEMA_VERSION,
        "source": source.model_dump(),
        "capabilities": caps.model_dump(),
    }
    sidecar_path(path).write_text(json.dumps(sidecar, indent=2) + "\n")


# --- Tardis.dev CSV (incremental_book_L2 + trades) -------------------------------------------

TARDIS_URL = "https://datasets.tardis.dev/v1/{exchange}/{kind}/{y}/{m}/{d}/{symbol}.csv.gz"
TARDIS_LICENSE = (
    "Tardis.dev Licence Agreement. Research/personal use. Raw or reformatted data may not be "
    "redistributed (clause 9.2); derived statistics may appear in research reports (9.3); "
    "models trained on the data may not be published (9.5). Not included in this repository."
)
# SHA-256 of the files used for the documented results, so a re-download can be verified.
TARDIS_REFERENCE_SHA256 = {
    "binance_incremental_book_L2_2024-03-01_LINKUSDT.csv.gz": (
        "ae286d01e7c163bcae11b17060fd27eb8a53a7f5b42ab385f1541558a14af6da"
    ),
    "binance_trades_2024-03-01_LINKUSDT.csv.gz": (
        "5992534932562e87fadc41ecaa01b030755fbbf2a809b3357efa625befc99b32"
    ),
    "binance_incremental_book_L2_2024-04-01_LINKUSDT.csv.gz": (
        "3a398f647536511ec79a9f9884b47ac6d73c4b9cc17c1d7e50bd6db383f9d92c"
    ),
    "binance_trades_2024-04-01_LINKUSDT.csv.gz": (
        "5a56eccb4faa57a0c44c77a34064e97d1b599e1e5e91a080ccfe0524d3de967a"
    ),
}


def tardis_filename(exchange: str, kind: str, date: str, symbol: str) -> str:
    return f"{exchange}_{kind}_{date}_{symbol}.csv.gz"


def fetch_tardis(exchange: str, symbol: str, date: str, out_dir: Path) -> list[Path]:
    """Download the free first-of-month sample (no API key) for one symbol-day."""
    y, m, d = date.split("-")
    if d != "01":
        raise ValueError("Tardis.dev only serves the first day of each month without an API key")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ("incremental_book_L2", "trades"):
        path = out_dir / tardis_filename(exchange, kind, date, symbol)
        if not path.exists():
            url = TARDIS_URL.format(exchange=exchange, kind=kind, y=y, m=m, d=d, symbol=symbol)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
            tmp.rename(path)
        expected = TARDIS_REFERENCE_SHA256.get(path.name)
        if expected is not None and sha256_file(path) != expected:
            raise RuntimeError(f"{path.name} differs from the reference download")
        paths.append(path)
    return paths


def _tardis_times() -> list[pl.Expr]:
    # Tardis copies local_timestamp into timestamp when the venue sends none (e.g. REST
    # snapshots). Equality to the microsecond is that fallback, so it becomes null.
    return [
        pl.when(pl.col("timestamp") != pl.col("local_timestamp"))
        .then(pl.col("timestamp") * 1000)
        .alias("event_time_ns"),
        (pl.col("local_timestamp") * 1000).alias("recv_time_ns"),
    ]


def read_tardis(
    book_paths: Sequence[Path], trade_paths: Sequence[Path], exchange: str, dates: Sequence[str]
) -> Imported:
    """One or more symbol-days, in date order. Each day's file starts with its own snapshot."""
    overrides = {"price": pl.Float64, "amount": pl.Float64}
    book = pl.concat([pl.read_csv(p, schema_overrides=overrides) for p in book_paths])
    trades = pl.concat([pl.read_csv(p, schema_overrides=overrides) for p in trade_paths])
    symbols = set(book["symbol"].unique()) | set(trades["symbol"].unique())
    if len(symbols) != 1:
        raise SchemaError(f"expected one symbol, found {sorted(symbols)}")

    # A message is a run of rows with one arrival timestamp; a whole snapshot run (which can
    # span several arrivals) is one message, because it must replace the book atomically.
    snap = pl.col("is_snapshot")
    new_message = (
        (pl.col("local_timestamp") != pl.col("local_timestamp").shift(1)) & ~(snap & snap.shift(1))
    ) | (snap != snap.shift(1))
    book_events = book.with_columns(
        *_tardis_times(),
        stream=pl.lit("depth"),
        seq=new_message.fill_null(True).cum_sum().cast(pl.Int64),
        kind=pl.when(snap).then(pl.lit(SNAPSHOT)).otherwise(pl.lit(BOOK)),
        side=pl.col("side").replace_strict({"bid": 1, "ask": -1}, return_dtype=pl.Int8),
        size=pl.col("amount"),
    )
    trade_events = trades.with_columns(
        *_tardis_times(),
        stream=pl.lit("trades"),
        seq=pl.col("id").cast(pl.Int64),
        kind=pl.lit(TRADE),
        side=pl.col("side").replace_strict(
            {"buy": 1, "sell": -1, "unknown": 0}, return_dtype=pl.Int8
        ),
        size=pl.col("amount"),
    )
    events = pl.concat([conform(book_events), conform(trade_events)])
    return Imported(
        events=events,
        source=SourceInfo(
            name=f"Tardis.dev samples: {exchange} {symbols.pop()} {', '.join(dates)}",
            license=TARDIS_LICENSE,
            synthetic=False,
            url="https://docs.tardis.dev/downloadable-csv-files",
            notes=(
                "recv_time = Tardis collector arrival time (their network position, not ours). "
                "Depth messages are numbered by the adapter: the CSV omits venue update ids, so "
                "loss is only visible through snapshot resets. The venue's REST snapshot caps "
                "initial depth (~1000 levels/side); levels far from the touch may be incomplete."
            ),
        ),
        capabilities=Capabilities(
            book_depth=None,
            has_trades=True,
            recv_time="measured",
            event_time="partial",
            streams={
                "depth": StreamInfo(sequencing="derived", description="L2 diff depth messages"),
                "trades": StreamInfo(sequencing="contiguous", description="venue trade id"),
            },
        ),
        adapter="tardis-csv",
        params={
            "exchange": exchange,
            "dates": list(dates),
            "files": [p.name for p in [*book_paths, *trade_paths]],
        },
        input_sha256=[sha256_file(p) for p in [*book_paths, *trade_paths]],
    )
