"""Time book reconstruction alone on a catalog dataset (median of N runs).

Usage: uv run python benchmarks/reconstruct_timing.py <dataset> [symbol] [repeats]
"""

from __future__ import annotations

import statistics
import sys
import time

import polars as pl

from qrp.book import reconstruct
from qrp.catalog import Catalog

name = sys.argv[1]
symbol = sys.argv[2] if len(sys.argv) > 2 else None
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
catalog = Catalog("data/catalog")
manifest = catalog.resolve(name)
events = (
    catalog.scan(name, manifest.version, [symbol] if symbol else None)
    .drop("date")
    .collect()
    .sort("stream", "seq", maintain_order=True)
)
book_stream = events.filter(pl.col("kind") != "trade")["stream"][0]
rows = events.filter(pl.col("stream") == book_stream)
sequencing = {k: v.sequencing for k, v in manifest.capabilities.streams.items()}
times = []
for _ in range(repeats):
    t0 = time.perf_counter()
    quotes, stats = reconstruct(rows, sequencing, levels=5)
    times.append(time.perf_counter() - t0)
median = statistics.median(times)
print(
    f"{name}: {rows.height} rows, {stats.messages} messages, "
    f"median {median:.2f}s ({[round(t, 2) for t in times]}), "
    f"{rows.height / median / 1e6:.2f} M rows/s, quotes hash {quotes.hash_rows().sum()}"
)
