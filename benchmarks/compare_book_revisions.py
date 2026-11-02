"""Interleaved A/B timing of book reconstruction across git revisions.

Each measurement runs in a fresh process (so peak RSS belongs to that run alone), revisions
alternate (ABAB...) so drift in machine load affects both equally, and output row hashes must
match across revisions.

Usage:
  uv run python benchmarks/compare_book_revisions.py <dataset> <rev> [<rev> ...] [--repeats N]
Example (the measurement in docs/benchmarks.md):
  uv run python benchmarks/compare_book_revisions.py tardis_binance_linkusdt a620b7b HEAD
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r"""
import importlib.util, json, resource, sys, time
import polars as pl
from qrp.catalog import Catalog
path, name = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("book_under_test", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
catalog = Catalog("data/catalog")
manifest = catalog.resolve(name)
events = catalog.scan(name, manifest.version).drop("date").collect()
events = events.sort("stream", "seq", maintain_order=True)
stream = events.filter(pl.col("kind") != "trade")["stream"][0]
rows = events.filter(pl.col("stream") == stream)
sequencing = {k: v.sequencing for k, v in manifest.capabilities.streams.items()}
unit = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
quotes, _ = mod.reconstruct(rows, sequencing, levels=5)
seconds = time.perf_counter() - t0
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"s": seconds, "rss_mb": (after - before) * unit / 2**20,
                  "hash": int(quotes.hash_rows().sum())}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("revisions", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp())
    files = {}
    for rev in args.revisions:
        source = subprocess.run(
            ["git", "show", f"{rev}:src/qrp/book.py"], capture_output=True, text=True, check=True
        ).stdout
        files[rev] = tmp / f"book_{rev.replace('/', '_')}.py"
        files[rev].write_text(source)
    results: dict[str, list[dict[str, float]]] = {rev: [] for rev in args.revisions}
    for _ in range(args.repeats):
        for rev in args.revisions:
            out = subprocess.run(
                [sys.executable, "-c", CHILD, str(files[rev]), args.dataset],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            results[rev].append(json.loads(out.strip().splitlines()[-1]))
    hashes = {r["hash"] for runs in results.values() for r in runs}
    for rev, runs in results.items():
        times = [r["s"] for r in runs]
        print(
            f"{rev:>10}: median {statistics.median(times):.3f} s "
            f"(min {min(times):.3f}, max {max(times):.3f}), "
            f"peak RSS +{statistics.median(r['rss_mb'] for r in runs):.0f} MB"
        )
    print("outputs identical" if len(hashes) == 1 else f"OUTPUTS DIFFER: {hashes}")


if __name__ == "__main__":
    main()
