# Benchmarks

Measured 2026-11-02 on an Apple-silicon MacBook (macOS 27, 16 cores), Python 3.12.14, polars
2.1.0, duckdb 2.0.0, pyarrow 26.0.0, numpy 2.5.4, pandas 3.1.0. Raw numbers:
[`benchmarks/results.json`](../benchmarks/results.json). Reproduce with

```
uv run qrp bench --rows 5000000 --repeats 3
```

## Method

* **Data.** The synthetic generator, 4 symbols × 8 sessions of 4 h, 6,064,371 rows. CSV:
  448 MB. Parquet (zstd): 66 MB as one file, 66 MB partitioned by `symbol=/date=` (6.8× smaller
  than CSV). It is regenerated deterministically, so anyone can rerun the exact workload.
* **Isolation.** Every (case, implementation, repeat) runs in a fresh Python process. The engine
  is imported before the clock starts, so import cost is excluded; the "RSS after import" column
  shows that fixed cost.
* **Metrics.** Wall time from `perf_counter`; CPU time = user + system from `getrusage`, summed
  over all threads, so it exceeds wall time when an engine parallelises; peak RSS from
  `ru_maxrss` (process lifetime, so it includes the import baseline). Median of 3 repeats.
* **Correctness gate.** Each implementation returns a small summary (row counts, sums); the run
  fails unless all implementations of a case agree to 1e-9 relative tolerance.
  `tests/test_bench.py` runs the same gate in CI on a small dataset.
* **Caveats.** The OS page cache is warm (data was just written). Single machine, no
  background-load control beyond running nothing else. Sub-50 ms timings are dominated by fixed
  costs (file opens, planning) and should be read as "fast enough", not as precise ratios.

## Results

**CSV → Parquet partitioned by symbol/date** (6.06 M rows)

| implementation | wall | CPU | peak RSS | RSS after import |
|---|---|---|---|---|
| pandas (`read_csv`, `to_parquet(partition_cols=...)`) | 1711 ms | 2941 ms | 1679 MB | 91 MB |
| polars (`read_csv`, one `write_parquet` per partition) | 361 ms | 2524 ms | 1759 MB | 52 MB |

Polars is 4.7× faster in wall time for about the same CPU: the gain is parallel CSV parsing,
not less work. Peak memory is the same; both materialise the full table.

**One symbol-day, per-minute VWAP** (out of 32 symbol-days in the partitioned dataset)

| implementation | wall | CPU | peak RSS | RSS after import |
|---|---|---|---|---|
| pandas, read everything then filter | 119 ms | 563 ms | 1021 MB | 90 MB |
| pandas, `read_parquet(filters=...)` | 35 ms | 44 ms | 128 MB | 90 MB |
| DuckDB SQL with hive partitioning | 22 ms | 22 ms | 62 MB | 33 MB |
| Polars lazy scan | 7 ms | 19 ms | 90 MB | 52 MB |

Most of the improvement is partition pruning and predicate pushdown, which pandas also gets
when given `filters`: 3.4× faster and 8× less memory without changing engine. The query
engines then add another 1.6–5×. The point of the catalog's `symbol=/date=` layout is this
first factor: a day's research query reads one file instead of the whole history.

**As-of join: each trade to the prevailing quote** (161,149 trades into 1,291,484 quotes)

| implementation | wall | CPU | peak RSS | RSS after import |
|---|---|---|---|---|
| pandas `merge_asof` | 43 ms | 79 ms | 197 MB | 90 MB |
| DuckDB `ASOF LEFT JOIN` | 32 ms | 105 ms | 224 MB | 33 MB |
| Polars `join_asof` | 11 ms | 43 ms | 149 MB | 52 MB |

**Trailing 5 s traded volume on a 1 s grid** (619,199 grid points)

| implementation | wall | CPU | peak RSS | RSS after import |
|---|---|---|---|---|
| pandas: 1 s right-closed resample, then `rolling(5)` | 53 ms | 57 ms | 171 MB | 90 MB |
| Polars: cumulative sum as-of `t` minus as-of `t − 5 s` (the feature code's method) | 12 ms | 28 ms | 122 MB | 52 MB |

The pandas version is exact only because the window is a whole number of grid steps; the
cumulative-sum method works for any window and any grid, which is why the feature code uses it.

## Where the pipeline actually spends time

Profiled on the real dataset (Tardis LINKUSDT, 2 days, 9.29 M rows) with `cProfile` and
wall-clock timings:

| Stage | Wall time |
|---|---|
| `qrp import` (CSV parse, normalise, validation checks, dedupe, partitioned write, SHA-256) | 5.3 s |
| Catalog scan + sort of 9.29 M rows | 0.2 s |
| Book reconstruction, 8.86 M book rows → 1.29 M messages | 3.97 s now (5.91 s originally) |
| All 18 features + labels on 172,674 grid points | 0.3 s |
| Full `qrp run` (selection, single test evaluation, bootstrap, regimes, 4 backtests, 7 controls), measured before the reconstruction speed-up | 33 s |

Book reconstruction is the one inherently sequential stage, and it dominated the profile
(~95% of data preparation). What was tried, and what the measurements showed:

1. **First attempt** (commit 4917a93): compute per-message flags once with
   `numpy.logical_and.reduceat` instead of reducing millions of tiny slices, and iterate plain
   Python lists instead of indexing numpy scalars. I reported this as 5.49 s → 5.13 s (−7%).
   That figure was measured while another process was loading the machine. Re-measured
   properly (below), this version is within 2% of the original on real data, and storing the
   ladder as a list of Python float lists added ~600 MB of peak memory. `cProfile` had also
   overstated the per-call overhead it targeted: it charges a fixed cost to every call, which
   inflated ~2.6 M tiny numpy calls.
2. **Second change** (commit 63e99e9): write the ladder into a flat `array("d")` buffer and
   keep sequence numbers only at message starts instead of a per-row Python int list. The gain
   comes from not allocating 1.3 M small lists and 8.9 M int objects.

Method for the final numbers: `benchmarks/compare_book_revisions.py`. It extracts `book.py`
from each git revision, runs each measurement in a fresh process, alternates revisions
(ABAB…) so drift in machine load hits both equally, and requires identical output row hashes.
Five runs each, median:

| Data | Original (a620b7b) | Now | Peak RSS (reconstruction) |
|---|---|---|---|
| Tardis LINKUSDT, 8.86 M rows | 5.91 s | 3.97 s (−33%) | unchanged (~1.56 GB over input) |
| Synthetic, 0.49 M rows | 1.14 s | 0.68 s (−40%) | unchanged (~0.36 GB) |

```
uv run python benchmarks/compare_book_revisions.py tardis_binance_linkusdt a620b7b HEAD
```

The remaining cost is per-row dict and `bisect` work in CPython: 3.97 s / 8.86 M rows ≈ 0.45 µs
per row. A compiled implementation (Rust/C++ extension or numba) would remove the interpreter
overhead; the gain was not measured here. It was not done, because 4 s for two days of
full-depth data does not justify a second toolchain in a research repo. It is the first thing
to do if datasets grow by an order of magnitude, and the C++ engine in Project 1 is the
natural home for it.
