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

Profiled on the real dataset (Tardis LINKUSDT, 2 days, 9.29 M rows; `cProfile` plus wall-clock
timings, median of 3 where stated):

| Stage | Wall time |
|---|---|
| `qrp import` (CSV parse, normalise, 18 validation checks, dedupe, partitioned write, SHA-256) | 5.3 s |
| Catalog scan + sort of 9.29 M rows | 0.2 s |
| Book reconstruction, 8.86 M book rows → 1.29 M messages | 5.1 s (median of 3) |
| All 18 features + labels on 172,674 grid points | 0.3 s |
| Full `qrp run` (selection, single test evaluation, bootstrap, regimes, 4 backtests, 7 controls) | 33 s |

Book reconstruction is the one stage that is inherently sequential, and it dominates. One
optimisation was made and measured:

* Per-message flags computed once with `numpy.logical_and.reduceat` instead of per-message
  slice reductions, and the loop switched from indexing numpy arrays to plain Python lists.
  Output is byte-identical (row hash compared). Median of 3: synthetic feed 1.11 s → 0.85 s
  (−23%); real feed 5.49 s → 5.13 s (−7%).
* Why the real-data gain was small: `cProfile` adds overhead to every call, which inflated the
  apparent cost of ~2.6 M tiny numpy calls. The real feed also has ~7 rows per message versus
  ~1.2 in the synthetic one, so per-row work (dict update plus `bisect` on a ~1,000-level
  side) dominates: 5.13 s / 8.86 M rows ≈ 0.58 µs per row in CPython.
* Not done: a compiled implementation (Rust/C++ extension or numba), which would remove the
  interpreter overhead (gain not measured here). 5 s for two days of full-depth data does not
  justify a second toolchain in a research repo; it is the first thing to do if datasets grow
  by an order of magnitude, and the C++ engine in Project 1 is the natural home for it.
