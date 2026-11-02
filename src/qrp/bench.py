"""Benchmarks: the platform's Polars/DuckDB/Parquet paths against straightforward pandas.

Every (case, implementation) pair runs in a fresh Python process, so peak RSS and CPU time are
that run's alone. CPU time is user + system over all threads (``getrusage``), so it can exceed
wall time for multithreaded engines. Before timings are reported, every implementation's result
is checked against the others; a benchmark whose implementations disagree is an error, not a
data point.

Usage: ``qrp bench --rows 3000000`` (synthetic data, regenerated deterministically).
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

NS = 1_000_000_000


# --- data -------------------------------------------------------------------------------------


def prepare_data(workdir: Path, rows: int) -> dict[str, Any]:
    """Synthetic multi-symbol feed of roughly ``rows`` rows in the layouts the cases read."""
    import polars as pl

    from qrp.book import reconstruct
    from qrp.synth import SynthConfig, generate

    workdir.mkdir(parents=True, exist_ok=True)
    marker = workdir / f"data-{rows}.json"
    if marker.exists():
        return dict(json.loads(marker.read_text()))
    per_symbol_day = 165_000  # rows for one 4 h synthetic session
    n_symbols = 4
    n_days = max(1, round(rows / per_symbol_day / n_symbols))
    frames = []
    for i in range(n_symbols):
        cfg = SynthConfig(symbol=f"S{i}", seed=100 + i, n_days=n_days, gaps_per_day=0)
        frames.append(generate(cfg))
    events = pl.concat(frames)
    events.write_csv(workdir / "events.csv")
    events.write_parquet(workdir / "events.parquet")
    one = events.filter(pl.col("symbol") == "S0")
    quotes, _ = reconstruct(one, {"feed": "contiguous"}, levels=1)
    quotes.select(
        t=pl.col("recv_time_ns"), bid=pl.col("bid_px_0"), ask=pl.col("ask_px_0")
    ).write_parquet(workdir / "quotes.parquet")
    one.filter(pl.col("kind") == "trade").select(
        t=pl.col("recv_time_ns"), price="price", size="size"
    ).write_parquet(workdir / "trades.parquet")
    # Partitioned copy for the query case, laid out like the catalog (symbol=/date=).
    part = workdir / "partitioned"
    dated = events.with_columns(
        date=pl.from_epoch("recv_time_ns", time_unit="ns").dt.date().cast(pl.String)
    )
    for (symbol, date), frame in dated.group_by(["symbol", "date"]):
        path = part / f"symbol={symbol}" / f"date={date}" / "part-0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.drop("symbol", "date").write_parquet(path, compression="zstd")
    meta = {
        "rows": events.height,
        "symbols": n_symbols,
        "days": n_days,
        "query_date": str(dated["date"].min()),
        "bytes": {
            "csv": (workdir / "events.csv").stat().st_size,
            "parquet_single_file": (workdir / "events.parquet").stat().st_size,
            "parquet_partitioned": sum(f.stat().st_size for f in part.rglob("*.parquet")),
        },
    }
    marker.write_text(json.dumps(meta, indent=2))
    return meta


# --- cases (run inside the child process) ------------------------------------------------------
# One function per (case, implementation). Each returns a small summary that must agree across
# implementations of the same case.

Meta = dict[str, Any]


def _ingest_pandas(d: Path, meta: Meta) -> Meta:
    import pandas as pd

    out = d / "ingest-pandas"
    df = pd.read_csv(d / "events.csv")
    df["date"] = pd.to_datetime(df["recv_time_ns"], unit="ns").dt.strftime("%Y-%m-%d")
    df.to_parquet(out, partition_cols=["symbol", "date"], compression="zstd")
    return {"rows": len(df), "files": len(list(out.rglob("*.parquet")))}


def _ingest_polars(d: Path, meta: Meta) -> Meta:
    import polars as pl

    out = d / "ingest-polars"
    df = pl.read_csv(d / "events.csv").with_columns(
        date=pl.from_epoch("recv_time_ns", time_unit="ns").dt.date().cast(pl.String)
    )
    for (symbol, date), part in df.group_by(["symbol", "date"]):
        path = out / f"symbol={symbol}" / f"date={date}" / "part-0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.drop("symbol", "date").write_parquet(path, compression="zstd")
    return {"rows": df.height, "files": len(list(out.rglob("*.parquet")))}


def _vwap_summary(volume: list[float], notional: list[float]) -> Meta:
    return {
        "minutes": len(volume),
        "volume": float(sum(volume)),
        "vwap_sum": float(sum(n / v for n, v in zip(notional, volume, strict=True))),
    }


def _query_pandas(d: Path, meta: Meta, pushdown: bool) -> Meta:
    import pandas as pd

    part, day = d / "partitioned", meta["query_date"]
    if pushdown:
        filters = [("symbol", "==", "S0"), ("date", "==", day), ("kind", "==", "trade")]
        df = pd.read_parquet(part, filters=filters)
    else:
        df = pd.read_parquet(part)
        df = df[(df["symbol"] == "S0") & (df["date"] == day) & (df["kind"] == "trade")]
    df = df.assign(minute=df["recv_time_ns"] // (60 * NS), notional=df["price"] * df["size"])
    g = df.groupby("minute")[["size", "notional"]].sum().sort_index()
    return _vwap_summary(g["size"].tolist(), g["notional"].tolist())


def _query_duckdb(d: Path, meta: Meta) -> Meta:
    import duckdb

    rel = duckdb.sql(
        f"""
        SELECT recv_time_ns // {60 * NS} AS minute, sum(size), sum(price * size)
        FROM read_parquet('{d / "partitioned"}/**/*.parquet', hive_partitioning = true)
        WHERE symbol = 'S0' AND date = '{meta["query_date"]}' AND kind = 'trade'
        GROUP BY minute ORDER BY minute
        """
    )
    assert rel is not None
    rows = rel.fetchall()
    return _vwap_summary([r[1] for r in rows], [r[2] for r in rows])


def _query_polars(d: Path, meta: Meta) -> Meta:
    import polars as pl

    g = (
        pl.scan_parquet(d / "partitioned" / "**" / "*.parquet", hive_partitioning=True)
        .filter(
            (pl.col("symbol") == "S0")
            & (pl.col("date").cast(pl.String) == meta["query_date"])
            & (pl.col("kind") == "trade")
        )
        .group_by(minute=pl.col("recv_time_ns") // (60 * NS))
        .agg(pl.col("size").sum(), (pl.col("price") * pl.col("size")).sum().alias("notional"))
        .sort("minute")
        .collect()
    )
    return _vwap_summary(g["size"].to_list(), g["notional"].to_list())


def _asof_pandas(d: Path, meta: Meta) -> Meta:
    import pandas as pd

    quotes, trades = pd.read_parquet(d / "quotes.parquet"), pd.read_parquet(d / "trades.parquet")
    j = pd.merge_asof(trades, quotes, on="t", direction="backward")
    cost = ((j["price"] - (j["bid"] + j["ask"]) / 2) * j["size"]).sum()
    return {"rows": len(j), "signed_cost_vs_mid": float(cost)}


def _asof_duckdb(d: Path, meta: Meta) -> Meta:
    import duckdb

    rel = duckdb.sql(
        f"""
        SELECT count(*), sum((tr.price - (q.bid + q.ask) / 2) * tr.size)
        FROM read_parquet('{d / "trades.parquet"}') tr
        ASOF LEFT JOIN read_parquet('{d / "quotes.parquet"}') q ON tr.t >= q.t
        """
    )
    assert rel is not None
    ((n, cost),) = rel.fetchall()
    return {"rows": int(n), "signed_cost_vs_mid": float(cost)}


def _asof_polars(d: Path, meta: Meta) -> Meta:
    import polars as pl

    quotes, trades = pl.read_parquet(d / "quotes.parquet"), pl.read_parquet(d / "trades.parquet")
    j = trades.join_asof(quotes, on="t", strategy="backward")
    cost = j.select(
        ((pl.col("price") - (pl.col("bid") + pl.col("ask")) / 2) * pl.col("size")).sum()
    )
    return {"rows": j.height, "signed_cost_vs_mid": float(cost.item())}


WINDOW, STEP = 5 * NS, NS


def _window_pandas(d: Path, meta: Meta) -> Meta:
    """Traded volume over (g - 5 s, g] on a 1 s grid g spanning the trades."""
    import numpy as np
    import pandas as pd

    trades = pd.read_parquet(d / "trades.parquet")
    size = pd.Series(trades["size"].to_numpy(), index=pd.to_datetime(trades["t"], unit="ns"))
    # Bins (k-1 s, k s] labelled k; five consecutive bins make the window (g - 5 s, g].
    per_second = size.resample("1s", closed="right", label="right").sum()
    rolled = per_second.rolling(5, min_periods=1).sum()
    start = (int(trades["t"].min()) // STEP + 1) * STEP
    grid = pd.to_datetime(np.arange(start, int(trades["t"].max()), STEP), unit="ns")
    values = rolled.reindex(grid).fillna(0.0)
    return {"grid_points": len(values), "volume_sum": float(values.sum())}


def _window_polars(d: Path, meta: Meta) -> Meta:
    """Same window via the feature code's method: cumulative sum as-of g minus as-of g - 5 s."""
    import polars as pl

    trades = pl.read_parquet(d / "trades.parquet").with_columns(cum=pl.col("size").cum_sum())
    start = (int(trades["t"].min()) // STEP + 1) * STEP  # type: ignore[arg-type]
    end = int(trades["t"].max())  # type: ignore[arg-type]
    grid = pl.DataFrame({"g": pl.int_range(start, end, STEP, eager=True)})
    cum = trades.select("t", "cum")
    upper = grid.join_asof(cum, left_on="g", right_on="t")["cum"].fill_null(0)
    lower = grid.with_columns(pl.col("g") - WINDOW).join_asof(cum, left_on="g", right_on="t")
    values = upper - lower["cum"].fill_null(0)
    return {"grid_points": values.len(), "volume_sum": float(values.sum())}


IMPLEMENTATIONS: dict[str, dict[str, Callable[[Path, Meta], Meta]]] = {
    "ingest_csv_to_partitioned_parquet": {"pandas": _ingest_pandas, "polars": _ingest_polars},
    "query_one_symbol_day_vwap_by_minute": {
        "pandas_naive": lambda d, m: _query_pandas(d, m, pushdown=False),
        "pandas_filters": lambda d, m: _query_pandas(d, m, pushdown=True),
        "duckdb": _query_duckdb,
        "polars": _query_polars,
    },
    "asof_join_trades_to_quotes": {
        "pandas": _asof_pandas,
        "duckdb": _asof_duckdb,
        "polars": _asof_polars,
    },
    "trailing_volume_on_1s_grid": {"pandas": _window_pandas, "polars": _window_polars},
}


def _maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def child(case: str, impl: str, workdir: str) -> None:
    """Entry point inside the measurement process; prints one JSON line."""
    d = Path(workdir)
    meta = json.loads(next(d.glob("data-*.json")).read_text())
    # Import the engine before measuring so library import cost is not counted as work.
    if impl.startswith("pandas"):
        import pandas  # noqa: F401
    elif impl == "duckdb":
        import duckdb  # noqa: F401
    else:
        import polars  # noqa: F401
    baseline = _maxrss_bytes()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    result = IMPLEMENTATIONS[case][impl](d, meta)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    print(
        json.dumps(
            {
                "wall_s": wall,
                "cpu_s": cpu,
                "peak_rss_mb": _maxrss_bytes() / 2**20,
                "rss_after_import_mb": baseline / 2**20,
                "result": result,
            }
        )
    )


def run_benchmarks(rows: int, repeats: int, workdir: Path, out: Path) -> dict[str, Any]:
    import shutil

    meta = prepare_data(workdir, rows)
    results: dict[str, Any] = {}
    for case, impls in IMPLEMENTATIONS.items():
        results[case] = {}
        for impl in impls:
            runs = []
            for _ in range(repeats):
                shutil.rmtree(workdir / f"ingest-{impl}", ignore_errors=True)
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"from qrp.bench import child; child({case!r}, {impl!r}, {str(workdir)!r})",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
            results[case][impl] = {
                "wall_s_median": statistics.median(r["wall_s"] for r in runs),
                "wall_s_all": [round(r["wall_s"], 4) for r in runs],
                "cpu_s_median": statistics.median(r["cpu_s"] for r in runs),
                "peak_rss_mb_max": max(r["peak_rss_mb"] for r in runs),
                "rss_after_import_mb": min(r["rss_after_import_mb"] for r in runs),
                "result": runs[0]["result"],
            }
        answers = [v["result"] for v in results[case].values()]
        if not all(_agree(answers[0], other) for other in answers[1:]):
            raise RuntimeError(f"{case}: implementations disagree: {answers}")
    report = {
        "data": meta,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "packages": _versions(),
        "repeats": repeats,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _agree(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same keys and numerically equal values (engines sum floats in different orders)."""
    return a.keys() == b.keys() and all(
        math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-6) for k in a
    )


def _versions() -> dict[str, str]:
    from importlib import metadata

    names = ("polars", "duckdb", "pyarrow", "numpy", "pandas")
    return {name: metadata.version(name) for name in names}
