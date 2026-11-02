from __future__ import annotations

from pathlib import Path

from qrp.bench import IMPLEMENTATIONS, run_benchmarks


def test_all_benchmark_implementations_agree(tmp_path: Path) -> None:
    # run_benchmarks raises if any implementation's answer differs from the others.
    report = run_benchmarks(rows=150_000, repeats=1, workdir=tmp_path, out=tmp_path / "r.json")
    assert set(report["results"]) == set(IMPLEMENTATIONS)
    for case, by_impl in report["results"].items():
        assert set(by_impl) == set(IMPLEMENTATIONS[case])
        assert all(m["wall_s_median"] > 0 and m["peak_rss_mb_max"] > 0 for m in by_impl.values())
    assert report["data"]["bytes"]["parquet_partitioned"] < report["data"]["bytes"]["csv"]
