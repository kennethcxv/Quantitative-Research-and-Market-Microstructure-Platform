"""Markdown research report generated from a run directory.

Every number comes from the run's artifacts, and every conclusion line is computed from those
numbers by a fixed rule, so the report cannot say more than the evidence does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from qrp.experiment import load_run_config


def _ci(x: dict[str, float], scale: float = 1.0, digits: int = 4) -> str:
    return (
        f"{x['estimate'] * scale:.{digits}f} "
        f"[{x['low'] * scale:.{digits}f}, {x['high'] * scale:.{digits}f}]"
    )


def _break_even_fee_bps(s: dict[str, Any], notional: float) -> float:
    """Taker fee at which these trades would have broken even (edge left after spread/latency)."""
    return float((s["edge_at_decision"] - s["latency_cost"] - s["spread_cost"]) / notional * 1e4)


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    backtest = json.loads((run_dir / "backtest.json").read_text())
    selection = pl.read_csv(run_dir / "selection.csv")
    controls_path = run_dir / "controls.json"
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else None
    config = (run_dir / "config.toml").read_text()
    cfg = load_run_config(run_dir)
    ds = manifest["dataset"]
    source, caps = ds["source"], ds["capabilities"]
    test, dev = metrics["test"], metrics["dev_oos"]
    out: list[str] = [f"# {manifest['experiment']}", ""]

    if source["synthetic"]:
        out += [
            "> **SYNTHETIC DATA.** Every result below comes from a simulated order book. It "
            "shows the pipeline works; it says nothing about real markets or real profitability.",
            "",
        ]
    out += [
        f"Run `{manifest['run_id']}` · dataset `{ds['name']}@{ds['version']}` · "
        f"commit `{manifest['provenance']['git_commit'][:10]}`"
        + (" (dirty tree)" if manifest["provenance"]["git_dirty"] else ""),
        "",
        "## Data",
        "",
        f"* Source: {source['name']}",
        f"* Licence: {source['license']}",
        f"* Notes: {source['notes'] or '-'}",
        f"* Book depth: {caps['book_depth'] or 'full'}; trades: {_yes(caps['has_trades'])}; "
        f"receive times: {caps['recv_time']}; exchange times: {caps['event_time']}",
        "* Stream sequencing: "
        + ", ".join(f"`{k}` {v['sequencing']}" for k, v in caps["streams"].items()),
        "* Validation warnings: "
        + (
            ", ".join(f"{w['name']}={w['violations']}" for w in ds["validation"]["warnings"])
            or "none"
        ),
        f"* Duplicate rows removed at import: {ds.get('duplicates_removed', 'n/a')}",
        f"* Book reconstruction: {manifest['book']}",
        f"* Rows: {manifest['rows']}",
        "",
        "## Features",
        "",
        _table(["feature", "definition"], [[f"`{k}`", v] for k, v in manifest["features"].items()]),
        "",
        "## Model selection (walk-forward on the development period only)",
        "",
    ]
    summary = selection.group_by("model", maintain_order=True).agg(
        folds=pl.len(),
        mean_r2_vs_zero=(1 - pl.col("mse") / pl.col("zero_mse")).mean(),
        mean_rank_ic=pl.col("rank_ic").mean(),
        folds_beating_zero=(pl.col("mse") < pl.col("zero_mse")).sum(),
    )
    out += [
        _table(
            ["model", "folds", "mean OOS R² vs zero", "mean rank IC", "folds beating zero"],
            [[r[0], r[1], f"{r[2]:.5f}", f"{r[3]:.4f}", r[4]] for r in summary.iter_rows()],
        ),
        "",
        f"Selected by lowest mean fold MSE: **{manifest['selected_model']}**. "
        f"Test-period evaluations of this experiment before this run: "
        f"{manifest['test_evaluations_before_this_run']}.",
        "",
        "## Out-of-sample results",
        "",
        "Intervals are 95% circular block bootstrap intervals "
        f"(block = {metrics['block_len']} samples); p-values are against "
        f"{cfg.evaluation.n_null} circular-shift nulls, so the smallest attainable p is "
        f"{1 / (cfg.evaluation.n_null + 1):.3f}.",
        "",
        _table(
            [
                "split",
                "n",
                "rank IC [95% CI]",
                "OOS R² vs zero [95% CI]",
                "hit rate",
                "calib. slope",
                "null p",
            ],
            [
                [
                    name,
                    m["n"],
                    _ci(m["rank_ic"]),
                    _ci(m["oos_r2"], digits=5),
                    f"{m['hit_rate']:.3f}",
                    f"{m['calibration_slope']:.3f}",
                    f"{m['null_p_value']:.3f}",
                ]
                for name, m in (
                    ("dev walk-forward (selected model; optimistic)", dev),
                    ("test (once)", test),
                )
            ],
        ),
        "",
    ]
    if test.get("regimes"):
        out += [
            f"Test results by `{cfg.evaluation.regime_feature}` tercile "
            "(cut points from development data):",
            "",
            _table(
                ["regime", "n", "mean |ret|", "rank IC [95% CI]", "OOS R² [95% CI]"],
                [
                    [
                        k,
                        v["n"],
                        f"{v['mean_abs_ret']:.2e}",
                        _ci(v["rank_ic"]),
                        _ci(v["oos_r2"], digits=5),
                    ]
                    for k, v in test["regimes"].items()
                ],
            ),
            "",
        ]
    out += [
        "Test calibration (forecast deciles):",
        "",
        _table(
            ["decile", "n", "mean forecast", "mean outcome"],
            [
                [c["bin"], c["n"], f"{c['mean_pred']:.2e}", f"{c['mean_y']:.2e}"]
                for c in test["calibration"]
            ],
        ),
        "",
        "## Simulated execution",
        "",
        "Simulated marketable orders against the reconstructed book at arrival time; no passive "
        "fills, no persistent market impact. Fills below are simulations, not historical trades.",
        "",
        "Strategies (thresholds on the forecast, set from development data): "
        + ", ".join(f"`{k}` = {v:.3e}" for k, v in backtest["thresholds"].items())
        + ". `cost_rule` trades only when the forecast exceeds the estimated round-trip cost; "
        "diagnostic strategies ignore costs on purpose, to show where the edge goes.",
        "",
    ]
    rows = []
    for split, by_rule in backtest["results"].items():
        for rule, s in by_rule.items():
            ci = s["pnl_per_minute_mean_ci"]
            traded = s["fills"] > 0
            rows.append(
                [
                    split,
                    rule,
                    s["orders"],
                    s["fills"],
                    f"{s['traded_notional']:.0f}",
                    f"{s['edge_at_decision']:.2f}",
                    f"{s['latency_cost']:.2f}",
                    f"{s['spread_cost']:.2f}",
                    f"{s['fees']:.2f}",
                    f"{s['pnl_total']:.2f}",
                    f"{_break_even_fee_bps(s, s['traded_notional']):.2f}" if traded else "-",
                    _ci(ci, digits=3) if ci and traded else "-",
                ]
            )
    out += [
        _table(
            [
                "split",
                "strategy",
                "orders",
                "fills",
                "notional",
                "edge at decision",
                "latency cost",
                "spread+slippage",
                "fees",
                "net P&L",
                "break-even fee (bps)",
                "P&L/min [95% CI]",
            ],
            rows,
        ),
        "",
        f"Costs: {cfg.backtest.costs.model_dump()}; latencies (ms): decision "
        f"{cfg.backtest.decision_latency_ms}, order {cfg.backtest.order_latency_ms}, "
        f"ack {cfg.backtest.ack_latency_ms} (assumptions, not measurements); position size "
        f"{cfg.backtest.position_size}, limit {cfg.backtest.max_position}.",
        "",
    ]
    if controls is not None:
        out += [
            "## Negative controls",
            "",
            _table(
                ["control", "expectation", "observed", "passed"],
                [[c["name"], c["expectation"], c["observed"], _yes(c["passed"])] for c in controls],
            ),
            "",
        ]
    cost = backtest["results"]["test"]["cost_rule"]
    t_ci = cost["pnl_per_minute_mean_ci"]
    if cost["fills"]:
        trading = (
            "* Tradeable (cost-rule strategy's per-minute net P&L CI entirely above 0): "
            f"{_yes(bool(t_ci) and t_ci['low'] > 0)}; net simulated P&L on test "
            f"{cost['pnl_total']:.2f}"
        )
    else:
        trading = (
            "* Tradeable: no. The cost-rule strategy did not trade on test: the largest forecast "
            f"({cost['max_abs_forecast']:.2e}) never exceeded the cost threshold "
            f"({backtest['thresholds']['cost_rule']:.2e})"
        )
    diagnostic = [
        f"* `{rule}` on test: edge before costs {s['edge_at_decision']:.2f}, "
        f"net {s['pnl_total']:.2f} after {s['spread_cost'] + s['latency_cost']:.2f} "
        f"spread/latency and {s['fees']:.2f} fees"
        for rule, s in backtest["results"]["test"].items()
        if rule != "cost_rule" and s["fills"]
    ]
    out += [
        "## Conclusions (mechanical)",
        "",
        f"* Signal exists (test rank IC CI excludes 0 and null p < "
        f"{cfg.evaluation.significance}): "
        f"{_yes(test['rank_ic']['low'] > 0 or test['rank_ic']['high'] < 0)}"
        f" and {_yes(test['null_p_value'] < cfg.evaluation.significance)}",
        f"* Usable (beats the zero forecast: test OOS R² CI entirely above 0): "
        f"{_yes(test['oos_r2']['low'] > 0)}",
        trading,
        *diagnostic,
        "",
        "## Reproduce",
        "",
        "```",
        f"qrp run --config <this run's config.toml>   # dataset version {ds['version']}",
        "```",
        "",
        f"Provenance: `{json.dumps(manifest['provenance'])}`",
        "",
        "<details><summary>config.toml</summary>",
        "",
        "```toml",
        config.rstrip(),
        "```",
        "",
        "</details>",
        "",
    ]
    path = run_dir / "report.md"
    path.write_text("\n".join(out))
    return path
