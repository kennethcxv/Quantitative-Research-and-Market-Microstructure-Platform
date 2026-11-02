#!/usr/bin/env bash
# End-to-end demonstration on the bundled synthetic feed: generate -> import -> validate ->
# experiment (selection, one test evaluation, backtest, negative controls) -> report.
# Everything it produces is SYNTHETIC. Runs in about a minute on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run qrp fixture --out data/raw/synthetic/synth.parquet
uv run qrp import --name synthetic --qrp-v1 data/raw/synthetic/synth.parquet
uv run qrp validate --name synthetic
report=$(uv run qrp run --config configs/synthetic_demo.toml)
echo
echo "Report: ${report}"
sed -n '/## Out-of-sample results/,/## Reproduce/p' "${report}"
