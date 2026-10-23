#!/usr/bin/env bash
# Generate the bigger test set, then run the benchmark. Safe to rerun: both steps resume from where they stopped.
#   nohup experiments/run_all.sh > experiments/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -u experiments/generate_emails.py --per-person 20 --manual 30
$PY -u experiments/bench.py --max-hours "${MAX_HOURS:-6}"
