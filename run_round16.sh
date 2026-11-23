#!/usr/bin/env bash
# Round 16: DeepSeek-V4.1-Flash as the second opinion (instead of gemma), with and without full profiles.
# DeepSeek allows at most 2 requests at once: routing.toml [model_limits] enforces it in every process, and
# the tests below run one after another, never in parallel.
#   nohup ./run_round16.sh > round16.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round16
mkdir -p $OUT
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
summ() {
  echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//'), $(grep -E "^Search misses" $2 | sed 's/ (.*//; s/Search misses: */search misses /')$(grep -E "second opinion:" $2 | sed 's/ *second opinion: */; second opinion /')"
  grep -E "auto-forward:|wider list:" $2 | sed 's/^ */      /'
}
each() {  # name, extra args...: every set
  name=$1; shift
  for f in $SETS; do
    short=$(basename $f .csv | sed 's/test_emails/handwritten/; s/_emails//')
    log=$OUT/${name}_$short.log
    $PY -u test_routing.py $f --spam "$@" > $log 2>&1; summ "$name $short" $log
  done
}
stamp() { echo; echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "0. compile + unit tests"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -2

stamp "baseline: current default"
each default

DS=DeepSeek-V4.1-Flash
stamp "1. DeepSeek decides instead of gemma (8 candidates, id only)"
each ds --second-opinion-model $DS

stamp "2. DeepSeek with full profiles (every duty line)"
each ds_full --second-opinion-model $DS --second-opinion-full-profiles
each ds_full_5cand --second-opinion-model $DS --second-opinion-full-profiles --second-opinion-candidates 0

stamp "3. DeepSeek with one sentence of reasoning first (full profiles)"
each ds_full_brief --second-opinion-model $DS --second-opinion-full-profiles --second-opinion-reasoning brief

stamp "done"
