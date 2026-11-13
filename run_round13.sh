#!/usr/bin/env bash
# Round 13: gemma decides (second opinion on every system one decision); Qwen3-30B and/or GaMS only check
#   whether an agreed decision (system one = gemma) may go out automatically. They never change the decision.
# Each on the handwritten, generated and both reworded sets (round11/reworded_*.csv, half of them hard).
#   nohup ./run_round13.sh > round13.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round13
mkdir -p $OUT
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
summ() {
  echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//'), $(grep -E "^Search misses" $2 | sed 's/ (.*//; s/Search misses: */search misses /')$(grep -E "second opinion:" $2 | sed 's/ *second opinion: */; second opinion /')"
  grep -E "model votes:|auto-forward:" $2 | sed 's/^ */      /'
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

QWEN=hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL
GAMS=cjvt/GaMS3-12B-Instruct

stamp "1. gemma alone decides and checks (the round 12 default)"
each gemma_only

stamp "2. check models on agreed decisions"
each check_qwen_gams --auto-check-models "$QWEN,$GAMS"
each check_qwen --auto-check-models "$QWEN"
each check_gams --auto-check-models "$GAMS"

stamp "3. fresh answers (nothing cached), Qwen3-30B + GaMS checking: real time per email"
SETS="data/test_emails.csv experiments/data/generated_emails.csv" CACHE_SYSTEMONE=0 each fresh_check_qwen_gams --auto-check-models "$QWEN,$GAMS"

stamp "done"
