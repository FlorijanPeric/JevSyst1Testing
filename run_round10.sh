#!/usr/bin/env bash
# Round 10: a bigger model on hivecore as second opinion for system one decisions.
#   1. "all": the bigger model re-decides every system one decision (is it better at all?)
#   2. "unclear": only when search overruled system one or the top two are close (cheaper)
#   nohup ./run_round10.sh > round10.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round10
mkdir -p $OUT
H=data/test_emails.csv
G=experiments/data/generated_emails.csv
summ() { echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//'), $(grep -E "llm  path" $2 | sed 's/ *llm  path: */system one path /; s/ correct//')$(grep -E "second opinion:" $2 | sed 's/ *second opinion: */; second opinion /')"; }
both() {
  name=$1; shift
  $PY -u test_routing.py $H --spam "$@" > $OUT/${name}_h.log 2>&1; summ "$name handwritten" $OUT/${name}_h.log
  $PY -u test_routing.py $G --spam "$@" > $OUT/${name}_g.log 2>&1; summ "$name generated  " $OUT/${name}_g.log
}
stamp() { echo; echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "0. compile + unit tests"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -2

stamp "baseline (system one only)"
both baseline

MODELS="gemma-4-12b cjvt/GaMS3-12B-Instruct qwen3.6:35b hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL"
stamp "1. bigger model re-decides every system one decision"
for m in $MODELS; do
  both "all_$(echo $m | tr '/:' '__' | cut -c1-30)" --second-opinion all --second-opinion-model "$m"
done

stamp "2. bigger model only for unclear decisions (overruled, or top two within 0.15)"
for m in $MODELS; do
  both "unclear_$(echo $m | tr '/:' '__' | cut -c1-30)" --second-opinion unclear --second-opinion-model "$m"
done

stamp "done"
