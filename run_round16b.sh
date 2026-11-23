#!/usr/bin/env bash
# Round 16b: is DeepSeek faster, and is it better? Gemma (default) vs DeepSeek on the same 4 sets, with the
# second-opinion cache off, so every second opinion and check is a fresh call and the times are real.
# System one stays cached (same for both). DeepSeek never gets more than 2 requests at once ([model_limits]).
#   nohup ./run_round16b.sh > round16b.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round16b
mkdir -p $OUT
export CACHE_SECOND_OPINION=0
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
each() {
  name=$1; shift
  for f in $SETS; do
    short=$(basename $f .csv | sed 's/test_emails/handwritten/; s/_emails//')
    log=$OUT/${name}_$short.log
    $PY -u test_routing.py $f --spam "$@" > $log 2>&1
    echo "  $name $short:"
    grep -E "score:|model time:|auto-forward:" $log | sed 's/^ */      /'
  done
}
stamp() { echo; echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "0. compile"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -1

stamp "gemma (default), fresh answers"
each gemma
stamp "DeepSeek, fresh answers"
each ds --second-opinion-model DeepSeek-V4.1-Flash
stamp "gemma again (does a second run give the same answers and times?)"
each gemma2
stamp "done"
