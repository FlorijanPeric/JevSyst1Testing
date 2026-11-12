#!/usr/bin/env bash
# Round 12: three ideas on top of the default (gemma-4-12b second opinion on unclear decisions).
#   1. agreement: system one picks the second-opinion model agreed with -> safe to forward automatically?
#   2. vote: gemma + Qwen3-30B + GaMS pick on unclear decisions, the majority wins
#   3. summary search: search on the email plus a one-line summary of what it asks for
# Each on the handwritten, generated and both reworded sets (round11/reworded_*.csv, half of them hard).
#   nohup ./run_round12.sh > round12.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round12
mkdir -p $OUT
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
VOTERS="gemma-4-12b,hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL,cjvt/GaMS3-12B-Instruct"
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

stamp "baseline: default (gemma on unclear)"
each default

stamp "1. agreement: gemma asked on every system one decision, agreed picks counted as auto-forward"
each agree_all --second-opinion all

stamp "2. vote: gemma + Qwen3-30B + GaMS, majority wins"
each vote_unclear --second-opinion-model "$VOTERS"
each vote_all --second-opinion all --second-opinion-model "$VOTERS"

stamp "3. summary search (gemma writes one line; weight = its share of the search vector)"
each nofast_plain --min-score 9
each nofast_summary0.5 --min-score 9 --search-summary
for w in 0.3 0.5 0.7; do
  each summary$w --search-summary --search-summary-weight $w
done

stamp "done"
