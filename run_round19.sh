#!/usr/bin/env bash
# Round 19, on top of the round 18 profiles ("not for them" lines in data/colleagues.csv, 926 of 1069):
#   1. a second set of "not for them" lines from the mistakes that are left, learned on one half, tested on the other
#   2. more auto-forwards without more wrong ones: every rule judged on the same decisions (experiments/auto_tradeoff.py)
#   nohup ./run_round19.sh > round19.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round19
mkdir -p $OUT
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
summ() {
  echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//')"
  grep -E "auto-forward:|score:" $2 | sed 's/^ */      /'
}
people() { sed -n '/Per person/,/^$/p' $1 | grep -E "^ +(jernej|jan|ana) " | tr -s ' ' | tr '\n' ' '; }
stamp() { echo; echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "0. compile + unit tests"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -1

stamp "baseline with the round 18 profiles (should be 926), every decision saved"
for f in $SETS; do
  short=$(basename $f .csv | sed 's/test_emails/handwritten/; s/_emails//')
  $PY -u test_routing.py $f --spam --jsonl $OUT/auto_$short.jsonl > $OUT/default_$short.log 2>&1
  summ "default $short" $OUT/default_$short.log
done

stamp "2. auto-forward rules on the same decisions (no model asked again)"
$PY experiments/auto_tradeoff.py $OUT/auto_handwritten.jsonl $OUT/auto_generated.jsonl \
    $OUT/auto_reworded_qwen.jsonl $OUT/auto_reworded_gemma_gams.jsonl | sed 's/^/  /'

stamp "1. second set of 'not for them' lines: learned on one half, tested on the other"
$PY -u experiments/group_split.py $SETS --out-dir $OUT 2>&1 | grep -v -i warn | sed 's/^/  /'
for k in 0 1; do
  t=$((1 - k))
  echo "--- learn on half $((k + 1)), test on half $((t + 1))"
  $PY -u test_routing.py $OUT/fold$k.csv --spam --out $OUT/fold${k}_run.csv > $OUT/fold${k}_learn.log 2>&1
  $PY -u suggest_profiles.py --contrast $OUT/fold${k}_run.csv 2>&1 | grep -v -i warn > $OUT/contrast_$k.log
  grep -E "mistakes between|^ +[0-9]+\. " $OUT/contrast_$k.log | cut -c1-160 | sed 's/^/    /'
  $PY suggest_profiles.py --apply-all --out $OUT/colleagues_c$k.csv > /dev/null
  $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_before.log 2>&1
  COLLEAGUES_COLLECTION=JevTry_r19_c$k $PY ingest_colleagues.py $OUT/colleagues_c$k.csv > /dev/null
  COLLEAGUES_COLLECTION=JevTry_r19_c$k $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_c.log 2>&1
  summ "half $((t + 1)) round 18 profiles   " $OUT/fold${t}_before.log; echo "      $(people $OUT/fold${t}_before.log)"
  summ "half $((t + 1)) + second set of lines" $OUT/fold${t}_c.log;      echo "      $(people $OUT/fold${t}_c.log)"
  curl -s -X DELETE localhost:6333/collections/JevTry_r19_c$k > /dev/null
done
rm -f data/profile_suggestions.csv

stamp "done"
