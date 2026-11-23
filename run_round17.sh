#!/usr/bin/env bash
# Round 17: ideas 1-5 after round 16 (Jernej's emails and wrong auto-forwards on the hard sets)
#   2. system one chooses from more people (6 / 8 instead of 5)
#   3. DeepSeek as a third check model before auto-forwarding
#   1. track record: no auto-forward to a person whose recent reviewed decisions were often wrong (learning curve)
#   4+5. "not for them" lines and new duty lines, learned on one half and tested on the other (rewordings of the
#        same email always in the same half); per person for jernej, jan, ana
#   nohup ./run_round17.sh > round17.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round17
mkdir -p $OUT
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
QWEN=hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL
GAMS=cjvt/GaMS3-12B-Instruct
DS=DeepSeek-V4.1-Flash
summ() {
  echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//'), $(grep -E "^Search misses" $2 | sed 's/ (.*//; s/Search misses: */search misses /')"
  grep -E "auto-forward:|score:" $2 | sed 's/^ */      /'
}
people() { sed -n '/Per person/,/^$/p' $1 | grep -E "^ +(jernej|jan|ana) " | tr -s ' ' | tr '\n' ' '; }
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
$PY -m pytest tests -q 2>&1 | tail -1

stamp "baseline: current default (round 16: 896 right, 43 wrong auto-forwards)"
each default

stamp "2. system one chooses from more people"
each cand6 --candidates 6
each cand8 --candidates 8

stamp "3. DeepSeek as a third check model before auto-forwarding"
each check_ds --auto-check-models "$QWEN,$GAMS,$DS"

stamp "1. track record (learning curve, feedback on everything, seed 2)"
curve() {  # name, sets, extra args
  name=$1; sets=$2; shift 2
  ROUTED_COLLECTION=routed_curve17 TRACK_RECORD=$OUT/track_$name.jsonl \
    $PY -u experiments/learn_curve.py $sets --feedback all --seed 2 "$@" > $OUT/curve_$name.log 2>&1
  echo "  $name: $(grep -E "^total" $OUT/curve_$name.log)"
  echo "      $(grep -E "^auto-forwarded" $OUT/curve_$name.log); $(grep -E "^last 100 auto" $OUT/curve_$name.log)"
  grep -E "^held back" $OUT/curve_$name.log | sed 's/^/      /'
  curl -s -X DELETE localhost:6333/collections/routed_curve17 > /dev/null
}
EASY="data/test_emails.csv experiments/data/generated_emails.csv"
HARD="round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
curve easy_off "$EASY"
curve easy_track90 "$EASY" --auto-track-min-precision 0.9
curve hard_off "$HARD"
curve hard_track90 "$HARD" --auto-track-min-precision 0.9
curve hard_track80 "$HARD" --auto-track-min-precision 0.8

stamp "4+5. 'not for them' + new duty lines: learned on one half, tested on the other"
$PY -u experiments/group_split.py $SETS --out-dir $OUT 2>&1 | grep -v -i warn | sed 's/^/  /'
for k in 0 1; do
  t=$((1 - k))
  echo "--- learn on half $((k + 1)), test on half $((t + 1))"
  $PY -u test_routing.py $OUT/fold$k.csv --spam --out $OUT/fold${k}_run.csv > $OUT/fold${k}_learn.log 2>&1
  $PY -u suggest_profiles.py --contrast $OUT/fold${k}_run.csv 2>&1 | grep -v -i warn > $OUT/contrast_$k.log
  grep -E "mistakes between|^ +[0-9]+\. " $OUT/contrast_$k.log | sed 's/^/    /'
  $PY suggest_profiles.py --apply-all --out $OUT/colleagues_c$k.csv > /dev/null
  $PY -u suggest_profiles.py --labels $OUT/fold$k.csv 2>&1 | grep -v -i warn > $OUT/duties_$k.log
  grep -E "labeled emails|^ +[0-9]+\. " $OUT/duties_$k.log | sed 's/^/    /'
  $PY suggest_profiles.py --apply-all --colleagues $OUT/colleagues_c$k.csv --out $OUT/colleagues_cd$k.csv > /dev/null
  $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_before.log 2>&1
  for v in c cd; do
    COLLEAGUES_COLLECTION=JevTry_r17_$v$k $PY ingest_colleagues.py $OUT/colleagues_$v$k.csv > /dev/null
    COLLEAGUES_COLLECTION=JevTry_r17_$v$k $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_$v.log 2>&1
  done
  summ "half $((t + 1)) before            " $OUT/fold${t}_before.log; echo "      $(people $OUT/fold${t}_before.log)"
  summ "half $((t + 1)) + not for them    " $OUT/fold${t}_c.log;      echo "      $(people $OUT/fold${t}_c.log)"
  summ "half $((t + 1)) + both + duties   " $OUT/fold${t}_cd.log;     echo "      $(people $OUT/fold${t}_cd.log)"
  for v in c cd; do curl -s -X DELETE localhost:6333/collections/JevTry_r17_$v$k > /dev/null; done
done
rm -f data/profile_suggestions.csv

stamp "done"
