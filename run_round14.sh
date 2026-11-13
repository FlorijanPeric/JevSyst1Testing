#!/usr/bin/env bash
# Round 14: four fixes on top of the default (gemma on every system one decision, Qwen3-30B + GaMS check auto-forward)
#   1. "not for them" lines from mistakes (shown to the models, not embedded), learned on one half, tested on the other
#   2. the second opinion's "nobody" only counts when system one also leaned that way
#   3. stricter auto-forward for clear matches: bigger lead over #2, or a check model must agree
#   4. learning curve with the current default
#   nohup ./run_round14.sh > round14.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round14
mkdir -p $OUT
SETS="data/test_emails.csv experiments/data/generated_emails.csv round11/reworded_qwen.csv round11/reworded_gemma_gams.csv"
QWEN=hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL
GAMS=cjvt/GaMS3-12B-Instruct
summ() {
  echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//'), $(grep -E "^Search misses" $2 | sed 's/ (.*//; s/Search misses: */search misses /')$(grep -E "second opinion:" $2 | sed 's/ *second opinion: */; second opinion /')"
  grep -E "auto-forward:" $2 | sed 's/^ */      /'
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

stamp "2. second opinion 'nobody' only when system one gave nobody at least ..."
for m in 0.2 0.4; do each nobody_min$m --second-opinion-nobody-min $m; done

stamp "3. clear matches: auto-forward only with a bigger lead, or when a check model agrees"
for m in 0.10 0.15; do each fast_margin$m --auto-fast-min-margin $m; done
each fast_check_gams --auto-fast-check-models "$GAMS"
each fast_check_qwen_gams --auto-fast-check-models "$QWEN,$GAMS"

stamp "1. 'not for them' lines: learned from the mistakes on one half, tested on the other half"
$PY - <<'PYEOF'
import csv
rows = []
for path in ("data/test_emails.csv", "experiments/data/generated_emails.csv"):
    with open(path, encoding="utf-8", newline="") as f:
        rows += [{"expected": r["expected"], "email": r["email"]} for r in csv.DictReader(f) if r.get("email")]
for k in (0, 1):
    with open(f"round14/fold{k}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["expected", "email"]); w.writeheader(); w.writerows(rows[k::2])
print(f"  {len(rows)} emails -> two halves of {len(rows[0::2])} and {len(rows[1::2])}")
PYEOF
for k in 0 1; do
  t=$((1 - k))
  echo "--- learn on half $((k + 1)), test on half $((t + 1))"
  $PY -u test_routing.py $OUT/fold$k.csv --spam --out $OUT/fold${k}_run.csv > $OUT/fold${k}_learn.log 2>&1
  $PY -u suggest_profiles.py --contrast $OUT/fold${k}_run.csv 2>&1 | grep -v -i warn | tee $OUT/contrast_$k.log \
      | grep -E "mistakes between|^ +[0-9]+\. " | sed 's/^/    /'
  $PY suggest_profiles.py --apply-all --out $OUT/colleagues_c$k.csv > /dev/null
  $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_before.log 2>&1
  COLLEAGUES_COLLECTION=JevTry_contrast$k $PY ingest_colleagues.py $OUT/colleagues_c$k.csv > /dev/null
  COLLEAGUES_COLLECTION=JevTry_contrast$k $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_after.log 2>&1
  summ "half $((t + 1)) before" $OUT/fold${t}_before.log; summ "half $((t + 1)) after " $OUT/fold${t}_after.log
  sed -n '/Per person/,/^$/p' $OUT/fold${t}_after.log | grep -E "jernej|eva|sara|nina" | sed 's/^/    after: /'
  sed -n '/Per person/,/^$/p' $OUT/fold${t}_before.log | grep -E "jernej|eva|sara|nina" | sed 's/^/    before:/'
  curl -s -X DELETE localhost:6333/collections/JevTry_contrast$k > /dev/null
done
rm -f data/profile_suggestions.csv

stamp "4. learning curve with the current default (memory + feedback, seed 2; round 11: 386 off, 403 gemma unclear)"
for fb in all corrections; do
  ROUTED_COLLECTION=routed_curve14 $PY -u experiments/learn_curve.py data/test_emails.csv \
      experiments/data/generated_emails.csv --feedback $fb --seed 2 > $OUT/curve_$fb.log 2>&1
  echo "  curve $fb: $(grep -E "^total" $OUT/curve_$fb.log); $(grep -E "^memory decisions" $OUT/curve_$fb.log)"
  echo "      $(grep -E "^last 100 emails" $OUT/curve_$fb.log); $(grep -E "^auto-forwarded" $OUT/curve_$fb.log); $(grep -E "^last 100 auto" $OUT/curve_$fb.log)"
done
curl -s -X DELETE localhost:6333/collections/routed_curve14 > /dev/null

stamp "done"
