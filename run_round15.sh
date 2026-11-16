#!/usr/bin/env bash
# Round 15: what gemma (second opinion) sees and how it answers, plus shadow mode end to end.
#   1. full profiles: every duty line instead of the two best-matching ones
#   2. a longer candidate list for gemma (8 instead of system one's 5)
#   3. a little reasoning before the answer: "brief" (one sentence) or "low" (the model's own thinking)
#   4. shadow mode: route the test emails as if they were a real inbox, review them from the labels, report
#   nohup ./run_round15.sh > round15.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round15
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

stamp "1. full profiles for gemma"
each full --second-opinion-full-profiles

stamp "2. gemma chooses from 8 people"
each wide8 --second-opinion-candidates 8
each wide8_full --second-opinion-candidates 8 --second-opinion-full-profiles

stamp "3. reasoning before the answer"
each brief --second-opinion-reasoning brief
each low --second-opinion-reasoning low

stamp "4. shadow mode end to end (own log, labels and memory; reviews answered from the labels)"
export DECISIONS_LOG=$OUT/shadow_decisions.jsonl REAL_LABELS=$OUT/shadow_labels.csv ROUTED_COLLECTION=routed_shadow15 \
       SHADOW_REVIEWS=$OUT/shadow_reviews.jsonl SHADOW_RUN_CSV=$OUT/shadow_run.csv
rm -f $DECISIONS_LOG $REAL_LABELS $SHADOW_REVIEWS
curl -s -X DELETE localhost:6333/collections/routed_shadow15 > /dev/null
for f in data/test_emails.csv experiments/data/generated_emails.csv; do
  echo "--- week with $(basename $f): route, review, report"
  $PY -u shadow.py $f > $OUT/shadow_route_$(basename $f .csv).log 2>&1; head -1 $OUT/shadow_route_$(basename $f .csv).log | sed 's/^/  /'
  $PY -u shadow.py --review --labels $f > $OUT/shadow_review_$(basename $f .csv).log 2>&1
  tail -n 1 $OUT/shadow_review_$(basename $f .csv).log | sed 's/^/  last review: /'
  $PY -u shadow.py --report 2>&1 | sed 's/^/  /'
done
$PY -u shadow.py experiments/data/generated_emails.csv > $OUT/shadow_again.log 2>&1
head -1 $OUT/shadow_again.log | sed 's/^/  again (nothing new should be routed): /'
curl -s -X DELETE localhost:6333/collections/routed_shadow15 > /dev/null
unset DECISIONS_LOG REAL_LABELS ROUTED_COLLECTION SHADOW_REVIEWS SHADOW_RUN_CSV

stamp "done"
