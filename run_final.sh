#!/usr/bin/env bash
# Final test: everything the router claims, checked in one run, with a PASS/FAIL report at the end.
#   nohup ./run_final.sh > final.log 2>&1 < /dev/null &
# Logs go to final/, the report to final/REPORT.md. Only temporary collections and files are written;
# the real JevTry profiles, memory (routed_emails), decision log and labels are not touched.
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=final
rm -rf ${OUT}_prev; [ -d $OUT ] && mv $OUT ${OUT}_prev  # keep the previous run to compare
mkdir -p $OUT
G=experiments/data/generated_emails.csv
H=data/test_emails.csv
quiet() { grep --line-buffered -v -i warn; }
stamp() { echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "1. code health: compile + unit tests"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile" | tee $OUT/compile.log
$PY -m pytest tests -q 2>&1 | tail -3 | tee $OUT/unit.log

stamp "2. profiles are up to date"
$PY ingest_colleagues.py | tee $OUT/ingest.log

stamp "3. accuracy on a fresh system (no memory yet)"
$PY -u test_routing.py $H --spam > $OUT/handwritten.log 2>&1; grep -E "^Accuracy|path:|Search misses" $OUT/handwritten.log
$PY -u test_routing.py $G --spam > $OUT/generated.log 2>&1; grep -E "^Accuracy|path:|Search misses" $OUT/generated.log

stamp "4. learning over time (436 emails as an inbox stream, scored before the label is known)"
export ROUTED_COLLECTION=routed_final
$PY -u experiments/learn_curve.py $H $G --no-memory --no-auto-learn --feedback none 2>&1 | quiet > $OUT/curve_none.log
$PY -u experiments/learn_curve.py $H $G --feedback corrections 2>&1 | quiet > $OUT/curve_corrections.log
$PY -u experiments/learn_curve.py $H $G --feedback all 2>&1 | quiet > $OUT/curve_all.log
tail -n 4 $OUT/curve_none.log $OUT/curve_corrections.log $OUT/curve_all.log

stamp "5. one correction carries over to reworded emails (two rewording sets, both cached)"
# With hints (system one sees the similar corrected email) and without (memory votes only).
for hints in memory-hints no-memory-hints; do
  $PY -u experiments/paraphrase_test.py $H $G --cache paraphrases.csv --$hints 2>&1 | quiet > $OUT/para_qwen_$hints.log
  $PY -u experiments/paraphrase_test.py $H $G --cache paraphrases_gemma_gams.csv --$hints \
      --models gemma-4-12b --models-sl cjvt/GaMS3-12B-Instruct 2>&1 | quiet > $OUT/para_gemma_gams_$hints.log
done
grep -E "before any fix|^ +0\.70 +0\.5 " $OUT/para_*.log

stamp "6. system one down: routing must still work (search only)"
# A model name that is never cached and an address that refuses connections, only for system one.
# Only unclear emails (no clear match), so each of them would really need system one.
for email in "Hi, how was your holiday at the seaside?" \
             "Can somebody look at this, not sure who handles it: the projector and the wifi in room 3 both fail" \
             "Kdo lahko pregleda pogodbo s partnerjem do petka?"; do
  CHAT_MODEL=systemone/down-test HIVECORE_BASE_URL=http://127.0.0.1:9 \
    $PY route_email.py --no-learn --no-memory "$email" 2>&1 | grep -E "Forward to|Decided by"
done | tee $OUT/down.log

stamp "7. web API end to end (temporary collections and files)"
export COLLEAGUES_COLLECTION=JevTry_final ROUTED_COLLECTION=routed_final_api \
       DECISIONS_LOG=$OUT/api_decisions.jsonl REAL_LABELS=$OUT/api_labels.csv LEARNED_EXAMPLES=$OUT/api_learned.csv
$PY ingest_colleagues.py > /dev/null
# System one answers not cached (embeddings still are), so system one is really asked and its speed is real.
CACHE_SYSTEMONE=0 $PY -m uvicorn service:app --host 127.0.0.1 --port 8092 > $OUT/service.log 2>&1 &
SVC=$!
for i in $(seq 1 60); do curl -s -m 2 localhost:8092/health > /dev/null && break; sleep 1; done
$PY -u experiments/api_selftest.py http://127.0.0.1:8092 2>&1 | quiet | tee $OUT/api.log | grep -E "health|->|learning|rejected"
kill $SVC
for c in JevTry_final routed_final_api routed_final; do curl -s -X DELETE localhost:6333/collections/$c > /dev/null; done
unset COLLEAGUES_COLLECTION ROUTED_COLLECTION DECISIONS_LOG REAL_LABELS LEARNED_EXAMPLES

stamp "8. report"
$PY experiments/final_report.py $OUT | tee $OUT/REPORT.md
stamp "done"
