#!/usr/bin/env bash
# Round 5: case-based profile edits reverted; memory of past emails (corrections, confirmations, agreed
# decisions) instead. Learning curves show whether accuracy rises and system one is needed less.
#   nohup ./run_round5.sh > round5.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p round5
summ() { grep -E "^Accuracy|^  (fast|llm |spam) +path|spam overruled|Search misses|^  manual_triage +[0-9]" "$1"; }

echo "=== compile + unit tests ($(date +%H:%M)) ==="
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -3

echo "=== sync profiles: round-4 edits reverted ($(date +%H:%M)) ==="
$PY ingest_colleagues.py

G=experiments/data/generated_emails.csv
echo "=== revert check, should match round 3 (95/106, 287/330) ($(date +%H:%M)) ==="
$PY -u test_routing.py data/test_emails.csv --spam > round5/handwritten.log 2>&1; summ round5/handwritten.log
$PY -u test_routing.py $G --spam > round5/generated.log 2>&1; summ round5/generated.log

echo "=== learning curves: 436 emails in a random order, scored before their label is known ==="
export ROUTED_COLLECTION=routed_curve
curve() {
  name=$1; shift
  echo "--- $name ($(date +%H:%M))"
  $PY -u experiments/learn_curve.py data/test_emails.csv $G "$@" > round5/curve_$name.log 2>&1
  tail -8 round5/curve_$name.log
}
curve baseline --no-memory --no-auto-learn --feedback none
curve auto_only --feedback none
curve corrections --feedback corrections
curve corrections_sim70 --feedback corrections --memory-min-sim 0.70
curve corrections_sim80 --feedback corrections --memory-min-sim 0.80
curve corrections_no_auto --feedback corrections --no-auto-learn
curve all_feedback --feedback all
curl -s -X DELETE localhost:6333/collections/routed_curve > /dev/null
unset ROUTED_COLLECTION

echo "=== service self-test (temporary collections and files) ($(date +%H:%M)) ==="
export COLLEAGUES_COLLECTION=JevTry_selftest ROUTED_COLLECTION=routed_selftest \
       DECISIONS_LOG=round5/api_decisions.jsonl REAL_LABELS=round5/api_labels.csv LEARNED_EXAMPLES=round5/api_learned.csv
rm -f "$DECISIONS_LOG" "$REAL_LABELS" "$LEARNED_EXAMPLES"
$PY ingest_colleagues.py
$PY -m uvicorn service:app --host 127.0.0.1 --port 8091 > round5/service.log 2>&1 &
SVC=$!
for i in $(seq 1 60); do curl -s -m 2 localhost:8091/health > /dev/null && break; sleep 1; done
$PY -u experiments/api_selftest.py http://127.0.0.1:8091 2>&1 | grep -v -i warn | sed -n '/^fix/,/^unknown/p'
kill $SVC
curl -s -X DELETE localhost:6333/collections/JevTry_selftest > /dev/null
curl -s -X DELETE localhost:6333/collections/routed_selftest > /dev/null
echo "=== done ($(date +%H:%M)) ==="
