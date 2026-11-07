#!/usr/bin/env bash
# Round 4: Jernej/Aleksander/Sara profiles, CEO hint in the question, spam override, summary and tune fixes.
#   nohup ./run_round4.sh > round4.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p round4
summ() { grep -E "^Accuracy|^  (fast|llm |spam) +path|spam overruled|Search misses|^  (manual_triage|jernej|aleksander|sara) +[0-9]" "$1"; }

echo "=== install new packages ($(date +%H:%M)) ==="
$PY -m pip install -q -r requirements.txt 2>&1 | tail -2

echo "=== compile all modules ==="
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"

echo "=== unit tests ==="
$PY -m pytest tests -q 2>&1 | tail -5

echo "=== sync profiles: jernej, aleksander, sara changed ($(date +%H:%M)) ==="
$PY ingest_colleagues.py

G=experiments/data/generated_emails.csv
echo "=== handwritten ($(date +%H:%M)) ==="
$PY -u test_routing.py data/test_emails.csv --spam > round4/handwritten.log 2>&1; summ round4/handwritten.log
echo "=== generated, run 1 ($(date +%H:%M)) ==="
$PY -u test_routing.py $G --spam > round4/generated_1.log 2>&1; summ round4/generated_1.log
echo "=== mistakes, generated ==="
sed -n "/^Most common mistakes/,\$p" round4/generated_1.log


echo "=== service self-test (temporary collection, log, label and learned files) ($(date +%H:%M)) ==="
export COLLEAGUES_COLLECTION=JevTry_selftest ROUTED_COLLECTION=routed_selftest \
       DECISIONS_LOG=round4/api_decisions.jsonl REAL_LABELS=round4/api_labels.csv LEARNED_EXAMPLES=round4/api_learned.csv
rm -f "$DECISIONS_LOG" "$REAL_LABELS" "$LEARNED_EXAMPLES"
$PY ingest_colleagues.py
$PY -m uvicorn service:app --host 127.0.0.1 --port 8091 > round4/service.log 2>&1 &
SVC=$!
for i in $(seq 1 60); do curl -s -m 2 localhost:8091/health > /dev/null && break; sleep 1; done
$PY -u experiments/api_selftest.py http://127.0.0.1:8091 2>&1 | grep -v -i warn
echo "learned examples file:"; cat "$LEARNED_EXAMPLES"
echo "labels file:"; cat "$REAL_LABELS"
kill $SVC
curl -s -X DELETE localhost:6333/collections/JevTry_selftest > /dev/null
curl -s -X DELETE localhost:6333/collections/routed_selftest > /dev/null
unset COLLEAGUES_COLLECTION ROUTED_COLLECTION DECISIONS_LOG REAL_LABELS LEARNED_EXAMPLES

echo "=== tune on the handwritten test set (nothing written) ($(date +%H:%M)) ==="
$PY -u tune.py data/test_emails.csv 2>&1 | grep -v -i warn | tail -7
echo "=== done ($(date +%H:%M)) ==="
