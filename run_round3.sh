#!/usr/bin/env bash
# Round 3: everything from the 13-point list, tested on this server.
#   nohup ./run_round3.sh > round3.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p round3
summ() { grep -E "^Accuracy|^  (fast|llm |spam) +path|Search misses|^  manual_triage +[0-9]" "$1"; }

echo "=== install new packages ($(date +%H:%M)) ==="
$PY -m pip install -q -r requirements.txt 2>&1 | tail -2

echo "=== compile all modules ==="
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"

echo "=== unit tests ==="
$PY -m pytest tests -q 2>&1 | tail -5

echo "=== sync profiles (colleagues + nobody_profiles.csv) ($(date +%H:%M)) ==="
$PY ingest_colleagues.py

G=experiments/data/generated_emails.csv
echo "=== handwritten ($(date +%H:%M)) ==="
$PY -u test_routing.py data/test_emails.csv --spam > round3/handwritten.log 2>&1; summ round3/handwritten.log
echo "=== generated, run 1 ($(date +%H:%M)) ==="
$PY -u test_routing.py $G --spam > round3/generated_1.log 2>&1; summ round3/generated_1.log
echo "=== generated, run 2: same setup, answers from the cache ($(date +%H:%M)) ==="
$PY -u test_routing.py $G --spam > round3/generated_2.log 2>&1; summ round3/generated_2.log
echo -n "runs 1 and 2 identical per email: "
diff <(grep "^\[" round3/generated_1.log | cut -c1-90) <(grep "^\[" round3/generated_2.log | cut -c1-90) > /dev/null && echo yes || echo NO

echo "=== CLI speed (one email, a new process each time) ==="
TIMEFORMAT="  cli total %Rs"
for i in 1 2 3; do time ($PY route_email.py --no-learn "The SSL certificate on our website expired" 2>&1 | grep "Forward to"); done

echo "=== service self-test (temporary collection, log, label and learned files) ($(date +%H:%M)) ==="
export COLLEAGUES_COLLECTION=JevTry_selftest ROUTED_COLLECTION=routed_selftest \
       DECISIONS_LOG=round3/api_decisions.jsonl REAL_LABELS=round3/api_labels.csv LEARNED_EXAMPLES=round3/api_learned.csv
rm -f "$DECISIONS_LOG" "$REAL_LABELS" "$LEARNED_EXAMPLES"
$PY ingest_colleagues.py
$PY -m uvicorn service:app --host 127.0.0.1 --port 8091 > round3/service.log 2>&1 &
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
$PY -u tune.py data/test_emails.csv 2>&1 | grep -v -i warn | tail -6
echo "=== done ($(date +%H:%M)) ==="
