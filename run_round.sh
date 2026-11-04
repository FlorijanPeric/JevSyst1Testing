#!/usr/bin/env bash
# Test round on this server (uses .env.local -> local Qdrant): test sets + fallback and logging self-tests.
#   nohup ./run_round.sh > round.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
for f in route_email.py test_routing.py rank_queries.py ingest_colleagues.py clients.py config.py summary.py inbox.py; do
  $PY -c "import ast; ast.parse(open('$f').read())" || { echo "syntax error in $f"; exit 1; }
done
echo "Qdrant in use: $($PY -c 'import config; print(config.QDRANT_URL)')"
mkdir -p round
echo "=== reload colleagues ($(date +%H:%M)) ==="
$PY ingest_colleagues.py data/colleagues.csv 2>&1 | tail -1

summ() { grep -E "^Accuracy|^  (fast|llm |spam) +path|Search misses|^  manual_triage +[0-9]" "$1"; }
echo "=== handwritten ($(date +%H:%M)) ==="
$PY -u test_routing.py data/test_emails.csv --spam > round/handwritten.log 2>&1; summ round/handwritten.log
echo "=== generated ($(date +%H:%M)) ==="
$PY -u test_routing.py experiments/data/generated_emails.csv --spam > round/generated.log 2>&1; summ round/generated.log

# ---- self-tests on temporary log/label files and a temporary cache collection ----
export DECISIONS_LOG=round/selftest_decisions.jsonl REAL_LABELS=round/selftest_labels.csv ROUTED_COLLECTION=routed_selftest
rm -f "$DECISIONS_LOG" "$REAL_LABELS"
show() { grep -E "Forward to|Decided by|Confidence|fallback" ; }

echo "=== fallback: system one down ($(date +%H:%M)) ==="
CHAT_MODEL=systemone/does-not-exist SYSTEMONE_TIMEOUT=5 $PY route_email.py --no-learn "Our Android app crashes when opening the camera" 2>&1 | show

echo "=== fallback: embedding endpoint down ==="
EMBEDDING_API_URL=http://hivecore.famnit.upr.si:6666/api/does-not-exist EMBEDDING_API_RETRIES=0 \
  $PY route_email.py --no-learn "The 3D printer is jammed again" 2>&1 | show

echo "=== logging, ok, fix, summary ==="
$PY route_email.py "Please renew the SSL certificate on our website" 2>&1 | grep "Forward to"
$PY route_email.py "Can you book a hotel in Vienna for the conference?" 2>&1 | grep "Forward to"
$PY route_email.py --ok "Please renew the SSL certificate on our website"
$PY route_email.py --fix jan "The iOS app layout breaks on the new iPad"
$PY route_email.py --no-learn "The iOS app layout breaks on the new iPad" 2>&1 | grep -E "Forward to|Decided by"
echo "decisions logged: $(wc -l < $DECISIONS_LOG)"; echo "labels:"; cat "$REAL_LABELS"
$PY summary.py --days 1
curl -s -X DELETE localhost:6333/collections/routed_selftest > /dev/null
echo "=== done ($(date +%H:%M)) ==="
