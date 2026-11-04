#!/usr/bin/env bash
# Round 2: nobody candidates "ranked" (new default) vs "always", and the system-one-down fallback with an unclear email.
#   nohup ./run_round2.sh > round2.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
for f in route_email.py test_routing.py rank_queries.py; do
  $PY -c "import ast; ast.parse(open('$f').read())" || { echo "syntax error in $f"; exit 1; }
done
mkdir -p round2
summ() { grep -E "^Accuracy|^  (fast|llm |spam) +path|Search misses|^  manual_triage +[0-9]" "$1"; }
G=experiments/data/generated_emails.csv
echo "=== generated, nobody ranked + top-up ($(date +%H:%M)) ==="
$PY -u test_routing.py $G --spam --nobody-candidates ranked > round2/generated_ranked.log 2>&1; summ round2/generated_ranked.log
echo "=== generated, nobody always (repeat, to see run-to-run noise) ($(date +%H:%M)) ==="
$PY -u test_routing.py $G --spam --nobody-candidates always > round2/generated_always.log 2>&1; summ round2/generated_always.log
echo "=== handwritten, nobody ranked + top-up ($(date +%H:%M)) ==="
$PY -u test_routing.py data/test_emails.csv --spam --nobody-candidates ranked > round2/handwritten_ranked.log 2>&1; summ round2/handwritten_ranked.log

echo "=== fallback: system one down, unclear email ($(date +%H:%M)) ==="
export DECISIONS_LOG=round2/selftest_decisions.jsonl REAL_LABELS=round2/selftest_labels.csv ROUTED_COLLECTION=routed_selftest
rm -f "$DECISIONS_LOG"
CHAT_MODEL=systemone/does-not-exist SYSTEMONE_TIMEOUT=5 $PY route_email.py --no-learn \
  "Hi, our tablet app shows the website login page instead of the app screen, who can help?" 2>&1 | grep -E "Forward to|Decided by|Confidence|fallback"
echo "logged path: $(grep -o '"path": "[^"]*"' $DECISIONS_LOG)"
curl -s -X DELETE localhost:6333/collections/routed_selftest > /dev/null
echo "=== done ($(date +%H:%M)) ==="
