#!/usr/bin/env bash
# After run_compare.sh: switch to the Qdrant on this server, test the tie-break, flag doubtful labels.
#   nohup ./run_next.sh > next.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
while pgrep -f run_compare.sh > /dev/null; do sleep 30; done

# Local Qdrant (docker container "qdrant", 127.0.0.1:6333); overrides QDRANT_URL/KEY from .env
export QDRANT_URL=http://localhost:6333 QDRANT_API_KEY=
mkdir -p compare
run() {  # name, test file, extra args
  echo "=== $1 ($(date +%H:%M)) ==="
  .venv/bin/python -u test_routing.py "$2" "${@:3}" > "compare/$1.log" 2>&1
  grep -E "^Accuracy|fast path|llm  path|spam path|manual_triage  " "compare/$1.log"
}
echo "=== load colleagues into local Qdrant ($(date +%H:%M)) ==="
.venv/bin/python ingest_colleagues.py data/colleagues.csv 2>&1 | tail -1

G=experiments/data/generated_emails.csv
H=data/test_emails.csv
run local_generated_window_spam           $G --ranking window --spam
run local_generated_window_spam_tiebreak  $G --ranking window --spam --tiebreak
run local_handwritten_window_spam_tiebreak $H --ranking window --spam --tiebreak

echo "=== flag doubtful labels ($(date +%H:%M)) ==="
.venv/bin/python -u experiments/flag_labels.py 2>&1 | grep -v -i warn
echo "=== done ($(date +%H:%M)) ==="
