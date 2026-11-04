#!/usr/bin/env bash
# Test the "nobody" profiles: flush local Qdrant, re-embed, test both sets (+ two manual thresholds).
#   nohup ./run_nobody.sh > nobody.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export QDRANT_URL=http://localhost:6333 QDRANT_API_KEY=
for f in route_email.py test_routing.py rank_queries.py ingest_colleagues.py; do
  .venv/bin/python -c "import ast; ast.parse(open('$f').read())" || { echo "syntax error in $f"; exit 1; }
done
echo "=== flush local Qdrant ($(date +%H:%M)) ==="
for c in $(curl -s localhost:6333/collections | .venv/bin/python -c "import sys, json; print(' '.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))"); do
  curl -s -X DELETE "localhost:6333/collections/$c" > /dev/null && echo "deleted $c"
done
echo "=== re-embed colleagues ($(date +%H:%M)) ==="
.venv/bin/python ingest_colleagues.py data/colleagues.csv --force 2>&1 | tail -1
mkdir -p nobody
run() {  # name, file, extra args
  echo "=== $1 ($(date +%H:%M)) ==="
  .venv/bin/python -u test_routing.py "$2" --spam "${@:3}" > "nobody/$1.log" 2>&1
  grep -E "^Accuracy|^  (fast|llm |spam) +path|Search misses|^  manual_triage +[0-9]" "nobody/$1.log"
}
run handwritten  data/test_emails.csv
run generated    experiments/data/generated_emails.csv
run generated_manual06 experiments/data/generated_emails.csv --manual-threshold 0.6
run generated_manual08 experiments/data/generated_emails.csv --manual-threshold 0.8
echo "=== done ($(date +%H:%M)) ==="
