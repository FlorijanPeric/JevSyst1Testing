#!/usr/bin/env bash
# Clean run on this server: empty the local Qdrant, re-embed all colleagues, test both sets with current defaults.
#   nohup ./run_fresh.sh > fresh.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export QDRANT_URL=http://localhost:6333 QDRANT_API_KEY=
echo "=== flush local Qdrant ($(date +%H:%M)) ==="
for c in $(curl -s localhost:6333/collections | .venv/bin/python -c "import sys, json; print(' '.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))"); do
  curl -s -X DELETE "localhost:6333/collections/$c" > /dev/null && echo "deleted $c"
done
echo "collections left: $(curl -s localhost:6333/collections)"
echo "=== re-embed colleagues ($(date +%H:%M)) ==="
.venv/bin/python ingest_colleagues.py data/colleagues.csv --force 2>&1 | tail -1
mkdir -p fresh
for set in "handwritten data/test_emails.csv" "generated experiments/data/generated_emails.csv"; do
  name=${set%% *}; file=${set#* }
  echo "=== $name ($(date +%H:%M)) ==="
  .venv/bin/python -u test_routing.py "$file" --spam > "fresh/$name.log" 2>&1
  grep -E "^Accuracy|^  (fast|llm |spam) +path|avg system|Search misses|^  manual_triage +[0-9]" "fresh/$name.log"
done
echo "=== done ($(date +%H:%M)) ==="
