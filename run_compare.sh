#!/usr/bin/env bash
# Side-by-side: ranking (window vs grouped) x spam filter, on both test sets. Runs one after another.
#   nohup ./run_compare.sh > compare.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
mkdir -p compare
run() {  # name, test file, extra args
  echo "=== $1 ($(date +%H:%M)) ==="
  .venv/bin/python -u test_routing.py "$2" "${@:3}" > "compare/$1.log" 2>&1
  grep -E "^Accuracy|fast path|llm  path|spam path|manual_triage  " "compare/$1.log"
}
G=experiments/data/generated_emails.csv
H=data/test_emails.csv
run generated_grouped        $G --ranking grouped
run generated_window_spam    $G --ranking window  --spam
run generated_grouped_spam   $G --ranking grouped --spam
run handwritten_window_spam  $H --ranking window  --spam
run handwritten_grouped_spam $H --ranking grouped --spam
echo "=== done ($(date +%H:%M)) ==="
