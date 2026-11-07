#!/usr/bin/env bash
# Round 6: memory_min_sim 0.70, and the paraphrase test: does one correction carry over to reworded emails?
#   nohup ./run_round6.sh > round6.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p round6
echo "=== compile + unit tests ($(date +%H:%M)) ==="
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -2

export ROUTED_COLLECTION=routed_para
echo "=== paraphrase test ($(date +%H:%M)) ==="
$PY -u experiments/paraphrase_test.py data/test_emails.csv experiments/data/generated_emails.csv 2>&1 | grep --line-buffered -v -i warn

export ROUTED_COLLECTION=routed_curve
echo "=== learning curve with the new default 0.70, corrections + confirmations ($(date +%H:%M)) ==="
$PY -u experiments/learn_curve.py data/test_emails.csv experiments/data/generated_emails.csv --feedback all 2>&1 | tail -8
curl -s -X DELETE localhost:6333/collections/routed_curve > /dev/null
curl -s -X DELETE localhost:6333/collections/routed_para > /dev/null
echo "=== done ($(date +%H:%M)) ==="
