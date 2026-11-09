#!/usr/bin/env bash
# Round 9: four accuracy ideas, each against the same baseline.
#   A. calibration: how often system one is right per confidence -> auto_min_confidence
#   B. qwen3-reranker re-sorts a bigger search pool before system one
#   C. qwen3-embedding instead of bge-m3 (own collection), with and without its query instruction
#   D. profile suggestions learned from labels: learned on one half of the emails, tested on the other half
#   nohup ./run_round9.sh > round9.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round9
mkdir -p $OUT
H=data/test_emails.csv
G=experiments/data/generated_emails.csv
summ() { echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in .*//; s/Accuracy: *//'), $(grep -E "^Search misses" $2 | sed 's/ (.*//; s/Search misses: */search misses /'), $(grep -E "fast path" $2 | sed 's/ *fast path: */fast /; s/ correct//')"; }
both() {  # name, extra args...: run the handwritten and the generated set
  name=$1; shift
  $PY -u test_routing.py $H --spam "$@" > $OUT/${name}_h.log 2>&1
  $PY -u test_routing.py $G --spam "$@" > $OUT/${name}_g.log 2>&1
  summ "$name handwritten" $OUT/${name}_h.log; summ "$name generated  " $OUT/${name}_g.log
}
stamp() { echo; echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "0. compile + unit tests"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -2

stamp "A. baseline + calibration of system one confidence"
both baseline
$PY -u test_routing.py $H --spam --out $OUT/calib_h.csv > /dev/null 2>&1
$PY -u test_routing.py $G --spam --out $OUT/calib_g.csv > /dev/null 2>&1
$PY -u experiments/calibration.py $OUT/calib_h.csv $OUT/calib_g.csv 2>&1 | grep -v -i warn

stamp "B. reranker (qwen3-reranker) before system one"
both rerank10 --rerank --rerank-pool 10
both rerank_all --rerank --rerank-pool 18

stamp "C. qwen3-embedding vs bge-m3 (fast path off = search quality only, then normal)"
both bge_nofast --min-score 9
export EMBEDDING_API_MODEL=qwen3-embedding:latest EMBEDDING_FALLBACK_MODEL=qwen3-embedding:latest COLLEAGUES_COLLECTION=JevTry_qwen3emb
$PY ingest_colleagues.py | sed 's/(.*)//'
both qwen3_nofast --min-score 9
both qwen3
export EMBEDDING_QUERY_PREFIX='Instruct: Given an email sent to a company, find the colleague whose job is to handle it\nQuery: '
both qwen3_instr_nofast --min-score 9
both qwen3_instr
curl -s -X DELETE localhost:6333/collections/JevTry_qwen3emb > /dev/null
unset EMBEDDING_API_MODEL EMBEDDING_FALLBACK_MODEL COLLEAGUES_COLLECTION EMBEDDING_QUERY_PREFIX

stamp "D. profile suggestions: learned on one half, tested on the other"
$PY - <<'PYEOF'
import csv
rows = []
for path in ("data/test_emails.csv", "experiments/data/generated_emails.csv"):
    with open(path, encoding="utf-8", newline="") as f:
        rows += [{"expected": r["expected"], "email": r["email"]} for r in csv.DictReader(f) if r.get("email")]
for k in (0, 1):
    with open(f"round9/fold{k}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["expected", "email"]); w.writeheader(); w.writerows(rows[k::2])
print(f"{len(rows)} emails -> two halves of {len(rows[0::2])} and {len(rows[1::2])}")
PYEOF
for k in 0 1; do
  t=$((1 - k))
  echo "--- learn on half $((k + 1)), test on half $((t + 1))"
  $PY -u suggest_profiles.py --labels $OUT/fold$k.csv 2>&1 | grep -v -i warn | tee $OUT/suggest_$k.log | grep -E "labeled emails|^ +[0-9]+\. "
  cp data/profile_suggestions.csv $OUT/suggestions_$k.csv 2>/dev/null
  $PY suggest_profiles.py --apply-all --out $OUT/colleagues_s$k.csv > /dev/null
  $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_before.log 2>&1
  COLLEAGUES_COLLECTION=JevTry_sugg$k $PY ingest_colleagues.py $OUT/colleagues_s$k.csv > /dev/null
  COLLEAGUES_COLLECTION=JevTry_sugg$k $PY -u test_routing.py $OUT/fold$t.csv --spam > $OUT/fold${t}_after.log 2>&1
  summ "half $((t + 1)) before" $OUT/fold${t}_before.log; summ "half $((t + 1)) after " $OUT/fold${t}_after.log
  curl -s -X DELETE localhost:6333/collections/JevTry_sugg$k > /dev/null
done
rm -f data/profile_suggestions.csv

stamp "done"
