#!/usr/bin/env bash
# Round 11: shake up the new default (gemma-4-12b second opinion on unclear decisions).
#   1. fresh: system one and gemma really asked (no cached answers), twice -> how much is noise
#   2. unseen emails: the reworded versions (Qwen3-30B, and Gemma + GaMS) of the labeled emails
#   3. margin: what counts as "unclear" (0.10 / 0.15 / 0.25)
#   4. learning curve with memory + second opinion, new email order (seed 2)
#   nohup ./run_round11.sh > round11.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round11
mkdir -p $OUT
H=data/test_emails.csv
G=experiments/data/generated_emails.csv
summ() { echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in /, /; s/Accuracy: *//'), $(grep -E "llm  path" $2 | sed 's/ *llm  path: */system one path /; s/ correct//')$(grep -E "second opinion:" $2 | sed 's/ *second opinion: */; second opinion /')"; }
run() {  # name, file, extra args...
  name=$1; file=$2; shift 2; log="$OUT/${name// /}.log"
  $PY -u test_routing.py $file --spam "$@" > "$log" 2>&1; summ "$name" "$log"
}
stamp() { echo; echo "=== $1 ($(date '+%d.%m. %H:%M')) ==="; }

stamp "0. compile + unit tests"
$PY -m py_compile *.py experiments/*.py tests/*.py && echo "all compile"
$PY -m pytest tests -q 2>&1 | tail -2

stamp "1. fresh answers (CACHE_SYSTEMONE=0), two repeats each"
for rep in 1 2; do
  for mode in off unclear; do
    CACHE_SYSTEMONE=0 run "fresh${rep}_${mode}_handwritten" $H --second-opinion $mode
    CACHE_SYSTEMONE=0 run "fresh${rep}_${mode}_generated  " $G --second-opinion $mode
  done
done

stamp "2. unseen emails: rewordings of the labeled emails"
$PY - <<'PYEOF'
import csv
label = {}
for path in ("data/test_emails.csv", "experiments/data/generated_emails.csv"):
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("email"):
                label[" ".join(r["email"].split())] = r["expected"]
for src, dst in (("paraphrases.csv", "reworded_qwen.csv"), ("paraphrases_gemma_gams.csv", "reworded_gemma_gams.csv")):
    with open(f"experiments/data/{src}", encoding="utf-8", newline="") as f:
        rows = [{"expected": label[k], "email": r["paraphrase"]} for r in csv.DictReader(f)
                if (k := " ".join(r["original"].split())) in label and r["paraphrase"].strip()]
    rows = list({r["email"]: r for r in rows}.values())  # same rewording twice counts once
    with open(f"round11/{dst}", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["expected", "email"]); w.writeheader(); w.writerows(rows)
    print(f"  {dst}: {len(rows)} reworded emails")
PYEOF
for set in qwen gemma_gams; do
  for mode in off unclear; do
    CACHE_SYSTEMONE=0 run "reworded_${set}_${mode}" $OUT/reworded_$set.csv --second-opinion $mode
  done
done

stamp "3. margin for unclear (cached answers where possible)"
for m in 0.10 0.25; do
  run "margin${m}_handwritten" $H --second-opinion-margin $m
  run "margin${m}_generated  " $G --second-opinion-margin $m
done

stamp "4. learning curve: memory + feedback on everything, new order (seed 2)"
for mode in off unclear; do
  ROUTED_COLLECTION=routed_curve11 $PY -u experiments/learn_curve.py $H $G --feedback all --seed 2 \
      --second-opinion $mode > $OUT/curve_$mode.log 2>&1
  echo "  curve $mode: $(grep -E "^total" $OUT/curve_$mode.log); $(grep -E "^memory decisions" $OUT/curve_$mode.log); $(grep -E "^last 100" $OUT/curve_$mode.log)"
done
curl -s -X DELETE localhost:6333/collections/routed_curve11 > /dev/null

stamp "done"
