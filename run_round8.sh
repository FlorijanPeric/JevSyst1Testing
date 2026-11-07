#!/usr/bin/env bash
# Round 8: paraphrase test with rewordings from non-Qwen models: GaMS (Slovenian model) for Slovenian emails,
# Gemma for English ones. Separate from round 7 (DeepSeek): own cache file and own temporary collection.
#   nohup ./run_round8.sh > round8.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
export ROUTED_COLLECTION=routed_para8
echo "=== paraphrase test, Gemma (English) + GaMS (Slovenian) rewordings ($(date '+%d.%m. %H:%M')) ==="
$PY -u experiments/paraphrase_test.py data/test_emails.csv experiments/data/generated_emails.csv \
    --models gemma-4-12b --models-sl cjvt/GaMS3-12B-Instruct --cache paraphrases_gemma_gams.csv \
    2>&1 | grep --line-buffered -v -i warn
curl -s -X DELETE localhost:6333/collections/routed_para8 > /dev/null
echo "=== done ($(date '+%d.%m. %H:%M')) ==="
