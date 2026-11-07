#!/usr/bin/env bash
# Round 7: paraphrase test again with rewordings by DeepSeek only (own cache file). DeepSeek had no worker on
# hivecore on 2026-11-07, so this waits until hivecore lists it again (checks every 5 min, gives up after 48 h).
#   nohup ./run_round7.sh > round7.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=.venv/bin/python
MODEL=DeepSeek-V4-Flash
listed() {
  $PY - "$MODEL" <<'PYEOF' 2>/dev/null
import sys, config
from openai import OpenAI
llm = OpenAI(base_url=config.HIVECORE_BASE_URL, api_key=config.HIVECORE_API_KEY, timeout=30, max_retries=0)
sys.exit(0 if sys.argv[1] in {m.id for m in llm.models.list().data} else 1)
PYEOF
}
echo "=== waiting for $MODEL on hivecore ($(date '+%d.%m. %H:%M')) ==="
for i in $(seq 1 576); do
  listed && break
  [ "$i" = 576 ] && { echo "gave up: $MODEL not listed for 48 h ($(date '+%d.%m. %H:%M'))"; exit 1; }
  sleep 300
done
echo "$MODEL is back ($(date '+%d.%m. %H:%M'))"

export ROUTED_COLLECTION=routed_para
echo "=== paraphrase test, DeepSeek rewordings ($(date '+%d.%m. %H:%M')) ==="
$PY -u experiments/paraphrase_test.py data/test_emails.csv experiments/data/generated_emails.csv \
    --models $MODEL --cache paraphrases_deepseek.csv 2>&1 | grep --line-buffered -v -i warn
curl -s -X DELETE localhost:6333/collections/routed_para > /dev/null
echo "=== done ($(date '+%d.%m. %H:%M')) ==="
