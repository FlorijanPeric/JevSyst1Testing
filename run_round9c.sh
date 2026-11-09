#!/usr/bin/env bash
# Round 9, part C again: the qwen3-embedding runs that hivecore 502 errors broke.
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=round9
H=data/test_emails.csv
G=experiments/data/generated_emails.csv
summ() { echo "  $1: $(grep -E "^Accuracy" $2 | sed 's/  in .*//; s/Accuracy: *//'), $(grep -E "^Search misses" $2 | sed 's/ (.*//; s/Search misses: */search misses /'), $(grep -E "fast path" $2 | sed 's/ *fast path: */fast /; s/ correct//'), $(grep -E "search-only path" $2 | sed 's/ *search-only path: */system one failed: /; s/ correct//')"; }
one() { name=$1; set=$2; shift 2; $PY -u test_routing.py $set --spam "$@" > $OUT/$name.log 2>&1; summ "$name" $OUT/$name.log; }
echo "=== C again ($(date '+%d.%m. %H:%M')) ==="
export EMBEDDING_API_MODEL=qwen3-embedding:latest EMBEDDING_FALLBACK_MODEL=qwen3-embedding:latest COLLEAGUES_COLLECTION=JevTry_qwen3emb
$PY ingest_colleagues.py | sed 's/(.*)//'
one qwen3_nofast_h $H --min-score 9
one qwen3_nofast_g $G --min-score 9
export EMBEDDING_QUERY_PREFIX='Instruct: Given an email sent to a company, find the colleague whose job is to handle it\nQuery: '
one qwen3_instr_nofast_g $G --min-score 9
curl -s -X DELETE localhost:6333/collections/JevTry_qwen3emb > /dev/null
echo "=== done ($(date '+%d.%m. %H:%M')) ==="
