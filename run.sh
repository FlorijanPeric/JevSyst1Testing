#!/usr/bin/env bash
# Email router helper.
#   ./run.sh                     load colleagues, then type emails to sort (empty line quits)
#   ./run.sh "email text"        sort one email
#   ./run.sh email.txt           sort an email saved in a file
#   ./run.sh load                embed new/changed colleagues from data/colleagues.csv (only those)
#   ./run.sh load --force        re-embed everyone
#   ./run.sh forget              clear remembered routing decisions
#   ./run.sh fix "email" jan     correct a decision: this (and very similar) email goes to jan
#   ./run.sh ok "email"          confirm the last decision for this email was right
#   ./run.sh summary [--days 7]  overview of logged decisions (who got what, paths, spam, corrections)
#   ./run.sh test                check accuracy on data/test_emails.csv
#   ./run.sh rank [file.csv]     rank emails worst-first, export the worst to Postman
#                                (default: data/my_inputs.csv = every email you typed)
#   ./run.sh inbox [file.csv]    priority inbox: sort emails by urgency, spam last
#   ./run.sh serve               web API on http://127.0.0.1:8090 (see service.py)
#   ./run.sh tune [file.csv]     re-tune routing.toml from real labels (add --write to apply)
#   ./run.sh unit                unit tests of the decision logic (no network)
# Settings: routing.toml. Colleagues: data/colleagues.csv + data/nobody_profiles.csv + learned corrections.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

case "${1:-}" in
  load)
    shift
    python ingest_colleagues.py data/colleagues.csv "$@"
    ;;
  forget)
    python route_email.py --forget
    ;;
  fix)
    python route_email.py --fix "$3" "$2"
    ;;
  ok)
    python route_email.py --ok "$2"
    ;;
  summary)
    shift
    python summary.py "$@"
    ;;
  test)
    shift
    python test_routing.py "$@"
    ;;
  rank)
    shift
    python rank_queries.py "$@"
    ;;
  inbox)
    shift
    python inbox.py "$@"
    ;;
  serve)
    shift
    python -m uvicorn service:app --host 127.0.0.1 --port "${PORT:-8090}" "$@"
    ;;
  tune)
    shift
    python tune.py "$@"
    ;;
  unit)
    shift
    python -m pytest tests -q "$@"
    ;;
  "")
    python ingest_colleagues.py data/colleagues.csv
    while true; do
      echo
      read -r -p "Email (empty to quit): " email || break
      [ -z "$email" ] && break
      python route_email.py "$email"
    done
    ;;
  *)
    python route_email.py "$@"
    ;;
esac
