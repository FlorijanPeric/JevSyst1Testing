"""Overview of logged routing decisions (data/decisions.jsonl) and real labels (data/real_labels.csv).

    ./run.sh summary            last 7 days
    ./run.sh summary --days 30
Also served by the API: GET /summary?days=7
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import config


def summarize(days: int = 7) -> str:
    since = datetime.now() - timedelta(days=days)
    log = Path(config.DECISIONS_LOG)
    entries = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if datetime.fromisoformat(entry["time"]) >= since:
                entries.append(entry)
    if not entries:
        return f"No decisions logged in the last {days} days ({log})."

    out = [f"Routing summary, last {days} days: {len(entries)} emails "
           f"({entries[0]['time'][:10]} to {entries[-1]['time'][:10]})", "", "Emails per person:"]
    out += [f"  {person:<16} {n:>4}  {'#' * min(n, 50)}" for person, n in Counter(e["final"] for e in entries).most_common()]

    out += ["", "How decisions were made:"]
    out += [f"  {path:<16} {n:>4}  ({n / len(entries):.0%})" for path, n in Counter(e["path"] for e in entries).most_common()]

    sources = Counter(e.get("source", "cli") for e in entries)
    out += ["", "Where emails came from: " + ", ".join(f"{s} {n}" for s, n in sources.most_common())]

    urgent = [e for e in entries if e["final"] != "spam" and (e["answers"].get("urgency") or 0) >= 1.5]
    if urgent:
        out += ["", f"Urgent (urgency >= 1.5, spam left out): {len(urgent)}"]
        out += [f"  {person:<16} {n:>4}" for person, n in Counter(e["final"] for e in urgent).most_common(5)]

    langs = Counter(e["answers"].get("language") for e in entries if e["answers"].get("language"))
    if langs:
        out += ["", "Languages: " + ", ".join(f"{lang} {n}" for lang, n in langs.most_common())
                + f"  (asked for {sum(langs.values())} of {len(entries)}; the fast path skips the question)"]

    fallbacks = sum(e["path"] == "search-only" for e in entries)
    ms = sorted(e["ms"] for e in entries)
    out += ["", f"Speed: median {ms[len(ms) // 2] / 1000:.1f}s, slowest {ms[-1] / 1000:.1f}s. "
                f"System one down (search-only fallback): {fallbacks}"]

    labels = Path(config.REAL_LABELS)
    if labels.exists():
        with labels.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if datetime.fromisoformat(r["time"]) >= since]
        by_source = Counter(r["source"] for r in rows)
        out += ["", f"Feedback: {by_source.get('confirmed', 0)} confirmed right, {by_source.get('corrected', 0)} corrected. "
                    f"All real labels: {labels} (test with ./run.sh test {labels}, re-tune with ./run.sh tune)"]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summary of routing decisions.")
    parser.add_argument("--days", type=int, default=7)
    print(summarize(parser.parse_args().days))


if __name__ == "__main__":
    main()
