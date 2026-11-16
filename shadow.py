"""Shadow mode: route real emails and log every decision, but forward nothing. Then review and measure.

    ./run.sh shadow inbox.csv        route new emails (CSV with an "email" column, or a folder of .txt / .eml files)
    ./run.sh shadow --review         go through unreviewed decisions: Enter = right, a number or id = the right person
    ./run.sh shadow --report         how many were right, also of the ones that would have been auto-forwarded
    ./run.sh shadow --review --labels test.csv   answers from a labeled file instead of a human (for testing)

Decisions go to the normal decision log (source "shadow"), reviews to data/shadow_reviews.jsonl. A review is a real
`ok` or `fix`: it lands in memory and in data/real_labels.csv, so the router learns while you measure. The report
also writes data/shadow_run.csv, the input for `./run.sh suggest --contrast data/shadow_run.csv`.
"""
import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path

from qdrant_client import QdrantClient

import config
import settings
from decide import MANUAL, is_nobody
from route_email import confirm, email_id, fix, known_people, load_questions, log_decision, route

REVIEWS = Path(os.environ.get("SHADOW_REVIEWS", "data/shadow_reviews.jsonl"))
RUN_CSV = Path(os.environ.get("SHADOW_RUN_CSV", "data/shadow_run.csv"))


def read_emails(source: str) -> list[str]:
    path = Path(source)
    if path.is_dir():
        emails = []
        for f in sorted(path.iterdir()):
            if f.suffix.lower() == ".eml":
                msg = BytesParser(policy=policy.default).parse(f.open("rb"))
                body = msg.get_body(preferencelist=("plain",))
                emails.append(f"Subject: {msg['subject'] or ''}\n\n{body.get_content() if body else ''}".strip())
            elif f.suffix.lower() == ".txt":
                emails.append(f.read_text(encoding="utf-8").strip())
        return [e for e in emails if e]
    with path.open(encoding="utf-8", newline="") as f:
        return [r["email"].strip() for r in csv.DictReader(f) if (r.get("email") or "").strip()]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def shadow_decisions() -> dict:
    """The latest shadow decision per email id."""
    return {e["id"]: e for e in read_jsonl(Path(config.DECISIONS_LOG)) if e.get("source") == "shadow"}


def decided(entry: dict) -> str:
    return MANUAL if entry["final"] == "spam" or is_nobody(entry["final"]) else entry["final"]


def run(source: str, learn: bool) -> None:
    emails = list(dict.fromkeys(read_emails(source)))
    done = shadow_decisions()
    new = [e for e in emails if email_id(e) not in done]
    print(f"{len(emails)} emails, {len(emails) - len(new)} already routed in shadow mode, {len(new)} new", flush=True)
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    args, questions = settings.as_args(learn=learn), load_questions()

    def one(email: str) -> dict:
        result = route(client, email, args, questions=questions)
        log_decision(email, result, source="shadow")
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (email, result) in enumerate(zip(new, pool.map(one, new)), 1):
            person = result["person"]
            print(f"[{i:>3}/{len(new)}] {result['path']:<6} {'auto' if result.get('auto') else '    '} "
                  f"-> {(person or {}).get('person_id', MANUAL):<14} {' '.join(email.split())[:60]}", flush=True)
    print("Nothing was forwarded. Next: ./run.sh shadow --review")


def review(labels: str | None = None) -> None:
    reviewed = {r["id"] for r in read_jsonl(REVIEWS)}
    todo = [e for e in shadow_decisions().values() if e["id"] not in reviewed]
    if not todo:
        print("Nothing to review.")
        return
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    people = known_people(client)
    answers_from = None
    if labels:  # testing: the right person comes from a labeled file instead of a human
        with open(labels, encoding="utf-8", newline="") as f:
            answers_from = {email_id(r["email"].strip()): MANUAL if is_nobody(r["expected"]) else r["expected"]
                            for r in csv.DictReader(f) if (r.get("email") or "").strip()}
    print(f"{len(todo)} decisions to review. Enter = right, a number = that candidate, an id = that person, "
          f"n = nobody (manual triage), s = skip, q = quit.")
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(todo, 1):
        got = decided(entry)
        print(f"\n--- {i}/{len(todo)}  ({entry['path']}{', would auto-forward' if entry.get('auto') else ''})")
        print("  " + " ".join(entry["email"].split())[:500])
        print(f"  decision: {got}" + (f" ({people[got]['name']})" if got in people and not is_nobody(got) else ""))
        options = [c["id"] for c in entry.get("candidates", []) if c["id"] != got]
        for n, pid in enumerate(options, 1):
            print(f"    {n}. {pid}" + (f" ({people[pid]['name']})" if pid in people else ""))
        if answers_from is not None:
            should = answers_from.get(entry["id"])
            answer = "s" if should is None else "" if should == got else should
            print(f"  right? {answer or '(yes)'}")
        else:
            answer = input("  right? ").strip()
        if answer == "q":
            break
        if answer == "s":
            continue
        if answer == "":
            correct, message = got, confirm(entry["email"], client)
        else:
            correct = (options[int(answer) - 1] if answer.isdigit() and 0 < int(answer) <= len(options)
                       else MANUAL if answer == "n" else answer)
            if correct not in people:
                print(f"  unknown person '{correct}', skipped")
                continue
            message = fix(client, entry["email"], correct)
        print("  " + message)
        with REVIEWS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": entry["id"], "time": datetime.now().isoformat(timespec="seconds"),
                                "decided": got, "correct": correct, "right": correct == got,
                                "path": entry["path"], "auto": entry.get("auto", False)}, ensure_ascii=False) + "\n")


def report() -> None:
    decisions, reviews = shadow_decisions(), {r["id"]: r for r in read_jsonl(REVIEWS)}
    rows = [r | {"email": decisions[i]["email"]} for i, r in reviews.items() if i in decisions]
    print(f"{len(decisions)} emails routed in shadow mode, {len(rows)} reviewed, {len(decisions) - len(rows)} not yet")
    if not rows:
        return
    pct = lambda group: f"{sum(r['right'] for r in group)}/{len(group)} ({sum(r['right'] for r in group) / len(group):.0%})"
    print(f"right: {pct(rows)}")
    auto = [r for r in rows if r["auto"]]
    if auto:
        print(f"would have been auto-forwarded: {len(auto)} of {len(rows)} ({len(auto) / len(rows):.0%}), right {pct(auto)}")
    for path, n in Counter(r["path"] for r in rows).most_common():
        print(f"  {path:<12} {pct([r for r in rows if r['path'] == path])}")
    wrong = Counter((r["correct"], r["decided"]) for r in rows if not r["right"])
    if wrong:
        print("most common mistakes (should be -> was):")
        for (should, was), n in wrong.most_common(8):
            print(f"  {n}x  {should} -> {was}")
    for r in [r for r in auto if not r["right"]][:5]:
        print(f"  wrong auto-forward: {r['decided']} instead of {r['correct']}: {' '.join(r['email'].split())[:80]}")
    with RUN_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["expected", "got", "ok", "path", "email"])
        writer.writeheader()
        writer.writerows({"expected": r["correct"], "got": r["decided"], "ok": r["right"], "path": r["path"],
                          "email": r["email"]} for r in rows)
    print(f"Wrote {RUN_CSV}: ./run.sh suggest --contrast {RUN_CSV} turns the mistakes into 'not for them' lines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", help="CSV with an 'email' column, or a folder of .txt / .eml files")
    parser.add_argument("--review", action="store_true", help="Review unreviewed shadow decisions")
    parser.add_argument("--report", action="store_true", help="Accuracy of the reviewed shadow decisions")
    parser.add_argument("--labels", help="With --review: take the answers from a labeled CSV (for testing)")
    parser.add_argument("--no-learn", action="store_true", help="Don't remember automatic decisions while routing")
    args = parser.parse_args()
    if args.review:
        review(args.labels)
    elif args.report:
        report()
    elif args.source:
        run(args.source, learn=not args.no_learn)
    else:
        parser.error("give a CSV or folder of emails, --review or --report")


if __name__ == "__main__":
    main()
