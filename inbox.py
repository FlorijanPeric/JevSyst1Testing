"""Priority inbox: route a batch of emails and list them most urgent first, spam last.

Input: a CSV with an "email" column (e.g. data/random_emails.csv or data/my_inputs.csv).
Output: printed list + data/inbox_sorted.csv. Nothing is learned into the routing cache.
"""
import argparse
import csv
import json
from pathlib import Path

from qdrant_client import QdrantClient

import config
from route_email import add_routing_args, describe, route


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort a batch of emails by urgency, with spam filtered out.")
    parser.add_argument("path", nargs="?", default="data/random_emails.csv", help="CSV with an 'email' column")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--limit", type=int, help="Only process the first N emails")
    add_routing_args(parser)
    args = parser.parse_args()
    args.cache_threshold, args.learn = 0.95, False

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if "urgency" not in questions:
        parser.error(f"{args.questions} needs an 'urgency' score question to sort by")

    with Path(args.path).open(encoding="utf-8", newline="") as f:
        emails = [r["email"] for r in csv.DictReader(f) if r.get("email", "").strip()]
    emails = emails[: args.limit] if args.limit else emails

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    rows = []
    for i, email in enumerate(emails, 1):
        result = route(client, email, args, questions=questions)
        answers = result["answers"]
        rows.append({
            "urgency": answers.get("urgency") or 0.0,
            "spam": answers.get("spam") or 0.0,
            "language": answers.get("language") or "",
            "type": answers.get("type") or "",
            "forward_to": describe(result),
            "email": email,
            "is_spam": result["path"] == "spam",
        })
        if i % 10 == 0 or i == len(emails):
            print(f"  processed {i}/{len(emails)}", flush=True)

    # Most urgent first; spam always at the bottom.
    rows.sort(key=lambda r: (r["is_spam"], -r["urgency"]))
    labels = questions["urgency"]["criteria"]

    print(f"\n{'urg':>4}  {'lang':<5} {'type':<9} {'forward to':<40} email")
    for r in rows:
        if r["is_spam"] and not rows[rows.index(r) - 1]["is_spam"]:
            print("---- spam (not forwarded) ----")
        preview = " ".join(r["email"].split())[:60]
        print(f"{r['urgency']:4.2f}  {r['language']:<5} {r['type']:<9} {r['forward_to'][:40]:<40} {preview}")

    out = Path("data/inbox_sorted.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["urgency", "urgency_label", "spam", "language", "type", "forward_to", "email"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            label = labels[min(max(round(r["urgency"]), 0), len(labels) - 1)]
            writer.writerow({**r, "urgency": f"{r['urgency']:.2f}", "urgency_label": label, "spam": f"{r['spam']:.2f}"})

    spam = sum(r["is_spam"] for r in rows)
    print(f"\n{len(rows)} emails, {spam} spam. Saved to {out}")


if __name__ == "__main__":
    main()
