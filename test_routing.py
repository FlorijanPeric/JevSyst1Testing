"""Measure routing accuracy on labeled test emails (data/test_emails.csv: expected,email).

A file without an "expected" column (e.g. data/random_emails.csv) is just routed:
you get each decision plus how many emails each person received, to check by eye.

Nothing is written to Qdrant: no cache lookups and no learning, so every run is comparable.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient

import config
from clients import embed_texts
from route_email import MANUAL, add_routing_args, ask_system_one, is_confident, shortlist


def main() -> None:
    parser = argparse.ArgumentParser(description="Test email routing against expected recipients.")
    parser.add_argument("path", nargs="?", default="data/test_emails.csv")
    add_routing_args(parser)
    parser.add_argument("--only", help="Only test emails expected for this person id")
    parser.add_argument("--limit", type=int, help="Only test the first N emails")
    args = parser.parse_args()

    with Path(args.path).open(encoding="utf-8", newline="") as f:
        tests = [row for row in csv.DictReader(f) if not args.only or row["expected"] == args.only]
    tests = tests[: args.limit] if args.limit else tests
    labeled = bool(tests) and "expected" in tests[0]

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    start = time.perf_counter()
    vectors = embed_texts([t["email"] for t in tests])

    results = []
    for i, (test, vector) in enumerate(zip(tests, vectors), 1):
        people = shortlist(client, vector, args.candidates)
        if is_confident(people, args.min_score, args.min_margin):
            path, got, conf = "fast", people[0]["person_id"], None
        else:
            path = "llm"
            got, _, verdict = ask_system_one(test["email"], people, {}, args)
            conf = verdict["confidence"]
        conf_text = "  -  " if conf is None else f"{conf:4.0%} "
        preview = " ".join(test["email"].split())
        if not labeled:
            results.append({**test, "got": got, "path": path, "confidence": conf})
            print(f"[{i:>3}/{len(tests)}] {path:<4} conf {conf_text} -> {got:<14} {preview[:80]}")
            continue
        in_shortlist = test["expected"] == MANUAL or any(p["person_id"] == test["expected"] for p in people)
        ok = got == test["expected"]
        results.append({**test, "got": got, "path": path, "ok": ok, "in_shortlist": in_shortlist, "confidence": conf})
        print(f"[{i:>3}/{len(tests)}] {'ok  ' if ok else 'FAIL'} {path:<4} conf {conf_text} "
              f"expected {test['expected']:<14} got {got:<14} {preview[:45]}")

    if not labeled:
        print(f"\nRouted {len(results)} emails in {time.perf_counter() - start:.0f}s. Emails per person:")
        for pid, n in Counter(r["got"] for r in results).most_common():
            print(f"  {pid:<14} {n}")
        return

    total = len(results)
    correct = sum(r["ok"] for r in results)
    print(f"\nAccuracy:        {correct}/{total} ({correct / total:.0%})  in {time.perf_counter() - start:.0f}s")
    for path in ("fast", "llm"):
        rows = [r for r in results if r["path"] == path]
        if rows:
            print(f"  {path:<4} path:     {sum(r['ok'] for r in rows)}/{len(rows)} correct")
    for label, group in (("correct", [r for r in results if r["ok"]]), ("wrong", [r for r in results if not r["ok"]])):
        confs = [r["confidence"] for r in group if r["confidence"] is not None]
        if confs:
            print(f"  avg system one confidence when {label}: {sum(confs) / len(confs):.0%}")
    missed = [r for r in results if not r["in_shortlist"]]
    print(f"Search misses:   {len(missed)} (right person not in top {args.candidates}, system one never saw them)")

    print("\nPer person:")
    for pid in dict.fromkeys(r["expected"] for r in results):
        rows = [r for r in results if r["expected"] == pid]
        print(f"  {pid:<14} {sum(r['ok'] for r in rows)}/{len(rows)}")

    mistakes = Counter((r["expected"], r["got"]) for r in results if not r["ok"])
    if mistakes:
        print("\nMost common mistakes (expected -> got):")
        for (expected, got), n in mistakes.most_common(10):
            print(f"  {n}x  {expected} -> {got}")


if __name__ == "__main__":
    main()
