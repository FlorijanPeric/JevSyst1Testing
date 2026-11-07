"""Measure routing accuracy on labeled test emails (data/test_emails.csv: expected,email).

A file without an "expected" column (e.g. data/random_emails.csv) is just routed:
you get each decision plus how many emails each person received, to check by eye.

Nothing is written to Qdrant: no cache lookups and no learning, so every run is comparable.
Emails run in parallel (`workers` in routing.toml); system-one answers come from the disk cache
when the exact same question was asked before, so repeated runs of a setup give the same result.
"""
import argparse
import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qdrant_client import QdrantClient

import config
from clients import classify, embed_texts
from decide import clean_email, spam_overruled
from route_email import MANUAL, add_routing_args, ask_system_one, fast_pick, shortlist


def main() -> None:
    parser = argparse.ArgumentParser(description="Test email routing against expected recipients.")
    parser.add_argument("path", nargs="?", default="data/test_emails.csv")
    add_routing_args(parser)
    parser.add_argument("--only", help="Only test emails expected for this person id")
    parser.add_argument("--limit", type=int, help="Only test the first N emails")
    parser.add_argument("--spam", action="store_true",
                        help="Also apply the spam filter from data/questions.json, as normal routing does (spam counts as manual_triage)")
    args = parser.parse_args()

    with Path(args.path).open(encoding="utf-8", newline="") as f:
        tests = [row for row in csv.DictReader(f) if not args.only or row["expected"] == args.only]
    tests = tests[: args.limit] if args.limit else tests
    labeled = bool(tests) and "expected" in tests[0]
    spam_q = {"spam": json.loads(Path("data/questions.json").read_text(encoding="utf-8"))["spam"]} if args.spam else {}

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    start = time.perf_counter()
    texts = [clean_email(t["email"]) if args.clean_emails else t["email"] for t in tests]
    vectors = embed_texts(texts)

    def run_one(i: int) -> dict:
        test, text, vector = tests[i], texts[i], vectors[i]
        people = shortlist(client, vector, args.candidates, args.ranking, args.nobody_candidates)
        fast = fast_pick(people, args.min_score, args.min_margin)
        if fast:
            path, got, conf = "fast", fast, None
            answers = classify(text, spam_q) if spam_q else {}
        else:
            path = "llm"
            got, answers, verdict = ask_system_one(text, people, spam_q, args)
            conf = verdict["confidence"]
        overruled = False
        if (answers.get("spam") or 0) >= args.spam_threshold:
            overruled = spam_overruled(people, args.spam_override_score)
            if not overruled:
                path, got = "spam", MANUAL
        row = {**test, "got": got, "path": path, "confidence": conf, "spam_overruled": overruled}
        if labeled:
            row["in_shortlist"] = test["expected"] == MANUAL or any(p["person_id"] == test["expected"] for p in people)
            row["ok"] = got == test["expected"]
        return row

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, row in enumerate(pool.map(run_one, range(len(tests))), 1):
            results.append(row)
            conf_text = "  -  " if row["confidence"] is None else f"{row['confidence']:4.0%} "
            preview = " ".join(row["email"].split())
            if labeled:
                print(f"[{i:>3}/{len(tests)}] {'ok  ' if row['ok'] else 'FAIL'} {row['path']:<4} conf {conf_text} "
                      f"expected {row['expected']:<14} got {row['got']:<14} {preview[:45]}", flush=True)
            else:
                print(f"[{i:>3}/{len(tests)}] {row['path']:<4} conf {conf_text} -> {row['got']:<14} {preview[:80]}", flush=True)

    if not labeled:
        print(f"\nRouted {len(results)} emails in {time.perf_counter() - start:.0f}s. Emails per person:")
        for pid, n in Counter(r["got"] for r in results).most_common():
            print(f"  {pid:<14} {n}")
        return

    total = len(results)
    correct = sum(r["ok"] for r in results)
    print(f"\nAccuracy:        {correct}/{total} ({correct / total:.0%})  in {time.perf_counter() - start:.0f}s")
    for path in ("fast", "llm", "spam"):
        rows = [r for r in results if r["path"] == path]
        if rows:
            print(f"  {path:<4} path:     {sum(r['ok'] for r in rows)}/{len(rows)} correct")
    for label, group in (("correct", [r for r in results if r["ok"]]), ("wrong", [r for r in results if not r["ok"]])):
        confs = [r["confidence"] for r in group if r["confidence"] is not None]
        if confs:
            print(f"  avg system one confidence when {label}: {sum(confs) / len(confs):.0%}")
    overruled = [r for r in results if r["spam_overruled"]]
    if overruled:
        print(f"  spam overruled by a clear match: {sum(r['ok'] for r in overruled)}/{len(overruled)} correct")
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
