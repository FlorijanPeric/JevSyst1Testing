"""Does the router learn? Labeled emails arrive one after another, like a real inbox.

Each email is routed first (scored before its label is known), then the simulated user gives feedback:
  --feedback none         nobody corrects anything (only automatic learning, if on)
  --feedback corrections  wrong decisions are fixed (`./run.sh fix`)
  --feedback all          wrong ones fixed, right ones confirmed (`./run.sh ok`)
Prints accuracy and how many emails skipped system one, per block, so learning shows as a curve.

    ROUTED_COLLECTION=routed_curve .venv/bin/python experiments/learn_curve.py data/test_emails.csv \\
        experiments/data/generated_emails.csv --feedback corrections
Uses its own memory collection (must not be the real one) and wipes it at the start.
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

import config  # noqa: E402
from clients import embed_texts  # noqa: E402
from decide import MANUAL, clean_email, is_nobody  # noqa: E402
from route_email import add_routing_args, known_people, load_questions, remember, route  # noqa: E402


def final(result: dict) -> str:
    return MANUAL if result["path"] == "spam" or not result["person"] else result["person"]["person_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="CSV files with expected,email")
    parser.add_argument("--feedback", choices=["none", "corrections", "all"], default="corrections")
    parser.add_argument("--block", type=int, default=100, help="Emails per row of the curve")
    parser.add_argument("--seed", type=int, default=1, help="Order of the emails (same seed = same order)")
    add_routing_args(parser)
    args = parser.parse_args()
    args.learn = True

    if config.ROUTED_COLLECTION == "routed_emails":
        sys.exit("Set ROUTED_COLLECTION to a temporary collection; this test wipes it.")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    if client.collection_exists(config.ROUTED_COLLECTION):
        client.delete_collection(config.ROUTED_COLLECTION)

    rows = []
    for path in args.paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows += [{"expected": MANUAL if is_nobody(r["expected"]) else r["expected"], "email": r["email"]}
                     for r in csv.DictReader(f) if r.get("email")]
    random.Random(args.seed).shuffle(rows)
    spam_q = {k: v for k, v in load_questions().items() if k == "spam"}
    people = known_people(client)
    print(f"{len(rows)} emails, feedback={args.feedback}, memory={args.memory}, auto_learn={args.auto_learn}, "
          f"min_sim={args.memory_min_sim}, collection={config.ROUTED_COLLECTION}", flush=True)

    results, start = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        # A small batch is routed in parallel, then its feedback is given: the next batch can learn from it.
        for at in range(0, len(rows), args.workers):
            batch = rows[at:at + args.workers]
            outs = list(pool.map(lambda r: route(client, r["email"], args, questions=spam_q), batch))
            for row, out in zip(batch, outs):
                got = final(out)
                results.append({"ok": got == row["expected"], "path": out["path"]})
                wrong, right = got != row["expected"], got == row["expected"]
                if (args.feedback != "none" and wrong) or (args.feedback == "all" and right):
                    text = clean_email(row["email"])
                    remember(client, text, embed_texts([text])[0], people[row["expected"]],
                             source="corrected" if wrong else "confirmed")

    print(f"\n{'emails':>9}  {'accuracy':>8}  {'no system one':>13}  {'memory':>6}  {'memory ok':>9}  paths")
    for at in range(0, len(results), args.block):
        block = results[at:at + args.block]
        paths = Counter(r["path"] for r in block)
        mem = [r for r in block if r["path"] in ("memory", "cache")]
        skipped = sum(n for p, n in paths.items() if p != "llm")
        print(f"{at + 1:>4}-{at + len(block):<4}  {sum(r['ok'] for r in block) / len(block):>8.0%}  "
              f"{skipped / len(block):>13.0%}  {len(mem) / len(block):>6.0%}  "
              f"{(sum(r['ok'] for r in mem) / len(mem) if mem else 0):>9.0%}  "
              + ", ".join(f"{p} {n}" for p, n in paths.most_common()), flush=True)
    total_ok = sum(r["ok"] for r in results)
    print(f"total {total_ok}/{len(results)} ({total_ok / len(results):.1%}) in {time.perf_counter() - start:.0f}s")
    points = client.count(config.ROUTED_COLLECTION).count if client.collection_exists(config.ROUTED_COLLECTION) else 0
    print(f"memory holds {points} emails at the end")


if __name__ == "__main__":
    main()
