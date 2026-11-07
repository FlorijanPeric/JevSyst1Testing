"""Re-tune the decision values in routing.toml from labeled emails (normally data/real_labels.csv).

For every labeled email it records what the router sees (candidates, scores, system one's
probabilities; all cached on disk), then tries every combination of search_weight,
manual_threshold, min_score and min_margin. The honest score is 2-fold cross-validated:
tuned on half the emails, measured on the other half.

    ./run.sh tune                          # real labels, needs at least --min-labels rows
    ./run.sh tune data/test_emails.csv     # any expected,email file
    ./run.sh tune --write                  # also write the best values into routing.toml
"""
import argparse
import csv
import itertools
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qdrant_client import QdrantClient

import config
import decide
import settings
from clients import classify, embed_texts
from decide import MANUAL, clean_email
from route_email import recipient_question, shortlist

GRID = {
    "search_weight": [0, 1, 2, 5, 10, 20],
    "manual_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    "min_score": [0.45, 0.5, 0.55, 0.6, 9.0],  # 9.0 = fast path off
    "min_margin": [0.05, 0.07, 0.1],
}


def decide_with(rec: dict, p: dict) -> str:
    fast = decide.fast_pick(rec["people"], p["min_score"], p["min_margin"])
    if fast:
        return fast
    return decide.pick_recipient(rec["probs"], rec["people"], p["search_weight"], p["manual_threshold"])


def accuracy(records: list[dict], p: dict) -> float:
    return sum(decide_with(r, p) == r["expected"] for r in records) / len(records) if records else 0.0


def best_params(records: list[dict], current: dict | None = None) -> tuple[dict, float]:
    """The best grid values; on a tie the current settings win, so --write never changes values for nothing."""
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    if current is not None:
        combos.insert(0, current)
    best = max(combos, key=lambda p: accuracy(records, p))  # max keeps the first of equal values
    return best, accuracy(records, best)


def write_settings(values: dict, path: Path = settings.SETTINGS_PATH) -> None:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = re.sub(rf"^({key}\s*=\s*)[^#\n]*?(\s*#.*)?$", lambda m: f"{m.group(1)}{value}{m.group(2) or ''}",
                      text, count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-tune routing.toml from labeled emails.")
    parser.add_argument("path", nargs="?", default=config.REAL_LABELS, help="CSV with expected,email")
    parser.add_argument("--min-labels", type=int, default=30, help="Refuse to tune on fewer labeled emails")
    parser.add_argument("--write", action="store_true", help="Write the best values into routing.toml")
    args = parser.parse_args()
    s = settings.load()

    path = Path(args.path)
    if not path.exists():
        parser.error(f"{path} doesn't exist yet. Use ./run.sh ok / fix to collect real labels.")
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("email", "").strip() and r.get("expected")]
    # the newest label per email wins (a later fix overrides an earlier ok)
    rows = list({" ".join(r["email"].split()): r for r in rows}.values())
    if len(rows) < args.min_labels:
        parser.error(f"only {len(rows)} labeled emails in {path}; need {args.min_labels} for a meaningful result")

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    texts = [clean_email(r["email"]) if s.clean_emails else r["email"] for r in rows]
    vectors = embed_texts(texts)

    def record(i: int) -> dict:
        people = shortlist(client, vectors[i], s.candidates, s.ranking, s.nobody_candidates)
        answer = classify(texts[i], {"recipient": recipient_question(people)}, raw=True)["recipient"]
        expected = MANUAL if decide.is_nobody(rows[i]["expected"]) else rows[i]["expected"]
        return {"idx": i, "expected": expected, "people": people, "probs": answer.get("probabilities", {})}

    print(f"Recording {len(rows)} labeled emails (cached answers are reused)...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, s.workers)) as pool:
        records = list(pool.map(record, range(len(rows))))

    current = {k: getattr(s, k) for k in GRID}
    best, tuned_acc = best_params(records, current)
    folds = [[r for r in records if r["idx"] % 2 == k] for k in (0, 1)]
    cv = sum(accuracy(folds[1 - k], best_params(folds[k], current)[0]) * len(folds[1 - k]) for k in (0, 1)) / len(records)

    print(f"\n{len(records)} emails from {path}")
    print(f"  current settings:  {accuracy(records, current):.1%}   {current}")
    print(f"  best on all:       {tuned_acc:.1%}   {best}")
    print(f"  honest (2-fold):   {cv:.1%}   <- what to expect on new emails with re-tuning")
    if args.write:
        write_settings(best)
        print(f"Wrote {best} to {settings.SETTINGS_PATH.name}")
    else:
        print("Nothing written. Add --write to apply the best values to routing.toml.")


if __name__ == "__main__":
    main()
