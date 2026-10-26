"""Find generated test emails whose label is probably wrong, for a human to check.

For each email, system one is asked whether the labeled person (or "nobody" for manual_triage)
really fits. Emails it doubts most are written first to experiments/data/label_review.csv,
with an empty "correct_label" column to fill in by hand.

Run from the project folder:  .venv/bin/python experiments/flag_labels.py
"""
import argparse
import csv

from common import EXP_DIR, ROOT, Breaker, Cache, GracefulStop, Stop, log, system_one_cached

MANUAL = "manual_triage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(EXP_DIR / "data" / "generated_emails.csv"))
    parser.add_argument("--flag-below", type=float, default=0.3, help="Flag emails where P(label fits) is below this")
    args = parser.parse_args()

    stopper = GracefulStop()
    breaker = Breaker(stopper)
    cache = Cache()
    with (ROOT / "data" / "colleagues.csv").open(encoding="utf-8") as f:
        people = {r["id"]: r for r in csv.DictReader(f)}
    with open(args.path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    scored = []
    try:
        for i, row in enumerate(rows, 1):
            if row["expected"] == MANUAL:
                instructions = "Is this email spam, personal, or off-topic, so that no one at a tech company/lab needs to handle it?"
            else:
                p = people[row["expected"]]
                duties = "; ".join(p["responsibilities"].split("|")[:8])
                instructions = (f"Is this email something that this person should handle: "
                                f"{p['role'].replace(' (fictional)', '')} ({duties})?")
            answer = system_one_cached(row["email"], {"type": "noul", "instructions": instructions}, cache, breaker)
            scored.append({**row, "p_label_fits": answer.get("noul", 0.0)})
            if i % 50 == 0:
                log(f"checked {i}/{len(rows)}")
    except Stop as e:
        log(f"Stopped early: {e}")

    scored.sort(key=lambda r: r["p_label_fits"])
    out = EXP_DIR / "data" / "label_review.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p_label_fits", "expected", "correct_label", "email"], extrasaction="ignore")
        writer.writeheader()
        for r in scored:
            writer.writerow({**r, "p_label_fits": f"{r['p_label_fits']:.2f}", "correct_label": ""})
    flagged = [r for r in scored if r["p_label_fits"] < args.flag_below]
    log(f"{len(flagged)} of {len(scored)} labels look doubtful (P < {args.flag_below}). Review the top of {out}")
    for r in flagged[:15]:
        log(f"  {r['p_label_fits']:.2f}  {r['expected']:<14} {' '.join(r['email'].split())[:80]}")


if __name__ == "__main__":
    main()
