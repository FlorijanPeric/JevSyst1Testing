"""How often is system one right at each confidence level? Picks auto_min_confidence for auto-forwarding.

Input: per-email CSVs from `test_routing.py --out FILE` (labeled emails). Only system one decisions count
where search did not overrule it (the same rule as decide.safe_to_auto).

    .venv/bin/python experiments/calibration.py round9/calib_h.csv round9/calib_g.csv [--target 0.95] [--write]
Prints accuracy per confidence band and for "at least this confident", then the lowest threshold that
reaches --target, checked honestly: chosen on one half of the emails, measured on the other half.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings  # noqa: E402
from tune import write_settings  # noqa: E402

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(16)]  # 0.20 ... 0.95


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["path"] != "llm" or not r["confidence"]:
                    continue
                rows.append({"ok": r["ok"] == "True", "conf": float(r["confidence"]),
                             "agree": r["system_one_choice"] == r["got"]})
    return rows


def pick(rows: list[dict], target: float, min_n: int) -> float | None:
    """Lowest threshold where agreed decisions at least that confident are >= target right (and >= min_n of them)."""
    for t in THRESHOLDS:
        sel = [r for r in rows if r["agree"] and r["conf"] >= t]
        if len(sel) >= min_n and sum(r["ok"] for r in sel) / len(sel) >= target:
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--target", type=float, default=0.95, help="Accuracy wanted for auto-forwarding")
    parser.add_argument("--min-n", type=int, default=20, help="At least this many decisions above the threshold")
    parser.add_argument("--write", action="store_true", help="Write the honest threshold to routing.toml")
    args = parser.parse_args()

    rows = load(args.csvs)
    agreed = [r for r in rows if r["agree"]]
    print(f"{len(rows)} system one decisions, {len(agreed)} not overruled by search "
          f"({sum(r['ok'] for r in agreed) / max(len(agreed), 1):.0%} right); "
          f"overruled ones {sum(r['ok'] for r in rows if not r['agree'])}/{len(rows) - len(agreed)} right\n")

    print(f"{'confidence':>12}  {'in band':>8}  {'right':>6}  |  {'at least':>9}  {'n':>4}  {'right':>6}  {'share of system one':>19}")
    for lo, hi in zip(THRESHOLDS, THRESHOLDS[1:] + [1.01]):
        band = [r for r in agreed if lo <= r["conf"] < hi]
        above = [r for r in agreed if r["conf"] >= lo]
        if not above:
            break
        band_acc = f"{sum(r['ok'] for r in band) / len(band):.0%}" if band else "-"
        print(f"{lo:>5.2f}-{min(hi, 1):<5.2f}  {len(band):>8}  {band_acc:>6}  |  {lo:>9.2f}  {len(above):>4}  "
              f"{sum(r['ok'] for r in above) / len(above):>6.0%}  {len(above) / len(rows):>19.0%}")

    best = pick(rows, args.target, args.min_n)
    halves = [rows[0::2], rows[1::2]]
    honest = []
    for k in (0, 1):
        t = pick(halves[k], args.target, args.min_n // 2)
        test = [r for r in halves[1 - k] if r["agree"] and t is not None and r["conf"] >= t]
        honest.append((t, sum(r["ok"] for r in test), len(test)))
    print(f"\ntarget {args.target:.0%} right: lowest threshold on all emails = {best}")
    for k, (t, ok, n) in enumerate(honest):
        print(f"  chosen on half {k + 1}: {t} -> other half {ok}/{n} right" + (f" ({ok / n:.0%})" if n else ""))
    ok_all = sum(ok for _, ok, _ in honest)
    n_all = sum(n for _, _, n in honest)
    verdict = n_all > 0 and ok_all / n_all >= args.target - 0.02
    print(f"honest: {ok_all}/{n_all} right" + (f" ({ok_all / n_all:.0%})" if n_all else "")
          + (" -> holds up" if verdict else " -> does not hold up, keep auto-forward off for system one"))
    if args.write:
        value = best if (best is not None and verdict) else 2.0
        write_settings({"auto_min_confidence": value}, settings.SETTINGS_PATH)
        print(f"Wrote auto_min_confidence = {value} to routing.toml")


if __name__ == "__main__":
    main()
