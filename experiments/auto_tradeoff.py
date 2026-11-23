"""How many emails could be auto-forwarded, and how many of those would be right, under different rules.

Reads `test_routing.py --jsonl` files (every decision with its verdict). Clear matches (fast path) and memory keep
today's rule; only system one decisions are re-judged, from what is already in the verdict: system one's confidence,
the second opinion's pick and the check models' picks. No model is asked again.

    .venv/bin/python experiments/auto_tradeoff.py round19/auto_*.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decide import is_nobody  # noqa: E402


def facts(row: dict) -> dict | None:
    """What a rule may look at, for a system one decision that kept system one's pick (else None)."""
    v = row.get("verdict") or {}
    got = row["got"]
    if row["path"] != "llm" or is_nobody(got) or v.get("system_one_choice") != got:
        return None
    second = v.get("second_opinion") or {}
    picks = list((second.get("votes") or {"": second.get("pick")}).values()) if second else []
    checks = list((v.get("auto_check") or {}).values())
    return {"conf": v.get("confidence") or 0.0, "second_agrees": bool(picks) and all(p == got for p in picks),
            "checks_yes": sum(p == got for p in checks), "checks": len(checks)}


RULES = {
    "today: second opinion + every check model": lambda f: f["second_agrees"] and f["checks"] > 0 and f["checks_yes"] == f["checks"],
    "second opinion + at least one check model": lambda f: f["second_agrees"] and f["checks_yes"] >= 1,
    "second opinion only": lambda f: f["second_agrees"],
}
for t in (0.6, 0.7, 0.8, 0.9, 0.95):
    RULES[f"second opinion + confidence >= {t:.2f}"] = lambda f, t=t: f["second_agrees"] and f["conf"] >= t
    RULES[f"today, or one check + confidence >= {t:.2f}"] = (
        lambda f, t=t: RULES["today: second opinion + every check model"](f)
        or (f["second_agrees"] and f["checks_yes"] >= 1 and f["conf"] >= t))


def main() -> None:
    sets = {Path(p).stem: [json.loads(line) for line in open(p, encoding="utf-8")] for p in sys.argv[1:]}
    if not sets:
        sys.exit(__doc__)
    total = sum(len(rows) for rows in sets.values())
    print(f"{total} emails in {len(sets)} sets ({', '.join(f'{k} {len(v)}' for k, v in sets.items())})")
    print(f"\n{'rule':<44} {'auto-forwarded':>15} {'right':>6} {'wrong':>6} {'precision':>9}   wrong per set")
    for name, rule in RULES.items():
        n = right = 0
        per_set = []
        for rows in sets.values():
            wrong_here = 0
            for r in rows:
                f = facts(r)
                auto = (rule(f) if f is not None else r["path"] in ("fast", "memory", "cache") and r["auto"])
                if auto:
                    n += 1
                    right += r["ok"]
                    wrong_here += not r["ok"]
            per_set.append(wrong_here)
        print(f"{name:<44} {n:>6} ({n / total:>5.0%})  {right:>6} {n - right:>6} {right / max(n, 1):>9.1%}   "
              + " / ".join(map(str, per_set)))


if __name__ == "__main__":
    main()
