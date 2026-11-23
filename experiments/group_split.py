"""Split labeled emails into two halves without near-duplicates on both sides.

Rewordings of the same email land in the same half, so what is learned on one half (profile lines, contrast lines)
is tested on emails it has not seen in any wording.

    .venv/bin/python experiments/group_split.py a.csv b.csv --out-dir round17 --min-sim 0.8
writes round17/fold0.csv and round17/fold1.csv (expected,email).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clients import embed_emails  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-sim", type=float, default=0.8, help="Emails at least this similar are one group")
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        with open(path, encoding="utf-8", newline="") as f:
            rows += [{"expected": r["expected"], "email": r["email"]} for r in csv.DictReader(f) if r.get("email")]
    vectors = np.array(embed_emails([r["email"] for r in rows]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    group_of = [-1] * len(rows)  # union of everything at least min_sim similar
    sims = vectors @ vectors.T
    groups: list[list[int]] = []
    for i in range(len(rows)):
        if group_of[i] >= 0:
            continue
        group_of[i], todo, members = len(groups), [i], [i]
        while todo:
            j = todo.pop()
            for k in np.nonzero(sims[j] >= args.min_sim)[0]:
                if group_of[k] < 0:
                    group_of[k] = group_of[i]
                    todo.append(k)
                    members.append(k)
        groups.append(members)

    halves: list[list[int]] = [[], []]
    for members in sorted(groups, key=len, reverse=True):  # biggest first, always into the smaller half
        min(halves, key=len).extend(members)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(halves):
        with (out / f"fold{k}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["expected", "email"])
            writer.writeheader()
            writer.writerows(rows[i] for i in sorted(idx))
    print(f"{len(rows)} emails in {len(groups)} groups (largest {max(map(len, groups))}) -> halves of "
          f"{len(halves[0])} and {len(halves[1])}")


if __name__ == "__main__":
    main()
