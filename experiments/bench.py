"""Long-running routing benchmark: chunking strategies x scoring x system-one setups.

Phase 1 (search only, cheap): every chunking x aggregation, scored on all emails.
Phase 2 (system one): the best chunkings (+ the current one) x description style x candidates,
         run in rounds; after each round, configs that can no longer win are dropped.
Phase 3 (offline): search weight / manual threshold / fast path swept on the saved
         probabilities, with 2-fold cross-validation so the "honest" number isn't tuned on itself.

Smart exits: everything is cached (rerun = resume), --max-hours budget, Ctrl-C/SIGTERM
write the report first, and a circuit breaker stops cleanly if hivecore goes down.

Run from the project folder:  .venv/bin/python experiments/bench.py
"""
import argparse
import csv
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from common import EXP_DIR, ROOT, Breaker, Cache, GracefulStop, Stop, embed_cached, log, system_one_cached

MANUAL = "manual_triage"
RESULTS = EXP_DIR / "results"


# ---------- data ----------

def load_people() -> list[dict]:
    with (ROOT / "data" / "colleagues.csv").open(encoding="utf-8") as f:
        return [{
            "id": r["id"], "name": r["name"], "role": r["role"].replace(" (fictional)", ""),
            "duties": [s.strip() for s in r["responsibilities"].split("|") if s.strip()],
            "examples": [s.strip() for s in r["example_emails"].split("|") if s.strip()],
        } for r in csv.DictReader(f)]


def load_emails(seed: int) -> list[dict]:
    rows = []
    for path, source in ((ROOT / "data" / "test_emails.csv", "handwritten"),
                         (EXP_DIR / "data" / "generated_emails.csv", "generated")):
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                rows += [{"expected": r["expected"], "email": r["email"], "source": source} for r in csv.DictReader(f)]
    random.Random(seed).shuffle(rows)  # rounds then see a mix of people and sources
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


# ---------- chunking strategies: how a profile becomes vectors ----------

def windows(items: list[str], size: int = 3, stride: int = 2) -> list[str]:
    if len(items) <= size:
        return ["; ".join(items)]
    return ["; ".join(items[i : i + size]) for i in range(0, len(items) - size + 1, stride)]


CHUNKERS = {
    "lines_prefixed": lambda p: [f"{p['name']}, {p['role']}: {d}" for d in p["duties"]]
                                + [f"{p['name']}, {p['role']} handles emails like: {e}" for e in p["examples"]],
    "lines_role": lambda p: [f"{p['role']}: {d}" for d in p["duties"]] + [f"{p['role']}: {e}" for e in p["examples"]],
    "lines_clean": lambda p: p["duties"] + p["examples"],
    "duties_only": lambda p: p["duties"] or [p["role"]],
    "examples_only": lambda p: p["examples"] or [p["role"]],
    "grouped": lambda p: [f"{p['role']}: " + "; ".join(p["duties"]), "Emails like: " + " ".join(p["examples"])],
    "window3": lambda p: [f"{p['role']}: {w}" for w in windows(p["duties"] + p["examples"])],
    "single": lambda p: [f"{p['role']}: " + "; ".join(p["duties"]) + ". Emails like: " + " ".join(p["examples"])],
}

AGGREGATORS = {
    "max": lambda s: s.max(),                                   # = Qdrant MaxSim (what production uses)
    "top2mean": lambda s: np.sort(s)[-2:].mean(),
    "mean": lambda s: s.mean(),
}


def person_scores(email_vecs: np.ndarray, profile_vecs: dict, agg) -> np.ndarray:
    """(emails x people) similarity matrix for one chunking + aggregation."""
    return np.stack([np.apply_along_axis(agg, 1, email_vecs @ vecs.T) for vecs in profile_vecs.values()], axis=1)


# ---------- system one question styles ----------

def question_old(cands, people_by_id, line_sims):
    """Production before 2026-10-23: name + role + the 2 best-matching prefixed lines."""
    crit = {}
    for pid in cands:
        p = people_by_id[pid]
        lines = CHUNKERS["lines_prefixed"](p)
        best = np.argsort(-line_sims[pid])[:2]
        crit[pid] = f"{p['name']}, {p['role']}. Handles: " + "; ".join(lines[i] for i in best)
    crit[MANUAL] = "None of these people fit; needs a human to triage"
    return {"type": "choice", "instructions": "Which colleague should this email be forwarded to?", "criteria": crit}


def question_new(cands, people_by_id, line_sims, max_duties: int = 6):
    """Production now: role + duties, best-matching duty first, no names."""
    crit = {}
    for pid in cands:
        if pid == MANUAL:
            continue
        p = people_by_id[pid]
        n = len(p["duties"])
        order = [i for i in np.argsort(-line_sims[pid]) if i < n]  # duty lines come first in lines_prefixed
        crit[pid] = f"{p['role']}: " + "; ".join(p["duties"][i] for i in order[:max_duties])
    crit[MANUAL] = "Spam, personal or off-topic; none of these people handle it"
    return {"type": "choice", "instructions": "Who is responsible for this email? Pick the one whose duties match its main topic.",
            "criteria": crit}


STYLES = {"old": question_old, "new": question_new}


# ---------- decision rule (same as route_email.py, but parameterized) ----------

def decide(rec: dict, w: float, manual_th: float, min_score: float, min_margin: float) -> str:
    scores = rec["scores"]
    if len(scores) > 1 and scores[0] >= min_score and scores[0] - scores[1] >= min_margin:
        return rec["cands"][0]
    probs = rec["probs"]
    if probs.get(MANUAL, 0) >= manual_th:
        return MANUAL
    best = scores[0]
    combined = {pid: probs.get(pid, 0) + w * (s - best) for pid, s in zip(rec["cands"], scores)}
    return max(combined, key=combined.get)


GRID = list(itertools.product([0, 1, 2, 5, 10, 20], [0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
                              [0.6, 0.65, 0.7, 9.0], [0.05, 0.1, 0.15]))  # min_score 9.0 = fast path off
PROD_PARAMS = (5, 0.7, 0.65, 0.10)


def accuracy(records, params) -> float:
    return sum(decide(r, *params) == r["expected"] for r in records) / len(records) if records else 0.0


def tune(records) -> tuple[tuple, float]:
    best = max(GRID, key=lambda p: accuracy(records, p))
    return best, accuracy(records, best)


def cross_validated(records) -> float:
    """Tune on half the emails, score on the other half, and the other way round."""
    folds = [[r for r in records if r["idx"] % 2 == k] for k in (0, 1)]
    if not all(folds):
        return 0.0
    scores = [accuracy(folds[1 - k], tune(folds[k])[0]) for k in (0, 1)]
    return (scores[0] * len(folds[1]) + scores[1] * len(folds[0])) / len(records)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-hours", type=float, default=6.0, help="Time budget; the report is written when it runs out")
    parser.add_argument("--top-chunkings", type=int, default=3, help="How many chunkings go on to system one")
    parser.add_argument("--rounds", default="0.25,0.5,1.0", help="Fractions of the emails after which weak configs are dropped")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--search-only", action="store_true", help="Stop after phase 1")
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    stopper = GracefulStop()
    breaker = Breaker(stopper)
    cache = Cache()
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()

    people = load_people()
    people_by_id = {p["id"]: p for p in people}
    emails = load_emails(args.seed)
    log(f"{len(emails)} emails ({sum(e['source'] == 'generated' for e in emails)} generated), {len(people)} profiles")

    report = {"started": time.strftime("%Y-%m-%d %H:%M"), "emails": len(emails), "phase1": [], "phase2": [],
              "dropped": [], "exit": "completed"}
    try:
        # ---- phase 1: search only ----
        log("Phase 1: embedding emails and profiles")
        email_vecs = embed_cached([e["email"] for e in emails], cache, breaker)
        profile_vecs = {name: {p["id"]: embed_cached(fn(p), cache, breaker) for p in people}
                        for name, fn in CHUNKERS.items()}
        ids = [p["id"] for p in people]
        expected = np.array([ids.index(e["expected"]) for e in emails])

        sims = {}
        for chunk, agg in itertools.product(CHUNKERS, AGGREGATORS):
            m = person_scores(email_vecs, profile_vecs[chunk], AGGREGATORS[agg])
            sims[(chunk, agg)] = m
            rank = (m > m[np.arange(len(emails)), expected][:, None]).sum(1)  # 0 = correct person is first
            report["phase1"].append({"chunking": chunk, "agg": agg, "top1": float((rank == 0).mean()),
                                     "recall@3": float((rank < 3).mean()), "recall@5": float((rank < 5).mean()),
                                     "mrr": float((1 / (rank + 1)).mean())})
        # MRR rewards putting the right person first, which is what system one and the fast path need
        report["phase1"].sort(key=lambda r: (-r["mrr"], -r["top1"]))
        log("Phase 1 leaderboard (search only):")
        for r in report["phase1"][:10]:
            log(f"  {r['chunking']:<15} {r['agg']:<9} top1 {r['top1']:.1%}  recall@3 {r['recall@3']:.1%}  "
                f"recall@5 {r['recall@5']:.1%}  mrr {r['mrr']:.3f}")
        write_report(report, started)
        if args.search_only:
            report["exit"] = "search-only"
            return

        # ---- phase 2: system one, in rounds with early dropping ----
        # Distinct chunkings only (best aggregation each): with a single vector, max/top2/mean are identical.
        chosen = []
        for r in report["phase1"]:
            if len(chosen) < args.top_chunkings and r["chunking"] not in {c for c, _ in chosen}:
                chosen.append((r["chunking"], r["agg"]))
        if ("lines_prefixed", "max") not in chosen:
            chosen.append(("lines_prefixed", "max"))  # current production, as the baseline
        configs = [(c, a, style, k) for (c, a) in chosen for style in STYLES for k in (3, 5)]
        log(f"Phase 2: {len(configs)} configs: {configs}")

        # per-person similarity of each prefixed line to each email, for building questions
        line_vecs = profile_vecs["lines_prefixed"]
        records = {cfg: [] for cfg in configs}
        alive = list(configs)
        cuts = [int(len(emails) * float(x)) for x in args.rounds.split(",")]
        start = 0
        for cut in cuts:
            log(f"Round up to email {cut}/{len(emails)} with {len(alive)} configs")
            for e in emails[start:cut]:
                line_sims_all = {pid: line_vecs[pid] @ email_vecs[e["idx"]] for pid in ids}
                for cfg in alive:
                    if time.time() > deadline:
                        raise Stop(f"time budget of {args.max_hours}h used up")
                    chunk, agg, style, k = cfg
                    row = sims[(chunk, agg)][e["idx"]]
                    order = np.argsort(-row)[:k]
                    cands = [ids[i] for i in order]
                    q = STYLES[style](cands, people_by_id, line_sims_all)
                    ans = system_one_cached(e["email"], q, cache, breaker)
                    records[cfg].append({"idx": e["idx"], "expected": e["expected"], "source": e["source"],
                                         "cands": cands, "scores": [float(row[i]) for i in order],
                                         "probs": ans.get("probabilities", {}), "conf": ans.get("confidence", 0.0)})
                if (e["idx"] + 1) % 25 == 0:
                    log(f"  {e['idx'] + 1}/{len(emails)} emails, API failures so far {breaker.total_failures}")
            start = cut

            # smart exit per config: drop ones clearly behind the leader (2 standard errors)
            if cut < len(emails) and len(alive) > 2:
                tuned = {cfg: tune(records[cfg])[1] for cfg in alive}
                leader = max(tuned.values())
                margin = 2 * math.sqrt(max(leader * (1 - leader), 0.01) / cut)
                keep = [cfg for cfg in alive if tuned[cfg] >= leader - margin]
                keep = keep if len(keep) >= 2 else sorted(alive, key=tuned.get, reverse=True)[:2]
                for cfg in set(alive) - set(keep):
                    report["dropped"].append({"config": list(cfg), "after": cut, "acc": tuned[cfg], "leader": leader})
                    log(f"  dropped {cfg}: {tuned[cfg]:.1%} vs leader {leader:.1%} (margin {margin:.1%})")
                alive = keep
            report["phase2"] = summarize(records)
            write_report(report, started)

    except Stop as e:
        report["exit"] = f"stopped early: {e}"
        log(report["exit"])
    finally:
        if "records" in locals():
            report["phase2"] = summarize(records)
        write_report(report, started)
        log(f"Report: {RESULTS / 'report.md'}")


def summarize(records: dict) -> list[dict]:
    rows = []
    for cfg, recs in records.items():
        if not recs:
            continue
        params, tuned_acc = tune(recs)
        right = [r["conf"] for r in recs if decide(r, *params) == r["expected"]]
        wrong = [r["conf"] for r in recs if decide(r, *params) != r["expected"]]
        by_source = {s: accuracy([r for r in recs if r["source"] == s], params)
                     for s in ("handwritten", "generated") if any(r["source"] == s for r in recs)}
        rows.append({
            "config": dict(zip(["chunking", "agg", "style", "candidates"], cfg)),
            "emails": len(recs),
            "cv_accuracy": cross_validated(recs),
            "tuned_accuracy": tuned_acc,
            "prod_params_accuracy": accuracy(recs, PROD_PARAMS),
            "best_params": dict(zip(["search_weight", "manual_threshold", "fast_min_score", "fast_min_margin"], params)),
            "by_source": by_source,
            "conf_right": sum(right) / len(right) if right else 0.0,
            "conf_wrong": sum(wrong) / len(wrong) if wrong else 0.0,
            "manual_recall": (accuracy([r for r in recs if r["expected"] == MANUAL], params)
                              if any(r["expected"] == MANUAL for r in recs) else None),
        })
    rows.sort(key=lambda r: (-r["emails"], -r["cv_accuracy"]))
    return rows


def write_report(report: dict, started: float) -> None:
    report["minutes"] = round((time.time() - started) / 60, 1)
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Routing benchmark ({report['started']})", "",
             f"{report['emails']} emails, {report['minutes']} min, exit: {report['exit']}", "",
             "## Phase 1: search only (chunking x aggregation)", "",
             "| chunking | agg | top-1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['chunking']} | {r['agg']} | {r['top1']:.1%} | {r['recall@3']:.1%} | {r['recall@5']:.1%} | {r['mrr']:.3f} |"
              for r in report["phase1"]]
    if report["phase2"]:
        lines += ["", "## Phase 2: with system one", "",
                  "`cv` = tuned on half, scored on the other half (honest). `prod` = current settings. Configs dropped early have fewer emails.", "",
                  "| chunking | agg | style | k | emails | cv acc | tuned acc | prod acc | handwritten / generated | manual | conf right/wrong | best params |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in report["phase2"]:
            c, bp = r["config"], r["best_params"]
            lines.append(f"| {c['chunking']} | {c['agg']} | {c['style']} | {c['candidates']} | {r['emails']} | "
                         f"**{r['cv_accuracy']:.1%}** | {r['tuned_accuracy']:.1%} | {r['prod_params_accuracy']:.1%} | "
                         f"{' / '.join(f'{v:.0%}' for v in r['by_source'].values())} | "
                         f"{'-' if r['manual_recall'] is None else format(r['manual_recall'], '.0%')} | "
                         f"{r['conf_right']:.0%} / {r['conf_wrong']:.0%} | "
                         f"w={bp['search_weight']} manual>={bp['manual_threshold']} fast={bp['fast_min_score']}/{bp['fast_min_margin']} |")
    if report["dropped"]:
        lines += ["", "## Dropped early", ""]
        lines += [f"- {d['config']} after {d['after']} emails: {d['acc']:.1%} vs leader {d['leader']:.1%}" for d in report["dropped"]]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
