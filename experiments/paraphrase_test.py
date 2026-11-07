"""Does one correction carry over to differently worded emails about the same thing?

1. Route every labeled email without memory and split them into mistakes and (as many) correct ones.
2. Have another model reword each of them a few times (cached in experiments/data/paraphrases.csv).
3. Route the rewordings with an empty memory (before), fix every mistake once, route them again (after).
   Mistakes should now go right; the correct group is the control and must not get worse.
Repeated for several memory settings; system one answers come from the cache after the first pass.

    ROUTED_COLLECTION=routed_para .venv/bin/python experiments/paraphrase_test.py data/test_emails.csv \\
        experiments/data/generated_emails.csv
"""
import argparse
import csv
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

import config  # noqa: E402
from clients import embed_texts  # noqa: E402
from decide import MANUAL, clean_email, is_nobody  # noqa: E402
from route_email import add_routing_args, known_people, load_questions, remember, route  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
MODELS = ["DeepSeek-V4-Flash", "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL"]
SLOVENIAN = re.compile(r"[čšžČŠŽ]|\b(je|in|da|ali|sem|ki|za|pri|lahko|prosim|hvala|pozdravljeni)\b", re.IGNORECASE)
_benched: dict[str, float] = {}  # model -> time until which it is skipped
PROMPT = ("Rewrite this email so it asks for the same thing, in the same language, but with clearly different "
          "words and sentence structure, as if a different person wrote it. Output only the new email.\n\n{email}")


def final(result: dict) -> str:
    return MANUAL if result["path"] == "spam" or not result["person"] else result["person"]["person_id"]


def paraphrase(llm: OpenAI, email: str, models: list[str]) -> str:
    """Try the models in order; one that fails is skipped for 10 minutes so it doesn't slow every email."""
    for attempt in range(3):
        for model in models:
            if _benched.get(model, 0) > time.time():
                continue
            try:
                r = llm.chat.completions.create(model=model, messages=[{"role": "user", "content": PROMPT.format(email=email)}],
                                                temperature=0.9, max_tokens=400)
                text = (r.choices[0].message.content or "").strip()
                if text:
                    return text
            except Exception as e:  # next model, then retry after a pause
                print(f"  {model}: {str(e)[:80]} (skipped for 10 min)", flush=True)
            _benched[model] = time.time() + 600
        if all(_benched.get(m, 0) > time.time() for m in models):
            _benched.clear()
        time.sleep(10 * (attempt + 1))
    return ""


def is_slovenian(text: str) -> bool:
    return len(SLOVENIAN.findall(text)) >= 3


def load_paraphrases(llm: OpenAI, originals: list[str], per_email: int, workers: int, models: list[str],
                     cache: Path, models_sl: list[str] | None = None) -> dict[str, list[str]]:
    """models_sl: models for Slovenian emails (e.g. GaMS); the others get `models`."""
    have: dict[str, list[str]] = {}
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                have.setdefault(r["original"], []).append(r["paraphrase"])
    todo = [o for o in originals for _ in range(per_email - len(have.get(o, [])))]
    print(f"paraphrases: {sum(len(v) for v in have.values())} cached, {len(todo)} to generate", flush=True)
    new = not cache.exists()
    with cache.open("a", encoding="utf-8", newline="") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        writer = csv.writer(f)
        if new:
            writer.writerow(["original", "paraphrase"])
        for i, (original, text) in enumerate(zip(todo, pool.map(lambda o: paraphrase(llm, o, models_sl if models_sl and is_slovenian(o) else models), todo)), 1):
            if text:
                writer.writerow([original, text])
                f.flush()
                have.setdefault(original, []).append(text)
            if i % 50 == 0:
                print(f"  generated {i}/{len(todo)}", flush=True)
    return have


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--per-email", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--models", default=",".join(MODELS), help="Rewording models in order of preference")
    parser.add_argument("--models-sl", default="", help="Models for Slovenian emails (default: same as --models)")
    parser.add_argument("--cache", default="paraphrases.csv", help="File in experiments/data with the rewordings")
    add_routing_args(parser)
    args = parser.parse_args()

    if config.ROUTED_COLLECTION == "routed_emails":
        sys.exit("Set ROUTED_COLLECTION to a temporary collection; this test wipes it.")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    llm = OpenAI(base_url=config.HIVECORE_BASE_URL, api_key=config.HIVECORE_API_KEY, timeout=90, max_retries=0)
    spam_q = {k: v for k, v in load_questions().items() if k == "spam"}
    people = known_people(client)

    def wipe() -> None:
        if client.collection_exists(config.ROUTED_COLLECTION):
            client.delete_collection(config.ROUTED_COLLECTION)

    def route_all(emails: list[str], **overrides) -> list[str]:
        a = argparse.Namespace(**{**vars(args), "learn": False, **overrides})
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            return list(pool.map(lambda e: final(route(client, e, a, questions=spam_q)), emails))

    rows = []
    for path in args.paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows += [{"expected": MANUAL if is_nobody(r["expected"]) else r["expected"], "email": r["email"]}
                     for r in csv.DictReader(f) if r.get("email")]

    wipe()
    got = route_all([r["email"] for r in rows], memory=False)
    mistakes = [r for r, g in zip(rows, got) if g != r["expected"]]
    correct = [r for r, g in zip(rows, got) if g == r["expected"]]
    control = random.Random(args.seed).sample(correct, min(len(mistakes), len(correct)))
    print(f"{len(rows)} emails: {len(mistakes)} mistakes, {len(control)} correct ones as control", flush=True)

    texts = load_paraphrases(llm, [r["email"] for r in mistakes + control], args.per_email, args.workers,
                             args.models.split(","), DATA / args.cache,
                             [m for m in args.models_sl.split(",") if m])
    groups = {name: [(r, p) for r in group for p in texts.get(r["email"], [])]
              for name, group in (("mistakes", mistakes), ("control", control))}

    # How similar is a rewording to its original? This decides memory_min_sim.
    pairs = groups["mistakes"] + groups["control"]
    a = np.array(embed_texts([clean_email(r["email"]) for r, _ in pairs]))
    b = np.array(embed_texts([clean_email(p) for _, p in pairs]))
    sims = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print("similarity rewording vs original: " + ", ".join(
        f"p{q} {np.percentile(sims, q):.2f}" for q in (10, 25, 50, 75, 90)), flush=True)

    def score(results: list[str], group: list) -> str:
        ok = sum(g == r["expected"] for g, (r, _) in zip(results, group))
        return f"{ok:>3}/{len(group):<3} ({ok / len(group):>4.0%})"

    wipe()
    before = {name: route_all([p for _, p in g], memory=False) for name, g in groups.items()}
    print(f"\nbefore any fix (memory empty): mistakes {score(before['mistakes'], groups['mistakes'])}  "
          f"control {score(before['control'], groups['control'])}", flush=True)

    for r in mistakes:  # the user fixes each mistake once
        text = clean_email(r["email"])
        remember(client, text, embed_texts([text])[0], people[r["expected"]], source="corrected")
    print(f"fixed {len(mistakes)} mistakes (only the originals; the rewordings were never seen)\n")

    print(f"{'min_sim':>7}  {'weight':>6}  {'mistakes now right':>20}  {'control (must stay)':>20}")
    for min_sim in (0.60, 0.65, 0.70, 0.75):
        for weight in (0.5, 1.0, 2.0):
            res = {name: route_all([p for _, p in g], memory=True, memory_min_sim=min_sim, memory_weight=weight)
                   for name, g in groups.items()}
            print(f"{min_sim:>7.2f}  {weight:>6.1f}  {score(res['mistakes'], groups['mistakes']):>20}  "
                  f"{score(res['control'], groups['control']):>20}", flush=True)
    wipe()


if __name__ == "__main__":
    main()
