"""Suggest new duty lines for colleague profiles, learned from corrections. You approve; nothing changes by itself.

An email teaches something when it belongs to a person but search did not rank them first: their profile does
not describe that kind of request yet. Such emails are grouped per person (similar ones together); for every
group of at least --min-group emails, a model (gemma-4-12b on hivecore, not the system-one family) writes one
short duty line. Suggestions are saved to data/profile_suggestions.csv.

    ./run.sh suggest                        from your corrections and confirmations (memory)
    ./run.sh suggest --labels file.csv      from any labeled file (expected,email)
    ./run.sh suggest --apply 1,3            add suggestions 1 and 3 to data/colleagues.csv (then ./run.sh load)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient

import config
from clients import embed_emails
from decide import clean_email, is_nobody
from profiles import get_store

SUGGESTIONS = Path("data/profile_suggestions.csv")
MODEL = "gemma-4-12b"
PROMPT = """{name} ({role}) handles these kinds of requests:
{duties}

These emails were also meant for {name}, but nothing in the list above describes them:
{emails}

Write ONE new line for the list (at most 12 words, in English, same style as the list) that covers what these
emails have in common. Make it general enough for similar future emails, not about one specific case.
Output only the line."""


def labels_from_memory(client: QdrantClient) -> list[dict]:
    if not client.collection_exists(config.ROUTED_COLLECTION):
        return []
    points, offset, rows = [], None, []
    while True:
        batch, offset = client.scroll(config.ROUTED_COLLECTION, limit=1000, offset=offset, with_payload=True)
        points += batch
        if offset is None:
            break
    for p in points:
        if p.payload.get("source") in ("corrected", "confirmed"):
            rows.append({"expected": p.payload["person_id"], "email": p.payload.get("text", "")})
    return rows


def labels_from_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{"expected": r["expected"], "email": r["email"]} for r in csv.DictReader(f) if r.get("email")]


def duty_lines(payload: dict) -> list[str]:
    return [t.split(": ", 1)[-1] for t in payload.get("texts", []) if " handles emails like: " not in t]


def groups(vectors: np.ndarray, min_sim: float) -> list[list[int]]:
    """Greedy grouping: each email joins the first group whose first email is at least min_sim similar."""
    found: list[list[int]] = []
    for i, v in enumerate(vectors):
        for g in found:
            if float(vectors[g[0]] @ v) >= min_sim:
                g.append(i)
                break
        else:
            found.append([i])
    return sorted(found, key=len, reverse=True)


def suggest(client: QdrantClient, rows: list[dict], min_group: int, group_sim: float) -> list[dict]:
    store = get_store(client)
    payloads = store.payloads()
    rows = [r for r in rows if not is_nobody(r["expected"]) and r["expected"] in payloads]
    texts = [clean_email(r["email"]) for r in rows]
    vectors = np.array(embed_emails(texts), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    uncovered: dict[str, list[int]] = {}
    for i, (r, v) in enumerate(zip(rows, vectors)):
        ranked = store.shortlist(v, candidates=1, nobody="always")
        best_real = next((p["person_id"] for p in ranked if not is_nobody(p["person_id"])), None)
        if best_real != r["expected"]:
            uncovered.setdefault(r["expected"], []).append(i)
    print(f"{len(rows)} labeled emails, {sum(map(len, uncovered.values()))} where search did not rank the right "
          f"person first, spread over {len(uncovered)} people", flush=True)

    llm = OpenAI(base_url=config.HIVECORE_BASE_URL, api_key=config.HIVECORE_API_KEY, timeout=90, max_retries=2)
    out = []
    for pid, idx in sorted(uncovered.items()):
        payload = payloads[pid]
        for g in groups(vectors[idx], group_sim):
            if len(g) < min_group:
                continue
            emails = [texts[idx[j]] for j in g]
            prompt = PROMPT.format(name=payload["name"], role=payload["role"],
                                   duties="\n".join(f"- {d}" for d in duty_lines(payload)),
                                   emails="\n".join(f'- "{" ".join(e.split())[:300]}"' for e in emails[:8]))
            r = llm.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                            temperature=0.2, max_tokens=60)
            line = (r.choices[0].message.content or "").strip().strip('-"* ').splitlines()[0] if r.choices else ""
            if line:
                out.append({"id": len(out) + 1, "person_id": pid, "line": line, "emails": len(g),
                            "examples": " | ".join(" ".join(e.split())[:120] for e in emails[:3])})
    return out


def apply(ids: list[int], colleagues: str, out_path: str | None = None) -> None:
    with SUGGESTIONS.open(encoding="utf-8", newline="") as f:
        chosen = [s for s in csv.DictReader(f) if int(s["id"]) in ids]
    with open(colleagues, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, people = reader.fieldnames, list(reader)
    for s in chosen:
        for p in people:
            if p["id"] == s["person_id"]:
                p["responsibilities"] = "|".join(x for x in [p["responsibilities"], s["line"]] if x)
                print(f"  {s['person_id']}: + {s['line']}")
    with open(out_path or colleagues, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(people)
    print(f"Wrote {out_path or colleagues}. Run ./run.sh load to embed the changed profiles.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", help="Labeled CSV (expected,email) instead of the corrections in memory")
    parser.add_argument("--min-group", type=int, default=3, help="Similar emails needed for one suggestion")
    parser.add_argument("--group-sim", type=float, default=0.6, help="Similarity for emails to count as one group")
    parser.add_argument("--apply", help="Comma-separated suggestion numbers to add to the colleagues file")
    parser.add_argument("--apply-all", action="store_true", help="Add every suggestion (used by the evaluation)")
    parser.add_argument("--colleagues", default="data/colleagues.csv")
    parser.add_argument("--out", help="With --apply/--apply-all: write the changed colleagues file here instead")
    args = parser.parse_args()

    if args.apply or args.apply_all:
        if not SUGGESTIONS.exists():
            sys.exit("No suggestions yet; run without --apply first.")
        with SUGGESTIONS.open(encoding="utf-8", newline="") as f:
            all_ids = [int(s["id"]) for s in csv.DictReader(f)]
        apply(all_ids if args.apply_all else [int(x) for x in args.apply.split(",")], args.colleagues, args.out)
        return

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    rows = labels_from_csv(args.labels) if args.labels else labels_from_memory(client)
    found = suggest(client, rows, args.min_group, args.group_sim)
    SUGGESTIONS.parent.mkdir(parents=True, exist_ok=True)
    with SUGGESTIONS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "person_id", "line", "emails", "examples"])
        writer.writeheader()
        writer.writerows(found)
    print(f"\n{len(found)} suggestions (saved to {SUGGESTIONS}):")
    for s in found:
        print(f"  {s['id']:>2}. {s['person_id']:<12} + {s['line']}   ({s['emails']} emails, e.g. {s['examples'][:90]})")
    if found:
        print("\nAdd the good ones with: ./run.sh suggest --apply 1,3   (then ./run.sh load)")


if __name__ == "__main__":
    main()
