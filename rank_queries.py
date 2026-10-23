"""Rank emails from worst-routed to best and export only the worst ones to Postman.

"Worst" = the router is least sure about it. Each email gets a badness score 0-1+:
  0.35 * (1 - system one's probability for the final pick)
  0.25 * (top-2 search scores are close: margin under 0.15)
  0.20 * (best search score is low: under 0.70)
  0.20 * (search and system one picked different people)
  + 1.0 if the file has an "expected" column and the pick is wrong

Output:
  data/worst_queries.csv                         every email, worst first, with reasons
  postman/worst_queries.postman_collection.json  the N worst, as replayable requests
  postman/email_router.postman_environment.json  URLs + API keys for Postman (gitignored)
"""
import argparse
import csv
import json
from pathlib import Path

from qdrant_client import QdrantClient

import config
from clients import classify, embed_texts
from route_email import MANUAL, add_routing_args, is_confident, pick_recipient, recipient_question, shortlist

POSTMAN_DIR = Path("postman")
SCHEMA = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"


def badness(people: list[dict], probs: dict, final: str, expected: str | None) -> tuple[float, list[str]]:
    top = people[0]["score"] if people else 0.0
    margin = people[0]["score"] - people[1]["score"] if len(people) > 1 else 1.0
    llm_pick = max(probs, key=probs.get) if probs else MANUAL
    search_pick = people[0]["person_id"] if people else MANUAL
    p_final = probs.get(final, 0.0)

    parts = {
        f"system one only {p_final:.0%} sure": 0.35 * (1 - p_final),
        f"top-2 search scores {margin:.2f} apart": 0.25 * max(0.0, 1 - margin / 0.15),
        f"best search score only {top:.2f}": 0.20 * max(0.0, min(1.0, (0.70 - top) / 0.30)),
        f"search says {search_pick}, system one says {llm_pick}": 0.20 * (search_pick != llm_pick),
    }
    if expected is not None and final != expected:
        parts[f"WRONG: expected {expected}"] = 1.0
    reasons = [name for name, value in sorted(parts.items(), key=lambda kv: -kv[1]) if value >= 0.05]
    return sum(parts.values()), reasons


def postman_request(name: str, url: str, auth_header: tuple[str, str], body: dict) -> dict:
    return {
        "name": name,
        "request": {
            "method": "POST",
            "header": [
                {"key": auth_header[0], "value": auth_header[1]},
                {"key": "Content-Type", "value": "application/json"},
            ],
            "body": {"mode": "raw", "raw": json.dumps(body, ensure_ascii=False, indent=2),
                     "options": {"raw": {"language": "json"}}},
            "url": url,
        },
    }


def postman_folder(rank: int, row: dict, vector: list[float], people: list[dict], candidates: int) -> dict:
    preview = " ".join(row["email"].split())[:60]
    return {
        "name": f"#{rank} badness {row['badness']:.2f} conf {row['confidence']:.0%} -> {row['got']} | {preview}",
        "description": row["email"]
        + f"\n\nSystem one picked {row['system_one_choice']} with confidence {row['confidence']:.1%}; final pick {row['got']}."
        + "\nWhy it ranked badly: " + "; ".join(row["reasons"].split(" | ")),
        "item": [
            postman_request("1. Embed email", "{{embed_url}}", ("Authorization", "Bearer {{embed_key}}"),
                            {"model": config.EMBEDDING_MODEL, "input": [row["email"]]}),
            postman_request("2. Search colleagues in Qdrant",
                            "{{qdrant_url}}/collections/{{colleagues_collection}}/points/query",
                            ("api-key", "{{qdrant_key}}"),
                            {"query": [vector], "using": "grouped", "limit": candidates,
                             "with_payload": ["person_id", "name", "role"]}),
            postman_request("3. Ask system one", "{{hivecore_url}}/systemone",
                            ("Authorization", "Bearer {{hivecore_key}}"),
                            {"model": config.CHAT_MODEL, "state": row["email"],
                             "questions": {"recipient": recipient_question(people)}}),
        ],
    }


def write_postman(folders: list[dict], source: str) -> None:
    POSTMAN_DIR.mkdir(exist_ok=True)
    collection = {
        "info": {"name": f"Email router - worst queries ({source})", "schema": SCHEMA},
        "item": folders,
    }
    (POSTMAN_DIR / "worst_queries.postman_collection.json").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")

    env = {
        "embed_url": (config.EMBEDDING_API_URL, "default"),
        "embed_key": (config.EMBEDDING_API_KEY, "secret"),
        "hivecore_url": (config.HIVECORE_BASE_URL, "default"),
        "hivecore_key": (config.HIVECORE_API_KEY, "secret"),
        "qdrant_url": (config.QDRANT_URL, "default"),
        "qdrant_key": (config.QDRANT_API_KEY or "", "secret"),
        "colleagues_collection": (config.COLLEAGUES_COLLECTION, "default"),
    }
    (POSTMAN_DIR / "email_router.postman_environment.json").write_text(json.dumps({
        "name": "Email router",
        "values": [{"key": k, "value": v, "type": t, "enabled": True} for k, (v, t) in env.items()],
        "_postman_variable_scope": "environment",
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank emails worst-first and export the worst to Postman.")
    parser.add_argument("path", nargs="?", default="data/my_inputs.csv",
                        help="CSV with an 'email' column (and optionally 'expected')")
    parser.add_argument("--top", type=int, default=10, help="How many of the worst to put in Postman")
    add_routing_args(parser)
    args = parser.parse_args()

    with Path(args.path).open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("email", "").strip()]
    if not rows:
        parser.error(f"no emails in {args.path}")

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    vectors = embed_texts([r["email"] for r in rows])

    ranked = []
    for i, (row, vector) in enumerate(zip(rows, vectors), 1):
        people = shortlist(client, vector, args.candidates)
        # Always ask system one, even when the fast path would skip it, so every email gets the same signals.
        answer = classify(row["email"], {"recipient": recipient_question(people)}, raw=True)["recipient"]
        probs = answer["probabilities"]
        fast = is_confident(people, args.min_score, args.min_margin)
        final = people[0]["person_id"] if fast else pick_recipient(probs, people, args)
        score, reasons = badness(people, probs, final, row.get("expected"))
        ranked.append(({**row, "got": final, "path": "fast" if fast else "llm", "badness": score,
                        "system_one_choice": answer.get("choice"), "confidence": answer.get("confidence", 0.0),
                        "final_pick_prob": probs.get(final, 0.0), "reasons": " | ".join(reasons)}, vector, people))
        if i % 10 == 0 or i == len(rows):
            print(f"  scored {i}/{len(rows)}", flush=True)
    print()

    ranked.sort(key=lambda item: -item[0]["badness"])

    out = Path("data/worst_queries.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        fields = ["badness", "got", "path", *(["expected"] if "expected" in rows[0] else []),
                  "system_one_choice", "confidence", "final_pick_prob", "reasons", "email"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "badness": f"{r['badness']:.2f}", "confidence": f"{r['confidence']:.3f}",
                          "final_pick_prob": f"{r['final_pick_prob']:.3f}"} for r, _, _ in ranked)

    worst = ranked[: args.top]
    write_postman([postman_folder(rank, r, v, p, args.candidates) for rank, (r, v, p) in enumerate(worst, 1)],
                  Path(args.path).name)

    print(f"\nWorst {len(worst)} of {len(ranked)}:")
    for rank, (r, _, _) in enumerate(worst, 1):
        preview = " ".join(r["email"].split())[:70]
        print(f"{rank:>3}. {r['badness']:.2f}  conf {r['confidence']:4.0%} (picked {r['system_one_choice']})"
              f"  -> {r['got']:<14} {preview}")
        print(f"            {r['reasons']}")
    print(f"\nAll {len(ranked)} ranked:  {out}")
    print(f"Postman:  import {POSTMAN_DIR}/worst_queries.postman_collection.json "
          f"and {POSTMAN_DIR}/email_router.postman_environment.json")


if __name__ == "__main__":
    main()
