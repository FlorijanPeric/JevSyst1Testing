import argparse
import csv
import json
import time
from datetime import datetime
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import config
from clients import classify, embed_texts
from ingest import ensure_collection

MANUAL = "manual_triage"


class Timer:
    def __init__(self):
        self.stages: dict[str, float] = {}
        self._last = time.perf_counter()

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages[stage] = (now - self._last) * 1000
        self._last = now

    @property
    def total(self) -> float:
        return sum(self.stages.values())


def shortlist(client: QdrantClient, vector: list[float], candidates: int) -> list[dict]:
    """Score every colleague locally and return the best `candidates`.

    Score = average similarity to the person's two "grouped" vectors (duties, example emails);
    Qdrant's multivector search only does best-match (MaxSim), which ranked worse in the benchmark.
    Fetching all points is fine for a team of tens of people; for thousands, prefilter with a Qdrant query.
    "matched" = the person's two best-matching lines, which system one sees.
    """
    query = np.asarray(vector) / np.linalg.norm(vector)

    def sims(vecs) -> np.ndarray:
        vecs = np.asarray(vecs)
        return vecs @ query / np.linalg.norm(vecs, axis=1)

    # Small download for everyone (2 vectors each), line vectors only for the winners.
    points, _ = client.scroll(config.COLLEAGUES_COLLECTION, limit=10_000, with_payload=False, with_vectors=["grouped"])
    scores = {point.id: float(sims(point.vector["grouped"]).mean()) for point in points}
    top = sorted(scores, key=scores.get, reverse=True)[:candidates]
    details = {point.id: point for point in client.retrieve(config.COLLEAGUES_COLLECTION, ids=top,
                                                             with_payload=True, with_vectors=["lines"])}
    people = []
    for point_id in top:
        point = details[point_id]
        line_sims = sims(point.vector["lines"])
        people.append({
            "person_id": point.payload["person_id"],
            "score": scores[point_id],
            **{k: point.payload[k] for k in ("name", "email", "role")},
            "matched": [point.payload["texts"][i] for i in np.argsort(-line_sims)[:2]],
        })
    return people


def cache_lookup(client: QdrantClient, vector: list[float], threshold: float):
    if not client.collection_exists(config.ROUTED_COLLECTION):
        return None
    hits = client.query_points(collection_name=config.ROUTED_COLLECTION, query=vector, limit=1).points
    return hits[0] if hits and hits[0].score >= threshold else None


def is_confident(people: list[dict], min_score: float, min_margin: float) -> bool:
    if not people or people[0]["score"] < min_score:
        return False
    return len(people) == 1 or people[0]["score"] - people[1]["score"] >= min_margin


def recipient_question(people: list[dict]) -> dict:
    # Name + role + the two best-matching lines. Shorter role-and-duties descriptions were tried
    # (2026-10-23): same accuracy, but they caught only half as many emails that belong to nobody.
    criteria = {p["person_id"]: f"{p['name']}, {p['role']}. Handles: {'; '.join(p['matched'])}" for p in people}
    criteria[MANUAL] = "None of these people fit; needs a human to triage"
    return {
        "type": "choice",
        "instructions": "Which colleague should this email be forwarded to?",
        "criteria": criteria,
    }


def ask_system_one(email_text: str, people: list[dict], questions: dict, args) -> tuple[str, dict, dict]:
    """Ask system one for the recipient (plus any extra questions) and pick one.

    The small model alone picks "manual" too eagerly, so each candidate gets
    P(system one) + search_weight * (search score - best search score), and
    "manual" wins only if system one gives it at least manual_threshold.
    On data/test_emails.csv this beat both search-only and system-one-only.

    Returns (final pick, answers to extra questions, verdict). The verdict holds
    system one's own pick, its confidence in that pick, and the probability it
    gave the final pick (lower when the search score overruled it).
    """
    raw = classify(email_text, {"recipient": recipient_question(people), **questions}, raw=True)
    recipient = raw.pop("recipient", None) or {}
    probs = recipient.get("probabilities", {})
    answers = {qid: ans.get(ans.get("type")) for qid, ans in raw.items()}
    choice = pick_recipient(probs, people, args)
    verdict = {
        "system_one_choice": recipient.get("choice"),
        "confidence": recipient.get("confidence"),
        "final_pick_prob": probs.get(choice),
    }
    return choice, answers, verdict


def pick_recipient(probs: dict, people: list[dict], args) -> str:
    if not people or probs.get(MANUAL, 0) >= args.manual_threshold:
        return MANUAL
    best_score = people[0]["score"]
    combined = {p["person_id"]: probs.get(p["person_id"], 0) + args.search_weight * (p["score"] - best_score)
                for p in people}
    return max(combined, key=combined.get)


def format_answer(question: dict, value) -> str:
    if value is None:
        return "?"
    if question["type"] == "noul":
        return f"{'yes' if value >= 0.5 else 'no'} ({value:.2f})"
    if question["type"] == "score":
        # score is the expected criteria index: 0 = first label, len-1 = last
        labels = question["criteria"]
        return f"{labels[min(max(round(value), 0), len(labels) - 1)]} ({value:.2f})"
    return str(value)


def remember(client: QdrantClient, email_text: str, vector: list[float], person: dict) -> None:
    ensure_collection(client, config.ROUTED_COLLECTION, len(vector))
    client.upsert(
        collection_name=config.ROUTED_COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, email_text)),
            vector=vector,
            payload={k: person[k] for k in ("person_id", "name", "email", "role")} | {"text": email_text},
        )],
    )


def route(client: QdrantClient, email_text: str, args, smart: bool = True, questions: dict | None = None) -> dict:
    """smart=False is the baseline: always ask the LLM, like plain RAG.

    `questions` are extra system-one questions (data/questions.json) answered in
    the same call as the recipient, or in their own call when routing skips the LLM.
    """
    questions = questions or {}
    timer = Timer()
    vector = embed_texts([email_text])[0]
    timer.mark("embed")

    if smart:
        hit = cache_lookup(client, vector, args.cache_threshold)
        timer.mark("cache")
        if hit:
            answers = classify(email_text, questions) if questions else {}
            timer.mark("questions")
            if is_spam(answers, args):
                return {"path": "spam", "person": None, "score": None, "people": [],
                        "answers": answers, "verdict": {}, "timer": timer}
            return {"path": "cache", "person": hit.payload, "score": hit.score, "people": [],
                    "answers": answers, "verdict": {}, "timer": timer}

    people = shortlist(client, vector, args.candidates)
    timer.mark("search")

    if smart and is_confident(people, args.min_score, args.min_margin):
        path, choice, verdict = "fast", people[0]["person_id"], {}
        answers = classify(email_text, questions) if questions else {}
        timer.mark("questions")
    else:
        path = "llm"
        choice, answers, verdict = ask_system_one(email_text, people, questions, args)
        timer.mark("llm")

    if is_spam(answers, args):
        path, choice = "spam", MANUAL

    # data/colleagues.csv has a "manual_triage" profile (spam, off-topic) so search can spot those too
    person = None if choice == MANUAL else next((p for p in people if p["person_id"] == choice), None)
    if smart and args.learn and person:
        remember(client, email_text, vector, person)
        timer.mark("remember")

    return {"path": path, "person": person, "score": person["score"] if person else None, "people": people,
            "answers": answers, "verdict": verdict, "timer": timer}


def is_spam(answers: dict, args) -> bool:
    """The "spam" question in data/questions.json (if present) blocks forwarding above the threshold."""
    return (answers.get("spam") or 0) >= args.spam_threshold


def describe(result: dict) -> str:
    person = result["person"]
    if result["path"] == "spam":
        return f"nobody, spam ({result['answers']['spam']:.2f})"
    return f"{person['name']} <{person['email']}>" if person else "manual triage"


PATH_LABELS = {"cache": "similar past email", "fast": "clear match", "llm": "system one", "spam": "spam filter"}


INPUTS_LOG = Path("data/my_inputs.csv")


INPUTS_FIELDS = ["time", "email", "got", "path", "search_score", "system_one_choice", "confidence", "final_pick_prob"]


def fmt_prob(value) -> str:
    return "" if value is None else f"{value:.3f}"


def save_input(email_text: str, result: dict) -> None:
    """Keep every email typed in with its decision, so rank_queries.py can find the worst ones later."""
    new = not INPUTS_LOG.exists()
    verdict = result["verdict"]
    with INPUTS_LOG.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INPUTS_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({
            "time": datetime.now().isoformat(timespec="seconds"),
            "email": email_text,
            "got": "spam" if result["path"] == "spam" else (result["person"] or {}).get("person_id", MANUAL),
            "path": result["path"],
            "search_score": fmt_prob(result["score"]),
            "system_one_choice": verdict.get("system_one_choice") or "",
            "confidence": fmt_prob(verdict.get("confidence")),
            "final_pick_prob": fmt_prob(verdict.get("final_pick_prob")),
        })


def describe_confidence(result: dict) -> str:
    verdict = result["verdict"]
    if not verdict:
        return "- (system one was not needed)"
    final = (result["person"] or {}).get("person_id", MANUAL)
    text = f"{verdict['confidence']:.0%} (system one picked {verdict['system_one_choice']})"
    if verdict["system_one_choice"] != final:
        text += f"; it gave the final pick {final} only {(verdict['final_pick_prob'] or 0):.0%}"
    return text


def run_single(client: QdrantClient, email_text: str, args, questions: dict) -> None:
    result = route(client, email_text, args, questions=questions)
    save_input(email_text, result)
    person = result["person"]
    preview = " ".join(email_text.split())

    print("-" * 60)
    print(f"Email:       {preview[:70]}{'...' if len(preview) > 70 else ''}")
    print(f"Forward to:  {describe(result)}" + (f"  ({person['role']})" if person else ""))
    for qid, question in questions.items():
        print(f"{qid.replace('_', ' ').capitalize() + ':':<13}{format_answer(question, result['answers'].get(qid))}")
    print(f"Confidence:  {describe_confidence(result)}")
    print(f"Decided by:  {PATH_LABELS[result['path']]}, {result['timer'].total / 1000:.1f}s")

    if args.details:
        print("\nTop matches:")
        for p in result["people"]:
            print(f"  {p['score']:.3f}  {p['name']} ({p['role']})  <- {p['matched'][0]}")
        print("Timings:     " + ", ".join(f"{k} {v:.0f}ms" for k, v in result["timer"].stages.items()))


def run_benchmark(client: QdrantClient, emails: list[str], args) -> None:
    rows = []
    for email_text in emails:
        base = route(client, email_text, args, smart=False)
        fast = route(client, email_text, args, smart=True)
        rows.append((email_text, base, fast))
        agree = (base["person"] or {}).get("person_id") == (fast["person"] or {}).get("person_id")
        print(f"{fast['path']:>5}  {fast['timer'].total:7.0f}ms vs {base['timer'].total:7.0f}ms  "
              f"{'ok  ' if agree else 'DIFF'}  {describe(fast):<35} {email_text[:50]}")

    base_total = sum(b["timer"].total for _, b, _ in rows)
    fast_total = sum(f["timer"].total for _, _, f in rows)
    agreed = sum((b["person"] or {}).get("person_id") == (f["person"] or {}).get("person_id") for _, b, f in rows)
    paths = {p: sum(f["path"] == p for _, _, f in rows) for p in ("cache", "fast", "llm")}

    print(f"\nEmails:            {len(rows)}")
    print(f"Paths taken:       cache {paths['cache']}, fast {paths['fast']}, llm {paths['llm']}")
    print(f"Always-LLM total:  {base_total:.0f}ms  ({base_total / len(rows):.0f}ms/email)")
    print(f"Smart total:       {fast_total:.0f}ms  ({fast_total / len(rows):.0f}ms/email)")
    print(f"Speedup:           {base_total / fast_total:.2f}x")
    print(f"Agreement w/ LLM:  {agreed}/{len(rows)}")


def add_routing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates", type=int, default=5, help="How many colleagues system one chooses from")
    # 0.50 / 0.07: on 436 benchmark emails, 23% skip system one and 98% of those are right.
    parser.add_argument("--min-score", type=float, default=0.50, help="Fast path: top match must score at least this")
    parser.add_argument("--min-margin", type=float, default=0.07, help="Fast path: top match must beat #2 by this")
    parser.add_argument("--search-weight", type=float, default=5.0, help="How much the search score counts next to system one")
    parser.add_argument("--manual-threshold", type=float, default=0.7, help="System one must be this sure to pick manual triage")
    parser.add_argument("--spam-threshold", type=float, default=0.8, help="Don't forward emails whose spam probability is at least this")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide which colleague an email should be forwarded to.")
    parser.add_argument("email", nargs="?", help="Email text, or a path to a .txt/.eml file")
    add_routing_args(parser)
    parser.add_argument("--cache-threshold", type=float, default=0.95, help="Reuse a past decision above this similarity")
    parser.add_argument("--no-learn", dest="learn", action="store_false", help="Don't store decisions in the cache")
    parser.add_argument("--benchmark", metavar="FILE", help="File with one email per line; compare smart vs always-LLM")
    parser.add_argument("--forget", action="store_true", help="Delete all cached routing decisions and exit")
    parser.add_argument("--questions", default="data/questions.json", help="Extra system-one questions to ask per email")
    parser.add_argument("--details", action="store_true", help="Also show match scores and timings")
    args = parser.parse_args()

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    if args.forget:
        if client.collection_exists(config.ROUTED_COLLECTION):
            client.delete_collection(config.ROUTED_COLLECTION)
        print(f"Cleared {config.ROUTED_COLLECTION}.")
        return

    if args.benchmark:
        lines = Path(args.benchmark).read_text(encoding="utf-8").splitlines()
        run_benchmark(client, [l.strip() for l in lines if l.strip() and not l.startswith("#")], args)
        return

    if not args.email:
        parser.error("pass an email, --benchmark FILE, or --forget")
    path = Path(args.email)
    questions_path = Path(args.questions)
    questions = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.is_file() else {}
    run_single(client, path.read_text(encoding="utf-8") if path.is_file() else args.email, args, questions)


if __name__ == "__main__":
    main()
