import argparse
import csv
import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import config
import decide
from clients import SystemOneDown, classify, embed_texts
from decide import MANUAL, clean_email, fast_pick, is_nobody  # noqa: F401 (re-exported for other scripts)
from ingest import ensure_collection
from profiles import get_store
from settings import add_routing_args  # noqa: F401 (re-exported for other scripts)


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


def shortlist(client: QdrantClient, vector: list[float], candidates: int, ranking: str = "window",
              nobody: str = "ranked") -> list[dict]:
    """Candidates for one email, scored from profile vectors kept in memory (see profiles.py)."""
    return get_store(client).shortlist(vector, candidates, ranking, nobody)


def recall(client: QdrantClient, vector: list[float], k: int) -> list:
    """The k most similar past emails in memory (corrections, confirmations and trusted decisions)."""
    if not client.collection_exists(config.ROUTED_COLLECTION):
        return []
    return client.query_points(collection_name=config.ROUTED_COLLECTION, query=vector, limit=max(k, 1)).points


def recipient_question(people: list[dict], hints: dict | None = None) -> dict:
    # Name + role + the two best-matching lines. Shorter role-and-duties descriptions were tried
    # (2026-10-23): same accuracy, but they caught only half as many emails that belong to nobody.
    # hints: similar past emails you corrected or confirmed (decide.memory_hints), shown as examples.
    hints = hints or {}
    quote = lambda texts: "; ".join(f'"{" ".join(t.split())[:200]}"' for t in texts)
    criteria = {p["person_id"]: f"{p['name']}, {p['role']}. Handles: {'; '.join(p['matched'])}"
                + (f". Similar past emails that went to them: {quote(hints[p['person_id']])}" if p["person_id"] in hints else "")
                for p in people}
    criteria[MANUAL] = "None of these people fit; needs a human to triage" + (
        f". Similar past emails that went nowhere: {quote(hints[MANUAL])}" if MANUAL in hints else "")
    return {
        "type": "choice",
        "instructions": "Which colleague should this email be forwarded to?",
        "criteria": criteria,
    }


def combined_scores(probs: dict, people: list[dict], args, prior: dict | None = None) -> dict:
    return decide.combined_scores(probs, people, args.search_weight, prior, getattr(args, "memory_weight", 0.0))


def pick_recipient(probs: dict, people: list[dict], args, prior: dict | None = None) -> str:
    return decide.pick_recipient(probs, people, args.search_weight, args.manual_threshold,
                                 prior, getattr(args, "memory_weight", 0.0))


def ask_system_one(email_text: str, people: list[dict], questions: dict, args,
                   prior: dict | None = None, hints: dict | None = None) -> tuple[str, dict, dict]:
    """Ask system one for the recipient (plus any extra questions) and pick one.

    Each candidate gets P(system one) + search_weight * (search score - best search score);
    "nobody" options are summed and win at manual_threshold (decide.py). prior: memory vote shares,
    weighted by memory_weight. hints: past corrected emails shown to system one as examples.

    Returns (final pick, answers to extra questions, verdict). The verdict holds
    system one's own pick, its confidence in that pick, and the probability it
    gave the final pick (lower when the search score overruled it).
    """
    raw = classify(email_text, {"recipient": recipient_question(people, hints), **questions}, raw=True)
    recipient = raw.pop("recipient", None) or {}
    probs = recipient.get("probabilities", {})
    answers = {qid: ans.get(ans.get("type")) for qid, ans in raw.items()}
    choice = pick_recipient(probs, people, args, prior)
    verdict = {
        "system_one_choice": recipient.get("choice"),
        "confidence": recipient.get("confidence"),
        "final_pick_prob": probs.get(choice),
    }
    if prior:
        verdict["memory"] = {pid: round(share, 2) for pid, share in prior.items()}
    if hints:
        verdict["hints"] = sorted(hints)  # candidates shown a similar corrected email
    if getattr(args, "tiebreak", False) and not is_nobody(choice):
        choice, verdict["tiebreak"] = tiebreak(email_text, people, probs, args, choice)
    return choice, answers, verdict


def tiebreak(email_text: str, people: list[dict], probs: dict, args, choice: str) -> tuple[str, dict | None]:
    """When the top two are close, ask one narrow yes/no question about just those two (tested worse; off)."""
    ranked = sorted(combined_scores(probs, people, args).items(), key=lambda kv: -kv[1])
    if (len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= args.tiebreak_margin
            or is_nobody(ranked[0][0]) or is_nobody(ranked[1][0])):
        return choice, None
    by_id = {p["person_id"]: p for p in people}
    a, b = by_id[ranked[0][0]], by_id[ranked[1][0]]
    describe_person = lambda p: f"{p['role']} ({'; '.join(m.split(': ', 1)[-1] for m in p['matched'])})"
    p_a = classify(email_text, {"a_over_b": {
        "type": "noul",
        "instructions": f"Should this email go to the {describe_person(a)} rather than the {describe_person(b)}?",
    }})["a_over_b"]
    return (a if p_a >= 0.5 else b)["person_id"], {"between": [a["person_id"], b["person_id"]], "p_first": p_a}


def format_answer(question: dict, value) -> str:
    if value is None:
        return "-"
    if question["type"] == "noul":
        return f"{'yes' if value >= 0.5 else 'no'} ({value:.2f})"
    if question["type"] == "score":
        # score is the expected criteria index: 0 = first label, len-1 = last
        labels = question["criteria"]
        return f"{labels[min(max(round(value), 0), len(labels) - 1)]} ({value:.2f})"
    return str(value)


def remember(client: QdrantClient, email_text: str, vector: list[float], person: dict, source: str = "auto") -> None:
    """Add an email to memory. source: corrected / confirmed (from you) or auto (a decision two signals agreed on).
    The same text again replaces the old entry, so a correction overrides an automatic decision."""
    ensure_collection(client, config.ROUTED_COLLECTION, len(vector))
    client.upsert(
        collection_name=config.ROUTED_COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, email_text)),
            vector=vector,
            payload={k: person[k] for k in ("person_id", "name", "email", "role")}
            | {"text": email_text, "source": source, "time": datetime.now().isoformat(timespec="seconds")},
        )],
    )


def fix(client: QdrantClient, email_text: str, person_id: str) -> str:
    """Correct a decision: the email goes into memory as a correction (it outweighs automatic decisions) and
    becomes a real label (data/real_labels.csv) for honest testing and ./run.sh tune. Similar emails then
    follow it through the memory vote; no profile is edited.
    """
    people = known_people(client)
    if person_id not in people:
        return f"Unknown person '{person_id}'. Known: {', '.join(sorted(people))}"
    text = clean_email(email_text)
    remember(client, text, embed_texts([text])[0], people[person_id], source="corrected")
    save_label(email_text, MANUAL if is_nobody(person_id) else person_id, "corrected")
    return (f"Saved: emails like this now go to {people[person_id]['name']} ({person_id}). "
            f"Added to memory and to {config.REAL_LABELS}.")


def known_people(client: QdrantClient) -> dict:
    people = get_store(client).payloads()
    people[MANUAL] = {"person_id": MANUAL, "name": "Manual triage", "email": "", "role": "nobody"}
    return people


def route(client: QdrantClient, email_text: str, args, smart: bool = True, questions: dict | None = None,
          ask_all_questions: bool = False) -> dict:
    """smart=False is the baseline: always ask the LLM, like plain RAG.

    Order: cache (an almost identical past email) -> memory (similar past emails agree) -> fast path
    (clear search match) -> system one (with the memory vote as a prior). Trusted decisions are learned.

    `questions` are extra system-one questions (data/questions.json), answered in the same call as
    the recipient. When system one is skipped, only `fast_path_questions` (routing.toml, normally just spam)
    are asked, unless ask_all_questions (the inbox needs urgency).
    """
    questions = questions or {}
    timer = Timer()
    text = clean_email(email_text) if getattr(args, "clean_emails", True) else email_text
    skip_questions = questions if ask_all_questions else {
        k: v for k, v in questions.items() if k in getattr(args, "fast_path_questions", ["spam"])}
    use_memory = smart and getattr(args, "memory", False)

    def ask_skipped() -> dict:
        try:
            return classify(text, skip_questions) if skip_questions else {}
        except SystemOneDown:
            return {}

    vector = embed_texts([text])[0]
    timer.mark("embed")

    neighbors, hints = [], {}
    if smart:
        hits = recall(client, vector, args.memory_k if use_memory else 1)
        timer.mark("cache")
        hit = hits[0] if hits and hits[0].score >= args.cache_threshold else None
        if hit:
            answers = ask_skipped()
            timer.mark("questions")
            if is_spam(answers, args):
                return {"path": "spam", "person": None, "score": None, "people": [],
                        "answers": answers, "verdict": {}, "timer": timer}
            person = None if is_nobody(hit.payload.get("person_id")) else hit.payload  # a correction to "nobody"
            return {"path": "cache", "person": person, "score": hit.score, "people": [],
                    "answers": answers, "verdict": {}, "timer": timer}
        if use_memory:
            neighbors = [(h.payload["person_id"], h.score, h.payload.get("source", "auto")) for h in hits]
            if getattr(args, "memory_hints", False):
                hints = decide.memory_hints([n + (h.payload.get("text", ""),) for n, h in zip(neighbors, hits)],
                                            args.memory_hint_min_sim, args.memory_hints_per_person)

    people = shortlist(client, vector, args.candidates, args.ranking, args.nobody_candidates)
    timer.mark("search")
    prior = decide.memory_votes(neighbors, args.memory_min_sim) if use_memory else {}

    remembered = (decide.memory_pick(neighbors, args.memory_min_sim, args.memory_min_support, args.memory_min_share)
                  if use_memory else None)
    fast = fast_pick(people, args.min_score, args.min_margin) if smart else None
    if remembered:
        path, choice, verdict = "memory", remembered, {"memory": {pid: round(s, 2) for pid, s in prior.items()}}
        answers = ask_skipped()
        timer.mark("questions")
    elif fast:
        path, choice, verdict = "fast", fast, {}
        answers = ask_skipped()
        timer.mark("questions")
    else:
        path = "llm"
        try:
            choice, answers, verdict = ask_system_one(text, people, questions, args, prior, hints)
        except SystemOneDown as e:
            # Fallback: best search match (nobody profile -> manual), clearly marked in output and log.
            path, answers = "search-only", {}
            best = people[0]["person_id"] if people else MANUAL
            choice = MANUAL if is_nobody(best) else best
            verdict = {"fallback": f"system one down: {str(e)[:120]}"}
        timer.mark("llm")

    if is_spam(answers, args) and not decide.spam_overruled(people, args.spam_override_score):
        path, choice = "spam", MANUAL

    person = None
    if not is_nobody(choice):
        person = next((p for p in people if p["person_id"] == choice), None)
        if person is None and choice in known_people(client):  # a memory pick outside the search shortlist
            person = known_people(client)[choice] | {"score": None}
    if smart and getattr(args, "learn", True) and getattr(args, "auto_learn", True) and person \
            and decide.should_learn(path, verdict, choice):
        remember(client, text, vector, person, source="auto")
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


PATH_LABELS = {"cache": "similar past email", "memory": "past emails like it", "fast": "clear match", "llm": "system one", "spam": "spam filter",
               "search-only": "search only (system one was down)"}


INPUTS_LOG = Path("data/my_inputs.csv")
INPUTS_FIELDS = ["time", "email", "got", "path", "search_score", "system_one_choice", "confidence", "final_pick_prob"]


def fmt_prob(value) -> str:
    return "" if value is None else f"{value:.3f}"


def final_id(result: dict) -> str:
    return "spam" if result["path"] == "spam" else (result["person"] or {}).get("person_id", MANUAL)


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
            "got": final_id(result),
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
    if "fallback" in verdict:
        return f"- ({verdict['fallback']})"
    if result["path"] == "memory":
        return "- (past emails voted " + ", ".join(f"{p} {s:.0%}" for p, s in verdict["memory"].items()) + ")"
    final =(result["person"] or {}).get("person_id", MANUAL)
    text = f"{verdict['confidence']:.0%} (system one picked {verdict['system_one_choice']})"
    if verdict["system_one_choice"] != final:
        text += f"; it gave the final pick {final} only {(verdict['final_pick_prob'] or 0):.0%}"
    return text


def email_id(email_text: str) -> str:
    return hashlib.sha256(" ".join(email_text.split()).encode("utf-8")).hexdigest()[:16]


def log_decision(email_text: str, result: dict, source: str = "cli") -> None:
    """Append one JSON line per routed email: the audit trail behind `./run.sh summary`."""
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "id": email_id(email_text),
        "source": source,
        "email": email_text,
        "final": final_id(result),
        "path": result["path"],
        "search_score": result["score"],
        "candidates": [{"id": p["person_id"], "score": round(p["score"], 4)} for p in result["people"]],
        "verdict": result["verdict"],
        "answers": result["answers"],
        "model": config.CHAT_MODEL,
        "ms": round(result["timer"].total),
    }
    path = Path(config.DECISIONS_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def save_label(email_text: str, expected: str, source: str) -> None:
    """Real labeled data (expected,email) from corrections and confirmations; test it with ./run.sh test FILE."""
    path = Path(config.REAL_LABELS)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["expected", "email", "source", "time"])
        writer.writerow([expected, email_text, source, datetime.now().isoformat(timespec="seconds")])


def confirm(email_text: str, client: QdrantClient | None = None) -> str:
    """Mark the last logged decision for this email as right (and, with a client, add it to memory)."""
    path, wanted, last = Path(config.DECISIONS_LOG), email_id(email_text), None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["id"] == wanted:
                last = entry
    if last is None:
        return "No logged decision for this email. Route it first, or use fix to give the right person."
    expected = MANUAL if last["final"] == "spam" or is_nobody(last["final"]) else last["final"]
    save_label(email_text, expected, "confirmed")
    if client is not None and expected in known_people(client):
        text = clean_email(email_text)
        remember(client, text, embed_texts([text])[0], known_people(client)[expected], source="confirmed")
    return f"Confirmed: {expected} was right. Saved to memory and {config.REAL_LABELS}."


def load_questions(path: str = "data/questions.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def run_single(client: QdrantClient, email_text: str, args, questions: dict) -> None:
    result = route(client, email_text, args, questions=questions)
    save_input(email_text, result)
    log_decision(email_text, result)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide which colleague an email should be forwarded to.")
    parser.add_argument("email", nargs="?", help="Email text, or a path to a .txt/.eml file")
    add_routing_args(parser)
    parser.add_argument("--no-learn", dest="learn", action="store_false", help="Don't add decisions to memory")
    parser.add_argument("--benchmark", metavar="FILE", help="File with one email per line; compare smart vs always-LLM")
    parser.add_argument("--forget", action="store_true", help="Delete all cached routing decisions and exit")
    parser.add_argument("--fix", metavar="PERSON_ID", help="Correct the recipient for this email (memory + label)")
    parser.add_argument("--ok", action="store_true", help="Confirm the last decision for this email was right (adds a real label)")
    parser.add_argument("--questions", default="data/questions.json", help="Extra system-one questions to ask per email")
    parser.add_argument("--details", action="store_true", help="Also show match scores and timings")
    args = parser.parse_args()

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    read = lambda arg: Path(arg).read_text(encoding="utf-8") if Path(arg).is_file() else arg

    if args.ok:
        if not args.email:
            parser.error("--ok needs the email text (or file) too")
        print(confirm(read(args.email), client))
        return

    if args.fix:
        if not args.email:
            parser.error("--fix needs the email text (or file) too")
        print(fix(client, read(args.email), args.fix))
        return

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
    run_single(client, read(args.email), args, load_questions(args.questions))


if __name__ == "__main__":
    main()
