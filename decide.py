"""Pure routing logic: no network, no Qdrant, no hivecore. Unit-tested in tests/test_decide.py."""
import re

MANUAL = "manual_triage"


def is_nobody(person_id: str | None) -> bool:
    """manual_triage, or one of the "nobody" profiles (manual_spam, manual_personal, manual_wrong, ...)."""
    return person_id == MANUAL or (person_id or "").startswith("manual_")


# ---------- email cleaning ----------

_REPLY_HEADER = re.compile(
    r"^\s*(On .{4,200} wrote:|Dne .{4,200} (je )?(napisal|napisala|pisal|pisala).{0,20}:"
    r"|-{2,}\s*(Original Message|Forwarded message|Izvirno sporočilo|Posredovano sporočilo)\s*-{2,}"
    r"|(From|Od): .*\S+@\S+.*)\s*$",  # only a header line with an address, not a sentence starting "From:"
    re.IGNORECASE,
)
_SIGNATURE = re.compile(r"^\s*(--|__+|Sent from my \w+|Poslano z mojega \w+)\s*$", re.IGNORECASE)
_FOOTER = re.compile(
    r"(this (e-?mail|message) (and any attachments )?(is|are|may be) (confidential|intended)"
    r"|to je (zaupno|zaupna)|unsubscribe|odjava od prejemanja)",
    re.IGNORECASE,
)


def clean_email(text: str) -> str:
    """Keep only the new part of an email: drop quoted replies, signatures and legal or newsletter footers.

    Falls back to the original text if cleaning would leave almost nothing.
    """
    kept = []
    for line in text.splitlines():
        if _REPLY_HEADER.match(line) or _SIGNATURE.match(line):
            break  # everything below is the previous email or the signature
        if line.lstrip().startswith(">") or _FOOTER.search(line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned if len(cleaned) >= 15 else text.strip()


# ---------- candidates ----------

def select_candidates(scores: dict, nobody_ids: set, k: int, mode: str = "ranked",
                      must_include: str | None = None) -> list:
    """Ids shown to system one, best score first.

    ranked: the top k overall (nobody profiles only if they rank there), topped up with real people so a
            nobody profile never pushes the right person out (generated 281/330).
    always: k real people plus every nobody profile (fewer search misses, but 272-274/330).
    must_include: the fast-path pick, which has to be in the list it is picked from.
    """
    ranked = sorted(scores, key=scores.get, reverse=True)
    real = [pid for pid in ranked if pid not in nobody_ids]
    if mode == "always":
        top = real[:k] + [pid for pid in ranked if pid in nobody_ids]
    else:
        top = ranked[:k]
        top += real[sum(pid not in nobody_ids for pid in top):k]
    if must_include is not None and must_include not in top:
        top.append(must_include)
    return sorted(top, key=lambda pid: -scores[pid])


def rerank_select(pool: list[dict], rerank_scores: dict, k: int) -> list[dict]:
    """Candidates for system one after reranking a bigger search pool.

    Real people: the k best by reranker score. Nobody profiles: kept only if they were in the top k by search
    score (as without reranking). The best grouped match stays in, so the fast path sees the same list.
    Returned best search score first, like the search shortlist.
    """
    by_score = sorted(pool, key=lambda p: -p["score"])
    nobody = [p for p in by_score[:k] if is_nobody(p["person_id"])]
    real = sorted((p for p in pool if not is_nobody(p["person_id"])),
                  key=lambda p: -rerank_scores.get(p["person_id"], 0.0))[:k]
    chosen = real + nobody
    best_grouped = max(pool, key=lambda p: p["grouped_score"]) if pool else None
    if best_grouped is not None and best_grouped not in chosen:
        chosen.append(best_grouped)
    return sorted(chosen, key=lambda p: -p["score"])


# ---------- decisions ----------

def fast_pick(people: list[dict], min_score: float, min_margin: float) -> str | None:
    """The person to forward to without asking system one, or None if the match isn't clear enough."""
    if not people:
        return None
    best = max(people, key=lambda p: p["grouped_score"])
    if best["grouped_score"] >= min_score and best["grouped_margin"] >= min_margin:
        return MANUAL if is_nobody(best["person_id"]) else best["person_id"]
    return None


def spam_overruled(people: list[dict], override_score: float) -> bool:
    """A real person matches this clearly, so a high spam probability is more likely a report about spam
    (".exe invoice, is this safe?") or an odd but real request than spam itself."""
    real = [p["grouped_score"] for p in people if not is_nobody(p["person_id"])]
    return bool(real) and max(real) >= override_score


# ---------- memory: past labeled emails (corrections, confirmations, trusted decisions) ----------

TRUST = {"corrected": 1.0, "confirmed": 1.0, "auto": 0.5}


def memory_votes(neighbors: list[tuple[str, float, str]], min_sim: float) -> dict:
    """Weighted votes from the nearest past emails: (person_id, similarity, source) -> {person_id: share}.

    Each neighbor at least min_sim similar votes with similarity * trust of its source; every nobody
    label counts for MANUAL. Shares add up to 1 (empty when nothing is similar enough).
    """
    votes: dict[str, float] = {}
    for pid, sim, source in neighbors:
        if sim >= min_sim:
            key = MANUAL if is_nobody(pid) else pid
            votes[key] = votes.get(key, 0.0) + sim * TRUST.get(source, TRUST["auto"])
    total = sum(votes.values())
    return {pid: v / total for pid, v in votes.items()} if total else {}


def memory_pick(neighbors: list[tuple[str, float, str]], min_sim: float, min_support: float,
                min_share: float) -> str | None:
    """The person to forward to from memory alone, or None.

    Support is the summed trust of the similar past emails: one correction (1.0) or two automatic decisions
    (0.5 each) reach 1.0. The winner must also hold at least min_share of the weighted vote."""
    votes = memory_votes(neighbors, min_sim)
    support = sum(TRUST.get(source, TRUST["auto"]) for _, sim, source in neighbors if sim >= min_sim)
    if not votes or support < min_support:
        return None
    best = max(votes, key=votes.get)
    return best if votes[best] >= min_share else None


def memory_hints(neighbors: list[tuple[str, float, str, str]], min_sim: float, per_person: int = 1) -> dict:
    """Past emails you corrected or confirmed, to show system one as examples: (person_id, similarity, source,
    text) -> {person_id: [text, ...]}, most similar first. Lower bar than voting: system one still decides,
    so a correction that is too far away to vote (e.g. 0.63) can still help it."""
    hints: dict[str, list[str]] = {}
    for pid, sim, source, text in sorted(neighbors, key=lambda n: -n[1]):
        key = MANUAL if is_nobody(pid) else pid
        if sim >= min_sim and source in ("corrected", "confirmed") and text and len(hints.get(key, [])) < per_person:
            hints.setdefault(key, []).append(text)
    return hints


def should_learn(path: str, verdict: dict, choice: str) -> bool:
    """Which automatic decisions become memory: only ones two independent signals agree on.

    fast: the search match was clear. llm: system one's own pick is the final pick (search didn't overrule it).
    Never spam, nobody, fallbacks or memory decisions (those would only repeat what memory already holds).
    """
    if is_nobody(choice):
        return False
    if path == "fast":
        return True
    return path == "llm" and verdict.get("system_one_choice") == choice


def rerank_select(pool: list[dict], rerank_scores: dict, k: int) -> list[dict]:
    """Candidates for system one after reranking a bigger search pool.

    Real people: the k best by reranker score. Nobody profiles: kept only if they were in the top k by search
    score (as without reranking). The best grouped match stays in, so the fast path sees the same list.
    Returned best search score first, like the search shortlist.
    """
    by_score = sorted(pool, key=lambda p: -p["score"])
    nobody = [p for p in by_score[:k] if is_nobody(p["person_id"])]
    real = sorted((p for p in pool if not is_nobody(p["person_id"])),
                  key=lambda p: -rerank_scores.get(p["person_id"], 0.0))[:k]
    chosen = real + nobody
    best_grouped = max(pool, key=lambda p: p["grouped_score"]) if pool else None
    if best_grouped is not None and best_grouped not in chosen:
        chosen.append(best_grouped)
    return sorted(chosen, key=lambda p: -p["score"])


def safe_to_auto(path: str, verdict: dict, choice: str, min_confidence: float, on_agreement: bool = False,
                 fast_min_margin: float = 0.0) -> bool:
    """Whether a decision is safe to forward without a human: clear matches and memory (95-100% right in
    tests), and system one picks it was at least min_confidence sure of and that search did not overrule
    (min_confidence comes from experiments/calibration.py; above 1 means never). on_agreement: also system
    one picks that every second-opinion model agreed with (models_agree). A clear match needs a lead of
    fast_min_margin, and every check model asked about it (auto_check) must agree."""
    if is_nobody(choice):
        return False
    if path == "fast":
        checks = verdict.get("auto_check")
        return (verdict.get("fast_margin", 1.0) >= fast_min_margin
                and (checks is None or all(pick == choice for pick in checks.values())))
    if path in ("memory", "cache"):
        return True
    if path != "llm" or verdict.get("system_one_choice") != choice:
        return False
    return (verdict.get("confidence") or 0) >= min_confidence or (on_agreement and models_agree(verdict, choice))


def models_agree(verdict: dict, choice: str) -> bool:
    """System one and every second-opinion model picked `choice`, and so did every check model asked
    (auto_check: models that only confirm a decision for auto-forwarding, they never change it)."""
    second = verdict.get("second_opinion") or {}
    votes = second.get("votes") or ({second.get("model"): second.get("pick")} if second else {})
    votes = votes | (verdict.get("auto_check") or {})
    return (verdict.get("system_one_choice") == choice and bool(votes)
            and all(pick == choice for pick in votes.values()))


def nobody_share(probs: dict) -> float:
    """System one's summed probability for "nobody" (manual triage and nobody profiles)."""
    return sum(v for k, v in probs.items() if is_nobody(k))


def keep_real_person(pick: str, probs: dict, combined: dict, min_nobody: float) -> str:
    """A second opinion of "nobody" only counts when system one also gave nobody at least min_nobody;
    otherwise the best real person by combined score stays. min_nobody 0 = always accept."""
    if not is_nobody(pick) or nobody_share(probs) >= min_nobody:
        return pick
    real = {pid: s for pid, s in combined.items() if not is_nobody(pid)}
    return max(real, key=real.get) if real else pick


def majority(picks: list[str | None]) -> str | None:
    """The pick most models gave; a tie goes to the earliest model in the list. None = no model answered."""
    given = [p for p in picks if p]
    if not given:
        return None
    counts = {p: given.count(p) for p in given}
    return max(given, key=lambda p: (counts[p], -given.index(p)))


# ---------- decisions ----------

def is_unclear(verdict: dict, choice: str, combined: dict, margin: float) -> bool:
    """A system one decision worth a second opinion: search overruled system one's own pick, or the best two
    candidates are less than `margin` apart in the combined score."""
    if verdict.get("system_one_choice") != choice:
        return True
    top = sorted(combined.values(), reverse=True)
    return len(top) > 1 and top[0] - top[1] < margin


def combined_scores(probs: dict, people: list[dict], search_weight: float, prior: dict | None = None,
                    prior_weight: float = 0.0) -> dict:
    """System one's probability + search bonus + (optionally) the memory vote share."""
    best_score = max(p["score"] for p in people)
    prior = prior or {}
    return {p["person_id"]: probs.get(p["person_id"], 0) + search_weight * (p["score"] - best_score)
            + prior_weight * prior.get(p["person_id"], 0) for p in people}


def pick_recipient(probs: dict, people: list[dict], search_weight: float, manual_threshold: float,
                   prior: dict | None = None, prior_weight: float = 0.0) -> str:
    """System one's probability plus a search bonus; "nobody" options are summed for the manual decision.

    prior: memory vote shares; MANUAL's share counts towards the nobody sum."""
    prior = prior or {}
    nobody = sum(v for k, v in probs.items() if is_nobody(k)) + prior_weight * prior.get(MANUAL, 0)
    if not people or nobody >= manual_threshold:
        return MANUAL
    combined = combined_scores(probs, people, search_weight, prior, prior_weight)
    best = max(combined, key=combined.get)
    return MANUAL if is_nobody(best) else best
