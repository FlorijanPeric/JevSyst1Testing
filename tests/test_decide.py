"""Unit tests for the routing logic. No network: run with `python -m pytest tests`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings  # noqa: E402
from decide import (MANUAL, clean_email, combined_scores, fast_pick, is_nobody, is_unclear, majority,  # noqa: E402
                    memory_pick, memory_hints, models_agree, memory_votes, pick_recipient, rerank_select, safe_to_auto, select_candidates, should_learn, spam_overruled)


def person(pid, score, grouped=None, margin=-1.0):
    return {"person_id": pid, "score": score, "grouped_score": score if grouped is None else grouped, "grouped_margin": margin}


# ---------- is_nobody ----------

def test_is_nobody():
    assert is_nobody(MANUAL)
    assert is_nobody("manual_spam") and is_nobody("manual_learned")
    assert not is_nobody("jan") and not is_nobody("") and not is_nobody(None)


# ---------- select_candidates ----------

SCORES = {"a": 0.9, "manual_spam": 0.85, "manual_wrong": 0.8, "b": 0.7, "c": 0.6, "d": 0.5, "e": 0.4, "f": 0.3}
NOBODY = {"manual_spam", "manual_wrong"}


def test_ranked_tops_up_with_real_people():
    top = select_candidates(SCORES, NOBODY, 3, "ranked")
    # top 3 overall = a, manual_spam, manual_wrong; topped up with b and c to 3 real people
    assert top == ["a", "manual_spam", "manual_wrong", "b", "c"]
    assert sum(pid not in NOBODY for pid in top) == 3


def test_ranked_without_nobody_in_top():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "manual_spam": 0.1}
    assert select_candidates(scores, {"manual_spam"}, 2, "ranked") == ["a", "b"]


def test_always_adds_every_nobody_profile():
    top = select_candidates(SCORES, NOBODY, 2, "always")
    assert set(top) == {"a", "b", "manual_spam", "manual_wrong"}
    assert top == sorted(top, key=lambda pid: -SCORES[pid])


def test_must_include_is_added():
    assert "f" in select_candidates(SCORES, NOBODY, 2, "ranked", must_include="f")


# ---------- fast_pick ----------

def test_fast_pick_clear_match():
    people = [person("jan", 0.6, margin=0.1), person("tilen", 0.5)]
    assert fast_pick(people, 0.5, 0.07) == "jan"


def test_fast_pick_needs_score_and_margin():
    assert fast_pick([person("jan", 0.45, margin=0.2)], 0.5, 0.07) is None
    assert fast_pick([person("jan", 0.6, margin=0.03)], 0.5, 0.07) is None
    assert fast_pick([], 0.5, 0.07) is None


def test_fast_pick_nobody_profile_means_manual():
    assert fast_pick([person("manual_spam", 0.7, margin=0.2)], 0.5, 0.07) == MANUAL


# ---------- pick_recipient ----------

def test_pick_follows_system_one_when_scores_tie():
    people = [person("jan", 0.5), person("tilen", 0.5)]
    assert pick_recipient({"jan": 0.2, "tilen": 0.6}, people, 5, 0.7) == "tilen"


def test_search_score_can_overrule_system_one():
    people = [person("jan", 0.6), person("tilen", 0.4)]
    # tilen +0.2 probability, but 0.2 lower search score * weight 5 = -1.0
    assert pick_recipient({"jan": 0.3, "tilen": 0.5}, people, 5, 0.7) == "jan"


def test_nobody_probabilities_are_summed():
    people = [person("jan", 0.6), person("manual_spam", 0.5)]
    probs = {"jan": 0.25, "manual_spam": 0.4, MANUAL: 0.35}  # 0.75 nobody in total
    assert pick_recipient(probs, people, 5, 0.7) == MANUAL
    # below the threshold the combined score decides: manual_spam 0.4 - 5*0.1 < jan 0.25
    assert pick_recipient(probs, people, 5, 0.8) == "jan"


def test_clean_keeps_line_starting_with_from_without_address():
    text = "From: our side everything is ready, please deploy the website."
    assert clean_email(text) == text


def test_winning_nobody_profile_means_manual():
    people = [person("manual_wrong", 0.6), person("jan", 0.5)]
    assert pick_recipient({"manual_wrong": 0.4, "jan": 0.3}, people, 5, 0.9) == MANUAL


def test_combined_scores():
    people = [person("jan", 0.6), person("tilen", 0.5)]
    combined = combined_scores({"jan": 0.1, "tilen": 0.5}, people, 2)
    assert round(combined["jan"], 6) == 0.1 and round(combined["tilen"], 6) == 0.3


# ---------- clean_email ----------

def test_clean_keeps_plain_email():
    assert clean_email("The 3D printer is jammed again.") == "The 3D printer is jammed again."


def test_clean_drops_quoted_reply():
    text = "Thanks, please fix the website login too.\n\nOn Mon, 3 Nov 2026 at 10:00, Tilen wrote:\n> Done with the SSL renewal."
    assert clean_email(text) == "Thanks, please fix the website login too."


def test_clean_drops_slovenian_reply_and_signature():
    text = "Prosim za nov ključ za laboratorij.\n--\nJana Novak\nFAMNIT\n\nDne 2. 11. 2026 je Tina napisala:\n> Ok"
    assert clean_email(text) == "Prosim za nov ključ za laboratorij."


def test_clean_drops_footer_lines():
    text = "Can you review the NDA by Friday?\nThis email and any attachments are confidential and intended only for the recipient."
    assert clean_email(text) == "Can you review the NDA by Friday?"


def test_spam_overruled_only_by_a_clear_real_match():
    assert spam_overruled([person("sara", 0.6), person("manual_spam", 0.7)], 0.55)
    assert not spam_overruled([person("sara", 0.5), person("manual_spam", 0.7)], 0.55)
    assert not spam_overruled([person("manual_spam", 0.9)], 0.55)
    assert not spam_overruled([], 0.55)


def test_clean_falls_back_when_nothing_left():
    text = "> only a quote"
    assert clean_email(text) == text


# ---------- memory ----------

def test_memory_votes_weight_corrections_over_auto_and_skip_dissimilar():
    votes = memory_votes([("jan", 0.9, "auto"), ("tilen", 0.9, "corrected"), ("ana", 0.5, "corrected")], 0.75)
    assert set(votes) == {"jan", "tilen"} and votes["tilen"] > votes["jan"]
    assert abs(sum(votes.values()) - 1) < 1e-9


def test_memory_votes_nobody_labels_count_as_manual():
    assert memory_votes([("manual_spam", 0.8, "corrected"), (MANUAL, 0.8, "corrected")], 0.75) == {MANUAL: 1.0}


def test_memory_pick_needs_enough_trusted_support_and_a_clear_winner():
    agree = [("jan", 0.9, "auto"), ("jan", 0.8, "auto"), ("tilen", 0.5, "auto")]
    assert memory_pick(agree, 0.75, 1.0, 0.75) == "jan"
    assert memory_pick(agree[:1], 0.75, 1.0, 0.75) is None  # one automatic decision is not enough ...
    assert memory_pick([("jan", 0.9, "corrected")], 0.75, 1.0, 0.75) == "jan"  # ... one correction is
    split = [("jan", 0.9, "auto"), ("tilen", 0.9, "auto")]
    assert memory_pick(split, 0.75, 1.0, 0.75) is None
    assert memory_pick([], 0.75, 1.0, 0.75) is None


def test_memory_prior_can_change_the_pick():
    people = [person("jan", 0.6), person("tilen", 0.6)]
    probs = {"jan": 0.5, "tilen": 0.4}
    assert pick_recipient(probs, people, 5.0, 0.7) == "jan"
    assert pick_recipient(probs, people, 5.0, 0.7, prior={"tilen": 1.0}, prior_weight=0.5) == "tilen"
    assert pick_recipient(probs, people, 5.0, 0.7, prior={MANUAL: 1.0}, prior_weight=0.5) == "jan"  # 0.5 < 0.7
    assert combined_scores(probs, people, 5.0, {"tilen": 1.0}, 0.0) == combined_scores(probs, people, 5.0)


def test_memory_hints_only_trusted_and_similar_enough():
    n = [("tilen", 0.63, "corrected", "iPad app cuts off the menu"), ("jan", 0.9, "auto", "auto one"),
         ("tilen", 0.60, "corrected", "older"), ("manual_spam", 0.7, "confirmed", "buy now"), ("ana", 0.4, "corrected", "far")]
    assert memory_hints(n, 0.55, 1) == {"tilen": ["iPad app cuts off the menu"], MANUAL: ["buy now"]}
    assert memory_hints(n, 0.55, 2)["tilen"] == ["iPad app cuts off the menu", "older"]


def test_should_learn_only_agreed_decisions():
    assert should_learn("fast", {}, "jan")
    assert should_learn("llm", {"system_one_choice": "jan"}, "jan")
    assert not should_learn("llm", {"system_one_choice": "tilen"}, "jan")  # search overruled system one
    assert not should_learn("memory", {}, "jan") and not should_learn("search-only", {}, "jan")
    assert not should_learn("fast", {}, MANUAL)


# ---------- reranking and auto-forward ----------

def test_rerank_select_takes_reranker_order_for_real_people():
    pool = [person("a", 0.9), person("manual_spam", 0.85), person("b", 0.8), person("c", 0.7), person("d", 0.6)]
    chosen = rerank_select(pool, {"a": 0.1, "b": 0.2, "c": 0.9, "d": 0.8}, 2)
    ids = [p["person_id"] for p in chosen]
    assert set(ids) == {"c", "d", "manual_spam", "a"}  # c, d by reranker; spam was top 2 by search; a = best grouped
    assert ids == sorted(ids, key=lambda i: -{p["person_id"]: p["score"] for p in pool}[i])  # search order kept


def test_rerank_select_drops_nobody_outside_search_top_k():
    pool = [person("a", 0.9), person("b", 0.8), person("manual_spam", 0.5)]
    assert "manual_spam" not in [p["person_id"] for p in rerank_select(pool, {"a": 0.5, "b": 0.6}, 2)]


def test_safe_to_auto():
    assert safe_to_auto("fast", {}, "jan", 2.0) and safe_to_auto("memory", {}, "jan", 2.0)
    assert not safe_to_auto("llm", {"system_one_choice": "jan", "confidence": 0.9}, "jan", 2.0)  # never
    assert safe_to_auto("llm", {"system_one_choice": "jan", "confidence": 0.7}, "jan", 0.6)
    assert not safe_to_auto("llm", {"system_one_choice": "tilen", "confidence": 0.9}, "jan", 0.6)  # overruled
    assert not safe_to_auto("fast", {}, MANUAL, 0.6) and not safe_to_auto("search-only", {}, "jan", 0.0)


def test_safe_to_auto_on_agreement():
    agreed = {"system_one_choice": "jan", "confidence": 0.5, "second_opinion": {"pick": "jan", "votes": {"a": "jan", "b": "jan"}}}
    split = {"system_one_choice": "jan", "confidence": 0.5, "second_opinion": {"pick": "jan", "votes": {"a": "jan", "b": "tilen"}}}
    single = {"system_one_choice": "jan", "second_opinion": {"model": "a", "pick": "jan", "before": "jan"}}
    assert safe_to_auto("llm", agreed, "jan", 2.0, on_agreement=True) and models_agree(single, "jan")
    assert not safe_to_auto("llm", agreed, "jan", 2.0)  # off by default
    assert not safe_to_auto("llm", split, "jan", 2.0, on_agreement=True)
    assert not safe_to_auto("llm", {"system_one_choice": "jan"}, "jan", 2.0, on_agreement=True)  # nobody asked
    assert not safe_to_auto("llm", agreed | {"system_one_choice": "tilen"}, "jan", 2.0, on_agreement=True)


def test_check_models_must_agree_too():
    agreed = {"system_one_choice": "jan", "second_opinion": {"model": "gemma", "pick": "jan", "before": "jan"}}
    assert models_agree(agreed | {"auto_check": {"qwen": "jan", "gams": "jan"}}, "jan")
    assert not models_agree(agreed | {"auto_check": {"qwen": "jan", "gams": "tilen"}}, "jan")
    assert not models_agree(agreed | {"auto_check": {"qwen": None}}, "jan")  # a check that failed blocks


def test_majority():
    assert majority(["jan", "tilen", "tilen"]) == "tilen"
    assert majority(["jan", "tilen", "sara"]) == "jan"  # tie: first model
    assert majority([None, "tilen", None]) == "tilen" and majority([None, None]) is None


def test_is_unclear():
    assert is_unclear({"system_one_choice": "tilen"}, "jan", {"jan": 0.9, "tilen": 0.1}, 0.15)  # overruled
    assert is_unclear({"system_one_choice": "jan"}, "jan", {"jan": 0.5, "tilen": 0.4}, 0.15)  # close
    assert not is_unclear({"system_one_choice": "jan"}, "jan", {"jan": 0.8, "tilen": 0.3}, 0.15)
    assert not is_unclear({"system_one_choice": "jan"}, "jan", {"jan": 0.8}, 0.15)


# ---------- settings ----------

def test_routing_toml_has_every_setting():
    s = settings.load()
    for key in ("search_weight", "manual_threshold", "min_score", "min_margin", "spam_threshold", "spam_override_score", "candidates", "memory",
                "memory_k", "memory_min_sim", "memory_min_support", "memory_min_share", "memory_weight", "auto_learn",
                "memory_hints", "memory_hint_min_sim", "memory_hints_per_person", "rerank", "rerank_pool",
                "auto_min_confidence", "second_opinion", "second_opinion_model", "second_opinion_margin",
                "auto_on_agreement", "auto_check_models", "search_summary", "search_summary_weight", "summary_model",
                "ranking", "nobody_candidates", "tiebreak", "cache_threshold", "clean_emails",
                "fast_path_questions", "workers", "learned_examples_per_person"):
        assert hasattr(s, key), key
    assert s.ranking in ("window", "grouped") and s.nobody_candidates in ("ranked", "always")
    assert 0 < s.manual_threshold <= 1 and 0 < s.spam_threshold <= 1
