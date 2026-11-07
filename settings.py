"""Routing settings from routing.toml (no network, safe to import anywhere, including unit tests)."""
import argparse
import tomllib
from pathlib import Path
from types import SimpleNamespace

SETTINGS_PATH = Path(__file__).resolve().parent / "routing.toml"


def load(path: Path = SETTINGS_PATH) -> SimpleNamespace:
    with path.open("rb") as f:
        return SimpleNamespace(**tomllib.load(f))


def add_routing_args(parser: argparse.ArgumentParser, settings: SimpleNamespace | None = None) -> None:
    """Command-line options for the decision settings, defaulting to routing.toml.

    Settings without an option (clean_emails, fast_path_questions, ...) still end up on `args`.
    """
    s = settings or load()
    parser.add_argument("--candidates", type=int, default=s.candidates, help="Real people system one chooses from")
    parser.add_argument("--min-score", type=float, default=s.min_score, help="Fast path: best grouped score at least this")
    parser.add_argument("--min-margin", type=float, default=s.min_margin, help="Fast path: ahead of #2 by at least this")
    parser.add_argument("--search-weight", type=float, default=s.search_weight, help="Weight of search score next to system one")
    parser.add_argument("--manual-threshold", type=float, default=s.manual_threshold, help="Summed nobody probability for manual triage")
    parser.add_argument("--ranking", choices=["window", "grouped"], default=s.ranking,
                        help="Vectors that rank candidates for system one (the fast path always uses grouped)")
    parser.add_argument("--nobody-candidates", choices=["ranked", "always"], default=s.nobody_candidates,
                        help="Nobody profiles as candidates only when they rank in the top N, or always")
    parser.add_argument("--tiebreak", action=argparse.BooleanOptionalAction, default=s.tiebreak,
                        help="Yes/no question between the top two when they are close")
    parser.add_argument("--tiebreak-margin", type=float, default=s.tiebreak_margin)
    parser.add_argument("--spam-threshold", type=float, default=s.spam_threshold, help="Don't forward emails at least this spammy")
    parser.add_argument("--spam-override-score", type=float, default=s.spam_override_score,
                        help="Forward anyway when a real person's grouped score is at least this")
    parser.add_argument("--cache-threshold", type=float, default=s.cache_threshold, help="Reuse past decisions this similar")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=s.memory,
                        help="Use similar past emails (corrections, confirmations, trusted decisions)")
    parser.add_argument("--memory-k", type=int, default=s.memory_k)
    parser.add_argument("--memory-min-sim", type=float, default=s.memory_min_sim)
    parser.add_argument("--memory-min-support", type=float, default=s.memory_min_support)
    parser.add_argument("--memory-min-share", type=float, default=s.memory_min_share)
    parser.add_argument("--memory-weight", type=float, default=s.memory_weight)
    parser.add_argument("--auto-learn", action=argparse.BooleanOptionalAction, default=s.auto_learn,
                        help="Remember decisions that two signals agreed on")
    parser.add_argument("--workers", type=int, default=s.workers, help="Parallel emails in batch runs")
    parser.set_defaults(clean_emails=s.clean_emails, fast_path_questions=list(s.fast_path_questions),
                        learned_examples_per_person=s.learned_examples_per_person)


def as_args(**overrides) -> SimpleNamespace:
    """Settings as an args-like object, for code that isn't driven by the command line (service, tune)."""
    s = vars(load())
    s.setdefault("learn", True)
    s.update(overrides)
    return SimpleNamespace(**s)
