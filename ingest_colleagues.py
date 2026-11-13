import argparse
import csv
import hashlib
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

import config
import settings
from clients import embed_texts
from decide import is_nobody


# Bump when the vector layout changes, so everyone gets re-embedded.
LAYOUT = "window+grouped+lines-v1"


def profile_texts(person: dict) -> list[str]:
    """One text per responsibility/example. Shown to system one as the best-matching lines."""
    header = f"{person['name']}, {person['role']}"
    texts = [f"{header}: {r}" for r in person.get("responsibilities", [])]
    texts += [f"{header} handles emails like: {e}" for e in person.get("example_emails", [])]
    return texts or [header]


def grouped_texts(person: dict) -> list[str]:
    """Two texts: all duties in one, all example emails in the other. Used for scoring.

    The 436-email benchmark (experiments/) found this ranks the right person first
    81% of the time, against 73% for scoring by the single best line.
    """
    role = person["role"].replace(" (fictional)", "")
    duties, examples = person.get("responsibilities", []), person.get("example_emails", [])
    texts = [f"{role}: " + "; ".join(duties)] if duties else [role]
    if examples:
        texts.append("Emails like: " + " ".join(examples))
    return texts


def window_texts(person: dict, size: int = 3, stride: int = 2) -> list[str]:
    """Overlapping windows of 3 duty/example lines. Used to rank candidates for system one.

    Best setup in the 436-email benchmark with system one: 85.1% (cross-validated).
    """
    role = person["role"].replace(" (fictional)", "")
    items = person.get("responsibilities", []) + person.get("example_emails", [])
    if len(items) <= size:
        return [f"{role}: " + "; ".join(items or [role])]
    return [f"{role}: " + "; ".join(items[i : i + size]) for i in range(0, len(items) - size + 1, stride)]


def point_id(person_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"colleague::{person_id}"))


def profile_hash(person: dict) -> str:
    # Includes the embedding model, so switching models re-embeds everyone.
    blob = json.dumps([config.EMBEDDING_MODEL, LAYOUT, person], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_people(path: Path) -> list[dict]:
    if path.suffix.lower() != ".csv":
        return json.loads(path.read_text(encoding="utf-8"))
    # CSV: list columns (responsibilities, example_emails) are "|"-separated.
    with path.open(encoding="utf-8", newline="") as f:
        return [
            {
                **row,
                "responsibilities": [s.strip() for s in row.get("responsibilities", "").split("|") if s.strip()],
                "example_emails": [s.strip() for s in row.get("example_emails", "").split("|") if s.strip()],
                "boundaries": [s.strip() for s in (row.get("boundaries") or "").split("|") if s.strip()],
            }
            for row in csv.DictReader(f)
        ]


def ensure_collection_layout(client: QdrantClient, name: str, probe_text: str) -> None:
    """One point per colleague with three named multivectors: "window" (ranking), "grouped" (fast path)
    and "lines" (the lines shown to system one)."""
    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        if isinstance(vectors, dict) and {"window", "grouped", "lines"} <= set(vectors):
            return
        print(f"Recreating {name}: old vector layout.")
        client.delete_collection(name)
    size = len(embed_texts([probe_text])[0])
    multi = MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)
    client.create_collection(
        collection_name=name,
        vectors_config={
            name: VectorParams(size=size, distance=Distance.COSINE, multivector_config=multi)
            for name in ("window", "grouped", "lines")
        },
    )


def merge_learned(people: list[dict], learned_path: Path, per_person: int) -> list[dict]:
    """Add the newest corrections from `fix` to each person's example emails.

    Corrections to "nobody" go into a separate learned nobody profile (manual_learned).
    """
    if not learned_path.exists():
        return people
    with learned_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_person: dict[str, list[str]] = {}
    for row in rows:
        pid = "manual_learned" if is_nobody(row["person_id"]) else row["person_id"]
        by_person.setdefault(pid, []).append(row["email"].strip())
    people = [dict(p) for p in people]
    known = {p["id"] for p in people}
    for p in people:
        learned = by_person.get(p["id"], [])[-per_person:]
        p["example_emails"] = p.get("example_emails", []) + [e for e in learned if e not in p.get("example_emails", [])]
    if "manual_learned" in by_person and "manual_learned" not in known:
        people.append({"id": "manual_learned", "name": "Nobody (learned from corrections)", "email": "",
                       "role": "Not for anyone here: emails corrected to nobody",
                       "responsibilities": [], "example_emails": by_person["manual_learned"][-per_person:]})
    return people


def build_people(path: Path, nobody_path: Path | None = None, learned_path: Path | None = None,
                 per_person: int = 20) -> list[dict]:
    """Colleagues + tracked nobody profiles + learned examples, as one list of profiles."""
    people = load_people(path)
    if nobody_path and nobody_path.exists() and nobody_path.resolve() != path.resolve():
        ids = {p["id"] for p in people}
        people += [p for p in load_people(nobody_path) if p["id"] not in ids]
    if learned_path:
        people = merge_learned(people, learned_path, per_person)
    return people


def sync_profiles(client: QdrantClient, people: list[dict], force: bool = False) -> str:
    """Embed new or changed profiles, delete removed ones; unchanged ones cost nothing. Returns a summary line."""
    name = config.COLLEAGUES_COLLECTION
    ensure_collection_layout(client, name, profile_texts(people[0])[0])

    stored = {
        p.payload["person_id"]: p.payload.get("profile_hash")
        for p in client.scroll(name, limit=10_000, with_payload=["person_id", "profile_hash"])[0]
    }

    changed = [p for p in people if force or stored.get(p["id"]) != profile_hash(p)]
    if changed:
        texts_per_person = [{"lines": profile_texts(p), "grouped": grouped_texts(p), "window": window_texts(p)}
                            for p in changed]
        vectors = embed_texts([t for named in texts_per_person for texts in named.values() for t in texts])
        points, start = [], 0
        for person, named in zip(changed, texts_per_person):
            named_vecs = {}
            for vector_name, texts in named.items():
                named_vecs[vector_name] = vectors[start : start + len(texts)]
                start += len(texts)
            points.append(PointStruct(
                id=point_id(person["id"]),
                vector=named_vecs,
                payload={
                    "person_id": person["id"],
                    "name": person["name"],
                    "email": person["email"],
                    "role": person["role"],
                    "texts": named["lines"],
                    "boundaries": person.get("boundaries", []),  # "not for them" lines: shown, not embedded
                    "profile_hash": profile_hash(person),
                },
            ))
        client.upsert(collection_name=name, points=points)

    removed = sorted(set(stored) - {p["id"] for p in people})
    if removed:
        client.delete(collection_name=name, points_selector=[point_id(pid) for pid in removed])

    return (f"{len(people)} profiles: {len(changed)} embedded"
            f"{' (' + ', '.join(p['name'] for p in changed) + ')' if changed else ''}, "
            f"{len(people) - len(changed)} unchanged, {len(removed)} removed"
            f"{' (' + ', '.join(removed) + ')' if removed else ''}.")


def sync_default(client: QdrantClient, force: bool = False, path: str = "data/colleagues.csv") -> str:
    """The standard profile set (colleagues + nobody profiles + learned examples), synced to Qdrant."""
    s = settings.load()
    people = build_people(Path(path), Path(config.NOBODY_PROFILES), Path(config.LEARNED_EXAMPLES),
                          s.learned_examples_per_person)
    return sync_profiles(client, people, force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed colleague profiles into Qdrant for email routing.")
    parser.add_argument("path", nargs="?", default="data/colleagues.csv", help="JSON or CSV file")
    parser.add_argument("--force", action="store_true", help="Re-embed everyone, even if unchanged")
    args = parser.parse_args()

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    print(sync_default(client, args.force, args.path))


if __name__ == "__main__":
    main()
