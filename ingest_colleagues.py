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
from clients import embed_texts


# Bump when the vector layout changes, so everyone gets re-embedded.
LAYOUT = "grouped+lines-v1"


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
            }
            for row in csv.DictReader(f)
        ]


def ensure_collection_layout(client: QdrantClient, name: str, probe_text: str) -> None:
    """One point per colleague with two named multivectors: "grouped" (scoring) and "lines" (explanations)."""
    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        if isinstance(vectors, dict) and {"grouped", "lines"} <= set(vectors):
            return
        print(f"Recreating {name}: old vector layout.")
        client.delete_collection(name)
    size = len(embed_texts([probe_text])[0])
    multi = MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "grouped": VectorParams(size=size, distance=Distance.COSINE, multivector_config=multi),
            "lines": VectorParams(size=size, distance=Distance.COSINE, multivector_config=multi),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed colleague profiles into Qdrant for email routing.")
    parser.add_argument("path", nargs="?", default="data/colleagues.csv", help="JSON or CSV file")
    parser.add_argument("--force", action="store_true", help="Re-embed everyone, even if unchanged")
    args = parser.parse_args()

    people = load_people(Path(args.path))
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    name = config.COLLEAGUES_COLLECTION

    ensure_collection_layout(client, name, profile_texts(people[0])[0])

    stored = {
        p.payload["person_id"]: p.payload.get("profile_hash")
        for p in client.scroll(name, limit=10_000, with_payload=["person_id", "profile_hash"])[0]
    }

    changed = [p for p in people if args.force or stored.get(p["id"]) != profile_hash(p)]
    if changed:
        texts_per_person = [(profile_texts(p), grouped_texts(p)) for p in changed]
        vectors = embed_texts([t for lines, grouped in texts_per_person for t in lines + grouped])
        points, start = [], 0
        for person, (texts, grouped) in zip(changed, texts_per_person):
            line_vecs = vectors[start : start + len(texts)]
            grouped_vecs = vectors[start + len(texts) : start + len(texts) + len(grouped)]
            points.append(PointStruct(
                id=point_id(person["id"]),
                vector={"lines": line_vecs, "grouped": grouped_vecs},
                payload={
                    "person_id": person["id"],
                    "name": person["name"],
                    "email": person["email"],
                    "role": person["role"],
                    "texts": texts,
                    "profile_hash": profile_hash(person),
                },
            ))
            start += len(texts) + len(grouped)
        client.upsert(collection_name=name, points=points)

    removed = sorted(set(stored) - {p["id"] for p in people})
    if removed:
        client.delete(collection_name=name, points_selector=[point_id(pid) for pid in removed])

    print(f"{len(people)} colleagues: {len(changed)} embedded"
          f"{' (' + ', '.join(p['name'] for p in changed) + ')' if changed else ''}, "
          f"{len(people) - len(changed)} unchanged, {len(removed)} removed"
          f"{' (' + ', '.join(removed) + ')' if removed else ''}.")


if __name__ == "__main__":
    main()
