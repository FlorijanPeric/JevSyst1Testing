"""Colleague profile vectors kept in memory, so searching needs no Qdrant request per email.

Qdrant stays the store. A process loads all profiles once; afterwards a cheap check of the stored
profile hashes (payload only, no vectors) reloads them only when someone was added, removed or changed.
"""
import threading
import time

import numpy as np
from qdrant_client import QdrantClient

import config
from decide import is_nobody, select_candidates


def _unit(vecs) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class ProfileStore:
    def __init__(self, client: QdrantClient, collection: str | None = None, check_every: float = 30.0):
        self.client, self.collection, self.check_every = client, collection or config.COLLEAGUES_COLLECTION, check_every
        self._lock = threading.Lock()
        self._signature = None
        self._checked = 0.0
        self.profiles: list[dict] = []

    def _current_signature(self) -> tuple:
        points, _ = self.client.scroll(self.collection, limit=10_000, with_payload=["profile_hash"], with_vectors=False)
        return tuple(sorted((str(p.id), p.payload.get("profile_hash")) for p in points))

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            if not force and self.profiles and time.time() - self._checked < self.check_every:
                return
            signature = self._current_signature()
            self._checked = time.time()
            if not force and signature == self._signature:
                return
            points, _ = self.client.scroll(self.collection, limit=10_000, with_payload=True,
                                           with_vectors=["window", "grouped", "lines"])
            self.profiles = [{
                "id": point.id,
                "payload": point.payload,
                "window": _unit(point.vector["window"]),
                "grouped": _unit(point.vector["grouped"]),
                "lines": _unit(point.vector["lines"]),
            } for point in points]
            self._signature = signature

    def shortlist(self, vector, candidates: int, ranking: str = "window", nobody: str = "ranked") -> list[dict]:
        """The candidates for one email, best first (same scoring as before, now from memory).

        score         = average similarity to the "window" vectors; ranks candidates for system one.
        grouped_score = average similarity to the two "grouped" vectors; drives the fast path.
        "matched"     = the person's two best-matching lines, which system one sees.
        """
        self.refresh()
        query = np.asarray(vector, dtype=np.float32)
        query /= np.linalg.norm(query)
        profiles = {p["id"]: p for p in self.profiles}
        grouped = {pid: float((p["grouped"] @ query).mean()) for pid, p in profiles.items()}
        scores = grouped if ranking == "grouped" else {pid: float((p["window"] @ query).mean()) for pid, p in profiles.items()}
        nobody_ids = {pid for pid, p in profiles.items() if is_nobody(p["payload"]["person_id"])}
        top = select_candidates(scores, nobody_ids, candidates, nobody, must_include=max(grouped, key=grouped.get))

        people = []
        for pid in top:
            p = profiles[pid]
            line_sims = p["lines"] @ query
            people.append({
                "person_id": p["payload"]["person_id"],
                "score": scores[pid],
                "grouped_score": grouped[pid],
                # how far ahead of everyone else this person is on grouped scores (negative if not first)
                "grouped_margin": grouped[pid] - max(v for k, v in grouped.items() if k != pid),
                **{k: p["payload"][k] for k in ("name", "email", "role")},
                "matched": [p["payload"]["texts"][i] for i in np.argsort(-line_sims)[:2]],
                "boundaries": p["payload"].get("boundaries", []),
            })
        return people

    def payloads(self) -> dict:
        self.refresh()
        return {p["payload"]["person_id"]: p["payload"] for p in self.profiles}


_stores: dict = {}


def get_store(client: QdrantClient) -> ProfileStore:
    """One store per process (and collection)."""
    key = (id(client), config.COLLEAGUES_COLLECTION)
    if key not in _stores:
        _stores[key] = ProfileStore(client)
    return _stores[key]
