"""Shared helpers for long-running experiments: disk cache, API calls with a circuit breaker, graceful stop."""
import hashlib
import json
import signal
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from clients import _embed_batch, classify  # noqa: E402

EXP_DIR = ROOT / "experiments"
CACHE_PATH = EXP_DIR / "cache" / "cache.sqlite"


class Stop(Exception):
    """Raised to leave a run early but cleanly (time budget, Ctrl-C, API down)."""


class GracefulStop:
    """First Ctrl-C / SIGTERM asks the run to stop after the current step; a second one kills it."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        log("Stop requested - finishing the current step and writing results (signal again to kill).")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Cache:
    """Embeddings and system-one answers, keyed by content hash, so restarts and repeats are free."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS s1 (key TEXT PRIMARY KEY, answer TEXT)")
        self.db.commit()

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get_emb(self, k):
        row = self.db.execute("SELECT vec FROM emb WHERE key=?", (k,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def put_emb(self, k, vec):
        self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (k, np.asarray(vec, dtype=np.float32).tobytes()))

    def get_s1(self, k):
        row = self.db.execute("SELECT answer FROM s1 WHERE key=?", (k,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put_s1(self, k, answer):
        self.db.execute("INSERT OR REPLACE INTO s1 VALUES (?, ?)", (k, json.dumps(answer)))
        self.db.commit()

    def commit(self):
        self.db.commit()


class Breaker:
    """Retries a flaky call with growing waits; after too many failures in a row, stops the run cleanly."""

    def __init__(self, stopper: GracefulStop, max_consecutive: int = 8, base_wait: float = 5.0):
        self.stopper, self.max_consecutive, self.base_wait = stopper, max_consecutive, base_wait
        self.consecutive = 0
        self.total_failures = 0

    def call(self, fn, *args, **kwargs):
        while True:
            if self.stopper.requested:
                raise Stop("stop requested")
            try:
                result = fn(*args, **kwargs)
                self.consecutive = 0
                return result
            except Exception as e:  # network errors, 429/5xx after classify's own retries
                self.consecutive += 1
                self.total_failures += 1
                if self.consecutive >= self.max_consecutive:
                    raise Stop(f"API down: {self.consecutive} failures in a row, last: {str(e)[:150]}")
                wait = min(self.base_wait * 2 ** (self.consecutive - 1), 300)
                log(f"  call failed ({self.consecutive}/{self.max_consecutive}): {str(e)[:120]} - waiting {wait:.0f}s")
                time.sleep(wait)


def embed_cached(texts: list[str], cache: Cache, breaker: Breaker) -> np.ndarray:
    """Normalized embeddings for texts, only calling the API for ones not cached yet."""
    keys = [Cache.key("emb", config.EMBEDDING_MODEL, t) for t in texts]
    missing = sorted({(k, t) for k, t in zip(keys, texts) if cache.get_emb(k) is None})
    size = config.EMBEDDING_BATCH_SIZE
    for i in range(0, len(missing), size):
        batch = missing[i : i + size]
        vectors = breaker.call(_embed_batch, [t for _, t in batch])
        for (k, _), v in zip(batch, vectors):
            cache.put_emb(k, v)
        cache.commit()
        if len(missing) > size:
            log(f"  embedded {min(i + size, len(missing))}/{len(missing)} new texts")
    out = np.stack([cache.get_emb(k) for k in keys])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def system_one_cached(state: str, question: dict, cache: Cache, breaker: Breaker) -> dict:
    """Full system-one answer object for one question, cached by (model, text, question)."""
    k = Cache.key("s1", config.CHAT_MODEL, state, question)
    hit = cache.get_s1(k)
    if hit is not None:
        return hit
    answer = breaker.call(classify, state, {"q": question}, raw=True)["q"]
    cache.put_s1(k, answer)
    return answer
