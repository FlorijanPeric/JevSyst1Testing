import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import requests
from openai import OpenAI

import config


class _DiskCache:
    """Small thread-safe sqlite cache: embeddings by (model, text), system-one answers by (model, text, questions)."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
            self._db.commit()

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


_cache = _DiskCache(config.CACHE_DB) if config.CACHE_DB else None

_openai_client = OpenAI(base_url=config.HIVECORE_BASE_URL, api_key=config.HIVECORE_API_KEY)

# Reused session keeps the TLS connection to hivecore alive between calls.
_session = requests.Session()
_session.headers["Authorization"] = f"Bearer {config.HIVECORE_API_KEY}"


_embed_session = requests.Session()
_embed_session.headers["Authorization"] = f"Bearer {config.EMBEDDING_API_KEY}"


def _embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(config.EMBEDDING_API_RETRIES + 1):
        try:
            resp = _embed_session.post(
                config.EMBEDDING_API_URL,
                json={"model": config.EMBEDDING_MODEL, "input": texts},
                timeout=config.EMBEDDING_API_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except requests.RequestException:
            if attempt == config.EMBEDDING_API_RETRIES:
                raise
            time.sleep(config.EMBEDDING_API_BACKOFF_SEC * (attempt + 1))


def _embed_batch_openai(texts: list[str], model: str) -> list[list[float]]:
    resp = _openai_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def _embed_with_fallback(texts: list[str]) -> list[list[float]]:
    try:
        return _embed_batch(texts)
    except requests.RequestException as e:
        if not config.EMBEDDING_FALLBACK_MODEL:
            raise
        print(f"[fallback] embedding endpoint failed ({str(e)[:80]}); using /v1/embeddings {config.EMBEDDING_FALLBACK_MODEL}")
        return _embed_batch_openai(texts, config.EMBEDDING_FALLBACK_MODEL)


def _embed_uncached(texts: list[str]) -> list[list[float]]:
    if not config.EMBEDDING_API_URL:
        return _embed_batch_openai(texts, config.EMBEDDING_MODEL)
    size = config.EMBEDDING_BATCH_SIZE
    return [vec for i in range(0, len(texts), size) for vec in _embed_with_fallback(texts[i : i + size])]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeddings, from the disk cache where possible (repeats and reminders skip the slowest call)."""
    if _cache is None:
        return _embed_uncached(texts)
    keys = [_cache.key("emb", config.EMBEDDING_MODEL, t) for t in texts]
    found = {k: _cache.get(k) for k in set(keys)}
    missing = sorted({(k, t) for k, t in zip(keys, texts) if found[k] is None})
    if missing:
        for (k, _), vec in zip(missing, _embed_uncached([t for _, t in missing])):
            _cache.put(k, vec)
            found[k] = vec
    return [found[k] for k in keys]


def chat_complete(system_prompt: str, user_prompt: str) -> str:
    resp = _openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def classify(state: str, questions: dict, model: str | None = None, raw: bool = False) -> dict:
    """Call the /systemone structured-answer addon.

    `questions` follows the same schema shown in the assignment: a dict of
    {question_id: {type, instructions, criteria}}. Returns {question_id: value},
    where value is the answer field named after its type: the picked key for
    "choice", a 0-1 probability for "noul", and for "score" the expected criteria
    index (0 = first label, len-1 = last). raw=True returns the full answer
    objects instead (with "probabilities", "confidence", ...).
    """
    models = [model] if model else config.SYSTEMONE_MODELS
    errors = []
    for m in models:
        if _benched.get(m, 0) > time.time():
            continue
        key = _cache.key("s1", m, state, questions) if _cache is not None else None
        cached = _cache.get(key) if key else None
        if cached is not None:
            answers = cached
            break
        try:
            answers = _classify_once(state, questions, m)
            if key:
                _cache.put(key, answers)
            break
        except requests.RequestException as e:
            errors.append(f"{m}: {str(e)[:80]}")
            _benched[m] = time.time() + 600  # skip this model for 10 minutes
            if len(models) > 1:
                print(f"[fallback] system one model {m} failed, trying the next one")
    else:
        raise SystemOneDown("; ".join(errors) or "all system one models are benched")
    if raw:
        return answers
    return {qid: ans.get(ans.get("type")) for qid, ans in answers.items()}


class SystemOneDown(RuntimeError):
    """No system one model answered; callers can fall back to search-only routing."""


_benched: dict[str, float] = {}


def _classify_once(state: str, questions: dict, model: str) -> dict:
    url = f"{config.HIVECORE_BASE_URL}/systemone"
    payload = {"model": model, "state": state, "questions": questions}
    for attempt in range(4):
        resp = _session.post(url, json=payload, timeout=config.SYSTEMONE_TIMEOUT)
        # hivecore rate-limits (429) and occasionally has no free worker (503/504); back off and retry
        if resp.status_code not in (429, 503, 504) or attempt == 3:
            break
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()
    return resp.json().get("answers", {})
