import os

from dotenv import load_dotenv

load_dotenv()
# Optional per-machine overrides (gitignored), e.g. on server 125: QDRANT_URL=http://localhost:6333
load_dotenv(".env.local", override=True)

HIVECORE_BASE_URL = os.environ.get("HIVECORE_BASE_URL", "https://hivecore.famnit.upr.si/v1").rstrip("/")
HIVECORE_API_KEY = os.environ["HIVECORE_API_KEY"]

CHAT_MODEL = os.environ.get("CHAT_MODEL", "systemone/Qwen3-4B-Instruct-2507-Q4_K_M")
# System-one models to try in order; a model that fails is skipped for a while. As of 2026-10-26 only
# the first one has a worker on hivecore, so the real fallback is search-only routing (route_email.py).
SYSTEMONE_MODELS = [CHAT_MODEL] + [m.strip() for m in os.environ.get("SYSTEMONE_FALLBACK_MODELS", "").split(",") if m.strip()]
SYSTEMONE_TIMEOUT = float(os.environ.get("SYSTEMONE_TIMEOUT", "30"))
# Ollama-style /api/embed endpoint. When EMBEDDING_API_URL is empty, embeddings
# go through the OpenAI-compatible HIVECORE_BASE_URL/embeddings instead.
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "")
EMBEDDING_API_KEY = (
    os.environ.get("EMBEDDING_API_KEY")
    or os.environ.get("EMBEDDING_BEARER_TOKEN")
    or os.environ.get("BEARER_TOKEN")
    or HIVECORE_API_KEY
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_API_MODEL") or os.environ["EMBEDDING_MODEL"]
EMBEDDING_API_TIMEOUT = float(os.environ.get("EMBEDDING_API_TIMEOUT", "120"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_API_RETRIES = int(os.environ.get("EMBEDDING_API_RETRIES", "2"))
EMBEDDING_API_BACKOFF_SEC = float(os.environ.get("EMBEDDING_API_BACKOFF_SEC", "1.5"))
# Backup: the same model through hivecore's OpenAI-compatible /v1/embeddings (identical vectors, cosine 1.0).
# Must be the SAME model as above, or search quality silently breaks. Empty = no fallback.
EMBEDDING_FALLBACK_MODEL = os.environ.get("EMBEDDING_FALLBACK_MODEL", "bge-m3:latest")

# Audit trail and real labels (both gitignored; real emails)
DECISIONS_LOG = os.environ.get("DECISIONS_LOG", "data/decisions.jsonl")
REAL_LABELS = os.environ.get("REAL_LABELS", "data/real_labels.csv")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "support_kb")
COLLEAGUES_COLLECTION = os.environ.get("COLLEAGUES_COLLECTION", "colleagues")
ROUTED_COLLECTION = os.environ.get("ROUTED_COLLECTION", "routed_emails")
