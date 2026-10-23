import os

from dotenv import load_dotenv

load_dotenv()

HIVECORE_BASE_URL = os.environ.get("HIVECORE_BASE_URL", "https://hivecore.famnit.upr.si/v1").rstrip("/")
HIVECORE_API_KEY = os.environ["HIVECORE_API_KEY"]

CHAT_MODEL = os.environ.get("CHAT_MODEL", "systemone/Qwen3-4B-Instruct-2507-Q4_K_M")
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

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "support_kb")
COLLEAGUES_COLLECTION = os.environ.get("COLLEAGUES_COLLECTION", "colleagues")
ROUTED_COLLECTION = os.environ.get("ROUTED_COLLECTION", "routed_emails")
