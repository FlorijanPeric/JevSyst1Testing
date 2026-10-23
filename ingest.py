import argparse
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from chunking import chunk_text
from clients import embed_texts

# filenames whose stem doesn't match a known department fall back to "general"
KNOWN_DEPARTMENTS = {"returns", "shipping", "billing"}


def department_for(path: Path, override: str | None) -> str:
    if override:
        return override
    return path.stem if path.stem in KNOWN_DEPARTMENTS else "general"


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk, embed, and upsert text files into Qdrant.")
    parser.add_argument("path", nargs="?", default="data", help="File or directory of .txt files (default: data/)")
    parser.add_argument("--department", help="Force this department tag for every ingested file")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.path)
    files = [root] if root.is_file() else sorted(root.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found at {root}")

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    all_chunks: list[dict] = []
    for file in files:
        department = department_for(file, args.department)
        text = file.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
            all_chunks.append({
                "source": file.name,
                "department": department,
                "chunk_index": i,
                "text": chunk,
            })

    print(f"Prepared {len(all_chunks)} chunks from {len(files)} file(s).")

    # Probe the embedding dimension once so the collection is created with the right size.
    probe_vector = embed_texts([all_chunks[0]["text"]])[0]
    ensure_collection(client, config.QDRANT_COLLECTION, len(probe_vector))

    for start in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[start:start + args.batch_size]
        vectors = embed_texts([c["text"] for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['source']}::{c['chunk_index']}")),
                vector=vector,
                payload=c,
            )
            for c, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        print(f"Upserted {start + len(batch)}/{len(all_chunks)}")

    print("Done.")


if __name__ == "__main__":
    main()
