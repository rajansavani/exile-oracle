from __future__ import annotations
import json
import time
import unicodedata
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from src.config import settings

# paths
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")


# pinecone parameters
UPSERT_BATCH_SIZE = 50


# convert unicode chunk_id to pure-ASCII for Pinecone-safe vector ID
def ascii_safe_id(chunk_id: str) -> str:
    # NFKD decomposes e.g. 'ó' -> 'o' + combining acute accent
    normalized = unicodedata.normalize("NFKD", chunk_id)
    # encode to ASCII, dropping any bytes that aren't representable
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii")


# create the pinecone index if it doesn't exist, and return the index host for upsert
def ensure_index(pc: Pinecone) -> str:
    name = settings.pinecone_index_name
    if not pc.has_index(name):
        print(f"Creating Pinecone index '{name}'...")
        pc.create_index(
            name=name,
            dimension=settings.embedding_dimensions,
            # cosine similarity and dot product are same here cause vectors are normalized
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # poll until serverless index creation is complete and we can start upserting
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        print(f"Index '{name}' ready.")
    else:
        print(f"Using existing index '{name}'.")

    return pc.describe_index(name).host


# yield one embedding record at a time from the cache file
def iter_embeddings():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"{EMBEDDINGS_PATH} does not exist. Run `python -m src.embed` first.")
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# convert an embedding record into Pinecone's upsert payload format
def record_to_pinecone_vector(record: dict) -> dict:
    original_id = record["chunk_id"]
    return {
        "id": ascii_safe_id(original_id),
        "values": record["embedding"],
        "metadata": {
            "chunk_id": original_id, # keep the full-unicode original 
            "source_title": record["source_title"],
            "category": record["category"],
            "chunk_index": record["chunk_index"],
            "text": record["text"],
        },
    }

# upsert all cached embeddings to Pinecone in batches
def upsert_all() -> int:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    host = ensure_index(pc)
    index = pc.Index(host=host)

    batch: list[dict] = []
    total = 0
    pbar = tqdm(desc="Upserting vectors", unit="vec")

    # upsert one batch with retries for transient/rate-limit errors
    def flush(current_batch: list[dict]) -> int:
        if not current_batch:
            return 0
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                index.upsert(vectors=current_batch)
                return len(current_batch)
            except Exception as e:
                last_exc = e
                backoff = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                tqdm.write(
                    f"    ~ upsert error ({type(e).__name__}): "
                    f"retry {attempt + 1}/5 in {backoff}s"
                )
                time.sleep(backoff)
        # if 5 attempts fail, raise the last exception
        assert last_exc is not None
        raise last_exc

    for record in iter_embeddings():
        batch.append(record_to_pinecone_vector(record))
        if len(batch) >= UPSERT_BATCH_SIZE:
            total += flush(batch)
            pbar.update(len(batch))
            batch = []
    total += flush(batch)
    pbar.update(len(batch))
    pbar.close()

    stats = index.describe_index_stats()
    print("\nIndex stats after upsert:")
    print(f"  Total vectors: {stats.total_vector_count:,}")
    print(f"  Dimension: {stats.dimension}")
    return total

if __name__ == "__main__":
    print(f"Reading embeddings from {EMBEDDINGS_PATH.resolve()}")
    print(f"Pinecone index: {settings.pinecone_index_name}")
    print(f"Target: {settings.pinecone_cloud} / {settings.pinecone_region}")
    n = upsert_all()
    print("\n=== Done ===")
    print(f"  Upserted {n:,} vectors.")