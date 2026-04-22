# create the pinecone index and upsert cached embeddings
from __future__ import annotations
import json
import time
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from src.config import settings

# paths
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

# pinecone parameters
UPSERT_BATCH_SIZE = 100

# create the pinecone index if it doesn't exist and return index host
def ensure_index(pc: Pinecone) -> str:
    name = settings.pinecone_index_name

    if not pc.has_index(name):
        print(f"Creating Pinecone index '{name}'...")
        pc.create_index(
            name=name,
            # text-embedding-3-small outputs 1536-dim vectors; must match exactly
            dimension=settings.embedding_dimensions,
            # we will use cosine similarity to measure vector similarity
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # serverless index creation is not instant so poll until it's ready before returning
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        print(f"Index '{name}' ready.")
    else:
        print(f"Using existing index '{name}'.")

    return pc.describe_index(name).host

# yield one embedding record at a time from the cache file
def iter_embeddings():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} does not exist. Run `python -m src.embed` first."
        )
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# convert an embedding record into Pinecone's upsert payload format
def record_to_pinecone_vector(record: dict) -> dict:
    return {
        "id": record["chunk_id"],
        "values": record["embedding"],
        "metadata": {
            "source_title": record["source_title"],
            "category": record["category"],
            "chunk_index": record["chunk_index"],
            "text": record["text"],
        },
    }

# upsert every cached embedding to pinecone and return count upserted
def upsert_all() -> int:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    host = ensure_index(pc)
    index = pc.Index(host=host)

    # stream records in and buffer into upsert batches (don't load everything into memory at once)
    batch: list[dict] = []
    total = 0
    pbar = tqdm(desc="Upserting vectors", unit="vec")
    for record in iter_embeddings():
        batch.append(record_to_pinecone_vector(record))
        if len(batch) >= UPSERT_BATCH_SIZE:
            index.upsert(vectors=batch)
            total += len(batch)
            pbar.update(len(batch))
            batch = []
    # final partial batch
    if batch:
        index.upsert(vectors=batch)
        total += len(batch)
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