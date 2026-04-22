# embed chunked text using OpenAI and cache results to disk
from __future__ import annotations
import json
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from src.config import settings

# paths
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

# embedding parameters
BATCH_SIZE = 100
MAX_RETRIES = 5

# return the set of chunk_ids already present in the embeddings file
def load_cached_ids() -> set[str]:
    if not EMBEDDINGS_PATH.exists():
        return set()
    
    cached: set[str] = set()
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cached.add(record["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return cached

# yield chunk records from chunks.jsonl one at a time
def iter_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} does not exist. Run `python -m src.chunk` first."
        )
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# call OpenAI embeddings with retries on transient errors
# returns list of embedding vectors in the same order as input texts
def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
            # response has a data list with one embedding per input,
            # already in the same order we sent them
            return [item.embedding for item in response.data]
        except Exception as e:
            is_last = attempt == MAX_RETRIES - 1
            if is_last:
                raise
            backoff = 2 ** attempt
            tqdm.write(
                f"    ~ embed error ({type(e).__name__}): "
                f"retry {attempt + 1}/{MAX_RETRIES} in {backoff}s"
            )
            time.sleep(backoff)
    raise RuntimeError("exhausted retries")

# embed every chunk not already cached and return number of new embeddings
def embed_all() -> int:
    client = OpenAI(api_key=settings.openai_api_key)

    cached_ids = load_cached_ids()
    if cached_ids:
        print(f"Resuming: {len(cached_ids):,} chunks already cached.")

    # build a list of all chunks whose IDs are not in the cache
    to_embed = [c for c in iter_chunks() if c["chunk_id"] not in cached_ids]
    if not to_embed:
        print("Nothing new to embed — cache is up to date.")
        return 0

    print(f"Embedding {len(to_embed):,} new chunks in batches of {BATCH_SIZE}.")

    # append to the cache file as we go so partial progress is preserved
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    with EMBEDDINGS_PATH.open("a", encoding="utf-8") as cache_file:
        # iterate in batches of BATCH_SIZE with tqdm progress bar
        for i in tqdm(
            range(0, len(to_embed), BATCH_SIZE),
            desc="Embedding batches",
            unit="batch",
        ):
            batch = to_embed[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            try:
                vectors = embed_batch(client, texts)
            except Exception as e:
                tqdm.write(f"    ! batch at offset {i} failed permanently: {e}")
                continue

            # write one record per embedding, pairing the vector back to the source chunk's metadata
            for chunk, vector in zip(batch, vectors):
                record = {
                    "chunk_id": chunk["chunk_id"],
                    "source_title": chunk["source_title"],
                    "category": chunk["category"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": vector,
                }
                cache_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_count += 1
            cache_file.flush()

    return new_count


if __name__ == "__main__":
    print(f"Reading chunks from {CHUNKS_PATH.resolve()}")
    print(f"Writing embeddings to {EMBEDDINGS_PATH.resolve()}")
    print(f"Model: {settings.embedding_model} ({settings.embedding_dimensions} dims)")
    n = embed_all()
    print("\n=== Done ===")
    print(f"  Embedded {n:,} new chunks.")
    if EMBEDDINGS_PATH.exists():
        size_mb = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
        print(f"  Cache file: {EMBEDDINGS_PATH} ({size_mb:.1f} MB)")