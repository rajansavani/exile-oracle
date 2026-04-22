from __future__ import annotations
from importlib.resources import path
import json
from pathlib import Path
import tiktoken
from tqdm import tqdm
from src.config import settings

# paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# tokenizer: cl100k_base which is the encoding used by text-embeddding-3-small and gpt-4o-mini
# ensures our chunks fit exactly (500-token chunk is 500 tokens to the model)
ENCODER = tiktoken.get_encoding("cl100k_base")

# split text into overlapping chunks (measured in tokens)
def chunk_tokens(
        text: str,
        chunk_size: int = settings.chunk_size_tokens,
        overlap: int = settings.chunk_overlap_tokens,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    
    # encode the whole text once
    tokens = ENCODER.encode(text)
    if not tokens:
        return []
    
    stride = chunk_size - overlap
    chunks: list[str] = []

    # slide a window of chunk_size across the token list, advancing by stride each time
    # decode each window back to a string
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(ENCODER.decode(window))
        if start + chunk_size >= len(tokens):
            break  # stop if we've reached the end
    
    return chunks


# per-file processor: read one raw .txt file and return a list of chunk records
def process_file(path: Path, category: str) -> list[dict]:
    content = path.read_text(encoding="utf-8")

    # parse the title out of first line so we can attach it as metadata
    lines = content.split("\n", 2)
    if len(lines) >= 3 and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body = lines[2]
    else:
        # fallback if format is somehow off
        title = path.stem.replace("_", " ")
        body = content

    chunks = chunk_tokens(body)

    # create a record for each chunk with metadata
    # chunk_id is globally unique: category + title + index
    records = []
    for i, chunk_text in enumerate(chunks):
        records.append({
            "chunk_id": f"{category}::{path.stem}::{i}",
            "source_title": title,
            "category": category,
            "chunk_index": i,
            "text": chunk_text,
        })
    return records


# chunk every .txt file in data/raw/ and write chunks.jsonl
def chunk_all() -> int:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} does not exist. Run `python -m src.ingest` first."
        )
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # collect all .txt files grouped by their parent folder (category label)
    # skip manifest.json and any other non-txt files
    files_by_category: dict[str, list[Path]] = {}
    for txt_path in RAW_DIR.rglob("*.txt"):
        category = txt_path.parent.name
        files_by_category.setdefault(category, []).append(txt_path)

    if not files_by_category:
        raise RuntimeError(f"No .txt files found under {RAW_DIR}.")
    
    total_chunks = 0

    # open the output file once and stream writes
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for category, paths in files_by_category.items():
            print(f"\n=== Chunking category: {category} ({len(paths)} files) ===")
            for path in tqdm(paths, desc=f"  {category}", unit="file"):
                try:
                    records = process_file(path, category)
                except Exception as e:
                    tqdm.write(f"    ! {path.name}: {type(e).__name__}: {e}")
                    continue
                for record in records:
                    # JSONL: one JSON object per line, no commas, no outer array
                    # ensure_ascii=False preserves wiki unicode (accents, etc.) rather than escaping them, which halves the file size
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

    return total_chunks


if __name__ == "__main__":
    print(f"Chunking from {RAW_DIR.resolve()} -> {CHUNKS_PATH.resolve()}")
    print(
        f"Config: chunk_size={settings.chunk_size_tokens} tokens, "
        f"overlap={settings.chunk_overlap_tokens} tokens"
    )
    n = chunk_all()
    print("\n=== Done ===")
    print(f"  Wrote {n:,} chunks to {CHUNKS_PATH}")
    size_mb = CHUNKS_PATH.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")