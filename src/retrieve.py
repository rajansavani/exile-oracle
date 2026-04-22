from __future__ import annotations
import sys
from dataclasses import dataclass
from openai import OpenAI
from pinecone import Pinecone
from src.config import settings

# return type: one chunk returned from the vector store, with enough context to cite
@dataclass
class RetrievedChunk:
    chunk_id: str
    source_title: str
    category: str
    text: str
    score: float  # cosine similarity

    # convert to dict for easier serialization and consumption by LLMs
    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_title": self.source_title,
            "category": self.category,
            "text": self.text,
            "score": self.score,
        }


# retriever class: encapsulates OpenAI + Pinecone clients so we build them once and reuse
class Retriever:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=settings.openai_api_key)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        host = pc.describe_index(settings.pinecone_index_name).host
        self.index = pc.Index(host=host)

    # embed a query string with the same model used during indexing
    def embed_query(self, query: str) -> list[float]:
        response = self.openai.embeddings.create(
            model=settings.embedding_model,
            input=[query],
        )
        return response.data[0].embedding
    
    # return top-k most similar chunks for a query
    def retrieve(self, query: str, top_k: int | None = None, category_filter: str | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else settings.top_k
        query_vector = self.embed_query(query)

        # use $eq filter to restrict retrieval to a specific category if provided
        pinecone_filter = None
        if category_filter:
            pinecone_filter = {"category": {"$eq": category_filter}}

        response = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter=pinecone_filter,
        )

        # response.matches is a list of ScoredVector objects
        # we have to convert each to our RetrievedChunk dataclass so shape is consistent for LLMs
        results = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(
                RetrievedChunk(
                    # prefer original unicode chunk_id from metadata
                    # fallback to ASCII-safe ID if metadata is missing for some reason
                    chunk_id=meta.get("chunk_id", match.id),
                    source_title=meta.get("source_title", "unknown"),
                    category=meta.get("category", "unknown"),
                    text=meta.get("text", ""),
                    score=float(match.score),
                )
            )
        return results
    

# test retrieve from the terminal
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieve <query> [--top_k N] [--category LABEL]")
        sys.exit(1)

    args = sys.argv[1:]
    top_k = settings.top_k
    category = None
    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--top_k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        else:
            query_parts.append(args[i])
            i += 1
    query = " ".join(query_parts)

    retriever = Retriever()
    results = retriever.retrieve(query, top_k=top_k, category_filter=category)

    print(f"\nQuery: {query!r}")
    print(f"top_k={top_k}, category={category}")
    print(f"Got {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        preview = r.text.replace("\n", " ")[:200]
        print(f"  [{i}] score={r.score:.4f}  category={r.category}")
        print(f"      source: {r.source_title}")
        print(f"      text:   {preview}...")
        print()


if __name__ == "__main__":
    main()