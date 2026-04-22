from __future__ import annotations
from dataclasses import dataclass
from openai import OpenAI
from src.config import settings
from src.retrieve import RetrievedChunk

# return type: LLM's response plus the sources it was grounded on
@dataclass
class Answer:
    answer: str
    sources: list[RetrievedChunk]
    # for cost tracking and debugging
    prompt_tokens: int
    completion_tokens: int

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }
    
# prompting
SYSTEM_PROMPT = """You are Exile Oracle, a Path of Exile (PoE 1) knowledge assistant.

You answer questions about skills, items, passive tree, ascendancies, mechanics, and patches using the provided context snippets from the official PoE Wiki. Your answers must be grounded in that context only.

Rules:
1. Answer ONLY using the information in the context. Do not use outside knowledge, even if you know it.
2. If the context does not contain the answer, say so explicitly. Do not guess.
3. Cite the sources you used by their title in parentheses, like: (see: Avatar of Fire).
4. Be concise. PoE players are technical, prefer bullet points over prose for mechanics.
5. Use PoE terminology (e.g., "more" vs "increased" multipliers, "Shock", "Freeze") precisely.
6. If the question is ambiguous, answer the most likely interpretation and note alternatives.
"""

# format retrieved chunks into a numbered context block for the prompt
def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant context was retrieved.)"
    
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            f"[Source {i}] {chunk.source_title} (category: {chunk.category})\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)

# build the user message with the question and retrieved context for the LLM
def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return (
        f"Context from the PoE Wiki:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above. Cite sources by title."
    )

# LLM wrapper, one instance per app; reuse across requests
class Generator:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    # call the LLM to generate an answer given a query and retrieved chunks as context and return an Answer
    def generate(self, query: str, chunks: list[RetrievedChunk], temperature: float = 0.2) -> Answer:
        response = self.client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(query, chunks)},
            ],
            temperature=temperature,
        )

        message = response.choices[0].message
        usage = response.usage

        return Answer(
            answer=message.content or "",
            sources=chunks,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )