from __future__ import annotations
import pytest

def test_config_imports():
    """Settings module should import even with placeholder keys."""
    from src.config import settings
    assert settings.app_name == "exile-oracle"
    assert settings.embedding_dimensions == 1536


def test_chunk_tokens_basic():
    """Chunker splits long text into overlapping windows."""
    from src.chunk import chunk_tokens

    text = "The quick brown fox jumps over the lazy dog. " * 100
    chunks = chunk_tokens(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # every chunk should be a non-empty string
    assert all(isinstance(c, str) and c for c in chunks)


def test_chunk_tokens_short_input():
    """Short input returns a single chunk, not an empty list."""
    from src.chunk import chunk_tokens

    chunks = chunk_tokens("hello world", chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_chunk_tokens_overlap_validation():
    """Chunker should reject overlap >= chunk_size."""
    from src.chunk import chunk_tokens

    with pytest.raises(ValueError):
        chunk_tokens("some text", chunk_size=10, overlap=10)


def test_ascii_safe_id_strips_accents():
    """ASCII sanitizer should decompose accented characters."""
    from src.index import ascii_safe_id

    assert ascii_safe_id("bosses::The_Black_Mórrigan::0") == "bosses::The_Black_Morrigan::0"
    assert ascii_safe_id("plain::ascii::1") == "plain::ascii::1"


def test_fastapi_app_has_expected_routes():
    """The FastAPI app exposes the routes we expect."""
    from src.main import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/health" in paths
    assert "/ask" in paths
    assert "/docs" in paths