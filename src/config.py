from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # tell pydantic settings to read from a .env file in the project root
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore", # ignore any extra keys in .env
    )

    # OpenAI api key is required
    openai_api_key: str

    # embedding model: text-embedding-3-small is cheap
    # outputs 1536-dim vectors which is plenty accurate for this use case
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # chat model for final answer generation
    chat_model: str = "gpt-4o-mini"

    # pinecone config
    pinecone_api_key: str
    pinecone_index_name: str = "exile-oracle"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # chunking parameters
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # retrieval parameters
    top_k: int = 5

    # app metadata
    app_name: str = "exile-oracle"
    app_version: str = "0.1.0"

settings = Settings()