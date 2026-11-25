"""Application settings, read from environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Document Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    # PostgreSQL in production (see docker-compose.yml); SQLite works for quick local runs and tests.
    database_url: str = "postgresql+psycopg://dip:dip@localhost:5432/dip"

    # Any OpenAI-compatible endpoint: Groq (Llama 3.1, Qwen), Ollama (local), or OpenAI.
    # LLM_PROVIDER=mock gives deterministic offline answers for tests and demos without a key.
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0

    # "fastembed" uses a small open-source embedding model (BAAI/bge-small-en-v1.5) via ONNX.
    # "tfidf" is a dependency-free fallback that works offline.
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size_chars: int = 800
    chunk_overlap_chars: int = 120
    retrieval_top_k: int = 4
    min_retrieval_score: float = 0.05  # below this, the system says it cannot answer
    agent_max_steps: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
