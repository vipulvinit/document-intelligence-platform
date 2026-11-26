"""Application settings, read from environment variables or a .env file.

Defaults use only free tools: SQLite, a folder for object storage, fastembed for embeddings,
and any OpenAI-compatible language model API (a free online key, or a local Ollama server).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Medical Helpline"
    app_version: str = "0.2.0"
    debug: bool = False

    # SQLite needs no installation. PostgreSQL also works (pip install ".[postgres]").
    database_url: str = "sqlite:///./data/helpline.db"

    # Object storage for figure images and uploaded files. "local" = a folder on disk.
    storage_backend: str = "local"
    storage_dir: str = "./data/objects"

    # Language models through any OpenAI-compatible API: a free online tier (Groq, Google Gemini,
    # OpenRouter...) or a local Ollama server. See .env.example for ready-made presets.
    # LLM_PROVIDER=mock gives deterministic offline answers for tests and demos.
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 120.0
    # Free tiers rate-limit (HTTP 429): retry this many times, waiting longer each time
    # (the server's Retry-After header is respected).
    llm_max_retries: int = 5
    # Ask for JSON output (response_format). If a provider or model rejects it, the client
    # retries without it automatically; set False to skip the first attempt.
    llm_json_mode: bool = True

    # Vision model that describes figures. Empty = use the figure caption only.
    # It can be on a different provider: empty VISION_BASE_URL / VISION_API_KEY reuse the LLM ones.
    vision_model: str = ""
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_max_figures_per_article: int = 8

    # "fastembed" runs BAAI/bge-small-en-v1.5 on the CPU (free, ~130 MB download once).
    # "hashing" needs no download (weaker); the tests use it.
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size_chars: int = 800
    chunk_overlap_chars: int = 120
    retrieval_top_k: int = 5
    min_retrieval_score: float = 0.05  # below this, the system says it cannot answer
    agent_max_steps: int = 5

    # Literature sources (Europe PMC: free, no key; covers PubMed, PMC full text and preprints).
    europepmc_base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    pmc_figure_base_url: str = "https://pmc.ncbi.nlm.nih.gov/articles/instance"
    http_user_agent: str = "medical-helpline/0.2 (academic project)"
    http_timeout_seconds: float = 60.0
    max_articles_per_poll: int = 20
    initial_lookback_days: int = 30  # first poll of a new source fetches this far back
    download_figures: bool = True
    seed_default_sources: bool = True

    # Shown when a message looks like an emergency (India defaults: 112, Tele-MANAS 14416)
    emergency_number: str = "112"
    mental_health_helpline: str = "14416 (Tele-MANAS)"

    # Scheduler (python -m app.worker)
    scheduler_check_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
