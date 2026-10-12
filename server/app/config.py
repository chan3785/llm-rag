from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llama_server_url: str = "http://localhost:8080"
    llm_timeout_seconds: float = 300.0

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "images"

    # Changing the model requires re-ingesting the collection.
    clip_model: str = "clip-ViT-B-32"
    rag_top_k: int = 3
    # Cosine similarity cutoff; tune against your own data.
    rag_score_threshold: float = 0.8

    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
