from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llama_server_url: str = "http://localhost:8080"
    llm_timeout_seconds: float = 300.0
    # Applied when the client does not set max_tokens; llama-server otherwise never stops
    # if the model falls into a repetition loop.
    llm_max_tokens: int = 1024

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "landmarks"

    # Changing the model requires re-ingesting the collection.
    clip_model: str = "clip-ViT-L-14"
    # Landmark classification: retrieve the nearest reference images, then compare
    # landmarks by the mean similarity of their closest `classify_per_landmark` images.
    # Raise top_k when many landmarks compete, so each candidate keeps enough images.
    classify_top_k: int = 20
    classify_per_landmark: int = 3
    # Below this similarity the photo is treated as an unregistered place.
    # Tune with scripts/evaluate.py against your own data.
    classify_threshold: float = 0.8

    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
