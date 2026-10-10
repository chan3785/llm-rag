from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llama_server_url: str = "http://localhost:8080"
    llm_timeout_seconds: float = 300.0

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
