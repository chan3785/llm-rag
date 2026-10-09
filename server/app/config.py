from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llama_server_url: str = "http://localhost:8080"
    llm_timeout_seconds: float = 300.0

    # SQLAlchemy async URL, e.g. postgresql+asyncpg://user:password@localhost:5432/llm_rag
    # Leave unset to run without a database.
    database_url: str | None = None

    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
