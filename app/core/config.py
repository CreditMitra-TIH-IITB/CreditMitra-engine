import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Credit Mitra API"
    API_V1_STR: str = "/api/v1"

    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "payee-lora:latest")
    # --- enrichment tiers ---
    ENRICHMENT_LLM_ENABLED: bool = True
    ENRICHMENT_LLM_MODEL: str = "qwen2.5:1.5b-instruct"  # NOT payee-lora

    # Tier 4: web search. OFF by default — it is the only tier that leaves
    # the device. Turn on only when you accept merchant names going out.
    ENRICHMENT_WEBSEARCH_ENABLED: bool = False
    LANGSEARCH_API_KEY: str = ""
    ENRICHMENT_WEBSEARCH_TIMEOUT: float = 8.0

    # Local Data Storage for background tasks
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "tasks"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "uploads"), exist_ok=True)
