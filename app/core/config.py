import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Credit Mitra API"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "payee-lora:latest")
    # How long Ollama keeps the payee model resident after a request. Its own
    # default is 5 minutes, after which the next statement pays the full load.
    OLLAMA_KEEP_ALIVE: str = "30m"
    # Load the payee model and the ONNX classifier in the background at startup,
    # so the first statement does not wait ~45s + ~75s for cold loads.
    WARMUP_ON_STARTUP: bool = True
    # --- enrichment tiers ---
    ENRICHMENT_LLM_ENABLED: bool = True
    ENRICHMENT_LLM_MODEL: str = "qwen2.5:1.5b-instruct"  # NOT payee-lora
    # On-device fuzzy tier (app/services/merchant_vectors.py): near misses of a
    # dictionary merchant. 0.75 = 0.3% false matches on local shops/people.
    ENRICHMENT_VECTOR_ENABLED: bool = True
    ENRICHMENT_VECTOR_THRESHOLD: float = 0.75

    # Tier 4: web search. OFF by default — it is the only tier that leaves
    # the device. Turn on only when you accept merchant names going out.
    ENRICHMENT_WEBSEARCH_ENABLED: bool = False
    LANGSEARCH_API_KEY: str = ""
    ENRICHMENT_WEBSEARCH_TIMEOUT: float = 8.0

    # Standalone Merchant Server
    MERCHANT_ENRICHMENT_URL: str = os.getenv("MERCHANT_ENRICHMENT_URL", "")
    MERCHANT_API_KEY: str = os.getenv("MERCHANT_API_KEY", "")

    # CORS: comma-separated origins allowed to call the engine. The engine is a
    # local sidecar, so this stays an explicit allow-list rather than a wildcard.
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,tauri://localhost",
    )
    # Optional escape hatch for LAN testing, e.g. r"http://192\.168\.\d+\.\d+:5173"
    CORS_ORIGIN_REGEX: str = os.getenv("CORS_ORIGIN_REGEX", "")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Local Data Storage for background tasks
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "tasks"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "uploads"), exist_ok=True)
