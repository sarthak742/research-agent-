"""Central configuration.

Everything tunable lives here so there are no magic numbers scattered
through the codebase. Values can be overridden with environment variables
(or a .env file) via pydantic-settings.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Secrets ---
    groq_api_key: str = Field(default="", description="Groq API key")
    tavily_api_key: str = Field(default="", description="Tavily API key")

    # --- Models ---
    llm_model: str = "llama-3.3-70b-versatile"

    # Lower temperature for planning (we want consistent, parseable output);
    # slightly higher for the report (a little fluency helps readability).
    plan_temperature: float = 0.2
    report_temperature: float = 0.4

    # --- Agent loop bounds (these keep an "autonomous" loop from running away) ---
    min_queries: int = 2          # floor the planner is allowed to choose
    max_queries: int = 6          # ceiling on the FIRST round of queries
    max_rounds: int = 3           # plan round + up to (max_rounds-1) reflection rounds
    max_total_searches: int = 10  # hard cap across all rounds; cost/latency guard
    results_per_query: int = 4    # Tavily results to request per query
    max_sources_in_report: int = 15  # cap sources fed into synthesis

    # --- Reliability ---
    llm_max_retries: int = 3
    search_max_retries: int = 3
    retry_base_delay: float = 0.5  # seconds; exponential backoff base
    request_timeout: float = 60.0  # seconds per external call

    # --- Sessions ---
    session_ttl_seconds: int = 3600      # sessions expire after 1 hour
    session_sweep_seconds: int = 300     # cleanup sweep interval

    # --- Search cache ---
    search_cache_ttl_seconds: int = 900  # 15 min; identical queries reuse results

    # --- Server ---
    cors_allow_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so we build Settings once per process."""
    return Settings()
