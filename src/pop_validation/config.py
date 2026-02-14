"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env.local or environment variables."""

    gemini_api_key: str = Field(
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model_name: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL_NAME")

    model_config = {
        "env_file": ".env.local",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


# --- Constants ---

VALID_RECEIPT_TYPES: frozenset[str] = frozenset(
    {
        "official_receipt_paper",
        "official_receipt_digital",
        "invoice",
        "e_receipt",
        "order_confirmation",
    }
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "retailer_name",
    "retailer_location",
    "purchase_date",
    "product_counts",
    "product_prices",
)

SUPPORTED_IMAGE_FORMATS: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
    }
)

MIN_IMAGE_RESOLUTION: int = 100  # pixels (width or height)
MIN_FILE_SIZE_BYTES: int = 1024  # 1 KB
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
BLUR_THRESHOLD: float = 50.0  # Laplacian variance below this = blurry


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    logger.debug("Loading settings from environment / .env.local file...")
    settings = Settings()  # type: ignore[call-arg, unused-ignore]
    logger.info(
        "Settings loaded | model={} | api_key={}...{}",
        settings.gemini_model_name,
        settings.gemini_api_key[:4],
        settings.gemini_api_key[-4:],
    )
    return settings
