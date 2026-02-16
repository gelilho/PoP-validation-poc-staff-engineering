"""ExtractionAgent — Step 4: Extract structured fields from a receipt (Gemini #2).

Only called if QualityAgent passes. Wraps client.gemini_client + extraction.field_utils.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pop_validation.agents.base import AgentResult
from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.client.gemini_client import GeminiClient
from pop_validation.config import Settings, get_settings
from pop_validation.extraction.field_utils import build_receipt_fields
from pop_validation.imaging.loader import LoadedImage
from pop_validation.prompts.extraction_prompt import build_extraction_prompt


class ExtractionAgent:
    """Extract structured receipt fields using Gemini OCR."""

    def __init__(
        self,
        settings: Settings | None = None,
        product_catalog: ProductCatalogProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = GeminiClient(self._settings)
        self._catalog = product_catalog or JsonFileProductCatalogProvider()

    @property
    def name(self) -> str:
        return "Extraction"

    def execute(self, **kwargs: Any) -> AgentResult:
        """Extract receipt fields from the image.

        Args:
            loaded: A LoadedImage from ImageLoaderAgent.
            image_index: Index of the image in the batch.

        Returns:
            AgentResult with data=ReceiptFields on success, data=None on failure.
        """
        loaded: LoadedImage = kwargs["loaded"]
        image_index: int = kwargs.get("image_index", 0)

        logger.info("[{}] Extracting fields for image [{}]", self.name, image_index)

        try:
            prompt = build_extraction_prompt(self._catalog.get_products())
            raw = self._client.call(
                prompt=prompt,
                image_data=loaded.data,
                mime_type=loaded.mime_type,
                call_label="Field Extraction",
                image_index=image_index,
            )
            fields = build_receipt_fields(raw)
            logger.info(
                "[{}] Fields extracted | retailer={} | price={}",
                self.name,
                fields.retailer_name,
                fields.price,
            )
            return AgentResult(success=True, data=fields)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("[{}] Field extraction failed: {}", self.name, error_msg)
            return AgentResult(success=True, data=None, error=error_msg)
