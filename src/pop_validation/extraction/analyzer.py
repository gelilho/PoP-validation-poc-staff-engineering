"""Single-image analyzer — the 'recipe' that reads top to bottom.

This is THE entry point anyone reads first to understand what the system does.
Each step is one method. The analyze() method IS the recipe.
"""

from __future__ import annotations

from loguru import logger

from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.client.gemini_client import GeminiClient
from pop_validation.config import Settings
from pop_validation.extraction.field_utils import (
    build_quality_report,
    build_receipt_fields,
    get_language,
    infer_product_category,
)
from pop_validation.imaging.loader import LoadedImage, load_image
from pop_validation.imaging.validator import validate_image
from pop_validation.models import (
    ImageAnalysis,
    ImageQuality,
    ImageQualityReport,
    ReceiptFields,
)
from pop_validation.prompts.extraction_prompt import build_extraction_prompt
from pop_validation.prompts.quality_prompt import build_quality_prompt


class PopAnalyzer:
    """Analyzes a single receipt image — the readable 'recipe'.

    Read this class top-to-bottom to understand the entire analysis flow.
    Each step is one method call. The analyze() method IS the recipe.

    Usage:
        analyzer = PopAnalyzer(settings)
        result = analyzer.analyze("https://example.com/receipt.jpg", image_index=0)
    """

    def __init__(
        self,
        settings: Settings,
        product_catalog: ProductCatalogProvider | None = None,
    ) -> None:
        self._client = GeminiClient(settings)
        self._quality_prompt = build_quality_prompt()
        self._catalog = product_catalog or JsonFileProductCatalogProvider()
        logger.info(
            "PopAnalyzer ready | catalog={} products",
            len(self._catalog.get_products()),
        )

    def analyze(self, image_url: str, image_index: int) -> ImageAnalysis:
        """The recipe: load → check → assess → extract → build result.

        Each line is a step. If any step rejects, we return early.
        """
        logger.info("--- Image [{}] analysis started | source={}", image_index, image_url)

        try:
            # Step 1: Load the image
            loaded = load_image(image_url)

            # Step 2: Technical quality check (free, no API cost)
            tech = validate_image(loaded)
            if not tech.resolution_ok or not tech.file_size_ok:
                return self._reject(image_index, image_url, ImageQuality.LOW)

            # Step 3: AI quality assessment (Gemini call #1)
            quality = self._assess_quality(loaded, image_index)
            if quality.rejection_reason is not None:
                return self._reject_quality(image_index, image_url, quality)

            # Step 4: Field extraction (Gemini call #2)
            fields = self._extract_fields(loaded, image_index)

            # Step 5: Build final result
            return self._success(image_index, image_url, quality, fields)

        except Exception as e:
            logger.error("[{}] Analysis failed: {}: {}", image_index, type(e).__name__, e)
            return ImageAnalysis(image_index=image_index, image_url=image_url)

    # ── Step implementations ──────────────────────

    def _assess_quality(self, loaded: LoadedImage, idx: int) -> ImageQualityReport:
        """Gemini call #1: Is this a readable receipt?"""
        try:
            data = self._client.call(
                prompt=self._quality_prompt,
                image_data=loaded.data,
                mime_type=loaded.mime_type,
                call_label="Quality Assessment",
                image_index=idx,
            )
            return build_quality_report(data)
        except Exception as e:
            logger.error("[{}] Quality assessment failed: {}", idx, e)
            return ImageQualityReport(
                rejection_reason=f"QUALITY_ASSESSMENT_FAILED: {type(e).__name__}",
            )

    def _extract_fields(self, loaded: LoadedImage, idx: int) -> ReceiptFields | None:
        """Gemini call #2: Extract structured receipt fields."""
        try:
            prompt = build_extraction_prompt(self._catalog.get_products())
            data = self._client.call(
                prompt=prompt,
                image_data=loaded.data,
                mime_type=loaded.mime_type,
                call_label="Field Extraction",
                image_index=idx,
            )
            return build_receipt_fields(data)
        except Exception as e:
            logger.error("[{}] Field extraction failed: {}", idx, e)
            return None

    # ── Result builders ───────────────────────────

    @staticmethod
    def _reject(idx: int, url: str, quality: ImageQuality) -> ImageAnalysis:
        logger.warning("[{}] REJECTED by technical validation | Gemini SKIPPED", idx)
        return ImageAnalysis(image_index=idx, image_url=url, image_quality=quality)

    @staticmethod
    def _reject_quality(idx: int, url: str, q: ImageQualityReport) -> ImageAnalysis:
        logger.warning("[{}] REJECTED by quality assessment | Extraction SKIPPED", idx)
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=q.image_quality,
            image_category=q.image_category,
            is_ai_generated=q.is_ai_generated,
        )

    @staticmethod
    def _success(
        idx: int, url: str, quality: ImageQualityReport, fields: ReceiptFields | None,
    ) -> ImageAnalysis:
        logger.info("[{}] Analysis complete | fields_extracted={}", idx, fields is not None)
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=quality.image_quality,
            image_category=quality.image_category,
            product_category=infer_product_category(fields),
            language_category=get_language(quality, fields),
            is_ai_generated=quality.is_ai_generated,
            receipt_fields=fields,
        )
