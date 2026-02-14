"""OCR extraction orchestrator — coordinates image analysis steps.

Single Responsibility: orchestrate the 4-step analysis flow.
Delegates LLM calls to GeminiClient and field parsing to field_utils.
No raw JSON handling, no Gemini SDK imports, no helper logic.
"""

from __future__ import annotations

from loguru import logger

from pop_validation.config import Settings
from pop_validation.field_utils import (
    build_quality_report,
    build_receipt_fields,
    get_language,
    infer_product_category,
)
from pop_validation.gemini_client import GeminiClient
from pop_validation.image_loader import LoadedImage, load_image
from pop_validation.image_validator import validate_image
from pop_validation.models import (
    ImageAnalysis,
    ImageQuality,
    ImageQualityReport,
    ReceiptFields,
)
from pop_validation.product_catalog import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.prompts.extraction_prompt import build_extraction_prompt
from pop_validation.prompts.quality_prompt import build_quality_prompt


class OcrExtractor:
    """Orchestrates receipt analysis: load → validate → quality → extract.

    Each step is a clear delegation to a specialized module:
    - Image loading:      image_loader
    - Technical checks:   image_validator
    - LLM calls:          GeminiClient
    - Field parsing:      field_utils
    - Product catalog:    ProductCatalogProvider (injectable)
    """

    def __init__(
        self,
        settings: Settings,
        product_catalog: ProductCatalogProvider | None = None,
    ) -> None:
        logger.info("Initializing OcrExtractor...")
        self._client = GeminiClient(settings)
        self._quality_prompt = build_quality_prompt()
        self._product_catalog = product_catalog or JsonFileProductCatalogProvider()
        logger.info(
            "OcrExtractor ready | products_catalog={} items",
            len(self._product_catalog.get_products()),
        )

    def analyze(self, image_url: str, image_index: int) -> ImageAnalysis:
        """Full 4-step analysis: load → technical check → LLM quality → LLM extract.

        Args:
            image_url: URL or path to the image.
            image_index: Zero-based index of this image in the request.

        Returns:
            ImageAnalysis with all extracted data, or minimal result on error.
        """
        logger.info("--- Image [{}] analysis started | source={}", image_index, image_url)

        try:
            loaded = self._step1_load(image_url, image_index)
            if not self._step2_technical_check(loaded, image_index):
                return self._rejected_by_technical(image_index, image_url)

            quality = self._step3_quality_assessment(loaded, image_index)
            if quality.rejection_reason is not None:
                return self._rejected_by_quality(image_index, image_url, quality)

            fields = self._step4_field_extraction(loaded, image_index)
            return self._build_success(image_index, image_url, quality, fields)

        except Exception as e:
            logger.error(
                "[{}] FAILED to analyze image | source={} | error={}: {}",
                image_index, image_url, type(e).__name__, e,
            )
            return ImageAnalysis(image_index=image_index, image_url=image_url)

    # ──────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────

    def _step1_load(self, image_url: str, idx: int) -> LoadedImage:
        """Step 1: Load image from URL or path."""
        logger.info("[{}] Loading image from source...", idx)
        loaded = load_image(image_url)
        logger.info(
            "[{}] Image loaded | size={} bytes | mime={} | dimensions={}x{}",
            idx, len(loaded.data), loaded.mime_type,
            loaded.pil_image.width, loaded.pil_image.height,
        )
        return loaded

    def _step2_technical_check(self, loaded: LoadedImage, idx: int) -> bool:
        """Step 2: Technical validation (Pillow — free, no API cost).

        Returns:
            True if image passes, False if rejected.
        """
        logger.info("[{}] Running technical validation (Pillow, no API cost)...", idx)
        report = validate_image(loaded)

        if not report.resolution_ok or not report.file_size_ok:
            logger.warning(
                "[{}] REJECTED by technical validation | resolution_ok={} | "
                "file_size_ok={} | format_ok={} | blur_score={:.1f} | "
                "Gemini calls SKIPPED (cost saved)",
                idx, report.resolution_ok, report.file_size_ok,
                report.format_ok, report.blur_score,
            )
            return False

        logger.info(
            "[{}] Technical validation PASSED | resolution_ok={} | "
            "blur_score={:.1f} (blurry={}) | format_ok={} | file_size_ok={}",
            idx, report.resolution_ok, report.blur_score,
            report.is_blurry, report.format_ok, report.file_size_ok,
        )
        return True

    def _step3_quality_assessment(
        self, loaded: LoadedImage, idx: int,
    ) -> ImageQualityReport:
        """Step 3: LLM quality assessment (Gemini call #1)."""
        try:
            data = self._client.call(
                prompt=self._quality_prompt,
                image_data=loaded.data,
                mime_type=loaded.mime_type,
                call_label="GEMINI CALL #1 — Quality Assessment",
                image_index=idx,
            )
            report = build_quality_report(data)

            logger.info(
                "[{}] Quality assessment result | is_receipt={} | readable={} | "
                "quality={} | category={} | ai_generated={} | rejection={}",
                idx, report.is_receipt, report.is_readable,
                report.image_quality, report.image_category,
                report.is_ai_generated, report.rejection_reason,
            )
            return report

        except Exception as e:
            logger.error(
                "[{}] GEMINI CALL #1 FAILED | error={}: {}",
                idx, type(e).__name__, e,
            )
            return ImageQualityReport(
                rejection_reason=f"QUALITY_ASSESSMENT_FAILED: {type(e).__name__}",
            )

    def _step4_field_extraction(
        self, loaded: LoadedImage, idx: int,
    ) -> ReceiptFields | None:
        """Step 4: LLM field extraction (Gemini call #2, only if quality passes)."""
        try:
            extraction_prompt = build_extraction_prompt(
                self._product_catalog.get_products(),
            )
            data = self._client.call(
                prompt=extraction_prompt,
                image_data=loaded.data,
                mime_type=loaded.mime_type,
                call_label="GEMINI CALL #2 — Field Extraction",
                image_index=idx,
            )
            fields = build_receipt_fields(data)

            logger.info(
                "[{}] Field extraction result | retailer={} | date={} | "
                "products={} | price={} {} | type={}",
                idx, fields.retailer_name, fields.purchase_date,
                fields.product_counts, fields.price,
                fields.currency, fields.receipt_type,
            )
            return fields

        except Exception as e:
            logger.error(
                "[{}] GEMINI CALL #2 FAILED | error={}: {}",
                idx, type(e).__name__, e,
            )
            return None

    # ──────────────────────────────────────────────
    # Result builders
    # ──────────────────────────────────────────────

    @staticmethod
    def _rejected_by_technical(idx: int, url: str) -> ImageAnalysis:
        """Build result for images rejected by technical validation."""
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=ImageQuality.LOW,
        )

    @staticmethod
    def _rejected_by_quality(
        idx: int, url: str, quality: ImageQualityReport,
    ) -> ImageAnalysis:
        """Build result for images rejected by Gemini quality assessment."""
        logger.warning(
            "[{}] REJECTED by quality assessment | reason={} | "
            "category={} | ai_generated={} | Extraction call SKIPPED (cost saved)",
            idx, quality.rejection_reason,
            quality.image_category, quality.is_ai_generated,
        )
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=quality.image_quality,
            image_category=quality.image_category,
            is_ai_generated=quality.is_ai_generated,
        )

    @staticmethod
    def _build_success(
        idx: int,
        url: str,
        quality: ImageQualityReport,
        fields: ReceiptFields | None,
    ) -> ImageAnalysis:
        """Build result for successfully analyzed images."""
        product_category = infer_product_category(fields)
        language = get_language(quality, fields)

        logger.info(
            "[{}] Analysis complete | product_category={} | language={} | "
            "fields_extracted={}",
            idx, product_category, language, fields is not None,
        )
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=quality.image_quality,
            image_category=quality.image_category,
            product_category=product_category,
            language_category=language,
            is_ai_generated=quality.is_ai_generated,
            receipt_fields=fields,
        )
