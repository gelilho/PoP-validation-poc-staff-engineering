"""PoP Validation Pipeline — the full recipe, top to bottom.

Open this file. Read validate_image() and validate(). You understand the system.

Seven steps per image, then aggregate across all images:
    1. Load image         (free)
    2. Technical check    (free — Pillow)
    3. Quality assessment (Gemini call #1)
    4. Field extraction   (Gemini call #2)
    5. Apply rules        (business logic)
    6. Build result       (assemble output)
    7. Log result         (audit trail — CSV / BQ / Postgres)
"""

from __future__ import annotations

import time

from loguru import logger

from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.client.gemini_client import GeminiClient
from pop_validation.config import Settings, get_settings
from pop_validation.extraction.field_utils import (
    build_quality_report,
    build_receipt_fields,
    get_language,
    infer_product_category,
)
from pop_validation.imaging.loader import LoadedImage, load_image
from pop_validation.imaging.validator import validate_image as check_technical_quality
from pop_validation.models import (
    ImageAnalysis,
    ImageQuality,
    ImageQualityReport,
    ImageValidationResult,
    ReceiptFields,
    ValidationRequest,
    ValidationResponse,
)
from pop_validation.prompts.extraction_prompt import build_extraction_prompt
from pop_validation.prompts.quality_prompt import build_quality_prompt
from pop_validation.reporting.csv_logger import CsvResultLogger, ResultLogger
from pop_validation.validation.rules import apply_rules


class PopValidationPipeline:
    """End-to-end Proof of Purchase validation.

    Read validate_image() to understand per-image flow.
    Read validate() to understand multi-image orchestration.

    Usage:
        pipeline = PopValidationPipeline()
        response = pipeline.validate(ValidationRequest(
            warranty_id="W-123",
            image_urls=["https://example.com/receipt.jpg"],
        ))
    """

    def __init__(
        self,
        settings: Settings | None = None,
        product_catalog: ProductCatalogProvider | None = None,
        result_logger: ResultLogger | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = GeminiClient(self._settings)
        self._quality_prompt = build_quality_prompt()
        self._catalog = product_catalog or JsonFileProductCatalogProvider()
        self._result_logger = result_logger or CsvResultLogger()
        logger.info(
            "Pipeline ready | catalog={} products",
            len(self._catalog.get_products()),
        )

    # ── Per-image recipe (THE 7 steps) ───────────

    def validate_image(
        self, image_url: str, image_index: int, warranty_id: str = "",
    ) -> ImageValidationResult:
        """The full per-image recipe. Seven steps, read top to bottom.

        Step 1: Load image           (free)
        Step 2: Technical validation  (free — Pillow)
        Step 3: Quality assessment    (Gemini call #1)
        Step 4: Field extraction      (Gemini call #2, only if #1 passes)
        Step 5: Apply business rules  (chain of responsibility)
        Step 6: Build result          (assemble output)
        Step 7: Log result            (audit trail — CSV / BQ / Postgres)
        """
        logger.info("--- Image [{}] | source={}", image_index, image_url)
        comment: str | None = None

        try:
            # Step 1: Load the image
            loaded = load_image(image_url)

            # Step 2: Technical validation (free — Pillow checks)
            tech = check_technical_quality(loaded)
            if not tech.resolution_ok or not tech.file_size_ok:
                analysis = self._reject_technical(image_index, image_url)
                result = self._apply_rules_and_build(analysis)
                self._log_result(result, warranty_id, image_url, comment)
                return result

            # Step 3: Quality assessment (Gemini call #1)
            quality = self._assess_quality(loaded, image_index)
            if quality.rejection_reason is not None:
                analysis = self._reject_quality(image_index, image_url, quality)
                result = self._apply_rules_and_build(analysis)
                self._log_result(result, warranty_id, image_url, comment)
                return result

            # Step 4: Field extraction (Gemini call #2)
            fields = self._extract_fields(loaded, image_index)

            # Step 5 + 6: Apply rules and build result
            analysis = self._build_analysis(image_index, image_url, quality, fields)
            result = self._apply_rules_and_build(analysis)

        except Exception as e:
            logger.error("[{}] Failed: {}: {}", image_index, type(e).__name__, e)
            comment = f"{type(e).__name__}: {e}"
            analysis = ImageAnalysis(image_index=image_index, image_url=image_url)
            result = self._apply_rules_and_build(analysis)

        # Step 7: Log result (audit trail — always executes)
        self._log_result(result, warranty_id, image_url, comment)
        return result

    # ── Multi-image orchestration ────────────────

    def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Validate all images and aggregate results.

        For each image → validate_image() (7 steps).
        Then OR-aggregate: any valid → pop_valid=True.
        """
        start = time.perf_counter()
        logger.info(
            "PIPELINE START | warranty_id={} | images={}",
            request.warranty_id, len(request.image_urls),
        )

        # Process each image through the 7-step recipe
        pop_valid = False
        uncertain = False
        results: list[ImageValidationResult] = []

        for idx, url in enumerate(request.image_urls):
            result = self.validate_image(url, idx, request.warranty_id)
            results.append(result)

            if result.message == "VALID_RECEIPT_FOUND":
                pop_valid = True
            if result.message and "MISSING_REQUIRED" in result.message:
                uncertain = True

        response = ValidationResponse(
            pop_valid=pop_valid,
            uncertain=uncertain,
            pop_validation_results=results,
        )

        self._log_summary(request, response, start)
        return response

    # ── Step implementations ─────────────────────

    def _assess_quality(self, loaded: LoadedImage, idx: int) -> ImageQualityReport:
        """Step 3: Gemini call #1 — Is this a readable receipt?"""
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
        """Step 4: Gemini call #2 — Extract structured receipt fields."""
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

    def _apply_rules_and_build(self, analysis: ImageAnalysis) -> ImageValidationResult:
        """Steps 5+6: Apply business rules, then build the result."""
        message, _is_valid, _is_uncertain = apply_rules(analysis)
        logger.info("[{}] Rules → {}", analysis.image_index, message)
        return _build_result(analysis, message)

    def _log_result(
        self,
        result: ImageValidationResult,
        warranty_id: str,
        image_url: str,
        comment: str | None,
    ) -> None:
        """Step 7: Persist the result to the audit trail."""
        self._result_logger.log(
            result=result,
            warranty_id=warranty_id,
            image_url=image_url,
            comment=comment,
        )

    # ── Analysis builders ────────────────────────

    @staticmethod
    def _reject_technical(idx: int, url: str) -> ImageAnalysis:
        logger.warning("[{}] REJECTED by technical validation | Gemini SKIPPED", idx)
        return ImageAnalysis(image_index=idx, image_url=url, image_quality=ImageQuality.LOW)

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
    def _build_analysis(
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

    # ── Logging ──────────────────────────────────

    @staticmethod
    def _log_summary(
        request: ValidationRequest,
        response: ValidationResponse,
        start: float,
    ) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "PIPELINE DONE | warranty_id={} | pop_valid={} | uncertain={} | "
            "images={} | {:.0f}ms",
            request.warranty_id, response.pop_valid, response.uncertain,
            len(response.pop_validation_results), elapsed_ms,
        )
        for i, r in enumerate(response.pop_validation_results):
            logger.info("  [{}] {} | retailer={}", i, r.message, r.retailer_name)


def _build_result(analysis: ImageAnalysis, message: str) -> ImageValidationResult:
    """Build an ImageValidationResult by merging metadata with receipt fields."""
    result_data: dict[str, object] = {}

    if analysis.receipt_fields is not None:
        result_data.update(analysis.receipt_fields.model_dump())

    result_data["message"] = message
    result_data["image_category"] = analysis.image_category
    result_data["product_category"] = analysis.product_category
    result_data["is_ai_generated"] = analysis.is_ai_generated
    if analysis.language_category is not None:
        result_data["language_category"] = analysis.language_category

    return ImageValidationResult(**result_data)  # type: ignore[arg-type, unused-ignore]
