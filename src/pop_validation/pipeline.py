"""Main PoP validation pipeline — Facade pattern.

Single entry point that orchestrates OCR extraction, validation rules,
and result aggregation behind one clean `validate()` method.
"""

from __future__ import annotations

import time

from loguru import logger

from pop_validation.config import Settings, get_settings
from pop_validation.models import (
    ImageAnalysis,
    ImageValidationResult,
    ValidationRequest,
    ValidationResponse,
)
from pop_validation.ocr_extractor import OcrExtractor
from pop_validation.product_catalog import ProductCatalogProvider
from pop_validation.validation_rules import apply_rules


class PopValidationPipeline:
    """End-to-end Proof of Purchase validation pipeline.

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
        ocr_extractor: OcrExtractor | None = None,
        product_catalog: ProductCatalogProvider | None = None,
    ) -> None:
        logger.info("Initializing PopValidationPipeline...")
        self._settings = settings or get_settings()
        self._ocr = ocr_extractor or OcrExtractor(
            self._settings, product_catalog=product_catalog
        )
        logger.info("PopValidationPipeline ready")

    def validate(self, request: ValidationRequest) -> ValidationResponse:
        """
        Run the full PoP validation pipeline.

        Three-step flow:
            1. Extract: OCR each image -> ImageAnalysis
            2. Validate: Apply rule chain -> (message, valid, uncertain)
            3. Aggregate: OR across results, build response

        Args:
            request: Validation request with warranty_id and image URLs.

        Returns:
            ValidationResponse with pop_valid, uncertain, and per-image results.
        """
        total_start = time.perf_counter()
        image_count = len(request.image_urls)

        logger.info("=" * 70)
        logger.info(
            "PIPELINE START | warranty_id={} | images={}",
            request.warranty_id,
            image_count,
        )
        logger.info("=" * 70)

        # Step 1: Extract OCR data from each image
        logger.info("Step 1/2: Extracting OCR data from {} image(s)...", image_count)
        extract_start = time.perf_counter()
        analyses = self._extract_all(request.image_urls)
        extract_ms = (time.perf_counter() - extract_start) * 1000
        logger.info(
            "Step 1/2: Extraction complete | {:.0f}ms for {} image(s)", extract_ms, image_count
        )

        # Step 2 + 3: Validate and aggregate
        logger.info("Step 2/2: Applying validation rules and aggregating results...")
        response = self._validate_and_aggregate(analyses)

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info("=" * 70)
        logger.info(
            "PIPELINE DONE | warranty_id={} | pop_valid={} | uncertain={} | "
            "images_processed={} | total_time={:.0f}ms",
            request.warranty_id,
            response.pop_valid,
            response.uncertain,
            image_count,
            total_ms,
        )
        for i, result in enumerate(response.pop_validation_results):
            logger.info(
                "  Image [{}] -> {} | retailer={} | products={}",
                i,
                result.message,
                result.retailer_name,
                result.product_counts,
            )
        logger.info("=" * 70)

        return response

    def _extract_all(self, image_urls: list[str]) -> list[ImageAnalysis]:
        """Extract OCR data from all images."""
        analyses: list[ImageAnalysis] = []
        total = len(image_urls)
        for idx, url in enumerate(image_urls):
            logger.info("Processing image [{}/{}]: {}", idx + 1, total, url)
            analysis = self._ocr.analyze(url, idx)
            analyses.append(analysis)
        return analyses

    def _validate_and_aggregate(self, analyses: list[ImageAnalysis]) -> ValidationResponse:
        """Apply rules to each analysis and aggregate into final response."""
        pop_valid = False
        uncertain = False
        results: list[ImageValidationResult] = []

        for analysis in analyses:
            logger.info(
                "[{}] Applying validation rule chain...",
                analysis.image_index,
            )
            message, image_valid, image_uncertain = apply_rules(analysis)

            logger.info(
                "[{}] Rule chain result | message={} | valid={} | uncertain={}",
                analysis.image_index,
                message,
                image_valid,
                image_uncertain,
            )

            if image_valid:
                pop_valid = True
            if image_uncertain:
                uncertain = True

            result = _build_result(analysis, message)
            results.append(result)

        return ValidationResponse(
            pop_valid=pop_valid,
            uncertain=uncertain,
            pop_validation_results=results,
        )


def _build_result(analysis: ImageAnalysis, message: str) -> ImageValidationResult:
    """Build an ImageValidationResult by merging metadata with receipt fields.

    Analysis-level metadata takes precedence over receipt field values
    when both are present (e.g., language_category).
    """
    result_data: dict[str, object] = {}

    # First: flatten receipt fields (lower precedence)
    if analysis.receipt_fields is not None:
        result_data.update(analysis.receipt_fields.model_dump())

    # Then: overlay analysis metadata (higher precedence)
    result_data["message"] = message
    result_data["image_category"] = analysis.image_category
    result_data["product_category"] = analysis.product_category
    result_data["is_ai_generated"] = analysis.is_ai_generated
    if analysis.language_category is not None:
        result_data["language_category"] = analysis.language_category

    return ImageValidationResult(**result_data)  # type: ignore[arg-type, unused-ignore]
