"""OCR extraction using Google Gemini 2.5 Flash.

Strategy pattern: injectable extractor that can be swapped for mocks or other providers.
Two isolated LLM calls per image: quality assessment, then field extraction.
No retries — fail fast to avoid wasting API cost.
"""

from __future__ import annotations

import json
import time
from typing import Any

import google.generativeai as genai
from loguru import logger

from pop_validation.config import Settings, load_products
from pop_validation.image_loader import LoadedImage, load_image
from pop_validation.image_validator import validate_image
from pop_validation.models import (
    ImageAnalysis,
    ImageCategory,
    ImageQuality,
    ImageQualityReport,
    ProductCategory,
    ReceiptFields,
)
from pop_validation.prompts.extraction_prompt import build_extraction_prompt
from pop_validation.prompts.quality_prompt import build_quality_prompt


class OcrExtractor:
    """Extracts receipt data from images using Gemini 2.5 Flash."""

    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing OcrExtractor...")
        genai.configure(api_key=settings.gemini_api_key)  # type: ignore[attr-defined]
        self._model = genai.GenerativeModel(  # type: ignore[attr-defined]
            model_name=settings.gemini_model_name,
            generation_config=genai.GenerationConfig(  # type: ignore[attr-defined]
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        self._quality_prompt = build_quality_prompt()
        self._extraction_prompt = build_extraction_prompt(load_products())
        logger.info(
            "OcrExtractor ready | model={} | products_catalog={} items",
            settings.gemini_model_name,
            len(load_products()),
        )

    def analyze(self, image_url: str, image_index: int) -> ImageAnalysis:
        """
        Full analysis of a single image: load -> technical check -> LLM quality -> LLM extract.

        Args:
            image_url: URL or path to the image.
            image_index: Zero-based index of this image in the request.

        Returns:
            ImageAnalysis with all extracted data, or minimal result on error.
        """
        logger.info(
            "--- Image [{}/{}] analysis started | source={}",
            image_index,
            "?",
            image_url,
        )

        try:
            # Step 1: Load image
            logger.info("[{}] Loading image from source...", image_index)
            loaded = load_image(image_url)
            logger.info(
                "[{}] Image loaded | size={} bytes | mime={} | dimensions={}x{}",
                image_index,
                len(loaded.data),
                loaded.mime_type,
                loaded.pil_image.width,
                loaded.pil_image.height,
            )

            # Step 2: Technical validation (Pillow — fast, no API cost)
            logger.info("[{}] Running technical validation (Pillow, no API cost)...", image_index)
            tech_report = validate_image(loaded)

            if not tech_report.resolution_ok or not tech_report.file_size_ok:
                logger.warning(
                    "[{}] REJECTED by technical validation | resolution_ok={} | "
                    "file_size_ok={} | format_ok={} | blur_score={:.1f} | "
                    "Gemini calls SKIPPED (cost saved)",
                    image_index,
                    tech_report.resolution_ok,
                    tech_report.file_size_ok,
                    tech_report.format_ok,
                    tech_report.blur_score,
                )
                return ImageAnalysis(
                    image_index=image_index,
                    image_url=image_url,
                    image_quality=ImageQuality.LOW,
                )

            logger.info(
                "[{}] Technical validation PASSED | resolution_ok={} | "
                "blur_score={:.1f} (blurry={}) | format_ok={} | file_size_ok={}",
                image_index,
                tech_report.resolution_ok,
                tech_report.blur_score,
                tech_report.is_blurry,
                tech_report.format_ok,
                tech_report.file_size_ok,
            )

            # Step 3: LLM quality assessment (Gemini call #1)
            quality_report = self.assess_quality(loaded, image_index)

            # Short-circuit if quality check rejects the image
            if quality_report.rejection_reason is not None:
                logger.warning(
                    "[{}] REJECTED by Gemini quality assessment | reason={} | "
                    "category={} | ai_generated={} | "
                    "Extraction call SKIPPED (cost saved)",
                    image_index,
                    quality_report.rejection_reason,
                    quality_report.image_category,
                    quality_report.is_ai_generated,
                )
                return ImageAnalysis(
                    image_index=image_index,
                    image_url=image_url,
                    image_quality=quality_report.image_quality,
                    image_category=quality_report.image_category,
                    is_ai_generated=quality_report.is_ai_generated,
                )

            logger.info(
                "[{}] Quality assessment PASSED | is_receipt={} | readable={} | "
                "quality={} | category={} | ai_generated={}",
                image_index,
                quality_report.is_receipt,
                quality_report.is_readable,
                quality_report.image_quality,
                quality_report.image_category,
                quality_report.is_ai_generated,
            )

            # Step 4: LLM field extraction (only if quality passes — Gemini call #2)
            receipt_fields = self.extract_fields(loaded, image_index)

            product_category = _infer_product_category(receipt_fields)
            language = _get_language(quality_report, receipt_fields)

            logger.info(
                "[{}] Analysis complete | product_category={} | language={} | fields_extracted={}",
                image_index,
                product_category,
                language,
                receipt_fields is not None,
            )

            return ImageAnalysis(
                image_index=image_index,
                image_url=image_url,
                image_quality=quality_report.image_quality,
                image_category=quality_report.image_category,
                product_category=product_category,
                language_category=language,
                is_ai_generated=quality_report.is_ai_generated,
                receipt_fields=receipt_fields,
            )

        except Exception as e:
            logger.error(
                "[{}] FAILED to analyze image | source={} | error={}: {}",
                image_index,
                image_url,
                type(e).__name__,
                e,
            )
            return ImageAnalysis(image_index=image_index, image_url=image_url)

    def assess_quality(self, image: LoadedImage, image_index: int) -> ImageQualityReport:
        """LLM call #1: Assess image quality and receipt detection."""
        logger.info(
            "[{}] GEMINI CALL #1 — Quality Assessment | sending {} bytes to Gemini...",
            image_index,
            len(image.data),
        )
        start = time.perf_counter()

        try:
            image_part = {"mime_type": image.mime_type, "data": image.data}
            response = self._model.generate_content([self._quality_prompt, image_part])
            elapsed_ms = (time.perf_counter() - start) * 1000
            data = json.loads(response.text)

            logger.info(
                "[{}] GEMINI CALL #1 DONE | {:.0f}ms | "
                "is_receipt={} | readable={} | ai_generated={} | "
                "quality={} | category={} | rejection={}",
                image_index,
                elapsed_ms,
                data.get("is_receipt"),
                data.get("is_readable"),
                data.get("is_ai_generated"),
                data.get("image_quality"),
                data.get("image_category"),
                data.get("rejection_reason"),
            )
            logger.debug("[{}] Full quality response: {}", image_index, data)

            return ImageQualityReport(
                is_receipt=bool(data.get("is_receipt", False)),
                is_readable=bool(data.get("is_readable", True)),
                is_ai_generated=bool(data.get("is_ai_generated", False)),
                image_quality=(
                    _parse_enum(data.get("image_quality"), ImageQuality) or ImageQuality.HIGH
                ),
                image_category=(
                    _parse_enum(data.get("image_category"), ImageCategory) or ImageCategory.OTHER
                ),
                rejection_reason=data.get("rejection_reason"),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "[{}] GEMINI CALL #1 FAILED | {:.0f}ms | error={}: {}",
                image_index,
                elapsed_ms,
                type(e).__name__,
                e,
            )
            # Return with rejection_reason so caller skips extraction (cost saved)
            return ImageQualityReport(
                rejection_reason=f"QUALITY_ASSESSMENT_FAILED: {type(e).__name__}",
            )

    def extract_fields(self, image: LoadedImage, image_index: int) -> ReceiptFields | None:
        """LLM call #2: Extract structured receipt fields."""
        logger.info(
            "[{}] GEMINI CALL #2 — Field Extraction | sending {} bytes to Gemini...",
            image_index,
            len(image.data),
        )
        start = time.perf_counter()

        try:
            image_part = {"mime_type": image.mime_type, "data": image.data}
            response = self._model.generate_content([self._extraction_prompt, image_part])
            elapsed_ms = (time.perf_counter() - start) * 1000
            data = json.loads(response.text)

            fields = ReceiptFields(**_sanitize_fields(data))

            logger.info(
                "[{}] GEMINI CALL #2 DONE | {:.0f}ms | "
                "retailer={} | date={} | products={} | price={} {} | type={}",
                image_index,
                elapsed_ms,
                fields.retailer_name,
                fields.purchase_date,
                fields.product_counts,
                fields.price,
                fields.currency,
                fields.receipt_type,
            )
            logger.debug("[{}] Full extraction response: {}", image_index, data)

            return fields

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "[{}] GEMINI CALL #2 FAILED | {:.0f}ms | error={}: {}",
                image_index,
                elapsed_ms,
                type(e).__name__,
                e,
            )
            return None


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────


def _parse_enum(
    value: str | None,
    enum_cls: type[ImageQuality] | type[ImageCategory] | type[ProductCategory],
) -> Any:
    """Safely parse a string into an enum, returning None on failure."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        logger.warning("Unknown enum value '{}' for {}", value, enum_cls.__name__)
        return None


def _sanitize_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize extracted fields: convert types, handle nulls."""
    sanitized = dict(data)

    # Ensure price is float
    if "price" in sanitized and sanitized["price"] is not None:
        try:
            sanitized["price"] = float(sanitized["price"])
        except (ValueError, TypeError):
            logger.warning(
                "Could not parse price '{}' as float, setting to None", sanitized["price"]
            )
            sanitized["price"] = None

    return sanitized


def _infer_product_category(fields: ReceiptFields | None) -> ProductCategory | None:
    """Infer product category from extracted fields."""
    if fields is None or fields.product_counts is None:
        return None

    shoe_keywords = {"cloud", "roger", "monster", "surfer", "runner", "swift", "eclipse", "nova"}

    for product_name in fields.product_counts:
        name_lower = product_name.lower()
        if any(kw in name_lower for kw in shoe_keywords):
            return ProductCategory.SHOES

    return ProductCategory.SHOES  # Default for On Running


def _get_language(quality: ImageQualityReport, fields: ReceiptFields | None) -> str | None:
    """Get language category from quality report or receipt fields."""
    if fields and fields.language_category:
        return fields.language_category
    return None
