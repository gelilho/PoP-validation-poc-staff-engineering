"""Pure helper functions for field parsing and inference.

Stateless, side-effect-free utilities used by the analyzer.
Each function is independently testable with zero dependencies on external services.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pop_validation.models import (
    ImageCategory,
    ImageQuality,
    ImageQualityReport,
    ProductCategory,
    ReceiptFields,
)


def parse_enum(
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


def sanitize_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize extracted fields: convert types, handle nulls."""
    sanitized = dict(data)

    if "price" in sanitized and sanitized["price"] is not None:
        try:
            sanitized["price"] = float(sanitized["price"])
        except (ValueError, TypeError):
            logger.warning(
                "Could not parse price '{}' as float, setting to None",
                sanitized["price"],
            )
            sanitized["price"] = None

    return sanitized


def infer_product_category(fields: ReceiptFields | None) -> ProductCategory | None:
    """Infer product category from extracted fields."""
    if fields is None or fields.product_counts is None:
        return None

    shoe_keywords = {
        "cloud", "roger", "monster", "surfer", "runner", "swift", "eclipse", "nova",
    }

    for product_name in fields.product_counts:
        name_lower = product_name.lower()
        if any(kw in name_lower for kw in shoe_keywords):
            return ProductCategory.SHOES

    return ProductCategory.SHOES  # Default for [Insert your brand]


def get_language(
    quality: ImageQualityReport, fields: ReceiptFields | None,
) -> str | None:
    """Get language category from quality report or receipt fields."""
    if fields and fields.language_category:
        return fields.language_category
    return None


def build_quality_report(data: dict[str, Any]) -> ImageQualityReport:
    """Build an ImageQualityReport from raw Gemini JSON response."""
    return ImageQualityReport(
        is_receipt=bool(data.get("is_receipt", False)),
        is_readable=bool(data.get("is_readable", True)),
        is_ai_generated=bool(data.get("is_ai_generated", False)),
        image_quality=(
            parse_enum(data.get("image_quality"), ImageQuality)
            or ImageQuality.HIGH
        ),
        image_category=(
            parse_enum(data.get("image_category"), ImageCategory)
            or ImageCategory.OTHER
        ),
        rejection_reason=data.get("rejection_reason"),
    )


def build_receipt_fields(data: dict[str, Any]) -> ReceiptFields:
    """Build a ReceiptFields model from raw Gemini JSON response."""
    return ReceiptFields(**sanitize_fields(data))
