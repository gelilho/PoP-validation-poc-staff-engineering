"""Validation rules for Proof of Purchase — Chain of Responsibility pattern.

Each rule is a pure function: takes ImageAnalysis, returns a message string
if the rule triggers (reject), or None if the image passes that rule.

Rules are applied in strict order. First trigger short-circuits the chain.

Rule order:
    1. AI-generated image detection
    2. Low image quality
    3. No receipt fields found
    4. Invalid receipt type
    5. Unofficial retailer (name == "other")
    6. Retailer not official on purchase date (stub — deferred)
    7. No [Insert your brand] products found
    8. Missing required fields (sets uncertain=True)
    9. All pass -> VALID_RECEIPT_FOUND (sets pop_valid=True)
"""

from __future__ import annotations

from loguru import logger

from pop_validation.config import REQUIRED_FIELDS, VALID_RECEIPT_TYPES
from pop_validation.models import ImageAnalysis, ImageQuality

# ──────────────────────────────────────────────
# Individual Rule Functions (pure, no side effects)
# ──────────────────────────────────────────────


def check_ai_generated(analysis: ImageAnalysis) -> str | None:
    """Rule 1: Reject AI-generated images."""
    if analysis.is_ai_generated:
        logger.warning(
            "[{}] Rule 1 TRIGGERED: AI-generated image detected",
            analysis.image_index,
        )
        return "FRAUDULENT_IMAGE_DETECTED"
    return None


def check_image_quality(analysis: ImageAnalysis) -> str | None:
    """Rule 2: Reject low quality images."""
    if analysis.image_quality == ImageQuality.LOW:
        logger.warning(
            "[{}] Rule 2 TRIGGERED: Low image quality (quality={})",
            analysis.image_index,
            analysis.image_quality,
        )
        return "LOW_IMAGE_QUALITY"
    return None


def check_receipt_found(analysis: ImageAnalysis) -> str | None:
    """Rule 3: Check that receipt fields were extracted."""
    if analysis.receipt_fields is None:
        logger.warning(
            "[{}] Rule 3 TRIGGERED: No receipt fields found in image",
            analysis.image_index,
        )
        return "RECEIPT_NOT_FOUND"
    return None


def check_receipt_type(analysis: ImageAnalysis) -> str | None:
    """Rule 4: Validate receipt type is in the accepted set."""
    rf = analysis.receipt_fields
    if rf is None:
        return None

    receipt_type = rf.receipt_type
    if receipt_type not in VALID_RECEIPT_TYPES:
        type_display = (receipt_type or "none").upper()
        logger.warning(
            "[{}] Rule 4 TRIGGERED: Invalid receipt type '{}' (accepted: {})",
            analysis.image_index,
            type_display,
            ", ".join(sorted(VALID_RECEIPT_TYPES)),
        )
        return f"RECEIPT_FORMAT_NOT_VALID: {type_display}"
    return None


def check_unofficial_retailer(analysis: ImageAnalysis) -> str | None:
    """Rule 5: Reject if retailer is 'other' (not an official partner)."""
    rf = analysis.receipt_fields
    if rf is None:
        return None

    if rf.retailer_name and rf.retailer_name.lower() == "other":
        logger.warning(
            "[{}] Rule 5 TRIGGERED: Unofficial retailer (name='{}')",
            analysis.image_index,
            rf.retailer_name,
        )
        return "UNOFFICIAL_RETAILER"
    return None


def check_retailer_date(analysis: ImageAnalysis) -> str | None:
    """Rule 6: Check if retailer was official on the purchase date.

    STUB — BigQuery retailer data is deferred. Returns None (pass) always.
    Wire in RetailerStore later when BQ integration is added.
    """
    # TODO: Implement with RetailerStore when BigQuery is integrated
    return None


def check_brand_products(analysis: ImageAnalysis) -> str | None:
    """Rule 7: Reject if product_counts contains only 'other' (no [Insert your brand] products)."""
    rf = analysis.receipt_fields
    if rf is None or rf.product_counts is None:
        return None

    non_other_keys = set(rf.product_counts.keys()) - {"other"}
    if len(non_other_keys) == 0:
        logger.warning(
            "[{}] Rule 7 TRIGGERED: No [Insert your brand] products found (only 'other')",
            analysis.image_index,
        )
        return "RECEIPT_DOES_NOT_CONTAIN_BRAND_FOOTWEAR"
    return None


def check_required_fields(analysis: ImageAnalysis) -> str | None:
    """Rule 8: Check that all required fields are present.

    Returns the first missing field. Matches original behavior (breaks on first).
    """
    rf = analysis.receipt_fields
    if rf is None:
        return None

    field_values: dict[str, object] = {
        "retailer_name": rf.retailer_name,
        "retailer_location": rf.retailer_location,
        "purchase_date": rf.purchase_date,
        "product_counts": rf.product_counts,
        "product_prices": rf.product_prices,
    }

    for field_name in REQUIRED_FIELDS:
        if field_values.get(field_name) is None:
            logger.warning(
                "[{}] Rule 8 TRIGGERED: Missing required field '{}' (sets uncertain=True)",
                analysis.image_index,
                field_name.upper(),
            )
            return f"RECEIPT_MISSING_REQUIRED_INFORMATION: {field_name.upper()}"
    return None


# ──────────────────────────────────────────────
# Chain Orchestrator
# ──────────────────────────────────────────────

# Ordered chain — rules are applied top to bottom
_RULE_CHAIN = [
    check_ai_generated,  # Rule 1
    check_image_quality,  # Rule 2
    check_receipt_found,  # Rule 3
    check_receipt_type,  # Rule 4
    check_unofficial_retailer,  # Rule 5
    check_retailer_date,  # Rule 6 (stub)
    check_brand_products,  # Rule 7
]


def apply_rules(analysis: ImageAnalysis) -> tuple[str, bool, bool]:
    """
    Apply all validation rules in order to a single image analysis.

    Chain of Responsibility: first rule that triggers short-circuits the chain.

    Args:
        analysis: The OCR extraction result for one image.

    Returns:
        Tuple of (message, pop_valid, uncertain).
    """
    logger.debug(
        "[{}] Starting rule chain (7 rules + required fields check)...",
        analysis.image_index,
    )

    # Rules 1-7: short-circuit chain
    for rule_fn in _RULE_CHAIN:
        message = rule_fn(analysis)
        if message is not None:
            logger.info(
                "[{}] Rule chain short-circuited at {} -> {}",
                analysis.image_index,
                rule_fn.__name__,
                message,
            )
            return message, False, False

    # Rule 8: required fields (sets uncertain=True if triggered)
    message = check_required_fields(analysis)
    if message is not None:
        logger.info(
            "[{}] Rule chain: missing required field -> uncertain=True | {}",
            analysis.image_index,
            message,
        )
        return message, False, True

    # Rule 9: All pass
    logger.info(
        "[{}] Rule chain: ALL RULES PASSED -> VALID_RECEIPT_FOUND",
        analysis.image_index,
    )
    return "VALID_RECEIPT_FOUND", True, False
