"""Tests for field parsing utilities — pure functions, no mocks needed."""

from __future__ import annotations

from pop_validation.field_utils import (
    build_quality_report,
    build_receipt_fields,
    get_language,
    infer_product_category,
    parse_enum,
    sanitize_fields,
)
from pop_validation.models import (
    ImageCategory,
    ImageQuality,
    ImageQualityReport,
    ProductCategory,
    ReceiptFields,
)

# ──────────────────────────────────────────────
# parse_enum
# ──────────────────────────────────────────────


class TestParseEnum:
    def test_valid_value(self) -> None:
        result = parse_enum("high", ImageQuality)
        assert result == ImageQuality.HIGH

    def test_invalid_value(self) -> None:
        result = parse_enum("garbage", ImageQuality)
        assert result is None

    def test_none_value(self) -> None:
        result = parse_enum(None, ImageQuality)
        assert result is None

    def test_image_category(self) -> None:
        result = parse_enum("PROOF_OF_PURCHASE", ImageCategory)
        assert result == ImageCategory.PROOF_OF_PURCHASE

    def test_product_category(self) -> None:
        result = parse_enum("shoes", ProductCategory)
        assert result == ProductCategory.SHOES


# ──────────────────────────────────────────────
# sanitize_fields
# ──────────────────────────────────────────────


class TestSanitizeFields:
    def test_converts_price_to_float(self) -> None:
        data = {"price": "149.99", "retailer_name": "Test"}
        result = sanitize_fields(data)
        assert result["price"] == 149.99

    def test_handles_invalid_price(self) -> None:
        data = {"price": "not-a-number"}
        result = sanitize_fields(data)
        assert result["price"] is None

    def test_preserves_none_price(self) -> None:
        data = {"price": None}
        result = sanitize_fields(data)
        assert result["price"] is None

    def test_preserves_other_fields(self) -> None:
        data = {"retailer_name": "Test", "currency": "USD"}
        result = sanitize_fields(data)
        assert result == data

    def test_does_not_mutate_input(self) -> None:
        data = {"price": "99.99"}
        sanitize_fields(data)
        assert data["price"] == "99.99"  # Original unchanged


# ──────────────────────────────────────────────
# infer_product_category
# ──────────────────────────────────────────────


class TestInferProductCategory:
    def test_none_fields(self) -> None:
        assert infer_product_category(None) is None

    def test_none_product_counts(self) -> None:
        assert infer_product_category(ReceiptFields(product_counts=None)) is None

    def test_shoe_keyword_detected(self) -> None:
        fields = ReceiptFields(product_counts={"Cloud 5": 1})
        assert infer_product_category(fields) == ProductCategory.SHOES

    def test_default_to_shoes(self) -> None:
        fields = ReceiptFields(product_counts={"random_product": 1})
        assert infer_product_category(fields) == ProductCategory.SHOES


# ──────────────────────────────────────────────
# get_language
# ──────────────────────────────────────────────


class TestGetLanguage:
    def test_from_receipt_fields(self) -> None:
        quality = ImageQualityReport()
        fields = ReceiptFields(language_category="en")
        assert get_language(quality, fields) == "en"

    def test_none_fields(self) -> None:
        quality = ImageQualityReport()
        assert get_language(quality, None) is None

    def test_no_language_in_fields(self) -> None:
        quality = ImageQualityReport()
        fields = ReceiptFields()
        assert get_language(quality, fields) is None


# ──────────────────────────────────────────────
# build_quality_report
# ──────────────────────────────────────────────


class TestBuildQualityReport:
    def test_parses_valid_response(self) -> None:
        data = {
            "is_receipt": True,
            "is_readable": True,
            "is_ai_generated": False,
            "image_quality": "high",
            "image_category": "PROOF_OF_PURCHASE",
            "rejection_reason": None,
        }
        report = build_quality_report(data)
        assert report.is_receipt is True
        assert report.image_quality == ImageQuality.HIGH
        assert report.image_category == ImageCategory.PROOF_OF_PURCHASE
        assert report.rejection_reason is None

    def test_defaults_on_empty(self) -> None:
        report = build_quality_report({})
        assert report.is_receipt is False
        assert report.image_quality == ImageQuality.HIGH
        assert report.image_category == ImageCategory.OTHER


# ──────────────────────────────────────────────
# build_receipt_fields
# ──────────────────────────────────────────────


class TestBuildReceiptFields:
    def test_parses_valid_response(self) -> None:
        data = {
            "retailer_name": "Foot Locker",
            "price": "149.99",
            "currency": "USD",
        }
        fields = build_receipt_fields(data)
        assert fields.retailer_name == "Foot Locker"
        assert fields.price == 149.99
        assert fields.currency == "USD"

    def test_handles_empty_response(self) -> None:
        fields = build_receipt_fields({})
        assert fields.retailer_name is None
        assert fields.price is None
