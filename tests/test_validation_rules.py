"""Tests for validation rules — Chain of Responsibility.

This is the most critical test file. Every business rule must be tested
individually and in combination through the chain.

Target: 100% branch coverage on validation_rules.py.
"""

from __future__ import annotations

import pytest

from pop_validation.models import (
    ImageAnalysis,
    ImageCategory,
    ImageQuality,
    ProductCategory,
    ReceiptFields,
)
from pop_validation.validation_rules import (
    apply_rules,
    check_ai_generated,
    check_image_quality,
    check_on_products,
    check_receipt_found,
    check_receipt_type,
    check_required_fields,
    check_retailer_date,
    check_unofficial_retailer,
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_analysis(**overrides: object) -> ImageAnalysis:
    """Create an ImageAnalysis with valid defaults, overridable for tests."""
    defaults: dict[str, object] = {
        "image_index": 0,
        "image_url": "https://example.com/receipt.jpg",
        "image_quality": ImageQuality.HIGH,
        "image_category": ImageCategory.PROOF_OF_PURCHASE,
        "product_category": ProductCategory.SHOES,
        "language_category": "en",
        "is_ai_generated": False,
        "receipt_fields": ReceiptFields(
            retailer_name="Foot Locker",
            retailer_location="125 Main Street",
            purchase_date="2025-01-15",
            product_name="Cloud 5",
            transaction_number="12345",
            price=149.99,
            currency="USD",
            receipt_type="official_receipt_paper",
            product_counts={"Cloud 5": 1},
            product_prices={"Cloud 5": 149.99},
            confidence=0.95,
        ),
    }
    defaults.update(overrides)
    return ImageAnalysis(**defaults)  # type: ignore[arg-type]


# ──────────────────────────────────────────────
# Rule 1: AI Detection
# ──────────────────────────────────────────────


class TestCheckAiGenerated:
    def test_ai_generated_returns_fraud_message(self) -> None:
        analysis = _make_analysis(is_ai_generated=True)
        assert check_ai_generated(analysis) == "FRAUDULENT_IMAGE_DETECTED"

    def test_not_ai_generated_passes(self) -> None:
        analysis = _make_analysis(is_ai_generated=False)
        assert check_ai_generated(analysis) is None


# ──────────────────────────────────────────────
# Rule 2: Image Quality
# ──────────────────────────────────────────────


class TestCheckImageQuality:
    def test_low_quality_rejects(self) -> None:
        analysis = _make_analysis(image_quality=ImageQuality.LOW)
        assert check_image_quality(analysis) == "LOW_IMAGE_QUALITY"

    def test_high_quality_passes(self) -> None:
        analysis = _make_analysis(image_quality=ImageQuality.HIGH)
        assert check_image_quality(analysis) is None

    def test_medium_quality_passes(self) -> None:
        analysis = _make_analysis(image_quality=ImageQuality.MEDIUM)
        assert check_image_quality(analysis) is None

    def test_none_quality_passes(self) -> None:
        analysis = _make_analysis(image_quality=None)
        assert check_image_quality(analysis) is None


# ──────────────────────────────────────────────
# Rule 3: Receipt Found
# ──────────────────────────────────────────────


class TestCheckReceiptFound:
    def test_no_receipt_fields_rejects(self) -> None:
        analysis = _make_analysis(receipt_fields=None)
        assert check_receipt_found(analysis) == "RECEIPT_NOT_FOUND"

    def test_with_receipt_fields_passes(self) -> None:
        analysis = _make_analysis()
        assert check_receipt_found(analysis) is None


# ──────────────────────────────────────────────
# Rule 4: Receipt Type
# ──────────────────────────────────────────────


class TestCheckReceiptType:
    @pytest.mark.parametrize(
        "receipt_type",
        [
            "official_receipt_paper",
            "official_receipt_digital",
            "invoice",
            "e_receipt",
            "order_confirmation",
        ],
    )
    def test_valid_types_pass(self, receipt_type: str) -> None:
        rf = ReceiptFields(receipt_type=receipt_type)
        analysis = _make_analysis(receipt_fields=rf)
        assert check_receipt_type(analysis) is None

    def test_bank_statement_rejected(self) -> None:
        rf = ReceiptFields(receipt_type="bank_statement")
        analysis = _make_analysis(receipt_fields=rf)
        result = check_receipt_type(analysis)
        assert result is not None
        assert "RECEIPT_FORMAT_NOT_VALID" in result
        assert "BANK_STATEMENT" in result

    def test_other_type_rejected(self) -> None:
        rf = ReceiptFields(receipt_type="other")
        analysis = _make_analysis(receipt_fields=rf)
        result = check_receipt_type(analysis)
        assert result is not None
        assert "RECEIPT_FORMAT_NOT_VALID" in result

    def test_none_type_rejected(self) -> None:
        rf = ReceiptFields(receipt_type=None)
        analysis = _make_analysis(receipt_fields=rf)
        result = check_receipt_type(analysis)
        assert result is not None
        assert "NONE" in result

    def test_no_receipt_fields_passes(self) -> None:
        analysis = _make_analysis(receipt_fields=None)
        assert check_receipt_type(analysis) is None


# ──────────────────────────────────────────────
# Rule 5: Unofficial Retailer
# ──────────────────────────────────────────────


class TestCheckUnofficialRetailer:
    def test_other_retailer_rejected(self) -> None:
        rf = ReceiptFields(retailer_name="other")
        analysis = _make_analysis(receipt_fields=rf)
        assert check_unofficial_retailer(analysis) == "UNOFFICIAL_RETAILER"

    def test_other_case_insensitive(self) -> None:
        rf = ReceiptFields(retailer_name="Other")
        analysis = _make_analysis(receipt_fields=rf)
        assert check_unofficial_retailer(analysis) == "UNOFFICIAL_RETAILER"

    def test_other_uppercase(self) -> None:
        rf = ReceiptFields(retailer_name="OTHER")
        analysis = _make_analysis(receipt_fields=rf)
        assert check_unofficial_retailer(analysis) == "UNOFFICIAL_RETAILER"

    def test_real_retailer_passes(self) -> None:
        rf = ReceiptFields(retailer_name="Foot Locker")
        analysis = _make_analysis(receipt_fields=rf)
        assert check_unofficial_retailer(analysis) is None

    def test_none_retailer_passes(self) -> None:
        rf = ReceiptFields(retailer_name=None)
        analysis = _make_analysis(receipt_fields=rf)
        assert check_unofficial_retailer(analysis) is None

    def test_no_receipt_fields_passes(self) -> None:
        analysis = _make_analysis(receipt_fields=None)
        assert check_unofficial_retailer(analysis) is None


# ──────────────────────────────────────────────
# Rule 6: Retailer Date (stub)
# ──────────────────────────────────────────────


class TestCheckRetailerDate:
    def test_always_passes_stub(self) -> None:
        analysis = _make_analysis()
        assert check_retailer_date(analysis) is None

    def test_passes_with_no_receipt(self) -> None:
        analysis = _make_analysis(receipt_fields=None)
        assert check_retailer_date(analysis) is None


# ──────────────────────────────────────────────
# Rule 7: On Products
# ──────────────────────────────────────────────


class TestCheckOnProducts:
    def test_only_other_rejected(self) -> None:
        rf = ReceiptFields(product_counts={"other": 1})
        analysis = _make_analysis(receipt_fields=rf)
        assert check_on_products(analysis) == "RECEIPT_DOES_NOT_CONTAIN_ON_FOOTWEAR"

    def test_on_product_passes(self) -> None:
        rf = ReceiptFields(product_counts={"Cloud 5": 1})
        analysis = _make_analysis(receipt_fields=rf)
        assert check_on_products(analysis) is None

    def test_mixed_products_passes(self) -> None:
        rf = ReceiptFields(product_counts={"Cloud 5": 1, "other": 2})
        analysis = _make_analysis(receipt_fields=rf)
        assert check_on_products(analysis) is None

    def test_none_product_counts_passes(self) -> None:
        rf = ReceiptFields(product_counts=None)
        analysis = _make_analysis(receipt_fields=rf)
        assert check_on_products(analysis) is None

    def test_empty_product_counts_rejected(self) -> None:
        rf = ReceiptFields(product_counts={})
        analysis = _make_analysis(receipt_fields=rf)
        # Empty dict has no non-other keys → rejected
        assert check_on_products(analysis) == "RECEIPT_DOES_NOT_CONTAIN_ON_FOOTWEAR"

    def test_no_receipt_fields_passes(self) -> None:
        analysis = _make_analysis(receipt_fields=None)
        assert check_on_products(analysis) is None


# ──────────────────────────────────────────────
# Rule 8: Required Fields
# ──────────────────────────────────────────────


class TestCheckRequiredFields:
    def test_all_present_passes(self) -> None:
        analysis = _make_analysis()
        assert check_required_fields(analysis) is None

    def test_missing_retailer_name(self) -> None:
        rf = ReceiptFields(
            retailer_location="NYC",
            purchase_date="2025-01-15",
            product_counts={"Cloud 5": 1},
            product_prices={"Cloud 5": 149.99},
        )
        analysis = _make_analysis(receipt_fields=rf)
        result = check_required_fields(analysis)
        assert result is not None
        assert "RETAILER_NAME" in result

    def test_missing_purchase_date(self) -> None:
        rf = ReceiptFields(
            retailer_name="Foot Locker",
            retailer_location="NYC",
            product_counts={"Cloud 5": 1},
            product_prices={"Cloud 5": 149.99},
        )
        analysis = _make_analysis(receipt_fields=rf)
        result = check_required_fields(analysis)
        assert result is not None
        assert "PURCHASE_DATE" in result

    def test_missing_retailer_location(self) -> None:
        rf = ReceiptFields(
            retailer_name="Foot Locker",
            purchase_date="2025-01-15",
            product_counts={"Cloud 5": 1},
            product_prices={"Cloud 5": 149.99},
        )
        analysis = _make_analysis(receipt_fields=rf)
        result = check_required_fields(analysis)
        assert result is not None
        assert "RETAILER_LOCATION" in result

    def test_missing_product_counts(self) -> None:
        rf = ReceiptFields(
            retailer_name="Foot Locker",
            retailer_location="NYC",
            purchase_date="2025-01-15",
            product_prices={"Cloud 5": 149.99},
        )
        analysis = _make_analysis(receipt_fields=rf)
        result = check_required_fields(analysis)
        assert result is not None
        assert "PRODUCT_COUNTS" in result

    def test_missing_product_prices(self) -> None:
        rf = ReceiptFields(
            retailer_name="Foot Locker",
            retailer_location="NYC",
            purchase_date="2025-01-15",
            product_counts={"Cloud 5": 1},
        )
        analysis = _make_analysis(receipt_fields=rf)
        result = check_required_fields(analysis)
        assert result is not None
        assert "PRODUCT_PRICES" in result

    def test_no_receipt_fields_passes(self) -> None:
        analysis = _make_analysis(receipt_fields=None)
        assert check_required_fields(analysis) is None


# ──────────────────────────────────────────────
# Chain Orchestrator: apply_rules
# ──────────────────────────────────────────────


class TestApplyRules:
    def test_valid_receipt_returns_valid(self) -> None:
        analysis = _make_analysis()
        message, pop_valid, uncertain = apply_rules(analysis)
        assert message == "VALID_RECEIPT_FOUND"
        assert pop_valid is True
        assert uncertain is False

    def test_ai_generated_short_circuits(self) -> None:
        analysis = _make_analysis(is_ai_generated=True, image_quality=ImageQuality.LOW)
        message, pop_valid, _uncertain = apply_rules(analysis)
        assert message == "FRAUDULENT_IMAGE_DETECTED"  # Rule 1 wins
        assert pop_valid is False

    def test_low_quality_before_receipt_check(self) -> None:
        analysis = _make_analysis(image_quality=ImageQuality.LOW, receipt_fields=None)
        message, _pop_valid, _uncertain = apply_rules(analysis)
        assert message == "LOW_IMAGE_QUALITY"  # Rule 2 wins over Rule 3

    def test_no_receipt_fields(self) -> None:
        analysis = _make_analysis(
            image_quality=ImageQuality.HIGH,
            receipt_fields=None,
        )
        message, pop_valid, uncertain = apply_rules(analysis)
        assert message == "RECEIPT_NOT_FOUND"
        assert pop_valid is False
        assert uncertain is False

    def test_invalid_receipt_type(self) -> None:
        rf = ReceiptFields(receipt_type="bank_statement")
        analysis = _make_analysis(receipt_fields=rf)
        message, _pop_valid, _uncertain = apply_rules(analysis)
        assert "RECEIPT_FORMAT_NOT_VALID" in message

    def test_unofficial_retailer(self) -> None:
        rf = ReceiptFields(
            retailer_name="other",
            receipt_type="official_receipt_paper",
        )
        analysis = _make_analysis(receipt_fields=rf)
        message, _pop_valid, _uncertain = apply_rules(analysis)
        assert message == "UNOFFICIAL_RETAILER"

    def test_no_on_products(self) -> None:
        rf = ReceiptFields(
            retailer_name="Foot Locker",
            receipt_type="official_receipt_paper",
            product_counts={"other": 1},
        )
        analysis = _make_analysis(receipt_fields=rf)
        message, _pop_valid, _uncertain = apply_rules(analysis)
        assert message == "RECEIPT_DOES_NOT_CONTAIN_ON_FOOTWEAR"

    def test_missing_field_sets_uncertain(self) -> None:
        rf = ReceiptFields(
            retailer_name="Foot Locker",
            retailer_location="NYC",
            purchase_date="2025-01-15",
            receipt_type="official_receipt_paper",
            product_counts={"Cloud 5": 1},
            # product_prices is missing
        )
        analysis = _make_analysis(receipt_fields=rf)
        message, pop_valid, uncertain = apply_rules(analysis)
        assert "RECEIPT_MISSING_REQUIRED_INFORMATION" in message
        assert pop_valid is False
        assert uncertain is True

    def test_rule_order_is_strict(self) -> None:
        """AI detection (Rule 1) must trigger before quality (Rule 2)."""
        analysis = _make_analysis(
            is_ai_generated=True,
            image_quality=ImageQuality.LOW,
            receipt_fields=None,
        )
        message, _, _ = apply_rules(analysis)
        assert message == "FRAUDULENT_IMAGE_DETECTED"

    def test_quality_before_receipt_found(self) -> None:
        """Quality (Rule 2) must trigger before receipt found (Rule 3)."""
        analysis = _make_analysis(
            image_quality=ImageQuality.LOW,
            receipt_fields=None,
        )
        message, _, _ = apply_rules(analysis)
        assert message == "LOW_IMAGE_QUALITY"
