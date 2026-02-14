"""Tests for Pydantic data models — serialization, validation, defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pop_validation.models import (
    ImageAnalysis,
    ImageCategory,
    ImageQuality,
    ImageQualityReport,
    ImageValidationResult,
    ProductCategory,
    ReceiptFields,
    ReceiptType,
    ValidationRequest,
    ValidationResponse,
)


class TestReceiptType:
    def test_valid_receipt_types(self) -> None:
        assert ReceiptType.OFFICIAL_RECEIPT_PAPER == "official_receipt_paper"
        assert ReceiptType.INVOICE == "invoice"
        assert ReceiptType.E_RECEIPT == "e_receipt"

    def test_all_types_exist(self) -> None:
        assert len(ReceiptType) == 10


class TestImageCategory:
    def test_proof_of_purchase(self) -> None:
        assert ImageCategory.PROOF_OF_PURCHASE == "PROOF_OF_PURCHASE"

    def test_all_categories_exist(self) -> None:
        assert len(ImageCategory) == 6


class TestValidationRequest:
    def test_valid_request(self) -> None:
        req = ValidationRequest(
            warranty_id="W-123",
            image_urls=["https://example.com/receipt.jpg"],
        )
        assert req.warranty_id == "W-123"
        assert len(req.image_urls) == 1

    def test_max_images(self) -> None:
        urls = [f"https://example.com/img{i}.jpg" for i in range(10)]
        req = ValidationRequest(warranty_id="W-123", image_urls=urls)
        assert len(req.image_urls) == 10

    def test_empty_urls_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ValidationRequest(warranty_id="W-123", image_urls=[])

    def test_too_many_urls_rejected(self) -> None:
        urls = [f"https://example.com/img{i}.jpg" for i in range(11)]
        with pytest.raises(ValidationError):
            ValidationRequest(warranty_id="W-123", image_urls=urls)


class TestReceiptFields:
    def test_all_fields_optional(self) -> None:
        rf = ReceiptFields()
        assert rf.retailer_name is None
        assert rf.purchase_date is None
        assert rf.price is None
        assert rf.product_counts is None

    def test_full_receipt(self, valid_receipt_fields: ReceiptFields) -> None:
        assert valid_receipt_fields.retailer_name == "Foot Locker"
        assert valid_receipt_fields.price == 149.99
        assert valid_receipt_fields.product_counts == {"Cloud 5": 1}

    def test_serialization_roundtrip(self, valid_receipt_fields: ReceiptFields) -> None:
        data = valid_receipt_fields.model_dump()
        restored = ReceiptFields(**data)
        assert restored == valid_receipt_fields


class TestImageQualityReport:
    def test_defaults(self) -> None:
        report = ImageQualityReport()
        assert report.resolution_ok is True
        assert report.is_blurry is False
        assert report.format_ok is True
        assert report.image_quality == ImageQuality.HIGH
        assert report.rejection_reason is None

    def test_rejected_report(self) -> None:
        report = ImageQualityReport(
            resolution_ok=False,
            rejection_reason="LOW_IMAGE_QUALITY",
        )
        assert report.rejection_reason == "LOW_IMAGE_QUALITY"


class TestImageAnalysis:
    def test_minimal_analysis(self) -> None:
        analysis = ImageAnalysis(image_index=0, image_url="test.jpg")
        assert analysis.image_index == 0
        assert analysis.is_ai_generated is False
        assert analysis.receipt_fields is None

    def test_full_analysis(self, valid_image_analysis: ImageAnalysis) -> None:
        assert valid_image_analysis.image_quality == ImageQuality.HIGH
        assert valid_image_analysis.receipt_fields is not None
        assert valid_image_analysis.receipt_fields.retailer_name == "Foot Locker"


class TestImageValidationResult:
    def test_result_with_message(self) -> None:
        result = ImageValidationResult(message="VALID_RECEIPT_FOUND")
        assert result.message == "VALID_RECEIPT_FOUND"
        assert result.is_ai_generated is False

    def test_result_with_all_fields(self) -> None:
        result = ImageValidationResult(
            message="VALID_RECEIPT_FOUND",
            image_category=ImageCategory.PROOF_OF_PURCHASE,
            product_category=ProductCategory.SHOES,
            retailer_name="Foot Locker",
            purchase_date="2025-01-15",
            product_counts={"Cloud 5": 1},
        )
        assert result.retailer_name == "Foot Locker"
        assert result.product_counts == {"Cloud 5": 1}


class TestValidationResponse:
    def test_default_response(self) -> None:
        resp = ValidationResponse()
        assert resp.pop_valid is False
        assert resp.uncertain is False
        assert resp.pop_validation_results == []

    def test_valid_response(self) -> None:
        resp = ValidationResponse(
            pop_valid=True,
            uncertain=False,
            pop_validation_results=[
                ImageValidationResult(message="VALID_RECEIPT_FOUND"),
            ],
        )
        assert resp.pop_valid is True
        assert len(resp.pop_validation_results) == 1
