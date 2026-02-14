"""Integration tests for the PoP validation pipeline — Facade pattern."""

from __future__ import annotations

from unittest.mock import MagicMock

from pop_validation.config import Settings
from pop_validation.models import (
    ImageAnalysis,
    ImageCategory,
    ImageQuality,
    ProductCategory,
    ReceiptFields,
    ValidationRequest,
)
from pop_validation.pipeline import PopValidationPipeline, _build_result

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_mock_analyzer(analyses: list[ImageAnalysis]) -> MagicMock:
    """Create a mock PopAnalyzer that returns predefined analyses."""
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.side_effect = analyses
    return mock_analyzer


def _valid_analysis(index: int = 0) -> ImageAnalysis:
    """Create a valid ImageAnalysis."""
    return ImageAnalysis(
        image_index=index,
        image_url=f"https://example.com/img{index}.jpg",
        image_quality=ImageQuality.HIGH,
        image_category=ImageCategory.PROOF_OF_PURCHASE,
        product_category=ProductCategory.SHOES,
        language_category="en",
        is_ai_generated=False,
        receipt_fields=ReceiptFields(
            retailer_name="Foot Locker",
            retailer_location="NYC",
            purchase_date="2025-01-15",
            product_name="Cloud 5",
            transaction_number="12345",
            price=149.99,
            currency="USD",
            receipt_type="official_receipt_paper",
            product_counts={"Cloud 5": 1},
            product_prices={"Cloud 5": 149.99},
        ),
    )


def _invalid_analysis(index: int = 0) -> ImageAnalysis:
    """Create an invalid ImageAnalysis (no receipt)."""
    return ImageAnalysis(
        image_index=index,
        image_url=f"https://example.com/img{index}.jpg",
        image_quality=ImageQuality.HIGH,
        receipt_fields=None,
    )


# ──────────────────────────────────────────────
# Pipeline Tests
# ──────────────────────────────────────────────


class TestPopValidationPipeline:
    def test_single_valid_image(self, settings: Settings) -> None:
        mock_analyzer = _make_mock_analyzer([_valid_analysis(0)])
        pipeline = PopValidationPipeline(settings=settings, analyzer=mock_analyzer)

        request = ValidationRequest(
            warranty_id="W-123",
            image_urls=["https://example.com/receipt.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is True
        assert response.uncertain is False
        assert len(response.pop_validation_results) == 1
        assert response.pop_validation_results[0].message == "VALID_RECEIPT_FOUND"

    def test_single_invalid_image(self, settings: Settings) -> None:
        mock_analyzer = _make_mock_analyzer([_invalid_analysis(0)])
        pipeline = PopValidationPipeline(settings=settings, analyzer=mock_analyzer)

        request = ValidationRequest(
            warranty_id="W-456",
            image_urls=["https://example.com/bad.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.uncertain is False
        assert response.pop_validation_results[0].message == "RECEIPT_NOT_FOUND"

    def test_mixed_images_one_valid(self, settings: Settings) -> None:
        mock_analyzer = _make_mock_analyzer(
            [
                _invalid_analysis(0),
                _valid_analysis(1),
                _invalid_analysis(2),
            ]
        )
        pipeline = PopValidationPipeline(settings=settings, analyzer=mock_analyzer)

        request = ValidationRequest(
            warranty_id="W-789",
            image_urls=["img0.jpg", "img1.jpg", "img2.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is True  # At least one valid
        assert len(response.pop_validation_results) == 3

    def test_all_invalid_images(self, settings: Settings) -> None:
        mock_analyzer = _make_mock_analyzer(
            [
                _invalid_analysis(0),
                _invalid_analysis(1),
            ]
        )
        pipeline = PopValidationPipeline(settings=settings, analyzer=mock_analyzer)

        request = ValidationRequest(
            warranty_id="W-000",
            image_urls=["bad1.jpg", "bad2.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.uncertain is False

    def test_uncertain_when_missing_fields(self, settings: Settings) -> None:
        analysis = ImageAnalysis(
            image_index=0,
            image_url="test.jpg",
            image_quality=ImageQuality.HIGH,
            receipt_fields=ReceiptFields(
                retailer_name="Foot Locker",
                retailer_location="NYC",
                purchase_date="2025-01-15",
                receipt_type="official_receipt_paper",
                product_counts={"Cloud 5": 1},
                # product_prices missing
            ),
        )
        mock_analyzer = _make_mock_analyzer([analysis])
        pipeline = PopValidationPipeline(settings=settings, analyzer=mock_analyzer)

        request = ValidationRequest(warranty_id="W-UNC", image_urls=["test.jpg"])
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.uncertain is True
        assert "RECEIPT_MISSING_REQUIRED_INFORMATION" in response.pop_validation_results[0].message

    def test_ai_generated_rejection(self, settings: Settings) -> None:
        analysis = ImageAnalysis(
            image_index=0,
            image_url="fake.jpg",
            image_quality=ImageQuality.HIGH,
            is_ai_generated=True,
        )
        mock_analyzer = _make_mock_analyzer([analysis])
        pipeline = PopValidationPipeline(settings=settings, analyzer=mock_analyzer)

        request = ValidationRequest(warranty_id="W-AI", image_urls=["fake.jpg"])
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.pop_validation_results[0].message == "FRAUDULENT_IMAGE_DETECTED"


# ──────────────────────────────────────────────
# Build Result Tests
# ──────────────────────────────────────────────


class TestBuildResult:
    def test_builds_result_with_receipt_fields(self) -> None:
        analysis = _valid_analysis()
        result = _build_result(analysis, "VALID_RECEIPT_FOUND")

        assert result.message == "VALID_RECEIPT_FOUND"
        assert result.retailer_name == "Foot Locker"
        assert result.product_counts == {"Cloud 5": 1}
        assert result.image_category == ImageCategory.PROOF_OF_PURCHASE

    def test_builds_result_without_receipt_fields(self) -> None:
        analysis = _invalid_analysis()
        result = _build_result(analysis, "RECEIPT_NOT_FOUND")

        assert result.message == "RECEIPT_NOT_FOUND"
        assert result.retailer_name is None
        assert result.product_counts is None

    def test_preserves_metadata(self) -> None:
        analysis = _valid_analysis()
        result = _build_result(analysis, "VALID_RECEIPT_FOUND")

        assert result.product_category == ProductCategory.SHOES
        assert result.language_category == "en"
        assert result.is_ai_generated is False
