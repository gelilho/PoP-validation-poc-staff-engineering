"""Tests for PopValidationPipeline — facade + orchestration tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from PIL import Image

from pop_validation.config import Settings
from pop_validation.imaging.loader import LoadedImage
from pop_validation.models import (
    ImageCategory,
    ImageQualityReport,
    ImageValidationResult,
    ProductCategory,
    ValidationRequest,
)
from pop_validation.pipeline import PopValidationPipeline

# ──────────────────────────────────────────────
# Noop logger (avoids CSV file I/O in unit tests)
# ──────────────────────────────────────────────


class _NoOpLogger:
    """A result logger that does nothing — for unit tests."""

    def log(
        self,
        *,
        result: ImageValidationResult,
        warranty_id: str,
        image_url: str,
        comment: str | None = None,
    ) -> None:
        pass


_NOOP_LOGGER = _NoOpLogger()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _valid_result() -> ImageValidationResult:
    """A result that represents a fully valid receipt."""
    return ImageValidationResult(
        message="VALID_RECEIPT_FOUND",
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
        image_category=ImageCategory.PROOF_OF_PURCHASE,
        product_category=ProductCategory.SHOES,
        language_category="en",
        is_ai_generated=False,
    )


def _invalid_result() -> ImageValidationResult:
    """A result where no receipt was found."""
    return ImageValidationResult(message="RECEIPT_NOT_FOUND")


def _uncertain_result() -> ImageValidationResult:
    """A result with missing required fields."""
    return ImageValidationResult(
        message="RECEIPT_MISSING_REQUIRED_INFORMATION: PRODUCT_PRICES",
        retailer_name="Foot Locker",
        retailer_location="NYC",
        purchase_date="2025-01-15",
        receipt_type="official_receipt_paper",
        product_counts={"Cloud 5": 1},
    )


def _ai_generated_result() -> ImageValidationResult:
    """A result flagged as AI-generated fraud."""
    return ImageValidationResult(
        message="FRAUDULENT_IMAGE_DETECTED",
        is_ai_generated=True,
    )


def _fake_loaded_image() -> LoadedImage:
    """Create a LoadedImage suitable for testing."""
    img = Image.new("RGB", (800, 600))
    return LoadedImage(
        data=b"fake-image-data", mime_type="image/jpeg", source="test", pil_image=img,
    )


# ──────────────────────────────────────────────
# Pipeline Facade Tests (mock orchestrator.run_image)
# ──────────────────────────────────────────────


class TestPopValidationPipeline:
    """Test validate() orchestration by mocking run_image on the orchestrator."""

    @patch("pop_validation.client.gemini_client.genai")
    def test_single_valid_image(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        pipeline._orchestrator.run_image = MagicMock(return_value=_valid_result())  # type: ignore[method-assign]

        request = ValidationRequest(
            warranty_id="W-123",
            image_urls=["https://example.com/receipt.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is True
        assert response.uncertain is False
        assert len(response.pop_validation_results) == 1
        assert response.pop_validation_results[0].message == "VALID_RECEIPT_FOUND"

    @patch("pop_validation.client.gemini_client.genai")
    def test_single_invalid_image(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        pipeline._orchestrator.run_image = MagicMock(return_value=_invalid_result())  # type: ignore[method-assign]

        request = ValidationRequest(
            warranty_id="W-456",
            image_urls=["https://example.com/bad.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.uncertain is False
        assert response.pop_validation_results[0].message == "RECEIPT_NOT_FOUND"

    @patch("pop_validation.client.gemini_client.genai")
    def test_mixed_images_one_valid(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        pipeline._orchestrator.run_image = MagicMock(  # type: ignore[method-assign]
            side_effect=[_invalid_result(), _valid_result(), _invalid_result()],
        )

        request = ValidationRequest(
            warranty_id="W-789",
            image_urls=["img0.jpg", "img1.jpg", "img2.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is True  # At least one valid
        assert len(response.pop_validation_results) == 3

    @patch("pop_validation.client.gemini_client.genai")
    def test_all_invalid_images(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        pipeline._orchestrator.run_image = MagicMock(  # type: ignore[method-assign]
            side_effect=[_invalid_result(), _invalid_result()],
        )

        request = ValidationRequest(
            warranty_id="W-000",
            image_urls=["bad1.jpg", "bad2.jpg"],
        )
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.uncertain is False

    @patch("pop_validation.client.gemini_client.genai")
    def test_uncertain_when_missing_fields(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        pipeline._orchestrator.run_image = MagicMock(return_value=_uncertain_result())  # type: ignore[method-assign]

        request = ValidationRequest(warranty_id="W-UNC", image_urls=["test.jpg"])
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.uncertain is True

    @patch("pop_validation.client.gemini_client.genai")
    def test_ai_generated_rejection(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        pipeline._orchestrator.run_image = MagicMock(return_value=_ai_generated_result())  # type: ignore[method-assign]

        request = ValidationRequest(warranty_id="W-AI", image_urls=["fake.jpg"])
        response = pipeline.validate(request)

        assert response.pop_valid is False
        assert response.pop_validation_results[0].message == "FRAUDULENT_IMAGE_DETECTED"


# ──────────────────────────────────────────────
# Per-Image Agent Flow Tests (integration through all agents)
# ──────────────────────────────────────────────


class TestValidateImageFlow:
    """Test the full per-image agent flow by mocking external boundaries."""

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.image_loader_agent.load_image")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_full_success_path(
        self,
        mock_tech: MagicMock,
        mock_load: MagicMock,
        mock_genai: MagicMock,
        settings: Settings,
    ) -> None:
        """All agents succeed → VALID_RECEIPT_FOUND."""
        mock_load.return_value = _fake_loaded_image()
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=True, format_ok=True, file_size_ok=True, blur_score=100.0,
        )

        quality_json = json.dumps({
            "is_receipt": True, "is_readable": True, "is_ai_generated": False,
            "image_quality": "high", "image_category": "PROOF_OF_PURCHASE",
            "rejection_reason": None,
        })
        extraction_json = json.dumps({
            "retailer_name": "Foot Locker", "retailer_location": "NYC",
            "purchase_date": "2025-01-15", "product_name": "Cloud 5",
            "price": 149.99, "currency": "USD",
            "receipt_type": "official_receipt_paper",
            "product_counts": {"Cloud 5": 1}, "product_prices": {"Cloud 5": 149.99},
        })

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = [
            MagicMock(text=quality_json),
            MagicMock(text=extraction_json),
        ]

        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        result = pipeline.validate_image("https://example.com/receipt.jpg", 0)

        assert result.message == "VALID_RECEIPT_FOUND"
        assert result.retailer_name == "Foot Locker"

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.image_loader_agent.load_image")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_technical_rejection_skips_gemini(
        self,
        mock_tech: MagicMock,
        mock_load: MagicMock,
        mock_genai: MagicMock,
        settings: Settings,
    ) -> None:
        """Tech check fails → skip Gemini calls entirely."""
        mock_load.return_value = _fake_loaded_image()
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=False, format_ok=True, file_size_ok=True, blur_score=0.0,
        )

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        result = pipeline.validate_image("https://example.com/blurry.jpg", 0)

        assert result.message == "LOW_IMAGE_QUALITY"
        mock_model.generate_content.assert_not_called()

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.image_loader_agent.load_image")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_quality_rejection_skips_extraction(
        self,
        mock_tech: MagicMock,
        mock_load: MagicMock,
        mock_genai: MagicMock,
        settings: Settings,
    ) -> None:
        """Gemini quality rejects → skip field extraction."""
        mock_load.return_value = _fake_loaded_image()
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=True, format_ok=True, file_size_ok=True, blur_score=100.0,
        )

        quality_json = json.dumps({
            "is_receipt": False, "is_readable": False, "is_ai_generated": False,
            "image_quality": "low", "image_category": "OTHER",
            "rejection_reason": "NOT_A_RECEIPT",
        })
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = MagicMock(text=quality_json)

        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        result = pipeline.validate_image("https://example.com/cat.jpg", 0)

        assert result.message == "LOW_IMAGE_QUALITY"
        # Only 1 Gemini call (quality), not 2 (extraction skipped)
        assert mock_model.generate_content.call_count == 1

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.image_loader_agent.load_image")
    def test_load_failure_returns_graceful_result(
        self,
        mock_load: MagicMock,
        mock_genai: MagicMock,
        settings: Settings,
    ) -> None:
        """Image load throws → graceful fallback, no crash."""
        mock_load.side_effect = FileNotFoundError("Image not found")
        mock_genai.GenerativeModel.return_value = MagicMock()

        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        result = pipeline.validate_image("/nonexistent.jpg", 0)

        assert result.message == "RECEIPT_NOT_FOUND"

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.image_loader_agent.load_image")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_gemini_api_error_returns_quality_rejection(
        self,
        mock_tech: MagicMock,
        mock_load: MagicMock,
        mock_genai: MagicMock,
        settings: Settings,
    ) -> None:
        """Gemini call #1 throws → quality assessment failed message."""
        mock_load.return_value = _fake_loaded_image()
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=True, format_ok=True, file_size_ok=True, blur_score=100.0,
        )

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        result = pipeline.validate_image("https://example.com/receipt.jpg", 0)

        # Quality assessment failed → no receipt fields → rules say RECEIPT_NOT_FOUND
        assert result.message == "RECEIPT_NOT_FOUND"

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.image_loader_agent.load_image")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_extraction_error_still_applies_rules(
        self,
        mock_tech: MagicMock,
        mock_load: MagicMock,
        mock_genai: MagicMock,
        settings: Settings,
    ) -> None:
        """Gemini call #2 throws → fields=None, rules still applied."""
        mock_load.return_value = _fake_loaded_image()
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=True, format_ok=True, file_size_ok=True, blur_score=100.0,
        )

        quality_json = json.dumps({
            "is_receipt": True, "is_readable": True, "is_ai_generated": False,
            "image_quality": "high", "image_category": "PROOF_OF_PURCHASE",
            "rejection_reason": None,
        })
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = [
            MagicMock(text=quality_json),
            RuntimeError("Extraction failed"),
        ]

        pipeline = PopValidationPipeline(settings=settings, result_logger=_NOOP_LOGGER)
        result = pipeline.validate_image("https://example.com/receipt.jpg", 0)

        # No fields extracted → RECEIPT_NOT_FOUND from rules
        assert result.message == "RECEIPT_NOT_FOUND"
