"""Tests for Gemini OCR extraction with mocked API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from pop_validation.config import Settings
from pop_validation.models import ImageQuality
from pop_validation.ocr_extractor import OcrExtractor, _parse_enum, _sanitize_fields


class TestParseEnum:
    def test_valid_value(self) -> None:
        result = _parse_enum("high", ImageQuality)
        assert result == ImageQuality.HIGH

    def test_invalid_value(self) -> None:
        result = _parse_enum("garbage", ImageQuality)
        assert result is None

    def test_none_value(self) -> None:
        result = _parse_enum(None, ImageQuality)
        assert result is None


class TestSanitizeFields:
    def test_converts_price_to_float(self) -> None:
        data = {"price": "149.99", "retailer_name": "Test"}
        result = _sanitize_fields(data)
        assert result["price"] == 149.99

    def test_handles_invalid_price(self) -> None:
        data = {"price": "not-a-number"}
        result = _sanitize_fields(data)
        assert result["price"] is None

    def test_preserves_none_price(self) -> None:
        data = {"price": None}
        result = _sanitize_fields(data)
        assert result["price"] is None

    def test_preserves_other_fields(self) -> None:
        data = {"retailer_name": "Test", "currency": "USD"}
        result = _sanitize_fields(data)
        assert result == data


class TestOcrExtractor:
    @patch("pop_validation.ocr_extractor.genai")
    def test_initialization(self, mock_genai: MagicMock, settings: Settings) -> None:
        OcrExtractor(settings)
        mock_genai.configure.assert_called_once_with(api_key=settings.gemini_api_key)

    @patch("pop_validation.ocr_extractor.genai")
    def test_assess_quality_success(self, mock_genai: MagicMock, settings: Settings) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        quality_response = json.dumps(
            {
                "is_receipt": True,
                "is_readable": True,
                "is_ai_generated": False,
                "image_quality": "high",
                "image_category": "PROOF_OF_PURCHASE",
                "rejection_reason": None,
            }
        )
        mock_model.generate_content.return_value = MagicMock(text=quality_response)

        extractor = OcrExtractor(settings)

        from PIL import Image

        from pop_validation.image_loader import LoadedImage

        img = Image.new("RGB", (100, 100))
        loaded = LoadedImage(data=b"fake", mime_type="image/jpeg", source="test", pil_image=img)

        report = extractor.assess_quality(loaded, 0)
        assert report.is_receipt is True
        assert report.is_ai_generated is False
        assert report.rejection_reason is None

    @patch("pop_validation.ocr_extractor.genai")
    def test_assess_quality_error_returns_default(
        self, mock_genai: MagicMock, settings: Settings
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = Exception("API error")

        extractor = OcrExtractor(settings)

        from PIL import Image

        from pop_validation.image_loader import LoadedImage

        img = Image.new("RGB", (100, 100))
        loaded = LoadedImage(data=b"fake", mime_type="image/jpeg", source="test", pil_image=img)

        report = extractor.assess_quality(loaded, 0)
        # On error, rejection_reason is set to prevent wasted extraction calls (cost saved)
        assert report.rejection_reason is not None
        assert "QUALITY_ASSESSMENT_FAILED" in report.rejection_reason

    @patch("pop_validation.ocr_extractor.genai")
    def test_extract_fields_success(self, mock_genai: MagicMock, settings: Settings) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        extraction_response = json.dumps(
            {
                "retailer_name": "Foot Locker",
                "purchase_date": "2025-01-15",
                "product_name": "Cloud 5",
                "price": 149.99,
                "currency": "USD",
                "receipt_type": "official_receipt_paper",
                "product_counts": {"Cloud 5": 1},
                "product_prices": {"Cloud 5": 149.99},
            }
        )
        mock_model.generate_content.return_value = MagicMock(text=extraction_response)

        extractor = OcrExtractor(settings)

        from PIL import Image

        from pop_validation.image_loader import LoadedImage

        img = Image.new("RGB", (100, 100))
        loaded = LoadedImage(data=b"fake", mime_type="image/jpeg", source="test", pil_image=img)

        fields = extractor.extract_fields(loaded, 0)
        assert fields is not None
        assert fields.retailer_name == "Foot Locker"
        assert fields.price == 149.99

    @patch("pop_validation.ocr_extractor.genai")
    def test_extract_fields_error_returns_none(
        self, mock_genai: MagicMock, settings: Settings
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = Exception("API error")

        extractor = OcrExtractor(settings)

        from PIL import Image

        from pop_validation.image_loader import LoadedImage

        img = Image.new("RGB", (100, 100))
        loaded = LoadedImage(data=b"fake", mime_type="image/jpeg", source="test", pil_image=img)

        fields = extractor.extract_fields(loaded, 0)
        assert fields is None

    @patch("pop_validation.ocr_extractor.genai")
    def test_analyze_returns_empty_on_load_failure(
        self, mock_genai: MagicMock, settings: Settings
    ) -> None:
        mock_genai.GenerativeModel.return_value = MagicMock()

        extractor = OcrExtractor(settings)
        result = extractor.analyze("/nonexistent/path.jpg", 0)

        assert result.image_index == 0
        assert result.receipt_fields is None
