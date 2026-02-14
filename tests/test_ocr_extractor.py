"""Tests for OcrExtractor orchestration and GeminiClient integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from pop_validation.config import Settings
from pop_validation.ocr_extractor import OcrExtractor

# ──────────────────────────────────────────────
# GeminiClient (via OcrExtractor integration)
# ──────────────────────────────────────────────


class TestOcrExtractor:
    @patch("pop_validation.gemini_client.genai")
    def test_initialization(self, mock_genai: MagicMock, settings: Settings) -> None:
        OcrExtractor(settings)
        mock_genai.configure.assert_called_once_with(api_key=settings.gemini_api_key)

    @patch("pop_validation.gemini_client.genai")
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

        report = extractor._step3_quality_assessment(loaded, 0)
        assert report.is_receipt is True
        assert report.is_ai_generated is False
        assert report.rejection_reason is None

    @patch("pop_validation.gemini_client.genai")
    def test_assess_quality_error_returns_rejection(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = Exception("API error")

        extractor = OcrExtractor(settings)

        from PIL import Image

        from pop_validation.image_loader import LoadedImage

        img = Image.new("RGB", (100, 100))
        loaded = LoadedImage(data=b"fake", mime_type="image/jpeg", source="test", pil_image=img)

        report = extractor._step3_quality_assessment(loaded, 0)
        assert report.rejection_reason is not None
        assert "QUALITY_ASSESSMENT_FAILED" in report.rejection_reason

    @patch("pop_validation.gemini_client.genai")
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

        fields = extractor._step4_field_extraction(loaded, 0)
        assert fields is not None
        assert fields.retailer_name == "Foot Locker"
        assert fields.price == 149.99

    @patch("pop_validation.gemini_client.genai")
    def test_extract_fields_error_returns_none(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = Exception("API error")

        extractor = OcrExtractor(settings)

        from PIL import Image

        from pop_validation.image_loader import LoadedImage

        img = Image.new("RGB", (100, 100))
        loaded = LoadedImage(data=b"fake", mime_type="image/jpeg", source="test", pil_image=img)

        fields = extractor._step4_field_extraction(loaded, 0)
        assert fields is None

    @patch("pop_validation.gemini_client.genai")
    def test_analyze_returns_empty_on_load_failure(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_genai.GenerativeModel.return_value = MagicMock()

        extractor = OcrExtractor(settings)
        result = extractor.analyze("/nonexistent/path.jpg", 0)

        assert result.image_index == 0
        assert result.receipt_fields is None


# ──────────────────────────────────────────────
# Dependency Injection
# ──────────────────────────────────────────────


class TestOcrExtractorDI:
    @patch("pop_validation.gemini_client.genai")
    def test_custom_catalog_is_used(self, mock_genai: MagicMock, settings: Settings) -> None:
        """Injected catalog provider must be used instead of default."""
        mock_catalog = MagicMock()
        mock_catalog.get_products.return_value = ["Custom Product A", "Custom Product B"]

        extractor = OcrExtractor(settings, product_catalog=mock_catalog)

        # Verify catalog was called during init (for logging product count)
        mock_catalog.get_products.assert_called()

        # Verify the injected catalog is stored
        assert extractor._product_catalog is mock_catalog

    @patch("pop_validation.gemini_client.genai")
    def test_default_catalog_when_none(self, mock_genai: MagicMock, settings: Settings) -> None:
        """Default JsonFileProductCatalogProvider is used when no catalog injected."""
        from pop_validation.product_catalog import JsonFileProductCatalogProvider

        extractor = OcrExtractor(settings)
        assert isinstance(extractor._product_catalog, JsonFileProductCatalogProvider)
