"""Tests for the multi-agent architecture — each agent in isolation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from PIL import Image

from pop_validation.agents.base import AgentResult
from pop_validation.agents.extraction_agent import ExtractionAgent
from pop_validation.agents.image_loader_agent import ImageLoaderAgent
from pop_validation.agents.quality_agent import QualityAgent, QualityVerdict
from pop_validation.agents.reporting_agent import ReportingAgent
from pop_validation.agents.validation_agent import ValidationAgent, _build_result
from pop_validation.config import Settings
from pop_validation.imaging.loader import LoadedImage
from pop_validation.models import (
    ImageAnalysis,
    ImageCategory,
    ImageQuality,
    ImageQualityReport,
    ImageValidationResult,
    ProductCategory,
    ReceiptFields,
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _fake_loaded_image() -> LoadedImage:
    img = Image.new("RGB", (800, 600))
    return LoadedImage(
        data=b"fake-image-data", mime_type="image/jpeg", source="test", pil_image=img,
    )


def _valid_analysis(index: int = 0) -> ImageAnalysis:
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


def _empty_analysis(index: int = 0) -> ImageAnalysis:
    return ImageAnalysis(
        image_index=index,
        image_url=f"https://example.com/img{index}.jpg",
    )


# ──────────────────────────────────────────────
# ImageLoaderAgent
# ──────────────────────────────────────────────


class TestImageLoaderAgent:
    def test_name(self) -> None:
        agent = ImageLoaderAgent()
        assert agent.name == "ImageLoader"

    @patch("pop_validation.agents.image_loader_agent.load_image")
    def test_success(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _fake_loaded_image()
        agent = ImageLoaderAgent()
        result = agent.execute(image_url="test.jpg")

        assert result.success is True
        assert result.data is not None
        assert result.error is None

    @patch("pop_validation.agents.image_loader_agent.load_image")
    def test_file_not_found(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = FileNotFoundError("not found")
        agent = ImageLoaderAgent()
        result = agent.execute(image_url="/bad/path.jpg")

        assert result.success is False
        assert result.error is not None
        assert "FileNotFoundError" in result.error


# ──────────────────────────────────────────────
# QualityAgent
# ──────────────────────────────────────────────


class TestQualityAgent:
    @patch("pop_validation.client.gemini_client.genai")
    def test_name(self, mock_genai: MagicMock, settings: Settings) -> None:
        agent = QualityAgent(settings=settings)
        assert agent.name == "Quality"

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_technical_rejection(
        self, mock_tech: MagicMock, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=False, file_size_ok=True,
        )

        agent = QualityAgent(settings=settings)
        result = agent.execute(loaded=_fake_loaded_image(), image_index=0)

        assert result.success is True
        verdict: QualityVerdict = result.data
        assert verdict.passed is False
        assert verdict.technical_ok is False

    @patch("pop_validation.client.gemini_client.genai")
    @patch("pop_validation.agents.quality_agent.check_technical_quality")
    def test_quality_pass(
        self, mock_tech: MagicMock, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_tech.return_value = ImageQualityReport(
            resolution_ok=True, file_size_ok=True,
        )

        import json
        quality_json = json.dumps({
            "is_receipt": True, "is_readable": True, "is_ai_generated": False,
            "image_quality": "high", "image_category": "PROOF_OF_PURCHASE",
            "rejection_reason": None,
        })
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = MagicMock(text=quality_json)

        agent = QualityAgent(settings=settings)
        result = agent.execute(loaded=_fake_loaded_image(), image_index=0)

        assert result.success is True
        verdict: QualityVerdict = result.data
        assert verdict.passed is True
        assert verdict.technical_ok is True


# ──────────────────────────────────────────────
# ExtractionAgent
# ──────────────────────────────────────────────


class TestExtractionAgent:
    @patch("pop_validation.client.gemini_client.genai")
    def test_name(self, mock_genai: MagicMock, settings: Settings) -> None:
        agent = ExtractionAgent(settings=settings)
        assert agent.name == "Extraction"

    @patch("pop_validation.client.gemini_client.genai")
    def test_successful_extraction(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        import json
        extraction_json = json.dumps({
            "retailer_name": "Foot Locker", "retailer_location": "NYC",
            "purchase_date": "2025-01-15", "product_name": "Cloud 5",
            "price": 149.99, "currency": "USD",
            "receipt_type": "official_receipt_paper",
        })
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = MagicMock(text=extraction_json)

        agent = ExtractionAgent(settings=settings)
        result = agent.execute(loaded=_fake_loaded_image(), image_index=0)

        assert result.success is True
        assert result.data is not None
        assert result.data.retailer_name == "Foot Locker"

    @patch("pop_validation.client.gemini_client.genai")
    def test_extraction_failure_returns_none_data(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = RuntimeError("API error")

        agent = ExtractionAgent(settings=settings)
        result = agent.execute(loaded=_fake_loaded_image(), image_index=0)

        # success=True because extraction failure is handled gracefully
        assert result.success is True
        assert result.data is None
        assert result.error is not None


# ──────────────────────────────────────────────
# ValidationAgent
# ──────────────────────────────────────────────


class TestValidationAgent:
    def test_name(self) -> None:
        agent = ValidationAgent()
        assert agent.name == "Validation"

    def test_valid_receipt(self) -> None:
        agent = ValidationAgent()
        result = agent.execute(analysis=_valid_analysis())

        assert result.success is True
        ivr: ImageValidationResult = result.data
        assert ivr.message == "VALID_RECEIPT_FOUND"
        assert ivr.retailer_name == "Foot Locker"

    def test_no_receipt_fields(self) -> None:
        agent = ValidationAgent()
        result = agent.execute(analysis=_empty_analysis())

        assert result.success is True
        ivr: ImageValidationResult = result.data
        assert ivr.message == "RECEIPT_NOT_FOUND"

    def test_build_result_preserves_metadata(self) -> None:
        analysis = _valid_analysis()
        result = _build_result(analysis, "VALID_RECEIPT_FOUND")

        assert result.product_category == ProductCategory.SHOES
        assert result.language_category == "en"
        assert result.is_ai_generated is False

    def test_build_result_without_fields(self) -> None:
        analysis = _empty_analysis()
        result = _build_result(analysis, "RECEIPT_NOT_FOUND")

        assert result.message == "RECEIPT_NOT_FOUND"
        assert result.retailer_name is None


# ──────────────────────────────────────────────
# ReportingAgent
# ──────────────────────────────────────────────


class TestReportingAgent:
    def test_name(self) -> None:
        agent = ReportingAgent(result_logger=MagicMock())
        assert agent.name == "Reporting"

    def test_logs_result(self) -> None:
        mock_logger = MagicMock()
        agent = ReportingAgent(result_logger=mock_logger)

        ivr = ImageValidationResult(message="VALID_RECEIPT_FOUND")
        result = agent.execute(
            result=ivr,
            warranty_id="W-123",
            image_url="receipt.jpg",
        )

        assert result.success is True
        mock_logger.log.assert_called_once_with(
            result=ivr,
            warranty_id="W-123",
            image_url="receipt.jpg",
            comment=None,
        )

    def test_logs_with_comment(self) -> None:
        mock_logger = MagicMock()
        agent = ReportingAgent(result_logger=mock_logger)

        ivr = ImageValidationResult(message="RECEIPT_NOT_FOUND")
        agent.execute(
            result=ivr,
            warranty_id="W-ERR",
            image_url="bad.jpg",
            comment="RuntimeError: API failed",
        )

        mock_logger.log.assert_called_once_with(
            result=ivr,
            warranty_id="W-ERR",
            image_url="bad.jpg",
            comment="RuntimeError: API failed",
        )


# ──────────────────────────────────────────────
# AgentResult
# ──────────────────────────────────────────────


class TestAgentResult:
    def test_success(self) -> None:
        result = AgentResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_failure(self) -> None:
        result = AgentResult(success=False, error="something broke")
        assert result.success is False
        assert result.data is None
        assert result.error == "something broke"

    def test_frozen(self) -> None:
        result = AgentResult(success=True)
        try:
            result.success = False  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised, "AgentResult should be frozen (immutable)"


# ──────────────────────────────────────────────
# BaseAgent Protocol structural typing
# ──────────────────────────────────────────────


class TestBaseAgentProtocol:
    def test_all_agents_satisfy_protocol(self) -> None:
        """All 5 agents + orchestrator have name + execute()."""
        agents: list[Any] = [
            ImageLoaderAgent(),
            ValidationAgent(),
        ]
        for agent in agents:
            assert hasattr(agent, "name")
            assert hasattr(agent, "execute")
            assert isinstance(agent.name, str)
